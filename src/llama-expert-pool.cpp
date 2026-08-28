#include "llama-expert-pool.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "llama-graph.h"   // llm_graph_result (mini chain owner, deleted in reset)

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>

// ---- per-layer chain param registry (see llama-expert-pool.h) ----
namespace {
    std::vector<llama_chain_params> g_chain; // indexed by layer id
}

void llama_expert_pool_register_chain(int il, int type_op, bool norm_w, float w_scale, uint32_t gating_op) {
    if (il < 0) {
        return;
    }
    if ((size_t) il >= g_chain.size()) {
        g_chain.resize(il + 1);
    }
    g_chain[il] = { type_op, norm_w, w_scale, gating_op };
}

const llama_chain_params & llama_expert_pool_get_chain(int il) {
    static const llama_chain_params none {};
    if (il < 0 || (size_t) il >= g_chain.size() || g_chain[il].type_op < 0) {
        return none;
    }
    return g_chain[il];
}

void llama_expert_pool_clear_chain() {
    g_chain.clear();
}

// ---- direct-mount registry (see llama-expert-pool.h) ----
namespace {
    std::vector<llama_expert_pool_mount> g_mount; // indexed by layer id
}

void llama_expert_pool_register_mount(int il, const llama_expert_pool_mount & mount) {
    if (il < 0) {
        return;
    }
    if ((size_t) il >= g_mount.size()) {
        g_mount.resize(il + 1);
    }
    g_mount[il] = mount;
}

llama_expert_pool_mount & llama_expert_pool_get_mount(int il) {
    // auto-grow (like register_chain): a returned reference must be the real
    // per-layer cell, not a shared static - callers write fields into it
    // (expert_pool_init registration loop fills the mount in place)
    static llama_expert_pool_mount none {};
    if (il < 0) {
        return none;
    }
    if ((size_t) il >= g_mount.size()) {
        g_mount.resize(il + 1);
    }
    return g_mount[il];
}

void llama_expert_pool_clear_mount() {
    g_mount.clear();
}

void llama_expert_pool::init(int32_t n_layer, int32_t n_expert, int32_t n_slot, int64_t expert_size) {
    enabled   = true;
    this->n_layer   = n_layer;
    this->n_expert  = n_expert;
    this->n_slot    = n_slot;
    this->expert_size = expert_size;

    expert_to_slot.assign(n_layer, std::vector<int32_t>(n_expert, -1));
    slot_expert.assign(n_layer, std::vector<int32_t>(n_slot, -1));
    n_resident.assign(n_layer, 0);
}

int32_t llama_expert_pool::slot_of(int32_t il, int32_t expert) const {
    if (!enabled || il < 0 || il >= n_layer || expert < 0 || expert >= n_expert) {
        return -1;
    }
    return expert_to_slot[il][expert];
}

int32_t llama_expert_pool::expert_in_slot(int32_t il, int32_t slot) const {
    if (!enabled || il < 0 || il >= n_layer || slot < 0 || slot >= n_slot) {
        return -1;
    }
    return slot_expert[il][slot];
}

bool llama_expert_pool::is_resident(int32_t il, int32_t expert) const {
    return slot_of(il, expert) >= 0;
}

llama_expert_pool::swap_plan llama_expert_pool::plan_swap(const std::vector<std::vector<int32_t>> & topk) const {
    swap_plan plan;

    for (int32_t il = 0; il < n_layer; ++il) {
        if (il >= (int32_t) topk.size()) {
            continue;
        }
        const std::vector<int32_t> & want = topk[il];

        // evict residents that are no longer wanted
        for (int32_t s = 0; s < n_slot; ++s) {
            const int32_t e = slot_expert[il][s];
            if (e < 0) {
                continue;
            }
            const bool still_wanted = std::find(want.begin(), want.end(), e) != want.end();
            if (!still_wanted) {
                plan.evict_il.push_back(il);
                plan.evict_ex.push_back(e);
            }
        }

        // fill missing experts, reusing slots freed by eviction first.
        // a slot is free when empty or when its resident is being evicted
        // (stable reuse keeps the same slots across successive repins).
        std::vector<int32_t> free_slots;
        for (int32_t s = 0; s < n_slot; ++s) {
            const int32_t e = slot_expert[il][s];
            const bool to_evict = e >= 0 && std::find(want.begin(), want.end(), e) == want.end();
            if (e < 0 || to_evict) {
                free_slots.push_back(s);
            }
        }
        for (int32_t e : want) {
            if (expert_to_slot[il][e] >= 0) {
                continue; // already resident (check the entry, not the slot values)
            }
            if (free_slots.empty()) {
                break; // no room (should not happen: |want| <= n_slot)
            }
            const int32_t s = free_slots.back();
            free_slots.pop_back();
            plan.fill_il.push_back(il);
            plan.fill_ex.push_back(e);
        }
    }

    // swap cost = H2D copies only (evictions are LUT updates, the CPU keeps
    // the original weights), so the delta is the fill set size.
    plan.delta = (int32_t) plan.fill_il.size();
    return plan;
}

bool llama_expert_pool::gate_should_swap(float h_inc, float c_cand, int32_t delta,
                                         float cpu_ms, float swap_ms, float L) {
    if (delta <= 0) {
        return false;
    }
    const float benefit = (c_cand - h_inc) * cpu_ms;
    const float cost    = delta * swap_ms / L;
    return benefit > cost;
}

void llama_expert_pool::begin_swap(const swap_plan & plan) {
    // evict: unmap every expert scheduled for eviction
    for (size_t i = 0; i < plan.evict_il.size(); ++i) {
        const int32_t il = plan.evict_il[i];
        const int32_t e  = plan.evict_ex[i];
        const int32_t s  = expert_to_slot[il][e];
        if (s >= 0) {
            expert_to_slot[il][e] = -1;
            slot_expert[il][s]    = -1;
            if (n_resident[il] > 0) {
                --n_resident[il];
            }
        }
    }
}

void llama_expert_pool::commit_slot(int32_t il, int32_t slot, int32_t expert) {
    if (!enabled || il < 0 || il >= n_layer || slot < 0 || slot >= n_slot ||
        expert < 0 || expert >= n_expert) {
        return;
    }
    // slot must be empty; if it still holds another expert, unmap that one first
    const int32_t old = slot_expert[il][slot];
    if (old >= 0) {
        expert_to_slot[il][old] = -1;
    } else {
        ++n_resident[il];
    }
    slot_expert[il][slot] = expert;
    expert_to_slot[il][expert] = slot;
}

int32_t llama_expert_pool::total_resident() const {
    int32_t total = 0;
    for (int32_t r : n_resident) {
        total += r;
    }
    return total;
}

void llama_expert_pool_state::reset() {
    enabled = false;
    direct_mount = false;
    rtlog_only = false;
    w_pool_gate_up.clear();
    w_pool_up.clear();
    w_pool_gate.clear();
    w_pool_down.clear();
    resident.clear();
    pooled_layers.clear();

    // runtime routing log cleanup (GGML_EXPPOOL_ROUTING_LOG)
    if (rt_log != nullptr) {
        fclose(rt_log);
    }
    rt_log = nullptr;
    rt_log_tried = false;
    log_step = 0;
    logged_il = -1;
    rt_step_done = false;

    swap_auto = false;
    win_step = 0;
    win_cnt.clear();
    win_hist.clear();
    stat_hit.clear();
    stat_miss.clear();
    win_hit  = 0;
    win_miss = 0;
}

// parse seed csv: one line per layer "il,e1,e2,...". layers missing from the
// file get zero slots (full CPU fallback); the per-layer slot count is the
// line length, i.e. the pool file itself defines the budget distribution.
// returns false only on open failure.
bool llama_expert_pool_parse_init(const std::string & path, int32_t n_layer,
                                  int32_t n_expert,
                                  std::vector<std::vector<int32_t>> & resident) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    resident.assign(n_layer, std::vector<int32_t>());
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        // skip comment/empty lines
        if (tok.empty() || tok[0] == '#') {
            continue;
        }
        const int32_t il = std::atoi(tok.c_str());
        if (il < 0 || il >= n_layer) {
            continue;
        }
        std::vector<int32_t> experts;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) {
                continue;
            }
            int32_t e = std::atoi(tok.c_str());
            if (e == -1) {
                // explicit empty slot: keeps a zero slice in the pool table
                // (weight copy skips non-resident entries). maps nothing.
                experts.push_back(e);
                continue;
            }
            if (e < 0 || e >= n_expert) {
                continue;
            }
            experts.push_back(e);
        }
        // dedup, keep order (the file writer emits already-deduped lists)
        std::vector<int32_t> dedup;
        for (int32_t e : experts) {
            if (std::find(dedup.begin(), dedup.end(), e) == dedup.end()) {
                dedup.push_back(e);
            }
        }
        resident[il] = dedup;
    }
    return true;
}

// random resident set per layer (fixed seed for reproducibility); samples
// without replacement so every slot holds a distinct expert
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert, int32_t n_slot,
                              std::vector<std::vector<int32_t>> & resident) {
    if (n_slot > n_expert) {
        n_slot = n_expert;
    }
    resident.assign(n_layer, std::vector<int32_t>(n_slot, -1));
    std::mt19937 rng(0);
    std::vector<int32_t> perm(n_expert);
    for (int32_t e = 0; e < n_expert; ++e) {
        perm[e] = e;
    }
    for (int32_t il = 0; il < n_layer; ++il) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int32_t s = 0; s < n_slot; ++s) {
            resident[il][s] = perm[s];
        }
    }
}
// -----------------------------------------------------------------------------
// moe routing-log hook: called by the CPU MUL_MAT_ID kernel (ith==0) before
// row grouping. collects NO rows (nothing is skipped: the -1 ids zero the
// columns natively since PR #26631, both chains merge in the main graph);
// it only feeds GGML_EXPPOOL_ROUTING_LOG.
// -----------------------------------------------------------------------------

void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) ud;
    *skip_out = nullptr;
    if (st.pooled_layers.empty()) {
        return;
    }
    if (!st.rtlog_only && !st.direct_mount) {
        return;
    }

    // find the pooled layer/matrix this node corresponds to (ilx = index into
    // pooled_layers; il = actual layer number)
    int32_t il = -1;
    int32_t ilx = -1;
    for (size_t ix = 0; ix < st.pooled_layers.size(); ++ix) {
        const int32_t i = st.pooled_layers[ix];
        if      (st.orig_gate_up[i] == src0) { il = i; ilx = (int32_t) ix; break; }
        else if (st.orig_up[i]       == src0) { il = i; ilx = (int32_t) ix; break; }
        else if (st.orig_gate[i]     == src0) { il = i; ilx = (int32_t) ix; break; }
        else if (st.orig_down[i]     == src0) { il = i; ilx = (int32_t) ix; break; }
    }
    if (il < 0) {
        return;
    }
    // log only the layers that have an active mount (direct mount) or all
    // pooled layers in routing-log-only mode
    if (st.direct_mount) {
        const llama_expert_pool_mount & mnt = llama_expert_pool_get_mount(il);
        if (!mnt.active) {
            return;
        }
    }
    // single decode rows only (the log line is one token's ids)
    if (ids->ne[1] != 1) {
        return;
    }
    // --- runtime routing log (env-gated, B=1 decode rows only): write the
    // expert ids that reached this kernel, one line per (step, pooled layer).
    // step counting: the first begin of each step is the first pooled layer
    // (up fires before gate/down in build order), so ilx==0 starts a new step.
    if (!st.rt_log_tried) {
        st.rt_log_tried = true;
        const char * lp = getenv("GGML_EXPPOOL_ROUTING_LOG");
        if (lp != nullptr && lp[0] != '\0') {
            st.rt_log = fopen(lp, "w");
            if (st.rt_log != nullptr) {
                fprintf(st.rt_log, "step,layer,expert_ids\n");
            }
        }
    }
    // lazy window allocation: pooled_layers must be known (first decode row)
    if (st.swap_auto && st.win_cnt.empty() && !st.pooled_layers.empty()) {
        st.win_cnt.assign((size_t) st.pooled_layers.size() * st.n_expert, 0);
        st.win_hist.resize(st.swap_W);
    }
    if (st.stat_hit.empty() && !st.pooled_layers.empty()) {
        st.stat_hit.assign(st.pooled_layers.size(), 0);
        st.stat_miss.assign(st.pooled_layers.size(), 0);
    }
    // step-advance detection, independent of the routing log: the first
    // pooled layer of a step begins after the last layer of the previous
    // one (rt_step_done is set at the end of this hook)
    if (ilx == 0 && st.rt_step_done) {
        st.rt_step_done = false;
        if (st.swap_auto && !st.win_cnt.empty()) {
            if (st.win_step >= st.swap_W && st.win_step % st.swap_W == 0) {
                llama_expert_pool_run_swap(st);
            }
            if (st.win_step >= st.swap_W) {
                // evict the oldest step before its slot is reused
                std::vector<int32_t> & old = st.win_hist[st.win_step % st.swap_W];
                for (size_t i = 0; i + 1 < old.size(); i += 2) {
                    st.win_cnt[old[i] * st.n_expert + old[i+1]] --;
                }
                old.clear();
            }
            st.win_step += 1;
        }
    }
    if (st.rt_log != nullptr) {
        // new decode step when the FIRST pooled layer logs again after the
        // LAST one did (a layer-id-change test breaks with one pooled layer)
        if (ilx == 0 && st.rt_step_done) {
            st.log_step += 1;
            st.logged_il = -1;
            // step boundary: flush the PREVIOUS step's lines (1 syscall/step,
            // vs per-line fflush which cost measurable time on the hot path)
            fflush(st.rt_log);
        }
        if (st.logged_il != il) {
            st.logged_il = il;
            fprintf(st.rt_log, "%llu,%d", (unsigned long long) st.log_step, il);
            for (int id = 0; id < (int) ids->ne[0]; ++id) {
                const int32_t e = *((const int32_t *) ((const char *) ids->data + id*ids->nb[0]));
                fprintf(st.rt_log, ",%d", e);
            }
            fputc('\n', st.rt_log);
        }
    }

    // --- stage 3 swap window: count this row's expert ids ---
    if (st.swap_auto && !st.win_cnt.empty()) {
        std::vector<int32_t> & hist = st.win_hist[st.win_step % st.swap_W];
        for (int id = 0; id < (int) ids->ne[0]; ++id) {
            const int32_t e = *((const int32_t *) ((const char *) ids->data + id*ids->nb[0]));
            if (e < 0 || e >= st.n_expert) {
                continue;
            }
            st.win_cnt[ilx * st.n_expert + e] ++;
            hist.push_back(ilx);
            hist.push_back(e);
        }
    }
    // hit/miss counters (direct mount: ids come from remap_cpu, so -1 is a GPU
    // pool hit and a non-negative id is the expert computed on the CPU). idle
    // layers (active=false) are skipped by the mount gate above.
    if (st.direct_mount) {
        for (int id = 0; id < (int) ids->ne[0]; ++id) {
            const int32_t e = *((const int32_t *) ((const char *) ids->data + id*ids->nb[0]));
            if (e < 0) {
                st.stat_hit[ilx] ++;
                st.win_hit ++;
            } else if (e < st.n_expert) {
                st.stat_miss[ilx] ++;
                st.win_miss ++;
            }
        }
    }
    // the last pooled layer completes the step (single-layer-safe detection:
    // a step with only one pooled layer sets the flag on its own hook)
    if (ilx == (int32_t) st.pooled_layers.size() - 1) {
        st.rt_step_done = true;
    }
}

// ---------------------------------------------------------------
// stage 3: sliding-window resident-set refresh (--expert-pool-swap)

namespace {

void swap_write_tables(llama_expert_pool_state & st, int32_t il,
                       const std::vector<int32_t> & res,
                       const std::vector<int32_t> & pending_fill) {
    llama_expert_pool_mount & m = llama_expert_pool_get_mount(il);
    if (!m.active || m.remap == nullptr || m.remap_cpu == nullptr) {
        return;
    }
    const int32_t n_expert = st.n_expert;
    // resident -> slot; pending_fill entries stay unmapped (-1) so the hot
    // path falls back to the CPU chain until their weight copies land
    std::vector<int32_t> rmp(n_expert, -1);
    for (int32_t s = 0; s < (int32_t) res.size(); ++s) {
        const int32_t e = res[s];
        if (e >= 0 && e < n_expert) {
            rmp[e] = s;
        }
    }
    for (int32_t e : pending_fill) {
        rmp[e] = -1;
    }
    std::vector<int32_t> rmp_cpu(n_expert, -1);
    for (int32_t e = 0; e < n_expert; ++e) {
        rmp_cpu[e] = rmp[e] >= 0 ? -1 : e;
    }
    ggml_backend_tensor_set(m.remap,     rmp.data(),     0, n_expert * sizeof(int32_t));
    ggml_backend_tensor_set(m.remap_cpu, rmp_cpu.data(), 0, n_expert * sizeof(int32_t));
}

void swap_copy_one(ggml_tensor * src, ggml_tensor * pw, int32_t e, int32_t slot) {
    if (src == nullptr || pw == nullptr) {
        return;
    }
    const size_t sz = src->nb[2];
    ggml_backend_tensor_set(pw, (const char *) src->data + e * src->nb[2],
            slot * pw->nb[2], sz);
}

} // namespace

void llama_expert_pool_run_swap(llama_expert_pool_state & st) {
    const int32_t P = (int32_t) st.pooled_layers.size();
    const int32_t n_expert = st.n_expert;
    if (P <= 0 || st.win_cnt.empty()) {
        return;
    }

    std::vector<int32_t> idx(n_expert);
    std::vector<int32_t> topk;
    int32_t delta = 0;

    for (int32_t ilx = 0; ilx < P; ++ilx) {
        const int32_t il = st.pooled_layers[ilx];
        std::vector<int32_t> & res = st.resident[il];
        const int32_t K = (int32_t) res.size();
        if (K <= 0) {
            continue;
        }
        // top-K of the sliding-window activation count on this layer
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                [&](int32_t a, int32_t b) {
                    return st.win_cnt[ilx * n_expert + a] > st.win_cnt[ilx * n_expert + b];
                });
        topk.assign(idx.begin(), idx.begin() + K);

        // diff against the resident set; keep slots stable for surviving experts
        const auto in_topk = [&](int32_t e) {
            return std::find(topk.begin(), topk.end(), e) != topk.end();
        };
        std::vector<int32_t> evict;
        for (int32_t e : res) {
            if (!in_topk(e)) {
                evict.push_back(e);
            }
        }
        std::vector<int32_t> fill;
        for (int32_t e : topk) {
            if (std::find(res.begin(), res.end(), e) == res.end()) {
                fill.push_back(e);
            }
        }
        if (fill.empty()) {
            continue;
        }

        // new layout: survivors keep their slots, fills take the evicted ones
        std::vector<int32_t> new_res(K, -1);
        for (int32_t e : res) {
            if (in_topk(e)) {
                for (int32_t s = 0; s < K; ++s) {
                    if (res[s] == e) {
                        new_res[s] = e;
                        break;
                    }
                }
            }
        }
        size_t fi = 0;
        for (int32_t s = 0; s < K && fi < fill.size(); ++s) {
            if (new_res[s] < 0) {
                new_res[s] = fill[fi++];
            }
        }

        // safe swap sequence: unmap fills -> copy weights -> commit
        swap_write_tables(st, il, new_res, fill);
        for (int32_t e : fill) {
            int32_t slot = -1;
            for (int32_t s = 0; s < K; ++s) {
                if (new_res[s] == e) {
                    slot = s;
                    break;
                }
            }
            if (slot < 0) {
                continue;
            }
            swap_copy_one(st.orig_gate_up[il], st.w_pool_gate_up[il], e, slot);
            swap_copy_one(st.orig_up[il],      st.w_pool_up[il],      e, slot);
            swap_copy_one(st.orig_gate[il],    st.w_pool_gate[il],    e, slot);
            swap_copy_one(st.orig_down[il],    st.w_pool_down[il],    e, slot);
        }
        swap_write_tables(st, il, new_res, {});

        res.swap(new_res);
        delta += (int32_t) fill.size();
    }

    LLAMA_LOG_INFO("%s: swapped %d expert slots (step %d)\n", __func__, delta, st.win_step);
    if (st.win_hit + st.win_miss > 0) {
        LLAMA_LOG_INFO("%s: pool hit rate %.1f%% (%llu/%llu rows, swap window %d)\n", __func__,
                100.0 * st.win_hit / (double) (st.win_hit + st.win_miss),
                (unsigned long long) st.win_hit,
                (unsigned long long) (st.win_hit + st.win_miss), st.swap_W);
    }
    // a new window starts at the next swap
    st.win_hit  = 0;
    st.win_miss = 0;
}
