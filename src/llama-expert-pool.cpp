#include "llama-expert-pool.h"

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
    static llama_expert_pool_mount none {};
    if (il < 0 || (size_t) il >= g_mount.size()) {
        return none;
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
    delegate_ok = false;
    direct_mount = false;
    rtlog_only = false;
    w_pool_gate_up.clear();
    w_pool_up.clear();
    w_pool_gate.clear();
    w_pool_down.clear();
    resident.clear();
    pooled_layers.clear();
    slots.clear();
    delegate_ok = false;
    gpu_backend = nullptr;
    // free the per-layer chain graphs and their scratch buffers (built in
    // expert_pool_fill via build_moe_ffn(chain_only))
    for (auto & x : mini) {
        for (auto & m : x) {
            delete (llm_graph_result *) m.gres;
        }
    }
    mini.clear();
    for (ggml_backend_buffer_t b : layer_bufs) {
        ggml_backend_buffer_free(b);
    }
    layer_bufs.clear();
    layer_out_down.clear();

    async_d2h = getenv("GGML_EXPPOOL_ASYNC_D2H") != nullptr && getenv("GGML_EXPPOOL_ASYNC_D2H")[0] == '1';
    dg_buf = nullptr;
    t_cur = nullptr;
    t_out = nullptr;
    t_ids = nullptr;
    n_used = 0;
    n_hit = 0;
    hit_slots.clear();
    hit_cols.clear();

    // runtime routing log cleanup (GGML_EXPPOOL_ROUTING_LOG)
    if (rt_log != nullptr) {
        fclose(rt_log);
    }
    rt_log = nullptr;
    rt_log_tried = false;
    log_step = 0;
    logged_il = -1;
    rt_step_done = false;
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

// random resident set per layer (fixed seed for reproducibility)
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert, int32_t n_slot,
                              std::vector<std::vector<int32_t>> & resident) {
    resident.assign(n_layer, std::vector<int32_t>(n_slot, -1));
    std::mt19937 rng(0);
    for (int32_t il = 0; il < n_layer; ++il) {
        for (int32_t s = 0; s < n_slot; ++s) {
            resident[il][s] = (int32_t) (rng() % n_expert);
        }
    }
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// moe delegate: take over cache-hit expert rows inside the CPU MUL_MAT_ID kernel
//
// begin(): called by the CPU kernel (ith==0) before row grouping. Scans the ids,
// collects hit (slot >= 0) rows, submits ONE batched matvec on the pool weights
// to the GPU (async), returns true so the kernel skips the mask==0 rows.
// end(): called after the local row loop (ith==0); waits for the GPU, copies the
// hit results back into the CPU dst rows.
// -----------------------------------------------------------------------------

static inline ggml_tensor * pool_w_for(const llama_expert_pool_state & st, int32_t il, int which) {
    switch (which) {
        case 0: return st.w_pool_gate_up[il];
        case 1: return st.w_pool_up[il];
        case 2: return st.w_pool_gate[il];
        default: return st.w_pool_down[il];
    }
}

void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) ud;
    *skip_out = nullptr;
    const int64_t dbg_t0 = ggml_time_us();
    if (st.pooled_layers.empty()) {
        return;
    }
    if (!st.delegate_ok && !st.rtlog_only) {
        return;
    }

    // find the pooled layer/matrix this node corresponds to (ilx = index into
    // pooled_layers; il = actual layer number)
    int32_t il = -1;
    int32_t ilx = -1;
    int which = -1;
    for (size_t ix = 0; ix < st.pooled_layers.size(); ++ix) {
        const int32_t i = st.pooled_layers[ix];
        if      (st.orig_gate_up[i] == src0) { il = i; ilx = (int32_t) ix; which = 0; break; }
        else if (st.orig_up[i]       == src0) { il = i; ilx = (int32_t) ix; which = 1; break; }
        else if (st.orig_gate[i]     == src0) { il = i; ilx = (int32_t) ix; which = 2; break; }
        else if (st.orig_down[i]     == src0) { il = i; ilx = (int32_t) ix; which = 3; break; }
    }
    if (il < 0) {
        return;
    }
    // delegation targets the decode shape only (one token per call). larger
    // batches that stay on the CPU (e.g. speculative-verification, below the
    // scheduler's offload_min_batch) would be silently corrupted: the skip
    // table is per-expert while the mini graph carries a single token, so
    // non-first tokens' hit rows would be skipped and never computed. bail
    // out to the full CPU path for those. (must sit before the
    // dst_ilx_which insert below so end() ignores the node entirely.)
    // the routing-log block above must stay ahead of this check: ids rows are
    // only valid per single token, so B>1 calls must not be logged either.
    if (ids->ne[1] != 1) {
        return;
    }
    // --- runtime routing log (env-gated, B=1 decode rows only): write the raw
    // expert ids the delegate actually sees, one line per (step, pooled layer).
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
    if (st.rt_log != nullptr) {
        // new decode step when the FIRST pooled layer logs again after the
        // LAST one did (a layer-id-change test breaks with one pooled layer)
        if (ilx == 0 && st.rt_step_done) {
            st.log_step += 1;
            st.logged_il = -1;
            st.rt_step_done = false;
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
            if (ilx == (int32_t) st.pooled_layers.size() - 1) {
                st.rt_step_done = true;
            }
        }
    }
    // routing-log-only mode: the ids are logged above; delegate state stays
    // untouched (no slots exist to consult, nothing to submit or skip)
    if (st.rtlog_only) {
        return;
    }
    // --- direct-mount mode: no mini graph, no event, no D2H. the GPU chain
    // runs inside the main graph itself (build_moe_ffn mount branch); this
    // hook only zeroes the columns the CPU kernel will skip and hands out
    // the skip table so each column is computed exactly once.
    if (st.direct_mount) {
        // decode batches only (matches the main-graph mount gate on
        // n_tokens==1): larger batches keep the pure CPU chain, their ids are
        // [n_used, T] and the single-token scan/zeroing here would corrupt
        // both the skip table and unrelated dst regions
        if (!st.delegate_ok || ids->ne[1] != 1) {
            return;
        }
        const llama_expert_pool_mount & mnt = llama_expert_pool_get_mount(il);
        if (!mnt.active) {
            return; // this tensor's matrix has no mount: full CPU compute
        }
        // scan ids -> hit slots
        st.n_hit = 0;
        if (st.slots[il].empty()) {
            return;
        }
        if ((int) st.hit_slots.size() < (int) ids->ne[0]) {
            st.hit_slots.resize(ids->ne[0]);
            st.hit_cols.resize(ids->ne[0]);
        }
        for (int id = 0; id < (int) ids->ne[0]; ++id) {
            const int32_t e = *((const int32_t *) ((const char *) ids->data + id*ids->nb[0]));
            if (e < 0 || e >= (int32_t) st.slots[il].size()) {
                return;
            }
            const int32_t s = st.slots[il][e];
            if (s >= 0) {
                st.hit_slots[st.n_hit] = s;
                st.hit_cols[st.n_hit]  = id;
                st.n_hit += 1;
            }
        }
        if (st.n_hit == 0) {
            // no hit: the CPU kernel computes all rows (count once per layer)
            if (st.mini_submitted_il != il) {
                if ((size_t) ilx < st.s_layer.size()) {
                    st.s_layer[ilx].miss_rows += (uint64_t) ids->ne[0];
                }
                st.mini_submitted_il = il;
            }
            return;
        }
        // the in-graph merge reads every column of dst, but the kernel leaves
        // skipped rows holding stale arena data -> zero them so the masked
        // merge takes exactly the mounted chain's output
        const size_t rowb = (size_t) dst->ne[0] * sizeof(float);
        char * dbase = (char *) dst->data;
        for (int i = 0; i < st.n_hit; ++i) {
            memset(dbase + (size_t) st.hit_cols[i]*dst->nb[1], 0, rowb);
        }
        if ((size_t) ilx < st.s_layer.size()) {
            st.s_layer[ilx].submits   += 1;
            st.s_layer[ilx].hit_rows  += (uint64_t) st.n_hit;
            st.s_layer[ilx].miss_rows += (uint64_t) (ids->ne[0] - st.n_hit);
        }
        *skip_out = st.slots[il].data();
        return;
    }

    // remember (ilx, which) for end() to map a dst back to its output region
    st.dst_ilx_which[dst] = std::make_pair(ilx, which);

    ggml_tensor * pw = pool_w_for(st, il, which);
    if (pw == nullptr) {
        return;
    }

    // scan ids -> hit slots
    st.n_hit = 0;
    if (st.slots[il].empty()) {
        return;
    }
    if ((int) st.hit_slots.size() < (int) ids->ne[0]) {
        st.hit_slots.resize(ids->ne[0]);
        st.hit_cols.resize(ids->ne[0]);
    }
    for (int id = 0; id < (int) ids->ne[0]; ++id) {
        const int32_t e = *((const int32_t *) ((const char *) ids->data + id*ids->nb[0]));
        if (e < 0 || e >= (int32_t) st.slots[il].size()) {
            return;
        }
        const int32_t s = st.slots[il][e];
        if (s >= 0) {
            st.hit_slots[st.n_hit] = s;
            st.hit_cols[st.n_hit]  = id;
            st.n_hit += 1;
        }
    }
    if (st.n_hit == 0) {
        // no hit: the CPU kernel computes all rows. record the miss count
        // exactly once per layer (both gate/up nodes call begin; the second
        // begin of the layer must not double-count).
        if (st.mini_submitted_il != il) {
            if ((size_t) ilx < st.s_layer.size()) {
                st.s_layer[ilx].miss_rows += (uint64_t) ids->ne[0];
            }
            st.mini_submitted_il = il;
        }
        // nothing was delegated this call; clear a possible stale submit flag
        // so a hit on a later step cannot be answered with old GPU output
        st.mini_submitted = false;
        return; // nothing to delegate
    }
    if (st.host_buf == nullptr) {
        return; // pinned mirrors unavailable; fall back to full CPU compute
    }

    const llama_expert_pool_state::mini_graph_entry & m = st.mini[ilx][0];
    if (m.g == nullptr) {
        st.n_hit = 0;
        return;
    }
    // a new layer starts a new submit round: reset BEFORE deciding whether
    // this layer has to submit. the whole-chain graph is submitted ONCE per
    // layer, by whichever MoE kernel fires first this layer (up is always
    // first in build order); later begins only scan and return skip tables.
    if (st.mini_submitted_il != il) {
        st.mini_submitted = false;
    }
    if (st.mini_submitted) {
        *skip_out = st.slots[il].data();
        return;
    }

    {
        const int64_t dbg_t1 = ggml_time_us();
        // input copy: token activations (src1) into t_cur; the down stage's
        // swiglu input is produced on-GPU by this same graph (no H2D needed)
        ggml_backend_tensor_get(src1, st.host_cur, 0, ggml_nbytes(src1));
        ggml_backend_tensor_set(st.t_cur, st.host_cur, 0, ggml_nbytes(st.t_cur));
        const int64_t dbg_t2 = ggml_time_us();
        // hit slots -> ids scratch (padding zeros; extra columns are ignored via n_hit)
        memset(st.host_ids, 0, (size_t) st.n_used * sizeof(int32_t));
        memcpy(st.host_ids, st.hit_slots.data(), (size_t) st.n_hit * sizeof(int32_t));
        ggml_backend_tensor_set(st.t_ids, st.host_ids, 0, (size_t) st.n_used * sizeof(int32_t));
        const int64_t dbg_t3 = ggml_time_us();

        const ggml_status gstatus = ggml_backend_graph_compute_async(st.gpu_backend, m.g);
        if (gstatus != GGML_STATUS_SUCCESS) {
            st.n_hit = 0;
            return;
        }
        ggml_backend_event_record(st.ev, st.gpu_backend);
        st.mini_submitted = true;
        st.mini_submitted_il = il;

        // accumulate per-layer statistics (delegate runs on the ith==0 thread only)
        if ((size_t) ilx < st.s_layer.size()) {
            st.s_layer[ilx].submits   += 1;
            st.s_layer[ilx].hit_rows  += (uint64_t) st.n_hit;
            st.s_layer[ilx].miss_rows += (uint64_t) (ids->ne[0] - st.n_hit);
            st.s_layer[ilx].getset_us += (uint64_t) (dbg_t2 - dbg_t1);
            st.s_layer[ilx].ids_us    += (uint64_t) (dbg_t3 - dbg_t2);
            st.s_layer[ilx].comp_us   += (uint64_t) (ggml_time_us() - dbg_t3);
        }
    }

    // give the CPU kernel the slot table: experts with slot >= 0 are cache hits
    *skip_out = st.slots[il].data();
}

void llama_expert_pool_delegate_end(ggml_tensor * dst, void * ud) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) ud;
    if (st.direct_mount) {
        st.n_hit = 0; // merge happened in-graph; nothing to write back
        return;
    }
    if (st.n_hit == 0 || !st.delegate_ok) {
        return;
    }
    // map this dst back to (ilx, which); unknown dst -> nothing to write back.
    // with the whole-chain mini graph the ONLY end that does work is the down
    // kernel's (gate/up hit results live entirely on-GPU now); every other
    // end just drops n_hit so a stray call cannot double-write.
    auto it = st.dst_ilx_which.find(dst);
    if (it == st.dst_ilx_which.end()) {
        st.n_hit = 0;
        return;
    }
    const int32_t ilx = it->second.first;
    const int32_t which = it->second.second;
    const bool is_down = (which == 3);
    if (!is_down) {
        st.n_hit = 0;
        return;
    }
    const llama_expert_pool_state::mini_graph_entry & m = st.mini[ilx][0];
    const ggml_tensor * out_t = m.out_down;
    if (out_t == nullptr) {
        st.n_hit = 0;
        return;
    }

    const int64_t dbg_t0 = ggml_time_us();
    // down outputs are [n_embd, n_used]: row = n_embd floats, dst->nb[1] is
    // the column stride in bytes
    const size_t row_bytes = dst->nb[1];
    void * host_out = st.host_out_down;
    // wait for THIS layer's whole-chain graph (event recorded at submit), then
    // pull the hit columns synchronously; by now the GPU had the whole CPU
    // miss-kernel time to finish, so the wait is usually near-zero
    ggml_backend_event_synchronize(st.ev);
    ggml_backend_tensor_get(out_t, host_out, 0, (size_t) st.n_hit * row_bytes);
    const int64_t dbg_t1 = ggml_time_us();
    for (int i = 0; i < st.n_hit; ++i) {
        float * dst_col = (float *) ((char *) dst->data + st.hit_cols[i]*dst->nb[1]);
        memcpy(dst_col, (float *) host_out + (size_t) i * (row_bytes / sizeof(float)), row_bytes);
    }

    // accumulate per-layer statistics; hit_rows is recorded in begin (once per
    // layer), so end only records timings.
    if ((size_t) ilx < st.s_layer.size()) {
        st.s_layer[ilx].sync_us  += (uint64_t) (dbg_t1 - dbg_t0);
        st.s_layer[ilx].get_us   += (uint64_t) (ggml_time_us() - dbg_t1);
    }
    st.n_hit = 0;
    // re-arm for a possible next step with the same single pooled layer
    // (begin only re-arms on a layer change, which never fires in that case)
    st.mini_submitted = false;
}
