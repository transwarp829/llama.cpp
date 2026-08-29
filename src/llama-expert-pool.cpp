#include "llama-expert-pool.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>

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
    // auto-grow: a returned reference must be the real per-layer cell, not a
    // shared static - callers write fields into it (expert_pool_init
    // registration loop fills the mount in place)
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

    pool_ready = false;
    budget_slots = 0;
    seg_cnt.clear();

    pend.clear();
    pool_backend  = nullptr;
    swap_sum      = 0;
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
    // NOTE: no single-token gate here. the swap window and the hit/miss
    // counters must see EVERY token column of the batch: multi-sequence runs
    // (-np N) and speculative verify batches (T = 1 + n_draft) both arrive
    // with ids->ne[1] > 1, and dropping them silently disables the swap under
    // -np N or speculative decoding (the window is a global mix of all
    // sequences routed in this decode). only the routing log below keeps the
    // one-token-per-line format.
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
    // layer with an active mount of a step begins after the last one of the
    // previous step (rt_step_done is set at the end of this hook)
    if (ilx == st.first_active_ilx && st.rt_step_done) {
        st.rt_step_done = false;
        if (st.swap_auto && !st.win_cnt.empty()) {
            llama_expert_pool_run_swap(st);
            st.win_step += 1;
            // WRITE-TIME eviction, once per step: after the increment, the
            // slot win_step % W still holds the step from one full window
            // ago (win_step - W); decrement those pairs and clear it before
            // the counting block pushes this step's rows. the previous
            // implementation evicted at the OLD win_step (one step late):
            // it removed the newest step's data while a full window leaked
            // in the counters, and the gate's sigma collapsed once the
            // window slid (the DSV4 rebound to ~40/step).
            std::vector<int32_t> & old = st.win_hist[st.win_step % st.swap_W];
            if (!old.empty()) {
                for (size_t i = 0; i + 1 < old.size(); i += 2) {
                    st.win_cnt[old[i] * st.n_expert + old[i+1]] --;
                }
                old.clear();
            }
        }
    }
    // routing log: keep the one-token-per-line format (B=1 decode rows only)
    if (ids->ne[1] == 1 && st.rt_log != nullptr) {
        // new decode step when the FIRST mounted layer logs again after the
        // LAST one did (a layer-id-change test breaks with one active layer)
        if (ilx == st.first_active_ilx && st.rt_step_done) {
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

    // --- stage 3 swap window: count this row's expert ids for EVERY token
    // column of the batch (multi-seq / spec verify arrive with ne[1] > 1).
    // one decode call = one window step regardless of the token count.
    if (st.swap_auto && !st.win_cnt.empty()) {
        std::vector<int32_t> & hist = st.win_hist[st.win_step % st.swap_W];
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < (int) ids->ne[0]; ++id) {
                const int32_t e = *((const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]));
                if (e < 0 || e >= st.n_expert) {
                    continue;
                }
                st.win_cnt[ilx * st.n_expert + e] ++;
                st.seg_cnt[ilx * st.n_expert + e] ++;
                hist.push_back(ilx);
                hist.push_back(e);
            }
        }
    }
    // hit/miss counters (direct mount: ids come from remap_cpu, so -1 is a GPU
    // pool hit and a non-negative id is the expert computed on the CPU). idle
    // layers (active=false) are skipped by the mount gate above.
    if (st.direct_mount) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < (int) ids->ne[0]; ++id) {
                const int32_t e = *((const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]));
                if (e < 0) {
                    st.stat_hit[ilx] ++;
                    st.win_hit ++;
                } else if (e < st.n_expert) {
                    st.stat_miss[ilx] ++;
                    st.win_miss ++;
                }
            }
        }
    }
    // the LAST layer with an active mount completes the step (single-layer-
    // safe detection: a 0-slot layer has no mount and its hook early-returns,
    // so the anchor is not always the last pooled index - using the last
    // ACTIVE index keeps the step-boundary alive under the desert rule)
    if (ilx == st.last_active_ilx) {
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

void swap_copy_one(ggml_backend_t be, ggml_tensor * src, ggml_tensor * pw, int32_t e, int32_t slot) {
    if (src == nullptr || pw == nullptr) {
        return;
    }
    const size_t sz = src->nb[2];
    const char * data = (const char *) src->data + e * src->nb[2];
    const size_t off  = slot * pw->nb[2];
    // async on the backend's main stream: the copy runs while the step's
    // CPU chain computes. fall back to the blocking copy if no backend is
    // known (CPU pool etc.).
    if (be != nullptr) {
        ggml_backend_tensor_set_async(be, pw, data, off, sz);
    } else {
        ggml_backend_tensor_set(pw, data, off, sz);
    }
}

} // namespace

// allocate the per-layer slot widths from the cumulative activation counts
// (the infinite-window seg_cnt): the global top-N (layer, expert) pairs by
// count (N = -nep budget) determine both the widths and the seed content.
// two structural rules:
// - desert: layers that would get 1-2 slots drop to 0 (a mounted layer with
//   one or two slots pays the per-layer roundtrip tax for near-zero hits;
//   s = 0 is the pure-CPU baseline, free); the freed slots go to the next
//   ranked pairs.
// - sparse guard: when the counts are too sparse to rank (fewer than two
//   expected events per expert, or fewer observed pairs than half the
//   budget), the caller keeps the current layout.
// returns false when the counts are unusable.
bool llama_expert_pool_alloc_from_counts(
        const std::vector<int32_t> & counts, int32_t P, int32_t n_expert,
        int32_t budget, std::vector<std::vector<int32_t>> & resident) {
    resident.assign(P, {});
    int64_t total = 0;
    int32_t observed = 0;
    for (int32_t i = 0; i < P * n_expert; ++i) {
        if (counts[i] > 0) {
            total += counts[i];
            observed += 1;
        }
    }
    // resolution floor: the mean events per expert must allow the ranking to
    // separate (Poisson sigma ~ sqrt(mu) below mu ~= 2 -> the top pairs are
    // distinguishable); and the observed pairs should cover half the budget
    const double mu = (double) total / (double) (P > 0 && n_expert > 0 ? P * n_expert : 1);
    if (mu < 2.0 || observed < budget / 2) {
        return false;
    }
    // rank all (ilx, e) pairs by count, descending; keep the counting until
    // the budget or until the counts run out (zero-count pairs carry no
    // information; the residual budget stays unused)
    std::vector<std::pair<int32_t, int32_t>> pairs;
    pairs.reserve((size_t) P * n_expert);
    for (int32_t i = 0; i < P * n_expert; ++i) {
        if (counts[i] > 0) {
            pairs.emplace_back(counts[i], i);
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    size_t take0 = std::min<size_t>(pairs.size(), (size_t) budget);
    std::vector<int32_t> width(P, 0);
    for (size_t i = 0; i < take0; ++i) {
        const int32_t ilx = pairs[i].second / n_expert;
        width[ilx] += 1;
    }
    // desert pass: 1-2-slot layers drop to 0; the freed slots go to the next
    // ranked pairs (the list is already sorted; the top-N just widens)
    int32_t freed = 0;
    for (int32_t ilx = 0; ilx < P; ++ilx) {
        if (width[ilx] == 1 || width[ilx] == 2) {
            freed += width[ilx];
            width[ilx] = 0;
        }
    }
    size_t take = take0 + (size_t) std::min<int32_t>(freed, (int32_t) pairs.size() - (int32_t) take0);
    for (size_t i = take0; i < take; ++i) {
        const int32_t ilx = pairs[i].second / n_expert;
        width[ilx] += 1;
    }
    // build the resident vectors: slot s holds the s-th pair of the layer
    // (pairs are globally sorted, so the resident order = the layer's counts
    // descending; desert layers keep zero slots)
    for (size_t i = 0; i < take; ++i) {
        const int32_t idx  = pairs[i].second;
        const int32_t ilx  = idx / n_expert;
        const int32_t e    = idx % n_expert;
        if (width[ilx] > 0 && (int32_t) resident[ilx].size() < width[ilx]) {
            resident[ilx].push_back(e);
        }
    }
    return true;
}

void llama_expert_pool_run_swap(llama_expert_pool_state & st) {
    const int32_t P = (int32_t) st.pooled_layers.size();
    const int32_t n_expert = st.n_expert;
    if (P <= 0 || st.win_cnt.empty()) {
        return;
    }

    // --- commit the exchanges issued at the previous step boundary ---
    // the weight copies were queued on the pool backend's main stream, and
    // the sched's per-split synchronization guarantees the step's GPU work
    // (and with it the copies) completed before the first CPU MMID of this
    // step; the table writes below rebuild the mapping from the resident
    // vectors, so the incoming experts become visible from this step on.
    for (const auto & pe : st.pend) {
        std::vector<int32_t> & res = st.resident[pe.il];
        if (pe.slot >= 0 && pe.slot < (int32_t) res.size() && res[pe.slot] < 0) {
            res[pe.slot] = pe.e;
            swap_write_tables(st, pe.il, res, {});
        }
    }
    st.pend.clear();

    // --- marginal exchange: swap the worst resident expert with the best
    // non-resident one, at most one pair per pooled layer per step, gated by
    // the z-sigma confidence test below (the cost-gate/payback terms were
    // retired 8/30; the gate is a property of the estimator, not of the
    // model or hardware).
    int32_t delta = 0;
    for (int32_t ilx = 0; ilx < P; ++ilx) {
        const int32_t il = st.pooled_layers[ilx];
        std::vector<int32_t> & res = st.resident[il];
        if (res.empty()) {
            continue;
        }
        // the exchange only affects layers with an active mount; others run
        // the plain CPU chain, so swapping their resident set is a no-op
        const llama_expert_pool_mount & mnt = llama_expert_pool_get_mount(il);
        if (!mnt.active) {
            continue;
        }
        // worst resident: prefer an EMPTY slot (the sparse-prefill fallback
        // seeds free slots that the swap fills as the window accumulates;
        // an empty slot has no incumbent to lose); otherwise the mapped
        // expert with the lowest window count
        int32_t worst_s = -1;
        int32_t worst_cnt = 0;
        for (int32_t s = 0; s < (int32_t) res.size(); ++s) {
            if (res[s] < 0) {
                worst_s = s;
                worst_cnt = 0;
                break;
            }
            const int32_t c = st.win_cnt[ilx * n_expert + res[s]];
            if (worst_s < 0 || c < worst_cnt) {
                worst_s = s;
                worst_cnt = c;
            }
        }
        if (worst_s < 0) {
            continue;
        }
        // best non-resident (not mapped, not already pending for this layer)
        const auto pending_e = [&](int32_t e) {
            for (const auto & pe : st.pend) {
                if (pe.il == il && pe.e == e) {
                    return true;
                }
            }
            return false;
        };
        int32_t best_e = -1;
        int32_t best_cnt = 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            const int32_t c = st.win_cnt[ilx * n_expert + e];
            if (c <= best_cnt) {
                continue;
            }
            bool resident = false;
            for (int32_t s = 0; s < (int32_t) res.size(); ++s) {
                if (res[s] == e) {
                    resident = true;
                    break;
                }
            }
            if (resident || pending_e(e)) {
                continue;
            }
            best_e = e;
            best_cnt = c;
        }
        if (best_e < 0) {
            continue;
        }
        // gate: the count gap must exceed z sigma of its Poisson noise (two
        // independent window bins, Var(gap) = cnt_out + cnt_in). z = 3: the
        // statistical confidence level, independent of model and backend -
        // boundaries only move when the drift is real and > 3 sigma, so the
        // exchange frequency is governed by the actual drift rate, not by
        // noise. drift adaptation in the long run belongs to the segment-end
        // reallocation (cumulative seg_cnt), not to this per-step test.
        const double gap = (double) (best_cnt - worst_cnt);
        if (gap <= 3.0 * sqrt((double) best_cnt + (double) worst_cnt)) {
            continue;
        }
        // unmap the victim, copy the fill into the freed slot, publish both
        // at the next step boundary (the slot stays -1 in the tables until
        // the commit, so the step's graphs never read the in-flight slot)
        const int32_t victim_e = res[worst_s];
        res[worst_s] = -1;
        swap_write_tables(st, il, res, {});
        swap_copy_one(st.pool_backend, st.orig_gate_up[il], st.w_pool_gate_up[il], best_e, worst_s);
        swap_copy_one(st.pool_backend, st.orig_up[il],      st.w_pool_up[il],      best_e, worst_s);
        swap_copy_one(st.pool_backend, st.orig_gate[il],    st.w_pool_gate[il],    best_e, worst_s);
        swap_copy_one(st.pool_backend, st.orig_down[il],    st.w_pool_down[il],    best_e, worst_s);
        st.pend.push_back({ il, best_e, worst_s });
        LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_TRACE,
                "%s: exchange layer %d (step %d): evict e=%d (cnt %d), fill e=%d (cnt %d)",
                __func__, il, st.win_step, victim_e, worst_cnt, best_e, best_cnt);
        delta += 1;
    }

    // per-exchange lines: the pair detail (TRACE) and the per-step count
    // (DEBUG). the INFO level gets a PERIODIC average instead of per-step
    // noise: the swap runs every step, but by far most steps exchange 0 or 1
    // pair, so a per-step INFO line is either spam (churn) or silence
    // (converged). print_timings-style: every 64 steps, one average line.
    st.swap_sum += delta;
    if (st.win_step > 0 && st.win_step % 64 == 0) {
        LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_INFO,
                "%s: marginal swap avg %.1f expert slots/step (past 64 steps, step %d)\n",
                __func__, (float) st.swap_sum / 64.0f, st.win_step);
        st.swap_sum = 0;
    }
    LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_DEBUG,
            "%s: marginal step %d: swapped %d\n",
            __func__, st.win_step, delta);
    // the pool hit rate is printed once at the end of the generation segment
    // by llama_expert_pool_finalize; win_hit/win_miss accumulate across the
    // segment (no per-swap reset here)
}
