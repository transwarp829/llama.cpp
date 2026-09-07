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

// draft-context flag: the CPU moe delegate is a global hook and cannot tell
// graphs apart; a draft context (MTP shares the main model's tensors) must
// not feed the pool's routing statistics, so llama_context::decode marks the
// calling thread while it runs a draft graph
static thread_local bool g_draft_decode = false;

void llama_expert_pool_set_draft_decode(bool on) {
    g_draft_decode = on;
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
    w_pool_up_b.clear();
    w_pool_gate_b.clear();
    w_pool_down_b.clear();
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
    fill_done = false;
    hook_step = 0;
    count_ilx = -1;
    stat_ilx  = -1;
    win_step = 0;
    win_cnt.clear();
    win_hist.clear();
    swap_sum = 0;
    stat_hit.clear();
    stat_miss.clear();
    win_hit  = 0;
    win_miss = 0;

    pool_ready = false;
    budget_slots = 0;
    seg_cnt.clear();

    pool_backend  = nullptr;
    mirror_ready.store(-1);
    mirror_pub.store(-1);

    // stop the swap worker (if running): signal, join, drain the route queue
    {
        std::lock_guard<std::mutex> lk(route_mtx);
        route_stop = true;
        route_q.clear();
    }
    route_cv.notify_all();
    if (cp_worker.joinable()) {
        cp_worker.join();
    }
    route_stop = false;
}

// push a step marker: the rows previously queued now belong to a complete
// step and the worker may settle them. hook thread only, except the
// segment-end drain in llama_context::expert_pool_finalize (no decode in
// flight there, so no hook can race it).
void llama_expert_pool_push_marker(llama_expert_pool_state & st) {
    llama_expert_pool_state::route_block m;
    m.step = st.hook_step;
    m.ilx  = -1;
    {
        std::lock_guard<std::mutex> lk(st.route_mtx);
        st.route_q.push_back(std::move(m));
    }
    st.route_cv.notify_one();
}

// push the FULL activation row of one (step, layer): the ids the CPU chain
// actually used, with the -1 (pool hit) entries recovered from the original
// top-k ids before the inverse remap. hook thread only, inside the cpu mmid
// delegate (no locks held while walking/reading; queue lock is brief).
static void route_push_row(llama_expert_pool_state & st, int32_t step, int32_t ilx,
                           const ggml_tensor * ids) {
    const int64_t n_used = ids->ne[0];
    const int64_t n_tok  = ids->ne[1];
    // original top-k ids: the inverse-remap get_rows' src[1] is the private
    // continuous copy of selected_experts (ffn_moe_ids_flat_priv). walk:
    // ids (cpu chain) = reshape(get_rows(remap_inv_host, ids_flat)); the
    // scheduler rewrites cross-backend inputs to CPU-side copies (src[1]
    // points at a CPU#... copy tensor), so the data is host-readable here.
    const ggml_tensor * gr = ids;
    if (gr->op == GGML_OP_RESHAPE && gr->src[0] != nullptr) {
        gr = gr->src[0];
    }
    const ggml_tensor * raw_t = nullptr;
    if (gr->op == GGML_OP_GET_ROWS && gr->src[1] != nullptr) {
        raw_t = gr->src[1];
    }
    const int32_t * raw = raw_t != nullptr ? (const int32_t *) raw_t->data : nullptr;
    if (raw == nullptr) {
        // defensive: without the original row the -1 (resident) activations
        // cannot be attributed. the swap is degraded to miss-side counting
        // (the pre-9/6 behavior), but never crashes.
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("%s: cannot walk to original top-k ids (swap rows degrades to miss-side)\n", __func__);
        }
    }

    llama_expert_pool_state::route_block rb;
    rb.step = step;
    rb.ilx  = ilx;
    rb.ids.reserve((size_t) n_used * (size_t) n_tok);
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t j = 0; j < n_used; ++j) {
            const int32_t e = *((const int32_t *) ((const char *) ids->data + t*ids->nb[1] + j*ids->nb[0]));
            int32_t o = e;
            if (e < 0) {
                // pool hit: recover the resident expert from the original row
                // raw is the flat row-major [n_used, n_tok] top-k copy, so (j, t) = j + t*n_used
                o = raw != nullptr ? raw[j + t*n_used] : -1;
            }
            if (o >= 0 && o < st.n_expert) {
                rb.ids.push_back(o);
            }
        }
    }
    if (rb.ids.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(st.route_mtx);
        st.route_q.push_back(std::move(rb));
    }
    st.route_cv.notify_one();
}

// start the swap worker thread (one per pool). the worker consumes the
// route rows, owns the window counters and the marginal exchange, performs
// the H2D weight copies (sync tensor_set on its own thread - the inference
// thread and the main graph stream never wait for it), and rebuilds the
// table mirror for the hook to publish. runaway workers are drained on reset().
namespace { void swap_copy_one_sync(ggml_backend_t, ggml_tensor *, ggml_tensor *, int32_t, int32_t); }

void llama_expert_pool_start_worker(llama_expert_pool_state & st) {
    if (st.cp_worker.joinable()) {
        return;
    }
    st.route_stop = false;
    st.cp_worker = std::thread([&st]() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(st.route_mtx);
                st.route_cv.wait(lk, [&st]() { return st.route_stop || !st.route_q.empty(); });
                if (st.route_stop) {
                    // drain before exit: settle every marker queued before the
                    // stop so no counted step is lost (finalize pushes a final
                    // marker for the tail rows first; teardown paths clear the
                    // queue, so this is a no-op there)
                    lk.unlock();
                    while (llama_expert_pool_worker_settle(st)) {
                    }
                    return;
                }
            }
            // settle every marker queued so far (each marker completes the
            // previous step; a slow worker drains the backlog step by step)
            while (llama_expert_pool_worker_settle(st)) {
            }
        }
    });
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
// without replacement so every slot holds a distinct expert. `widths` is
// indexed by layer number (0 = layer not pooled)
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert,
                              const std::vector<int32_t> & widths,
                              std::vector<std::vector<int32_t>> & resident) {
    resident.assign(n_layer, std::vector<int32_t>());
    std::mt19937 rng(0);
    std::vector<int32_t> perm(n_expert);
    for (int32_t e = 0; e < n_expert; ++e) {
        perm[e] = e;
    }
    for (int32_t il = 0; il < n_layer; ++il) {
        int32_t n_slot = (size_t) il < widths.size() ? widths[il] : 0;
        if (n_slot > n_expert) {
            n_slot = n_expert;
        }
        if (n_slot <= 0) {
            continue;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        resident[il].assign(perm.begin(), perm.begin() + n_slot);
    }
}
// -----------------------------------------------------------------------------
// moe routing-log hook: called by the CPU MUL_MAT_ID kernel (ith==0) before
// row grouping. collects NO rows (nothing is skipped: the -1 skip ids zero
// the columns natively, both chains merge in the main graph);
// it only feeds GGML_EXPPOOL_ROUTING_LOG.
// -----------------------------------------------------------------------------

void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) ud;
    *skip_out = nullptr;
    if (g_draft_decode) {
        return;
    }
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
    // batch gate: prefill-scale batches (at/above the MoE offload threshold)
    // run the native path with the pool fully inert - no table publish, no
    // window rows, no hit/miss counting. same threshold as the graph-side
    // small-batch gate, never a literal.
    const bool small_batch = ids->ne[1] < llama_expert_pool_offload_min_batch();
    // NOTE: below the threshold the swap window and the hit/miss counters
    // must see EVERY token column of the batch: multi-sequence runs (-np N)
    // and speculative verify batches (T = 1 + n_draft) both arrive with
    // ids->ne[1] > 1, and dropping them silently disables the swap under
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
    // lazy window allocation is gone from the hook: the swap worker owns
    // win_cnt/win_hist and allocates them at its first settlement.
    if (st.stat_hit.empty() && !st.pooled_layers.empty()) {
        st.stat_hit.assign(st.pooled_layers.size(), 0);
        st.stat_miss.assign(st.pooled_layers.size(), 0);
    }
    // step-advance detection, independent of the routing log: the first
    // layer with an active mount of a step begins after the last one of the
    // previous step (rt_step_done is set at the end of this hook)
    if (ilx == st.first_active_ilx && st.rt_step_done) {
        st.rt_step_done = false;
        // new step: re-arm the per-layer counting dedup (a single active
        // layer would otherwise be skipped forever after its first count)
        st.count_ilx = -1;
        st.stat_ilx  = -1;
        if (st.rt_log != nullptr) {
            // advance the log's step counter and flush the previous step's
            // lines (the swap never lives in this block anymore)
            st.log_step += 1;
            st.logged_il = -1;
            fflush(st.rt_log);
        }
        if (st.swap_auto && small_batch) {
            // publish the worker's latest ready mirror: the only table write
            // point. if the worker is still copying (no new ready mirror),
            // the tables keep the old mapping - the swap decision takes
            // effect one or more steps after it was made, which the sigma
            // gate tolerates (the drift it tracks is slow).
            llama_expert_pool_tab_publish(st);
            // step marker: the rows queued before it are now a complete step
            // and the worker may settle them (count + exchange + mirror).
            llama_expert_pool_push_marker(st);
            st.hook_step += 1;
        }
    }
    // routing log: keep the one-token-per-line format (B=1 decode rows only);
    // the step advance happens in the main advance block above
    if (ids->ne[1] == 1 && st.rt_log != nullptr) {
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

    // --- stage 3 swap window: push the FULL activation row of this layer
    // (resident + non-resident, every token column of the batch) to the swap
    // worker. one decode step = one window step regardless of the token
    // count; pushed once per (step, layer): the hook fires per MUL_MAT_ID
    // node (2-3 per layer) with the same ids, and the worker attributes the
    // row from the -1 positions against the original top-k ids.
    if (st.swap_auto && small_batch && st.count_ilx != ilx) {
        st.count_ilx = ilx;
        route_push_row(st, st.hook_step, ilx, ids);
    }
    // hit/miss counters (direct mount: ids come from remap_inv, so -1 is a GPU
    // pool hit and a non-negative id is the expert computed on the CPU). idle
    // layers (active=false) are skipped by the mount gate above. same per
    // (step, layer) dedup as the window counting
    if (st.direct_mount && small_batch && st.stat_ilx != ilx) {
        st.stat_ilx = ilx;
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
// stage 3: merged mount-table mirror and its publish point

namespace {

// copy one expert weight slice into a pool slot (sync, worker's own thread)
void swap_copy_one_sync(ggml_backend_t be, ggml_tensor * src, ggml_tensor * pw, int32_t e, int32_t slot) {
    if (src == nullptr || pw == nullptr) {
        return;
    }
    const size_t sz = src->nb[2];
    const char * data = (const char *) src->data + e * src->nb[2];
    const size_t off  = slot * pw->nb[2];
    // SYNC copy: the worker blocks only its own thread, and the settled step
    // publishes the result into the mirror (the tables change at the next
    // hook publish, so no torn slot is ever readable by a graph).
    ggml_backend_tensor_set(pw, data, off, sz);
}

} // namespace

// rebuild the merged host mirror from resident[] (worker thread only) and
// mark it ready. mirror layout: [2*n_expert, n_layers]; layer il =
// [il*2*n_expert + e] remap, [+n_expert] remap_inv (the inv half feeds
// tab_cpu). the tables themselves are NOT written here.
void llama_expert_pool_tab_build(llama_expert_pool_state & st) {
    if (st.tab_all == nullptr) {
        return;
    }
    const int32_t n_expert = st.n_expert;
    const size_t n_total = (size_t) st.tab_all->ne[0] * (size_t) st.tab_all->ne[1];
    const size_t n_half  = n_total; // one half of the mirror (remap or inv) per table
    // pick a free mirror slot: never the ready one (the hook may not have
    // published it yet) and never the published one (its set_async may still
    // be in flight). three slots, so at most two are busy - no waiting.
    const int32_t ready = st.mirror_ready.load(std::memory_order_acquire);
    const int32_t pub   = st.mirror_pub.load(std::memory_order_acquire);
    int32_t k = -1;
    for (int32_t c = 0; c < 3; ++c) {
        const int32_t cand = pub < 0 ? c : (pub + 1 + c) % 3;
        if (cand != ready && cand != pub) {
            k = cand;
            break;
        }
    }
    if (k < 0) {
        return; // cannot happen: ready and pub occupy at most two of three
    }
    std::vector<int32_t> & mir = st.tab_mirror[k];
    if (mir.size() != 2 * n_total) {
        mir.assign(2 * n_total, -1);
    }
    for (int32_t il : st.pooled_layers) {
        const llama_expert_pool_mount & m = llama_expert_pool_get_mount(il);
        if (!m.active) {
            continue;
        }
        const std::vector<int32_t> & res = st.resident[il];
        int32_t * rmp    = mir.data() + il * n_expert;
        int32_t * rmp_cu = mir.data() + n_half + il * n_expert;
        // default: nothing resident (remap -1, inverse = its own id)
        for (int32_t e = 0; e < n_expert; ++e) {
            rmp[e] = -1;
            rmp_cu[e] = e;
        }
        for (int32_t s = 0; s < (int32_t) res.size(); ++s) {
            const int32_t e = res[s];
            if (e >= 0 && e < n_expert) {
                rmp[e] = s;
                rmp_cu[e] = -1;
            }
        }
    }
    st.mirror_ready.store(k, std::memory_order_release);
}

// publish the ready mirror (hook thread only, at a step boundary): the GPU
// table write uses the BACKEND iface so it is stream-ordered after all
// in-flight compute on the main stream, and it is queued before the current
// step's remaining GPU segments submit (the hook runs before the sched
// resumes, see run_swap in the previous form). no host sync: the worker only
// reuses a published slot after another publish cycle (mirror slot rules).
// the CPU-hosted copy is a sync set - same-thread ordering with the miss
// chain's get_rows reads, exactly as before.
void llama_expert_pool_tab_publish(llama_expert_pool_state & st) {
    if (st.tab_all == nullptr) {
        return;
    }
    const int32_t r = st.mirror_ready.load(std::memory_order_acquire);
    if (r < 0 || r == st.mirror_pub.load(std::memory_order_acquire)) {
        return;
    }
    const size_t n_half = (size_t) st.tab_all->ne[0] * (size_t) st.tab_all->ne[1];
    ggml_backend_tensor_set_async(st.pool_backend, st.tab_all, st.tab_mirror[r].data(), 0, n_half * sizeof(int32_t));
    if (st.tab_cpu != nullptr) {
        ggml_backend_tensor_set(st.tab_cpu, st.tab_mirror[r].data() + n_half, 0, n_half * sizeof(int32_t));
    }
    st.mirror_pub.store(r, std::memory_order_release);
}

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
        if (width[ilx] == 0) {
            continue; // do not re-create a 1-2 slot layer the desert pass just zeroed
        }
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

// rate-gated window top-k refresh + weight copies + mirror rebuild
// (worker thread only). each settled step, every pooled layer's resident
// set converges towards the window's k most used experts (k = slot count):
// the new top-k is the experts with positive window count ranked by count
// (zero-count experts never evict anything - a cold window only fills),
// incoming experts take the empty slots first, then the slots of resident
// experts that fell out of the top-k (coldest first). the resulting pairs
// are executed globally in window-count order, descending, until the
// per-step pair limit (swap_per_step, negative = unlimited) is exhausted -
// the hottest experts copy first, the tail waits for the next settled step.
static int32_t worker_decide_and_copy(llama_expert_pool_state & st) {
    const int32_t P = (int32_t) st.pooled_layers.size();
    const int32_t n_expert = st.n_expert;
    if (P <= 0 || st.win_cnt.empty()) {
        return 0;
    }
    struct pair_t {
        int32_t ilx;
        int32_t e;    // expert to swap in
        int32_t slot; // slot to fill
        int32_t cnt;  // window count of the incoming expert
    };
    std::vector<pair_t> queue;
    queue.reserve(P);
    for (int32_t ilx = 0; ilx < P; ++ilx) {
        const int32_t il = st.pooled_layers[ilx];
        const std::vector<int32_t> & res = st.resident[il];
        if (res.empty()) {
            continue;
        }
        // the refresh only affects layers with an active mount; others run
        // the plain CPU chain, so swapping their resident set is a no-op
        const llama_expert_pool_mount & mnt = llama_expert_pool_get_mount(il);
        if (!mnt.active) {
            continue;
        }
        const int32_t K = (int32_t) res.size();
        // positive-count experts, (count, id) descending = the top-k pool
        std::vector<std::pair<int32_t, int32_t>> hot; // (cnt, e)
        hot.reserve((size_t) n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            const int32_t c = st.win_cnt[ilx * n_expert + e];
            if (c > 0) {
                hot.emplace_back(c, e);
            }
        }
        std::sort(hot.begin(), hot.end(), [](const auto & a, const auto & b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        const int32_t n_new = std::min<int32_t>(K, (int32_t) hot.size());
        std::vector<bool> in_topk((size_t) n_expert, false);
        for (int32_t i = 0; i < n_new; ++i) {
            in_topk[hot[i].second] = true;
        }
        // incoming: hot experts of the new top-k that are not resident
        std::vector<std::pair<int32_t, int32_t>> in; // (cnt, e)
        for (int32_t i = 0; i < n_new; ++i) {
            const int32_t e = hot[i].second;
            bool resident = false;
            for (int32_t s = 0; s < K; ++s) {
                if (res[s] == e) {
                    resident = true;
                    break;
                }
            }
            if (!resident) {
                in.emplace_back(hot[i].first, e);
            }
        }
        // outgoing slots: empty slots first (an empty slot has no incumbent
        // to lose), then resident experts outside the new top-k, coldest
        // first; when the top-k is smaller than K, slots beyond it keep
        // their content - the pool only fills, it never shrinks
        std::vector<int32_t> empty;
        std::vector<std::pair<int32_t, int32_t>> out; // (cnt, slot)
        for (int32_t s = 0; s < K; ++s) {
            const int32_t e = res[s];
            if (e < 0) {
                empty.push_back(s);
            } else if (!in_topk[e]) {
                out.emplace_back(st.win_cnt[ilx * n_expert + e], s);
            }
        }
        std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
        const int32_t npairs = std::min<int32_t>((int32_t) in.size(),
                (int32_t) (empty.size() + out.size()));
        for (int32_t i = 0; i < npairs; ++i) {
            const int32_t slot = i < (int32_t) empty.size()
                ? empty[i] : out[i - (int32_t) empty.size()].second;
            queue.push_back({ilx, in[i].second, slot, in[i].first});
        }
    }
    // execute the queue in window-count order, descending, until the
    // per-step pair limit: the first pair over the limit stops the batch
    const int32_t limit = st.swap_per_step < 0 ? -1 : st.swap_per_step;
    std::sort(queue.begin(), queue.end(), [](const pair_t & a, const pair_t & b) {
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        if (a.ilx != b.ilx) return a.ilx < b.ilx;
        if (a.e != b.e)     return a.e < b.e;
        return a.slot < b.slot;
    });
    const int32_t total = (int32_t) queue.size();
    int32_t delta = 0;
    int32_t deferred = 0;
    for (const pair_t & p : queue) {
        if (limit >= 0 && delta >= limit) {
            deferred = total - delta;
            break;
        }
        const int32_t il = st.pooled_layers[p.ilx];
        // the weight copies run here, synchronously on the worker thread;
        // the resident set commits only after the copies returned, and the
        // tables keep the OLD mapping until the hook publishes the rebuilt
        // mirror - so no graph ever sees a half-filled slot (the
        // one-step-unmap of the old pipeline is gone).
        std::vector<int32_t> & res = st.resident[il];
        const int32_t victim_e = res[p.slot];
        swap_copy_one_sync(st.pool_backend, st.orig_gate_up[il], st.w_pool_gate_up[il], p.e, p.slot);
        swap_copy_one_sync(st.pool_backend, st.orig_up[il],      st.w_pool_up[il],      p.e, p.slot);
        swap_copy_one_sync(st.pool_backend, st.orig_gate[il],    st.w_pool_gate[il],    p.e, p.slot);
        swap_copy_one_sync(st.pool_backend, st.orig_down[il],    st.w_pool_down[il],    p.e, p.slot);
        swap_copy_one_sync(st.pool_backend, st.orig_up_b[il],    st.w_pool_up_b[il],    p.e, p.slot);
        swap_copy_one_sync(st.pool_backend, st.orig_gate_b[il],  st.w_pool_gate_b[il],  p.e, p.slot);
        swap_copy_one_sync(st.pool_backend, st.orig_down_b[il],  st.w_pool_down_b[il],  p.e, p.slot);
        res[p.slot] = p.e;
        delta += 1;
        LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_TRACE,
                "%s: exchange layer %d (step %d): evict e=%d, fill e=%d (cnt %d)",
                __func__, il, st.win_step, victim_e, p.e, p.cnt);
    }
    if (delta > 0) {
        llama_expert_pool_tab_build(st);
        st.swap_sum += delta;
    }
    if (deferred > 0) {
        LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_DEBUG,
                "%s: %d pair(s) deferred by the per-step swap limit (step %d)",
                __func__, deferred, st.win_step);
    }
    return delta;
}

// settle one queued step marker: drain the route rows pushed before it (the
// previous step's FULL activation counts), count them into the window
// window, run the marginal exchange, and rebuild the mirror. worker thread
// only; the hook never touches win_cnt/win_hist/resident.
bool llama_expert_pool_worker_settle(llama_expert_pool_state & st) {
    std::vector<llama_expert_pool_state::route_block> rows;
    {
        std::lock_guard<std::mutex> lk(st.route_mtx);
        // find the first marker WITHOUT consuming anything: rows are only
        // complete once their marker arrives, so an early wake (rows pushed,
        // marker not yet) must leave the queue untouched.
        size_t nrows = 0;
        for (; nrows < st.route_q.size() && st.route_q[nrows].ilx >= 0; ++nrows) {
        }
        if (nrows >= st.route_q.size()) {
            return false; // no marker queued - nothing to settle
        }
        rows.reserve(nrows);
        for (size_t i = 0; i < nrows; ++i) {
            rows.push_back(std::move(st.route_q.front()));
            st.route_q.pop_front();
        }
        st.route_q.pop_front(); // the step marker itself
    }
    if (rows.empty()) {
        // a marker with no rows (the first step marker): nothing settled.
        return true;
    }
    // lazy window allocation (the rows are the first thing the worker sees)
    if (st.win_cnt.empty() && !st.pooled_layers.empty()) {
        st.win_cnt.assign((size_t) st.pooled_layers.size() * st.n_expert, 0);
        st.win_hist.resize(st.swap_W);
    }
    // count the rows by their own step id. a lagging worker may drain rows
    // of several old steps at once (multiple markers queued), so group by
    // step: each step's slot is evicted exactly once, before its rows land.
    size_t i = 0;
    while (i < rows.size()) {
        const int32_t t = rows[i].step;
        // WRITE-TIME eviction, per step: the slot t % W still holds the rows
        // of step t - W (or of the previous incarnation of this slot);
        // decrement them before this step's rows land in the same slot.
        std::vector<int32_t> & old = st.win_hist[t % st.swap_W];
        if (!old.empty()) {
            for (size_t j = 0; j + 1 < old.size(); j += 2) {
                st.win_cnt[old[j] * st.n_expert + old[j+1]] --;
            }
            old.clear();
        }
        size_t j = i;
        for (; j < rows.size() && rows[j].step == t; ++j) {
            const llama_expert_pool_state::route_block & rb = rows[j];
            std::vector<int32_t> & hist = st.win_hist[t % st.swap_W];
            for (const int32_t e : rb.ids) {
                st.win_cnt[rb.ilx * st.n_expert + e] ++;
                st.seg_cnt[rb.ilx * st.n_expert + e] ++;
                hist.push_back(rb.ilx);
                hist.push_back(e);
            }
        }
        i = j;
        // the marginal exchange runs after this step's rows are counted: the
        // decision window now ends at this step, mirroring the old boundary
        // semantics (run_swap at the next step boundary with the window
        // including the settled step).
        const int32_t delta = worker_decide_and_copy(st);
        st.win_step = t + 1;
        // per-exchange lines: the pair detail (TRACE) and the per-step count
        // (DEBUG). the INFO level gets a PERIODIC average instead of per-step
        // noise. print_timings-style: every 64 steps, one average line.
        if (delta > 0) {
            LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_DEBUG,
                    "%s: marginal step %d: swapped %d\n",
                    __func__, st.win_step, delta);
        }
        if (st.win_step > 0 && st.win_step % 64 == 0) {
            LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_INFO,
                    "%s: marginal swap avg %.1f expert slots/step (past 64 steps, step %d)\n",
                    __func__, (float) st.swap_sum / 64.0f, st.win_step);
            st.swap_sum = 0;
        }
    }
    // the pool hit rate is printed once at the end of the generation segment
    // by llama_expert_pool_finalize; win_hit/win_miss accumulate across the
    // segment (no per-swap reset here)
    return true;
}
