#include "llama-expert-pool.h"
#include "llama-impl.h"
#include "llama-ext.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>

// ---- per-context pool plumbing ----

// per-context pool plumbing
static thread_local llama_expert_pool_state * g_current_pool = nullptr;

llama_expert_pool_state * llama_expert_pool_set_current(llama_expert_pool_state * st) {
    llama_expert_pool_state * prev = g_current_pool;
    g_current_pool = st;
    return prev;
}

// route observer (llama-ext.h, fork-private): gets the clean topk rows; set once at init.
static llama_expert_pool_route_fn g_route_cb = nullptr;
static void *                     g_route_ud = nullptr;

void llama_expert_pool_set_route_observer(llama_expert_pool_route_fn cb, void * user_data) {
    g_route_cb = cb;
    g_route_ud = user_data;
}

llama_expert_pool_params llama_expert_pool_default_params() {
    llama_expert_pool_params p = {};
    p.swap_cap   = 40;
    p.swap_decay = 96;
    return p;
}

void llama_expert_pool_state::set_mount(int il, const llama_expert_pool_mount & m) {
    // auto-grow: setup writes mounts for the pooled layers only
    if (il < 0) {
        return;
    }
    if ((size_t) il >= mounts.size()) {
        mounts.resize(il + 1);
    }
    mounts[il] = m;
}

const llama_expert_pool_mount & llama_expert_pool_state::mount(int il) const {
    static const llama_expert_pool_mount none {};
    if (il < 0 || (size_t) il >= mounts.size()) {
        return none;
    }
    return mounts[il];
}

void llama_expert_pool_state::reset() {
    phase = PHASE_NONE;
    direct_mount = false;
    layers.clear();
    mounts.clear();
    resident.clear();
    pooled_layers.clear();

    step_done = false;

    swap_auto = false;
    hook_step = 0;
    last_ilx = -1;
    settled_steps = 0;
    act_cnt.clear();
    swap_sum = 0;
    stat.clear();
    seg = {};
    pend_fill.clear();
    seq_ctr = 0;
    mirror_seq[0] = mirror_seq[1] = mirror_seq[2] = -1;
    pub_seq.store(-1);

    pool_backend  = nullptr;
    mirror_ready.store(-1);
    mirror_pub.store(-1);

    stop_worker();
}

void llama_expert_pool_state::stop_worker() {
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

// push a step marker: the queued rows are a complete step. hook thread only,
// except the segment-end drain in llama_context::expert_pool_finalize.
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

// push the layer's full activation row (clean topk ids). observer thread only.
static void route_push_row(llama_expert_pool_state & st, int32_t step, int32_t ilx,
                           const int32_t * ids, int32_t n_used, int32_t n_tok) {
    if (ids == nullptr || n_used <= 0 || n_tok <= 0) {
        return;
    }
    llama_expert_pool_state::route_block rb;
    rb.step  = step;
    rb.ilx   = ilx;
    rb.n_tok = n_tok;
    rb.ids.reserve((size_t) n_used * (size_t) n_tok);
    // the row is the contiguous [n_used*n_tok, 1] I32 copy of the topk output
    for (int64_t i = 0; i < (int64_t) n_used * (int64_t) n_tok; ++i) {
        const int32_t o = ids[i];
        if (o >= 0 && o < st.n_expert) {
            rb.ids.push_back(o);
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

// start the swap worker thread (one per pool): consumes the route rows, owns
// the counters and the refresh, performs the H2D copies on its own thread (the
// inference thread never waits), rebuilds the table mirror. drained by reset().
namespace { void swap_copy_one_sync(ggml_backend_t, ggml_tensor *, ggml_tensor *, int32_t, int32_t, int); }

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
                    // stop so no counted step is lost (no-op on teardown paths)
                    lk.unlock();
                    while (llama_expert_pool_worker_settle(st)) {
                    }
                    return;
                }
            }
            // settle every marker queued so far (a slow worker drains the backlog)
            while (llama_expert_pool_worker_settle(st)) {
            }
        }
    });
}

// parse seed csv: one line per layer "il,e1,e2,..."; the line length is that
// layer's slot count, missing layers get zero. false only on open failure.
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

// random resident set (fixed seed); `widths` indexed by layer (0 = not pooled)
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
// split-head observation: the scheduler reports every split's head right before it
// runs; the pool reads its statistics there (in the lane form via one small D2H).
// -----------------------------------------------------------------------------

// match a split head: registered tensors first, then a foreign expert mmid (layer from the weight name)
static bool split_head_match(llama_expert_pool_state & st, ggml_tensor * head,
                             int32_t & il, int32_t & ilx, bool & lane, bool & registered) {
    il = -1;
    ilx = -1;
    lane = false;
    registered = false;
    const auto it = st.split_head_refs.find(head);
    if (it != st.split_head_refs.end()) {
        il         = it->second.il;
        ilx        = it->second.ilx;
        lane       = it->second.lane;
        registered = true;
        return true;
    }
    if (head->op != GGML_OP_MUL_MAT_ID || head->src[0] == nullptr || head->src[2] == nullptr ||
        head->src[2]->type != GGML_TYPE_I32 || head->src[0]->name == nullptr) {
        return false;
    }
    // a "BACKEND#" prefix marks a staged copy (above the threshold only)
    if (sscanf(head->src[0]->name, "blk.%d.", &il) != 1 &&
        sscanf(head->src[0]->name, "%*[^#]#blk.%d.", &il) != 1) {
        il = -1;
    }
    return il >= 0;
}

// host-readable CLEAN rows of a matched head, flattened to [n_used*T] I32
static const int32_t * split_head_rows(llama_expert_pool_state & st, ggml_tensor * head, ggml_backend_t backend,
                                       int32_t il, bool lane, bool registered, int32_t n_used, int32_t & n_tok) {
    n_tok = 0;
    if (head == nullptr || n_used <= 0) {
        return nullptr;
    }
    if (lane) {
        const ggml_tensor * t = 0 <= il && il < (int32_t) st.ids_clean.size() ? st.ids_clean[il] : nullptr;
        if (t == nullptr || backend == nullptr || t->ne[1] != 1 || t->ne[0] % n_used != 0) {
            return nullptr;
        }
        st.ids_buf.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, st.ids_buf.data(), 0, ggml_nbytes(t));
        n_tok = (int32_t) (t->ne[0] / n_used);
        return st.ids_buf.data();
    }
    // cpu forms: the registered copy, or a CPU split's ids input (possibly strided)
    const ggml_tensor * row = registered ? head->src[1] : head->src[2];
    if (row == nullptr || row->type != GGML_TYPE_I32 || row->data == nullptr) {
        return nullptr;
    }
    if (row->ne[1] == 1) {
        if (row->ne[0] % n_used != 0) {
            return nullptr;
        }
        n_tok = (int32_t) (row->ne[0] / n_used);
        return (const int32_t *) row->data;
    }
    if (row->ne[0] != (int64_t) n_used || row->ne[1] <= 0) {
        return nullptr;
    }
    const int64_t nt  = row->ne[1];
    const int64_t nb0 = row->nb[0] / sizeof(int32_t);
    const int64_t nb1 = row->nb[1] / sizeof(int32_t);
    st.ids_buf.resize((size_t) n_used * (size_t) nt);
    for (int64_t s = 0; s < nt; ++s) {
        for (int32_t j = 0; j < n_used; ++j) {
            st.ids_buf[(size_t) s * n_used + j] =
                *(const int32_t *) ((const char *) row->data + (size_t) (s * nb1 + j * nb0) * sizeof(int32_t));
        }
    }
    n_tok = (int32_t) nt;
    return st.ids_buf.data();
}

void llama_expert_pool_observe_split_head(void * user_data, ggml_tensor * head, ggml_backend_t backend) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) user_data;
    if (head == nullptr) {
        return;
    }
    int32_t il  = -1;
    int32_t ilx = -1;
    bool lane = false;
    bool registered = false;
    if (!split_head_match(st, head, il, ilx, lane, registered)) {
        return;
    }
    const int32_t n_used = registered && 0 <= il && il < (int32_t) st.ids_n_used.size() ? st.ids_n_used[il]
                          : (head->op == GGML_OP_MUL_MAT_ID ? (int32_t) head->ne[1] : 0);
    int32_t n_tok = 0;
    const int32_t * ids = split_head_rows(st, head, backend, il, lane, registered, n_used, n_tok);

    // route observer: one row per (step, layer), one-token batches only
    if (ids != nullptr && g_route_cb != nullptr && n_tok == 1) {
        const int32_t prev = st.rtlog_prev_il;
        st.rtlog_prev_il = il;
        g_route_cb(g_route_ud, il, ids, n_used, prev >= 0 && il < prev ? 1 : 0);
    }

    // pool statistics: registered heads of a direct-mount pool only
    if (ids == nullptr || !registered || !st.direct_mount || st.pooled_layers.empty() || ilx < 0) {
        return;
    }
    if (st.last_ilx == ilx) {
        // once per (step, layer): the lane form reports every miss mmid of the layer
        return;
    }

    if (st.stat.empty()) {
        st.stat.assign(st.pooled_layers.size(), llama_expert_pool_counts{});
    }
    // step-advance: splits run in layer order, so this follows the previous step's last
    const bool first_of_step = ilx == st.first_active_ilx && st.step_done;
    if (first_of_step) {
        st.step_done = false;
        // new step: re-arm the per-layer gate (a single active layer would stick)
        st.last_ilx = -1;
        if (st.swap_auto) {
            // publish the ready mirror (the only table write point; none ready = keep the old mapping)
            llama_expert_pool_tab_publish(st);
            llama_expert_pool_push_marker(st);
            st.hook_step += 1;
        }
    }
    st.last_ilx = ilx;
    // the layer's full activation row: one tick per step regardless of the tokens
    if (st.swap_auto) {
        route_push_row(st, st.hook_step, ilx, ids, n_used, n_tok);
    }
    // hit/miss from the inverse half of the published table (what this graph read)
    const int32_t pub = st.mirror_pub.load(std::memory_order_acquire);
    if (st.direct_mount && pub >= 0 && pub < 3 && !st.tab_mirror[pub].empty()) {
        const int32_t * inv = st.tab_mirror[pub].data() +
                              (size_t) st.n_expert * st.layers.size() + (size_t) il * st.n_expert;
        for (int64_t i = 0; i < (int64_t) n_used * (int64_t) n_tok; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= st.n_expert) {
                continue;
            }
            if (inv[e] == -1) {
                st.stat[ilx].hit ++;
                st.seg.hit ++;
            } else {
                st.stat[ilx].miss ++;
                st.seg.miss ++;
            }
        }
    }
    // the LAST active-mount layer completes the step: a 0-slot layer has no mount
    // and is never reported, so the anchor is not always the last pooled index
    if (ilx == st.last_active_ilx) {
        st.step_done = true;
    }
}

// %il ascends within one build: a decreasing index means a new graph, and no pointer survives its graph
static void bind_new_build(const llama_expert_pool_state & st, int32_t il) {
    if (il < st.bind_last_il) {
        st.split_head_refs.clear();
        st.ids_clean.clear();
        st.ids_n_used.clear();
    }
    st.bind_last_il = il;
}

static int32_t bind_ilx(const llama_expert_pool_state & st, int32_t il) {
    for (size_t i = 0; i < st.pooled_layers.size(); ++i) {
        if (st.pooled_layers[i] == il) {
            return (int32_t) i;
        }
    }
    return -1;
}

void llama_expert_pool_bind_ids(const llama_expert_pool_state & st, int32_t il, int32_t n_used, const ggml_tensor * ids_clean) {
    if (il < 0 || ids_clean == nullptr) {
        return;
    }
    bind_new_build(st, il);
    if ((size_t) il >= st.ids_clean.size()) {
        st.ids_clean.resize((size_t) il + 1, nullptr);
        st.ids_n_used.resize((size_t) il + 1, 0);
    }
    st.ids_clean[il]   = ids_clean;
    st.ids_n_used[il]  = n_used;
}

void llama_expert_pool_bind_split_head(const llama_expert_pool_state & st, int32_t il, const ggml_tensor * head, bool lane) {
    if (il < 0 || head == nullptr) {
        return;
    }
    bind_new_build(st, il);
    const int32_t ilx = bind_ilx(st, il);
    if (ilx < 0) {
        return;
    }
    st.split_head_refs[head] = { il, ilx, lane };
}

// ---------------------------------------------------------------
// merged mount-table mirror and its publish point

namespace {

// copy one expert slice (kind k) into a pool slot (sync, worker's own thread)
void swap_copy_one_sync(ggml_backend_t be, ggml_tensor * src, ggml_tensor * pw, int32_t e, int32_t slot, int k) {
    if (src == nullptr || pw == nullptr) {
        return;
    }
    const size_t sz = llama_expert_pool_stride(src, k);
    const char * data = (const char *) src->data + e * llama_expert_pool_stride(src, k);
    const size_t off  = slot * llama_expert_pool_stride(pw, k);
    // SYNC copy: only the worker's thread blocks; the result reaches the tables
    // at the next hook publish, so no torn slot is ever readable by a graph.
    ggml_backend_tensor_set(pw, data, off, sz);
}

} // namespace

// rebuild the merged host mirror from resident[] (worker thread only), mark it
// ready, return its sequence number. layout: [2*n_expert, n_layers]; layer il =
// [il*2*n_expert + e] remap, [+n_expert] remap_inv. no tensor writes here.
int32_t llama_expert_pool_tab_build(llama_expert_pool_state & st) {
    if (st.tab_all == nullptr) {
        return st.seq_ctr;
    }
    const int32_t n_expert = st.n_expert;
    // one half of the mirror; the tables live in one tensor of 2*n_layers
    // columns, so derive the half from the layer count, not from its shape
    const size_t n_half = (size_t) st.n_expert * st.layers.size();
    // pick a free mirror slot: never the ready one (not published yet) nor the
    // published one (its set_async may be in flight); three slots, no waiting
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
        return st.seq_ctr; // cannot happen: ready and pub occupy at most two of three
    }
    std::vector<int32_t> & mir = st.tab_mirror[k];
    if (mir.size() != 2 * n_half) {
        mir.assign(2 * n_half, -1);
    }
    for (int32_t il : st.pooled_layers) {
        const llama_expert_pool_mount & m = st.mount(il);
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
    st.mirror_seq[k] = ++st.seq_ctr;
    st.mirror_ready.store(k, std::memory_order_release);
    return st.seq_ctr;
}

// publish the ready mirror (hook thread only, at a step boundary): the GPU table
// write goes through the BACKEND iface so it is stream-ordered after all in-flight
// compute and queued before the step's remaining GPU segments submit; no host sync
// (the mirror slot rules keep the source alive). the CPU copy is a sync set.
void llama_expert_pool_tab_publish(llama_expert_pool_state & st) {
    if (st.tab_all == nullptr) {
        return;
    }
    const int32_t r = st.mirror_ready.load(std::memory_order_acquire);
    if (r < 0 || r == st.mirror_pub.load(std::memory_order_acquire)) {
        return;
    }
    const size_t n_half = (size_t) st.n_expert * st.layers.size();
    // both halves in one stream-ordered copy (the table layout matches the mirror)
    ggml_backend_tensor_set_async(st.pool_backend, st.tab_all, st.tab_mirror[r].data(), 0, 2 * n_half * sizeof(int32_t));
    if (st.tab_cpu != nullptr) {
        ggml_backend_tensor_set(st.tab_cpu, st.tab_mirror[r].data() + n_half, 0, n_half * sizeof(int32_t));
    }
    st.mirror_pub.store(r, std::memory_order_release);
    st.pub_seq.store(st.mirror_seq[r], std::memory_order_release);
}

// top-k refresh + two-phase slot fill + mirror rebuild (worker thread only).
//
// every settled step each pooled layer's resident set converges towards the k
// highest-scoring experts (k = slot count): the top-k is the positive-count experts
// ranked descending, incoming experts take empty slots first, then the coldest
// incumbent that fell out; pairs run globally in count order until swap_per_step.
//
// an exchange spans two settles: settle N unmaps the victim (res = -1), the hook
// publishes that mirror at the step boundary, and only then - guarded by the
// published sequence - settle N+1 writes the weights and remaps the slot.
static int32_t worker_decide_and_copy(llama_expert_pool_state & st) {
    const int32_t P = (int32_t) st.pooled_layers.size();
    const int32_t n_expert = st.n_expert;
    if (P <= 0 || st.act_cnt.empty()) {
        return 0;
    }
    struct pair_t {
        int32_t ilx;
        int32_t e;    // expert to swap in
        int32_t slot; // slot to fill
        float   cnt;  // decayed activation count of the incoming expert
    };
    std::vector<pair_t> queue;
    queue.reserve(P);

    // a slot with a pending fill is owned by the pending queue - never plan it again
    auto slot_pending = [&](int32_t il, int32_t slot) {
        for (const auto & p : st.pend_fill) {
            if (p.il == il && p.slot == slot) {
                return true;
            }
        }
        return false;
    };

    // phase 1: fill the slots whose unmap reached the tables
    int32_t filled = 0;
    if (!st.pend_fill.empty()) {
        const int32_t pub_seq = st.pub_seq.load(std::memory_order_acquire);
        std::vector<llama_expert_pool_state::pending_fill> wait;
        for (const auto & p : st.pend_fill) {
            std::vector<int32_t> & res = st.resident[p.il];
            if ((size_t) p.slot >= res.size() || res[p.slot] >= 0) {
                continue; // the slot is no longer waiting (teardown or replan)
            }
            if (p.seq > pub_seq) {
                wait.push_back(p); // its unmap is not in the tables yet
                continue;
            }
            const llama_expert_pool_layer & l = st.layers[p.il];
            for (int k = 0; k < PK_N; ++k) {
                swap_copy_one_sync(st.pool_backend, l.orig[k], l.pool[k], p.e, p.slot, k);
            }
            res[p.slot] = p.e;
            filled += 1;
            LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_TRACE,
                    "%s: fill layer %d slot %d (step %d): expert e=%d (cnt %d)",
                    __func__, p.il, p.slot, st.settled_steps, p.e, p.cnt);
        }
        st.pend_fill.swap(wait);
    }

    // phase 2: plan this settle's exchanges and unmap the victim slots
    for (int32_t ilx = 0; ilx < P; ++ilx) {
        const int32_t il = st.pooled_layers[ilx];
        const std::vector<int32_t> & res = st.resident[il];
        if (res.empty()) {
            continue;
        }
        // only layers with an active mount are refreshed (others run the plain path)
        const llama_expert_pool_mount & mnt = st.mount(il);
        if (!mnt.active) {
            continue;
        }
        const int32_t K = (int32_t) res.size();
        // positive-count experts, (weight, id) descending = the top-k pool
        std::vector<std::pair<float, int32_t>> hot; // (cnt, e)
        hot.reserve((size_t) n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            const float c = st.act_cnt[ilx * n_expert + e];
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
        std::vector<std::pair<float, int32_t>> in; // (cnt, e)
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
        // outgoing slots: empty first (no incumbent to lose), then residents
        // outside the new top-k, coldest first; the pool only fills, never shrinks
        std::vector<int32_t> empty;
        std::vector<std::pair<float, int32_t>> out; // (cnt, slot)
        for (int32_t s = 0; s < K; ++s) {
            const int32_t e = res[s];
            if (e < 0) {
                if (!slot_pending(il, s)) {
                    empty.push_back(s);
                }
            } else if (!in_topk[e]) {
                out.emplace_back(st.act_cnt[ilx * n_expert + e], s);
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
    // execute in count order, descending, until the per-step limit is passed
    const int32_t limit = st.swap_per_step < 0 ? -1 : st.swap_per_step;
    std::sort(queue.begin(), queue.end(), [](const pair_t & a, const pair_t & b) {
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        if (a.ilx != b.ilx) return a.ilx < b.ilx;
        if (a.e != b.e)     return a.e < b.e;
        return a.slot < b.slot;
    });
    const int32_t total = (int32_t) queue.size();
    int32_t planned = 0;
    int32_t deferred = 0;
    for (const pair_t & p : queue) {
        if (limit >= 0 && planned >= limit) {
            deferred = total - planned;
            break;
        }
        const int32_t il = st.pooled_layers[p.ilx];
        std::vector<int32_t> & res = st.resident[il];
        const int32_t victim_e = res[p.slot];
        // unmap the victim: once the mirror built below is published, no
        // expert maps onto this slot, so the fill may write it next settle
        res[p.slot] = -1;
        st.pend_fill.push_back({ il, p.e, p.slot, 0, p.cnt });
        planned += 1;
        LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_TRACE,
                "%s: unmap layer %d slot %d (step %d): evict e=%d, fill e=%d (cnt %d)",
                __func__, il, p.slot, st.settled_steps, victim_e, p.e, p.cnt);
    }
    if (filled > 0 || planned > 0) {
        // tag the fills queued above with this mirror's sequence so they wait for it
        const int32_t seq = llama_expert_pool_tab_build(st);
        for (auto & p : st.pend_fill) {
            if (p.seq == 0) {
                p.seq = seq;
            }
        }
        st.swap_sum += filled;
    }
    if (deferred > 0) {
        LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_DEBUG,
                "%s: %d pair(s) deferred by the per-step swap limit (step %d)",
                __func__, deferred, st.settled_steps);
    }
    return filled;
}

// settle one queued step marker: drain the rows pushed before it, decay and count
// them, run the refresh, rebuild the mirror. worker thread only.
bool llama_expert_pool_worker_settle(llama_expert_pool_state & st) {
    std::vector<llama_expert_pool_state::route_block> rows;
    {
        std::lock_guard<std::mutex> lk(st.route_mtx);
        // find the first marker WITHOUT consuming: rows are only complete once
        // their marker arrives, so an early wake must leave the queue untouched
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
    // lazy allocation of the counters
    if (st.act_cnt.empty() && !st.pooled_layers.empty()) {
        st.act_cnt.assign((size_t) st.pooled_layers.size() * st.n_expert, 0.0f);
    }
    // count by the rows' own step id: a lagging worker may drain several old
    // steps at once, so each step gets one decay tick and then its rows land
    size_t i = 0;
    while (i < rows.size()) {
        const int32_t t = rows[i].step;
        // one decay tick per settled step; the rows land with 1/n_tok increments,
        // so a batch of n columns contributes one step's worth of evidence
        const float lam = st.swap_lambda;
        for (float & c : st.act_cnt) {
            c *= lam;
        }
        size_t j = i;
        for (; j < rows.size() && rows[j].step == t; ++j) {
            const llama_expert_pool_state::route_block & rb = rows[j];
            const float inc = 1.0f / (float) (rb.n_tok > 0 ? rb.n_tok : 1);
            for (const int32_t e : rb.ids) {
                st.act_cnt[rb.ilx * st.n_expert + e] += inc;
            }
        }
        i = j;
        // the refresh runs after this step's rows are counted, and reaches the
        // graph 1-2 steps later (the hook publishes at a step boundary)
        const int32_t delta = worker_decide_and_copy(st);
        st.settled_steps = t + 1;
        // per-exchange lines at TRACE/DEBUG; INFO gets a periodic average
        // instead of per-step noise (print_timings-style, every 64 steps)
        if (delta > 0) {
            LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_DEBUG,
                    "%s: swap step %d: swapped %d\n",
                    __func__, st.settled_steps, delta);
        }
        if (st.settled_steps > 0 && st.settled_steps % 64 == 0) {
            LLAMA_LOG_INFV(LLAMA_LOG_VERBOSITY_INFO,
                    "%s: swap avg %.1f expert slots/step (past 64 steps, step %d)\n",
                    __func__, (float) st.swap_sum / 64.0f, st.settled_steps);
            st.swap_sum = 0;
        }
    }
    // the hit rate is printed once at segment end by expert_pool_finalize
    return true;
}
