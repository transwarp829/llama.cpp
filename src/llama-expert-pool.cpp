#include "llama-expert-pool.h"
#include "llama-impl.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-graph.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <random>

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
    act_cnt.clear();
    swap_total.store(0);
    step_slots.clear();
    for (llama_expert_pool_state::slot_window & w : slot_windows) {
        w.open.store(false);
        w.pairs.store(0);
        w.swap_base = 0;
        w.steps = 0;
        w.hit = 0;
        w.miss = 0;
    }
    stat.clear();
    seg = {};
    for (std::unique_ptr<pool_dev> & dp : devs) {
        pool_dev & d = *dp;
        d.mirror_seq[0] = d.mirror_seq[1] = d.mirror_seq[2] = -1;
        d.mirror_ready.store(-1);
        d.mirror_pub.store(-1);
        d.pub_seq.store(-1);
        // the bundle's own worker state
        {
            std::lock_guard<std::mutex> lk(d.route_mtx);
            d.route_q.clear();
        }
        d.route_stop  = false;
        d.settled_steps = 0;
        d.pend_fill.clear();
        d.swap_sum   = 0;
        d.swap_steps = 0;
        d.seq_ctr    = 0;
        d.step_slots_hist.clear();
    }

    stop_worker();
}

// stop and join EVERY bundle's worker (one per pool): clearing a queue drops rows whose
// step will never be settled, which is what teardown wants
void llama_expert_pool_state::stop_worker() {
    for (std::unique_ptr<pool_dev> & dp : devs) {
        pool_dev & d = *dp;
        {
            std::lock_guard<std::mutex> lk(d.route_mtx);
            d.route_stop = true;
            d.route_q.clear();
        }
        d.route_cv.notify_all();
        if (d.cp_worker.joinable()) {
            d.cp_worker.join();
        }
        d.route_stop = false;
    }
}

// push a step marker to EVERY bundle: the queued rows are a complete step, and each pool
// settles its own copy of the step (its own decay tick, decision and mirror publish).
// hook thread only, except the segment-end read-back in llama_context::expert_pool_finalize.
void llama_expert_pool_push_marker(llama_expert_pool_state & st) {
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        llama_expert_pool_state::route_block m;
        m.step = st.hook_step;
        m.il   = -1;
        {
            std::lock_guard<std::mutex> lk(d.route_mtx);
            d.route_q.push_back(std::move(m));
        }
        d.route_cv.notify_one();
    }
}

// env-gated per-row route histogram dump (GGML_EXPPOOL_ROUTE_LOG=<path>): one line per
// (step, layer, ubatch) with the expert hit counts, so an experiment can replay any decay
// offline. the first line is a self-describing header, so a reader can tell this format
// (v2, explicit step) from the earlier one that had no step column. no file is opened and
// nothing is written unless the variable names a path.
static void route_dump_row(const llama_expert_pool_state & st, int32_t il, int32_t n_used, int32_t n_tok) {
    static FILE * f = []() -> FILE * {
        const char * p = getenv("GGML_EXPPOOL_ROUTE_LOG");
        if (p == nullptr || p[0] == '\0') {
            return nullptr;
        }
        FILE * fp = fopen(p, "w");
        if (fp != nullptr) {
            fprintf(fp, "# v2 step,layer,n_tok,n_used,c0..cN\n");
        }
        return fp;
    }();
    if (f == nullptr) {
        return;
    }
    std::vector<int32_t> cnt((size_t) st.n_expert, 0);
    for (size_t i = 0; i < st.ids_buf.size(); ++i) {
        const int32_t e = st.ids_buf[i];
        if (e >= 0 && e < st.n_expert) {
            cnt[(size_t) e] += 1;
        }
    }
    fprintf(f, "%d,%d,%d,%d", st.hook_step, il, n_tok, n_used);
    for (int32_t e = 0; e < st.n_expert; ++e) {
        fprintf(f, ",%d", cnt[(size_t) e]);
    }
    fputc('\n', f);
    fflush(f);
}

// push the layer's full activation row (clean topk ids). observer thread only.
static void route_push_row(llama_expert_pool_state & st, int32_t step, int32_t il,
                           const int32_t * ids, int32_t n_used, int32_t n_tok) {
    if (ids == nullptr || n_used <= 0 || n_tok <= 0) {
        return;
    }
    llama_expert_pool_state::route_block rb;
    rb.step  = step;
    rb.il   = il;
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
    // every row goes to the bundle that serves its layer: one pool never sees another
    // device's traffic (the row's device is the layer's serving device)
    const int32_t di = st.dev_of(il);
    if (di < 0 || (size_t) di >= st.devs.size()) {
        return;
    }
    llama_expert_pool_state::pool_dev & d = *st.devs[di];
    {
        std::lock_guard<std::mutex> lk(d.route_mtx);
        d.route_q.push_back(std::move(rb));
    }
    d.route_cv.notify_one();
}

// start the swap worker thread (one per pool): consumes the route rows, owns
// the counters and the refresh, performs the H2D copies on its own thread (the
// inference thread never waits), rebuilds the table mirror. consumed by reset().
namespace { void swap_copy_one_sync(ggml_backend_t, ggml_tensor *, ggml_tensor *, int32_t, int32_t, int); }

void llama_expert_pool_set_step_slots(llama_expert_pool_state & st, const int32_t * seq_ids, int32_t n_seq_ids) {
    if (st.phase == llama_expert_pool_state::PHASE_NONE) {
        return;
    }
    st.step_slots.assign(seq_ids, seq_ids + n_seq_ids);
}

// one worker per pool: each bundle consumes only its own rows, so the decay, the swap
// decision (with its own per-step cap) and the mirror build are device-local. the cap is
// the same configured value for every pool (a per-device PCIe link does not contend with
// the others; only the miss chain's memory bandwidth is shared).
void llama_expert_pool_start_worker(llama_expert_pool_state & st) {
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        if (d.cp_worker.joinable()) {
            continue;
        }
        d.route_stop = false;
        d.cp_worker = std::thread([&st, &d]() {
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(d.route_mtx);
                    d.route_cv.wait(lk, [&d]() { return d.route_stop || !d.route_q.empty(); });
                    if (d.route_stop) {
                        // consume the queue before exit: settle every marker queued before
                        // the stop so no counted step is lost (no-op on teardown paths)
                        lk.unlock();
                        while (llama_expert_pool_worker_settle(st, d)) {
                        }
                        return;
                    }
                }
                // settle every marker queued so far (a slow worker works through the backlog)
                while (llama_expert_pool_worker_settle(st, d)) {
                }
            }
        });
    }
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
    {
        // a host weight mmid staged above the offload threshold hands its node a device-side ids
        // copy, and above that threshold no pool is needed for this path to run: dereferencing the
        // device pointer on the host crashes, so read it through the split's backend, in its
        // stream (ordered after the kernels that produce those rows)
        const ggml_backend_buffer_t row_buf = row->view_src != nullptr ? row->view_src->buffer : row->buffer;
        if (row_buf == nullptr) {
            return nullptr;
        }
        if (!ggml_backend_buffer_is_host(row_buf)) {
            if (backend == nullptr) {
                return nullptr;
            }
            auto dev_read = [&](void * dst, size_t off, size_t size) {
                ggml_backend_tensor_get_async(backend, row, dst, off, size);
                ggml_backend_synchronize(backend);
            };
            if (row->ne[1] == 1) {
                if (row->ne[0] % n_used != 0) {
                    return nullptr;
                }
                st.ids_buf.resize(ggml_nelements(row));
                dev_read(st.ids_buf.data(), 0, ggml_nbytes(row));
                n_tok = (int32_t) (row->ne[0] / n_used);
                return st.ids_buf.data();
            }
            if (row->ne[0] != (int64_t) n_used || row->ne[1] <= 0) {
                return nullptr;
            }
            const int64_t nt   = row->ne[1];
            const size_t  span = (size_t) (nt - 1) * row->nb[1] + (size_t) n_used * row->nb[0];
            std::vector<uint8_t> raw(span);
            dev_read(raw.data(), 0, span);
            st.ids_buf.resize((size_t) n_used * (size_t) nt);
            for (int64_t s = 0; s < nt; ++s) {
                for (int32_t j = 0; j < n_used; ++j) {
                    st.ids_buf[(size_t) s * n_used + j] =
                        *(const int32_t *) (raw.data() + (size_t) s * row->nb[1] + (size_t) j * row->nb[0]);
                }
            }
            n_tok = (int32_t) nt;
            return st.ids_buf.data();
        }
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
    // the rows are read for the route observer only: the pool's own accounting reads the
    // copies the graph hands it, at the ubatch boundary (llama_expert_pool_account_ids)
    const int32_t * ids = g_route_cb != nullptr
        ? split_head_rows(st, head, backend, il, lane, registered, n_used, n_tok) : nullptr;

    // route observer: one row per (step, layer), ctx-level - all columns of the
    // step, so a T > 1 row carries n_used * n_tok ids
    if (ids != nullptr && g_route_cb != nullptr) {
        const int32_t prev = st.rtlog_prev_il;
        st.rtlog_prev_il = il;
        g_route_cb(g_route_ud, il, ids, n_used, prev >= 0 && il < prev ? 1 : 0);
    }

    // step boundary: splits run in layer order, so the first active-mount layer of a
    // step is where the mirror that step's tables show has to be in place
    if (!registered || !st.direct_mount || st.pooled_layers.empty() || ilx < 0) {
        return;
    }
    const bool first_of_step = ilx == st.first_active_ilx && st.step_done;
    if (first_of_step) {
        st.step_done = false;
        if (st.swap_auto) {
            // publish the ready mirror (the only table write point; none ready = keep the old mapping)
            llama_expert_pool_tab_publish(st);
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

// ubatch-boundary accounting for the pool's layers: the graph hands every pool layer's
// top-k rows over, this reads them back once per ubatch and feeds everything the
// split-head hook used to feed - the counters, the per-layer hit/miss, the slot windows
// and the step marker. the hook keeps only the mirror publish (it has to happen before
// the next step's graphs read the tables) and the route observer.
void llama_expert_pool_account_ids(llama_expert_pool_state & st) {
    if (!st.direct_mount || st.pooled_layers.empty() || st.phase != llama_expert_pool_state::PHASE_FILLED) {
        return;
    }
    // this ubatch is one step: open the windows of its slots and count it in each (the
    // step boundary the mount path's hook used to detect by the layer order)
    for (const int32_t s : st.step_slots) {
        if (s < 0 || s >= LLAMA_MAX_SEQ) {
            continue;
        }
        llama_expert_pool_state::slot_window & w = st.slot_windows[s];
        if (!w.open.load(std::memory_order_relaxed)) {
            w.swap_base = st.swap_total.load(std::memory_order_relaxed);
            w.pairs.store(0, std::memory_order_relaxed);
            w.steps = 0;
            w.hit = 0;
            w.miss = 0;
            w.open.store(true, std::memory_order_relaxed);
        }
        w.steps ++;
    }

    bool any = false;
    for (size_t ilx = 0; ilx < st.pooled_layers.size(); ++ilx) {
        const int32_t il = st.pooled_layers[ilx];
        const ggml_tensor * ids = (size_t) il < st.ids_clean.size() ? st.ids_clean[il] : nullptr;
        const int32_t n_used = (size_t) il < st.ids_n_used.size() ? st.ids_n_used[il] : 0;
        if (ids == nullptr || n_used <= 0 || ids->type != GGML_TYPE_I32 || ids->ne[0] <= 0) {
            continue;
        }
        const int64_t n_rows = ggml_nelements(ids);
        if (n_rows <= 0 || n_rows % n_used != 0) {
            continue;
        }
        const int32_t n_tok = (int32_t) (n_rows / n_used);
        // probe (GGML_EXPPOOL_IDS_LOG=1): which ids tensor the observer read, per layer per
        // step - the last layer's prefill rows come back as a 1-token tensor
        if (getenv("GGML_EXPPOOL_IDS_LOG") != nullptr && st.hook_step < 2) {
            LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO, "exppool_ids: step=%d il=%d name=%s ne=[%lld,%lld,%lld,%lld] n_used=%d n_tok=%d\n",
                    st.hook_step, il, ids->name,
                    (long long) ids->ne[0], (long long) ids->ne[1], (long long) ids->ne[2], (long long) ids->ne[3],
                    n_used, n_tok);
        }
        st.ids_buf.resize((size_t) n_rows);
        ggml_backend_tensor_get(ids, st.ids_buf.data(), 0, ggml_nbytes(ids));
        route_dump_row(st, il, n_used, n_tok);
        if (st.swap_auto) {
            route_push_row(st, st.hook_step, il, st.ids_buf.data(), n_used, n_tok);
        }
        // hit/miss from the inverse half of the published table (what this graph read).
        // a width-0 layer has no residents, so there is nothing to classify: its rows
        // feed the counters alone (a later layout may pool it)
        const int32_t di       = st.dev_of(il);
        const int32_t pub      = di >= 0 ? st.devs[di]->mirror_pub.load(std::memory_order_acquire) : -1;
        const bool have_mirror = di >= 0 && pub >= 0 && pub < 3 && !st.devs[di]->mirror[pub].empty();
        if (have_mirror && !st.resident[il].empty()) {
            const size_t n_half = (size_t) st.n_expert * st.layers.size();
            const int32_t * inv = st.devs[di]->mirror[pub].data() + n_half + (size_t) il * st.n_expert;
            uint64_t row_hit  = 0;
            uint64_t row_miss = 0;
            for (int64_t i = 0; i < n_rows; ++i) {
                const int32_t e = st.ids_buf[i];
                if (e < 0 || e >= st.n_expert) {
                    continue;
                }
                if (inv[e] == -1) {
                    row_hit ++;
                } else {
                    row_miss ++;
                }
            }
            if (st.stat.size() != st.pooled_layers.size()) {
                st.stat.assign(st.pooled_layers.size(), llama_expert_pool_counts{});
            }
            st.stat[ilx].hit  += row_hit;
            st.stat[ilx].miss += row_miss;
            st.seg.hit        += row_hit;
            st.seg.miss       += row_miss;
            // the window caliber is the pool's, so every slot taking part in this step
            // sees the same rows (their windows differ only in which steps)
            for (const int32_t s : st.step_slots) {
                if (s < 0 || s >= LLAMA_MAX_SEQ) {
                    continue;
                }
                llama_expert_pool_state::slot_window & w = st.slot_windows[s];
                if (!w.open.load(std::memory_order_relaxed)) {
                    continue;
                }
                w.hit  += row_hit;
                w.miss += row_miss;
            }
        }
        any = true;
    }
    if (any && st.swap_auto) {
        // the ubatch's rows are complete: mark the step in EVERY bundle (each pool settles
        // its own copy of it) and hand every worker the step's slot set for attribution
        llama_expert_pool_push_marker(st);
        for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
            std::lock_guard<std::mutex> lk(dp->route_mtx);
            dp->step_slots_hist.emplace_back(st.hook_step, st.step_slots);
        }
    }
    if (any) {
        // one accounted ubatch is one step, swap or not: the dump and any future reader need
        // the step to advance even in the frozen arm
        st.hook_step += 1;
    }
}

// ---------------------------------------------------------------
// merged mount-table mirror and its publish point

namespace {

// copy one expert's slice of a pool table into a slot (sync, worker's own thread).
// the weights and the per-expert biases move together: one expert, one slot.
void swap_copy_one_sync(ggml_tensor * src, ggml_tensor * pw, int32_t e, int32_t slot, int k) {
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

// rebuild ONE bundle's merged host mirror from resident[] (that bundle's worker only),
// mark it ready, return its sequence number. layout: [2*n_expert, n_layers]; layer il =
// [il*2*n_expert + e] remap, [+n_expert] remap_inv. no tensor writes here. the sequence
// counter is the bundle's own: the unmap -> publish -> fill handshake never crosses devices.
int32_t llama_expert_pool_tab_build_dev(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d) {
    if (d.tab_all == nullptr) {
        return d.seq_ctr;
    }
    const int32_t n_expert = st.n_expert;
    // one half of a mirror; each device's tables live in one tensor of 2*n_layer_all
    // columns, so derive the half from the layer count, not from the tensor's shape
    const size_t n_half = (size_t) st.n_expert * st.layers.size();
    {
        // pick a free mirror slot: never the ready one (not published yet) nor the
        // published one (its set_async may be in flight); three slots, no waiting
        const int32_t ready = d.mirror_ready.load(std::memory_order_acquire);
        const int32_t pub   = d.mirror_pub.load(std::memory_order_acquire);
        int32_t k = -1;
        for (int32_t c = 0; c < 3; ++c) {
            const int32_t cand = pub < 0 ? c : (pub + 1 + c) % 3;
            if (cand != ready && cand != pub) {
                k = cand;
                break;
            }
        }
        if (k < 0) {
            return d.seq_ctr; // cannot happen: ready and pub occupy at most two of three
        }
        std::vector<int32_t> & mir = d.mirror[k];
        if (mir.size() != 2 * n_half) {
            mir.assign(2 * n_half, -1);
        }
        for (int32_t il : d.layers) {
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
        d.mirror_seq[k] = ++d.seq_ctr;
        d.mirror_ready.store(k, std::memory_order_release);
    }
    return d.seq_ctr;
}

// rebuild every bundle's mirror (non-worker callers: the relayout path)
int32_t llama_expert_pool_tab_build(llama_expert_pool_state & st) {
    int32_t seq = 0;
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        seq = llama_expert_pool_tab_build_dev(st, *dp);
    }
    return seq;
}

// publish the ready mirror (hook thread only, at a step boundary): the GPU table
// write goes through the BACKEND iface so it is stream-ordered after all in-flight
// compute and queued before the step's remaining GPU segments submit; no host sync
// (the mirror slot rules keep the source alive). the CPU copy is a sync set.
void llama_expert_pool_tab_publish(llama_expert_pool_state & st) {
    const size_t n_half = (size_t) st.n_expert * st.layers.size();
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        if (d.tab_all == nullptr) {
            continue;
        }
        const int32_t r = d.mirror_ready.load(std::memory_order_acquire);
        if (r < 0 || r == d.mirror_pub.load(std::memory_order_acquire)) {
            continue;
        }
        // both halves in one stream-ordered copy (the table layout matches the mirror)
        ggml_backend_tensor_set_async(d.backend, d.tab_all, d.mirror[r].data(), 0, 2 * n_half * sizeof(int32_t));
        if (d.tab_cpu != nullptr) {
            ggml_backend_tensor_set(d.tab_cpu, d.mirror[r].data() + n_half, 0, n_half * sizeof(int32_t));
        }
        d.mirror_pub.store(r, std::memory_order_release);
        d.pub_seq.store(d.mirror_seq[r], std::memory_order_release);
    }
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
static int32_t worker_decide_and_copy(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d) {
    const int32_t P = (int32_t) d.layers.size();
    const int32_t n_expert = st.n_expert;
    if (P <= 0 || st.act_cnt.empty()) {
        return 0;
    }
    struct pair_t {
        int32_t il;
        int32_t e;    // expert to swap in
        int32_t slot; // slot to fill
        float   cnt;  // decayed activation count of the incoming expert
        float   cnt_out; // decayed count of the evicted expert (0 when the slot is unmapped)
    };
    std::vector<pair_t> queue;
    queue.reserve(P);

    // a slot with a pending fill is owned by the pending queue - never plan it again
    auto slot_pending = [&](int32_t il, int32_t slot) {
        for (const auto & p : d.pend_fill) {
            if (p.il == il && p.slot == slot) {
                return true;
            }
        }
        return false;
    };

    // phase 1: fill the slots whose unmap reached the tables
    int32_t filled = 0;
    if (!d.pend_fill.empty()) {
        std::vector<llama_expert_pool_state::pending_fill> wait;
        for (const auto & p : d.pend_fill) {
            std::vector<int32_t> & res = st.resident[p.il];
            if ((size_t) p.slot >= res.size() || res[p.slot] >= 0) {
                continue; // the slot is no longer waiting (teardown or replan)
            }
            // the fill waits for THIS bundle's mirror: its table unmapped the slot
            const int32_t pub_seq = d.pub_seq.load(std::memory_order_acquire);
            if (p.seq > pub_seq) {
                wait.push_back(p); // its unmap is not in the tables yet
                continue;
            }
            const llama_expert_pool_layer & l = st.layers[p.il];
            for (int k = 0; k < PK_N; ++k) { // the whole expert payload: weights + per-expert biases
                swap_copy_one_sync(l.orig[k], l.pool[k], p.e, p.slot, k);
            }
            res[p.slot] = p.e;
            filled += 1;
            LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_TRACE,
                    "%s: fill layer %d slot %d (step %d): expert e=%d (cnt %.3f)\n",
                    __func__, p.il, p.slot, d.settled_steps, p.e, p.cnt);
        }
        d.pend_fill.swap(wait);
    }

    // phase 2: plan this settle's exchanges and unmap the victim slots
    for (int32_t ilx = 0; ilx < P; ++ilx) {
        const int32_t il = d.layers[ilx];
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
            const float c = st.act_cnt[il * n_expert + e];
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
        // victims in one list, coldest first: an unmapped slot counts as 0, so the old
        // "empty first" preference falls out of the ordering. the pool only fills.
        std::vector<std::pair<float, int32_t>> victim; // (cnt, slot)
        for (int32_t s = 0; s < K; ++s) {
            const int32_t e = res[s];
            if (e < 0) {
                if (!slot_pending(il, s)) {
                    victim.emplace_back(0.0f, s);
                }
            } else if (!in_topk[e]) {
                victim.emplace_back(st.act_cnt[il * n_expert + e], s);
            }
        }
        std::sort(victim.begin(), victim.end(), [](const auto & a, const auto & b) {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
        const int32_t npairs = std::min<int32_t>((int32_t) in.size(), (int32_t) victim.size());
        for (int32_t i = 0; i < npairs; ++i) {
            queue.push_back({il, in[i].second, victim[i].second, in[i].first, victim[i].first});
        }
    }
    // spend the per-step limit on the largest gain first: a swap costs one payload, so
    // rank by (in - out) rather than by the incoming count alone
    const int32_t limit = st.swap_per_step;
    std::sort(queue.begin(), queue.end(), [](const pair_t & a, const pair_t & b) {
        const float ga = a.cnt - a.cnt_out;
        const float gb = b.cnt - b.cnt_out;
        if (ga != gb)       return ga > gb;
        if (a.il != b.il)   return a.il < b.il;
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
        const int32_t il = p.il;
        std::vector<int32_t> & res = st.resident[il];
        const int32_t victim_e = res[p.slot];
        // unmap the victim: once the mirror built below is published, no
        // expert maps onto this slot, so the fill may write it next settle
        res[p.slot] = -1;
        d.pend_fill.push_back({ il, p.e, p.slot, 0, p.cnt });
        planned += 1;
        LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_TRACE,
                "%s: unmap layer %d slot %d (step %d): evict e=%d, fill e=%d (cnt %.3f), evict cnt %.3f\n",
                __func__, il, p.slot, d.settled_steps, victim_e, p.e, p.cnt, p.cnt_out);
    }
    if (filled > 0 || planned > 0) {
        // tag the fills queued above with this bundle's mirror sequence so they wait for it
        const int32_t seq = llama_expert_pool_tab_build_dev(st, d);
        for (auto & p : d.pend_fill) {
            if (p.seq == 0) {
                p.seq = seq;
            }
        }
        d.swap_sum += filled;
    }
    if (deferred > 0) {
        LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_DEBUG,
                "%s: %d pair(s) deferred by the per-step swap limit (step %d)",
                __func__, deferred, d.settled_steps);
    }
    return filled;
}

// settle one queued step marker of ONE bundle: consume that bundle's rows pushed before
// it, decay and count them, run the refresh, rebuild that bundle's mirror. worker thread.
bool llama_expert_pool_worker_settle(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d) {
    std::vector<llama_expert_pool_state::route_block> rows;
    {
        std::lock_guard<std::mutex> lk(d.route_mtx);
        // find the first marker WITHOUT consuming: rows are only complete once
        // their marker arrives, so an early wake must leave the queue untouched
        size_t nrows = 0;
        for (; nrows < d.route_q.size() && d.route_q[nrows].il >= 0; ++nrows) {
        }
        if (nrows >= d.route_q.size()) {
            return false; // no marker queued - nothing to settle
        }
        rows.reserve(nrows);
        for (size_t i = 0; i < nrows; ++i) {
            rows.push_back(std::move(d.route_q.front()));
            d.route_q.pop_front();
        }
        d.route_q.pop_front(); // the step marker itself
    }
    if (rows.empty()) {
        // a marker with no rows (the first step marker): nothing settled.
        return true;
    }
    // the counters are allocated once, before the workers start (see the build path):
    // every bundle owns its own layer columns, so no lock is needed here
    if (st.act_cnt.empty()) {
        return true;
    }
    // count by the rows' own step id: a lagging worker may settle several old
    // steps at once, so each step gets one decay tick and then its rows land
    size_t i = 0;
    while (i < rows.size()) {
        const int32_t t = rows[i].step;
        // one decay tick per settled step, over THIS bundle's layer columns only; the
        // rows land with 1/n_tok increments, so a batch of n columns contributes one
        // step's worth of evidence
        const float lam = st.swap_lambda;
        for (const int32_t il : d.layers) {
            float * c = st.act_cnt.data() + (size_t) il * (size_t) st.n_expert;
            for (int32_t e = 0; e < st.n_expert; ++e) {
                c[e] *= lam;
            }
        }
        size_t j = i;
        for (; j < rows.size() && rows[j].step == t; ++j) {
            const llama_expert_pool_state::route_block & rb = rows[j];
            const float inc = 1.0f / (float) (rb.n_tok > 0 ? rb.n_tok : 1);
            for (const int32_t e : rb.ids) {
                st.act_cnt[rb.il * st.n_expert + e] += inc;
            }
        }
        i = j;
        // the refresh runs after this step's rows are counted, and reaches the
        // graph 1-2 steps later (the hook publishes at a step boundary)
        const int32_t delta = worker_decide_and_copy(st, d);
        d.settled_steps = t + 1;
        st.swap_total.fetch_add(delta, std::memory_order_relaxed);
        d.swap_steps += 1;
        // attribute this settle's pairs to the windows of the step's slots (a
        // window closed by a report simply misses its tail - the accepted +-1)
        {
            std::lock_guard<std::mutex> lk(d.route_mtx);
            while (!d.step_slots_hist.empty() && d.step_slots_hist.front().first <= t) {
                const std::pair<int32_t, std::vector<int32_t>> & e = d.step_slots_hist.front();
                if (e.first == t) {
                    for (const int32_t s : e.second) {
                        if (s < 0 || s >= LLAMA_MAX_SEQ) {
                            continue;
                        }
                        llama_expert_pool_state::slot_window & w = st.slot_windows[s];
                        if (w.open.load(std::memory_order_relaxed)) {
                            w.pairs.fetch_add(delta, std::memory_order_relaxed);
                        }
                    }
                }
                d.step_slots_hist.pop_front();
            }
        }
        // per-exchange lines at TRACE/DEBUG; INFO gets a periodic average
        // instead of per-step noise (print_timings-style, every 64 steps)
        if (delta > 0) {
            LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_DEBUG,
                    "%s: swap step %d: swapped %d\n",
                    __func__, d.settled_steps, delta);
        }
    }
    // the hit rate is printed once at segment end by expert_pool_finalize
    return true;
}


// ---------------------------------------------------------------
// the pool driver: llama_context holds one instance and hands it the handles the engine
// needs. the entry points below were moved verbatim from llama-context.cpp; the five
// context-owned names they reach (sched, the reserve flag, the cached graph result and
// synchronize()) go through `ctx`.

llama_expert_pool::llama_expert_pool(
        llama_context & ctx,
        const llama_model & model,
        const llama_cparams & cparams,
        const std::vector<ggml_backend_t> & backend_ptrs,
        const std::vector<ggml_backend_buffer_type_t> & backend_buft) :
    ctx(ctx), model(model), cparams(cparams), backend_ptrs(backend_ptrs), backend_buft(backend_buft) {
}

// one bundle per selected serving device, in the given order, with the given layers and
// budgets: the single place the bundles are created. called by expert_pool_init (which
// resolves the budgets) and by the relayout path, which frees the bundles first because the
// widths - and with them the tensor shapes - change.
void llama_expert_pool::expert_pool_make_devs(
        const std::vector<ggml_backend_dev_t> &   dv_order,
        const std::vector<std::vector<int32_t>> & layers_by_dev,
        const std::vector<int32_t> &              slots_by_dev,
        const std::vector<int32_t> &              scope_by_dev) {
    llama_expert_pool_state & st = this->st;
    const int32_t n_expert    = (int32_t) model.hparams.n_expert;
    const int32_t n_layer_all = (int32_t) model.hparams.n_layer_all;

    st.layer_dev.assign(n_layer_all, -1);
    st.devs.clear();
    st.dev_slots.clear();
    st.dev_scope.clear();
    for (size_t si = 0; si < dv_order.size(); ++si) {
        const ggml_backend_dev_t layer_dev = dv_order[si];
        // the device's first non-host buft; a device without one gets the CPU buft, which is
        // the degenerate (nothing to cache) configuration
        ggml_backend_buffer_type_t bt = nullptr;
        for (const ggml_backend_buffer_type_t cand : backend_buft) {
            if (cand != nullptr && !ggml_backend_buft_is_host(cand) &&
                    ggml_backend_buft_get_device(cand) == layer_dev) {
                bt = cand;
                break;
            }
        }
        if (bt == nullptr) {
            bt = ggml_backend_cpu_buffer_type();
        }
        std::unique_ptr<llama_expert_pool_state::pool_dev> d(new llama_expert_pool_state::pool_dev);
        d->buft = bt;
        for (size_t i = 0; i < backend_buft.size() && i < backend_ptrs.size(); ++i) {
            if (backend_buft[i] == bt) {
                d->backend = backend_ptrs[i];
                break;
            }
        }
        d->w_ctx       = ggml_init({ 4u*1024u*1024u, nullptr, true }); // no_alloc
        d->tab_ctx     = ggml_init({ 4u*1024u*1024u, nullptr, true });
        d->tab_cpu_ctx = ggml_init({ 64u*1024u, nullptr, true });
        // CPU-hosted copy of the inverse table: the miss chain's get_rows reads it HERE, so
        // its ids are not tied to the pool segment's output slot
        d->tab_cpu = ggml_new_tensor_2d(d->tab_cpu_ctx, GGML_TYPE_I32, n_expert, (int64_t) n_layer_all);
        ggml_set_name(d->tab_cpu, "mnt_tab_cpu");
        // the device table: ONE [n_expert, 2*n_layer_all] I32 tensor holding both halves in the
        // mirror's layout (remap, then inv); each layer gets a 1KB view, so a publish is one
        // stream-ordered copy
        d->tab_all = ggml_new_tensor_2d(d->tab_ctx, GGML_TYPE_I32, n_expert, 2 * (int64_t) n_layer_all);
        ggml_set_name(d->tab_all, "mnt_tab_all");
        st.dev_slots.push_back(si < slots_by_dev.size() ? slots_by_dev[si] : 0);
        st.dev_scope.push_back(si < scope_by_dev.size() ? scope_by_dev[si] : 0);
        st.devs.push_back(std::move(d));
        for (int32_t il : layers_by_dev[si]) {
            st.layer_dev[il] = (int32_t) si;
            st.devs.back()->layers.push_back(il);
        }
    }
    int32_t n_layers_pooled = 0;
    for (const std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        n_layers_pooled += (int32_t) dp->layers.size();
    }
    LLAMA_LOG_INFO("%s: pool serves %d layer(s) on %d device bundle(s)\n",
            __func__, n_layers_pooled, (int32_t) st.devs.size());
    for (size_t bi = 0; bi < st.devs.size(); ++bi) {
        LLAMA_LOG_INFO("%s: bundle %zu: %s, %zu layer(s), %d slots (-nep)\n",
                __func__, bi, ggml_backend_buft_name(st.devs[bi]->buft), st.devs[bi]->layers.size(),
                st.dev_slots[bi]);
    }
}

// expert pool initialization:
// - find the host-resident MoE layers, allocate one pool weight copy per matrix
//   (same type/layout, expert dim = S), seed the resident sets, fill the tables
void llama_expert_pool::expert_pool_init() {
    // already built (a reserve after a rebuild re-enters here): a reset() would
    // wipe the fresh pool
    if (this->st.phase != llama_expert_pool_state::PHASE_NONE) {
        return;
    }
    llama_expert_pool_state & st = this->st;
    st.reset();

    if (cparams.expert_pool <= 0) {
        return;
    }
    // one bundle per serving device (grouped below): a layer-split run is dispatched
    // per device, each with its own weight buffer, tables and mirror. the swap decision
    // is still taken by ONE worker across all bundles, so the per-step budget is shared
    // between devices until the per-pool worker lands.
    if (model.n_devices() > 1) {
        LLAMA_LOG_WARN("%s: %d devices: the pool is dispatched per serving device (one bundle "
                       "each), but this path is NOT verified on this build (single-GPU host); "
                       "the per-step swap budget is currently shared between the bundles\n",
                __func__, model.n_devices());
    }
    const int32_t n_layer = model.hparams.n_layer();
    const int32_t n_layer_all = (int32_t) model.hparams.n_layer_all;
    const int32_t n_expert = model.hparams.n_expert;
    if (n_expert <= 0) {
        return;
    }
    // the method governs the range below the offload threshold only: at/above it
    // the scheduler's lane runs the miss chain on the pool device
    LLAMA_LOG_INFO("%s: expert pool: miss method %s, offload threshold %d\n", __func__,
            cparams.expert_pool_miss_method == LLAMA_EXPERT_POOL_MISS_CPU_PARALLEL ? "cpu-parallel" : "cpu-serial",
            llama_expert_pool_offload_min_batch());
    // unified memory (iGPU / APUs, coherent CPU-GPU links): nothing to cache - the
    // same memory already holds the experts, and -cmoe/-ncmoe would only serialize
    // CPU and GPU work. informational only.
    // GGML_CUDA_ENABLE_UNIFIED_MEMORY is the same case but the device still
    // reports itself as discrete, so it is not detected here.
    for (int32_t il = 0; il < n_layer_all; ++il) {
        if (ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            LLAMA_LOG_WARN("%s: expert pool is used on an integrated/unified-memory GPU (layer %d) - "
                           "consider disabling the pool and running without CPU offload for better performance\n",
                           __func__, il);
            break;
        }
    }
    // an MTP draft context executes only the appended NextN block(s): its pooled set
    // is that range instead of the trunk
    const bool    mtp_ctx  = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP;
    const int32_t il_begin = mtp_ctx ? n_layer : 0;
    const int32_t il_end   = mtp_ctx ? n_layer_all : n_layer;
    st.n_expert = n_expert;
    // 0 pairs/step freezes the resident set (the static A/B control arm)
    st.swap_per_step = cparams.expert_pool_swap_cap;
    st.swap_auto     = st.swap_per_step > 0;
    // decaying activation counter (--expert-pool-swap-decay H): lambda = 2^(-1/H) per
    // settled step; a step's increment is its activation count divided by its token
    // columns
    st.swap_decay_hl = cparams.expert_pool_swap_decay > 0 ? cparams.expert_pool_swap_decay : st.swap_decay_hl;
    st.swap_lambda   = std::pow(2.0f, -1.0f / (float) st.swap_decay_hl);
    LLAMA_LOG_INFO("%s: expert pool: decaying counter, half-life %d steps (lambda %.6f)\n",
            __func__, st.swap_decay_hl, st.swap_lambda);

    // --- find the pooled layers: every present expert matrix of the layer in a host
    // buffer, the rest of the layer on a device ---
    // the placement is read from the tensors themselves (the loader resolved
    // -ot/-cmoe/-ncmoe into buffers, fallbacks included), never re-derived from the
    // override patterns; one device-pinned matrix skips the whole layer.
    // CPU-attn layers keep the pure -cmoe path (the pool chain reads the layer's
    // FFN input after the GPU segment).
    std::vector<int32_t> pooled_ils;
    // pooling stays opt-in: without a host-placement override the pool has no reason
    // to build chains, and a device that fell back to the host for mul_mat_id must
    // not be handed a device-side pool
    if (model.params.tensor_buft_overrides != nullptr) {
        for (int32_t il = il_begin; il < il_end; ++il) {
            const llama_layer & L = model.layers[il];
            // CPU-attn layers are never pooled (see comment above)
            if (ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            // the same for a meta device (tensor parallel / -sm tensor): the pool's weight
            // copies and its mount chain would have to run through the meta buft and the
            // meta backend, which the pool does not serve. reject by TYPE first - the buft
            // identity tests below would accept a meta buft, and it is not a host buffer
            if (ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_META) {
                LLAMA_LOG_WARN("%s: layer %d is placed on a meta device (tensor parallel): "
                               "the pool cannot serve it, layer skipped\n", __func__, il);
                continue;
            }
            ggml_tensor * ws[4] = {L.ffn_gate_up_exps, L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps};
            bool any        = false;
            bool any_host   = false;
            bool all_host   = true;
            bool any_repack = false; // a CPU-device buffer that is not host-visible: the pool cannot read it
            for (ggml_tensor * w : ws) {
                if (w == nullptr) {
                    continue;
                }
                any = true;
                const bool host = w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer);
                any_host = any_host || host;
                all_host = all_host && host;
                if (!host && w->buffer != nullptr &&
                        ggml_backend_dev_type(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(w->buffer))) ==
                            GGML_BACKEND_DEVICE_TYPE_CPU) {
                    any_repack = true;
                }
            }
            if (any && all_host) {
                pooled_ils.push_back(il);
            } else if (any_host) {
                LLAMA_LOG_WARN("%s: layer %d has mixed CPU/GPU expert matrices, skipped by the pool (keep the whole layer on one device)\n",
                        __func__, il);
            } else if (any_repack) {
                // repacked expert weights (CPU extra bufts, e.g. --no-host with repacking on) live in
                // a layout that is not the model's: the pool copies raw host bytes, so it would read
                // the repacked form. TODO: upload from the GGUF file offsets (pread) instead of host memory
                LLAMA_LOG_WARN("%s: layer %d expert matrices sit in a repacked CPU buffer: the pool cannot read that layout, layer skipped\n",
                        __func__, il);
            }
        }
    }
    if (pooled_ils.empty()) {
        LLAMA_LOG_WARN("%s: no CPU-resident MoE layers found, expert pool disabled\n", __func__);
        return;
    }

    // --- per device budgets -------------------------------------------------------------
    // group the eligible layers by serving device, in the model's device order (= the
    // --device order); resolve the -nep entries against those devices; then give every pool
    // its OWN layer set and widths. nothing is shared between pools except the rate knobs
    // and the batch-regime threshold.
    auto model_dev_pos = [&](ggml_backend_dev_t dv) -> int32_t {
        for (size_t i = 0; i < model.devices.size(); ++i) {
            if (model.devices[i].dev == dv) {
                return (int32_t) i;
            }
        }
        return -1;
    };
    std::vector<ggml_backend_dev_t>   dv_order;
    std::vector<std::vector<int32_t>> dv_layers;
    auto dv_slot_of = [&](ggml_backend_dev_t dv) -> int32_t {
        const int32_t pos = model_dev_pos(dv);
        size_t at = 0;
        while (at < dv_order.size() && dv_order[at] != dv && model_dev_pos(dv_order[at]) <= pos) {
            at += 1;
        }
        if (at < dv_order.size() && dv_order[at] == dv) {
            return (int32_t) at;
        }
        dv_order.insert(dv_order.begin() + at, dv);
        dv_layers.insert(dv_layers.begin() + at, std::vector<int32_t>());
        return (int32_t) at;
    };
    for (int32_t il : pooled_ils) {
        dv_layers[dv_slot_of(model.dev_layer(il))].push_back(il);
    }
    const int32_t n_dv = (int32_t) dv_order.size();

    // resolve the entries: an empty name means "the i-th device"; a name matches the device
    // name or the buffer type name (what the allocation log prints); the first entry wins
    std::vector<int32_t> dv_slots(n_dv, 0);
    std::vector<int32_t> dv_scope(n_dv, 0);
    std::vector<char>    dv_seen (n_dv, 0);
    std::vector<llama_cparams::expert_pool_dev_budget> entries = cparams.expert_pool_devs;
    if (entries.empty() && cparams.expert_pool > 0) {
        // the legacy single value (-nep N, or an API caller setting slots/layers): one
        // positional entry, i.e. the FIRST device only
        llama_cparams::expert_pool_dev_budget b = {};
        b.slots  = cparams.expert_pool;
        b.layers = cparams.expert_pool_layers;
        entries.push_back(b);
    }
    for (size_t ei = 0; ei < entries.size(); ++ei) {
        const llama_cparams::expert_pool_dev_budget & e = entries[ei];
        int32_t at = -1;
        if (e.dev[0] == '\0') {
            at = (int32_t) ei;
        } else {
            for (int32_t b = 0; b < n_dv && at < 0; ++b) {
                if (strcmp(e.dev, ggml_backend_dev_name(dv_order[b])) == 0) {
                    at = b;
                    break;
                }
                for (const ggml_backend_buffer_type_t bt : backend_buft) {
                    if (bt != nullptr && ggml_backend_buft_get_device(bt) == dv_order[b] &&
                            strcmp(e.dev, ggml_backend_buft_name(bt)) == 0) {
                        at = b;
                        break;
                    }
                }
            }
        }
        if (at < 0 || at >= n_dv) {
            LLAMA_LOG_ERROR("%s: -nep entry %d ('%s') does not match any of the %d device(s) the pool "
                            "serves; pool disabled\n", __func__, (int) ei,
                    e.dev[0] != '\0' ? e.dev : "(by position)", n_dv);
            st.reset();
            return;
        }
        if (dv_seen[at]) {
            LLAMA_LOG_WARN("%s: -nep: %s already has a budget entry (the first one wins)\n",
                    __func__, ggml_backend_dev_name(dv_order[at]));
            continue;
        }
        dv_seen[at]  = 1;
        dv_slots[at] = e.slots;
        dv_scope[at] = e.layers;
    }

    // per device: keep (its layers, its deepest M), then its own widths
    const bool    layout_whole = llama_expert_pool_layout_whole();
    const int32_t min_slots    = llama_expert_pool_min_slots(n_expert);
    std::vector<int32_t> widths(n_layer_all, 0);
    std::vector<int32_t> kept_ils;
    std::vector<ggml_backend_dev_t>   sel_dev;     // the devices that end up with a pool
    std::vector<std::vector<int32_t>> sel_layers;
    std::vector<int32_t>              sel_slots;
    std::vector<int32_t>              sel_scope;
    for (int32_t b = 0; b < n_dv; ++b) {
        std::vector<int32_t> keep = dv_layers[b];   // ascending
        const int32_t N = dv_slots[b];
        const int32_t M = dv_scope[b];
        if (N <= 0) {
            LLAMA_LOG_INFO("%s: %s: no budget entry, its %d eligible MoE layer(s) stay at width 0\n",
                    __func__, ggml_backend_dev_name(dv_order[b]), (int32_t) keep.size());
            continue;
        }
        if (M > 0 && (int32_t) keep.size() > M) {
            const int32_t n_eligible = (int32_t) keep.size();
            keep.erase(keep.begin(), keep.end() - M);
            LLAMA_LOG_INFO("%s: %s: pooling the deepest %d of %d eligible MoE layers (layer %d..%d)\n",
                    __func__, ggml_backend_dev_name(dv_order[b]), (int32_t) keep.size(), n_eligible,
                    keep.front(), keep.back());
        }
        if (keep.empty()) {
            continue;
        }
        if (layout_whole) {
            const int32_t n_full = std::min<int32_t>((int32_t) keep.size(), N / n_expert);
            if (n_full <= 0) {
                LLAMA_LOG_WARN("%s: %s: budget %d is below one full layer (%d experts), its pool stays empty\n",
                        __func__, ggml_backend_dev_name(dv_order[b]), N, n_expert);
                continue;
            }
            for (int32_t i = 0; i < n_full; ++i) {
                widths[keep[i]] = n_expert;
            }
            LLAMA_LOG_INFO("%s: %s: whole-layer layout, %d full layer(s) (layer %d..%d), %d slots unused\n",
                    __func__, ggml_backend_dev_name(dv_order[b]), n_full, keep.front(), keep[n_full - 1],
                    N - n_full * n_expert);
        } else {
            int32_t n_slot = N / (int32_t) keep.size();
            int32_t rem    = N % (int32_t) keep.size();
            if (n_slot > n_expert) {
                LLAMA_LOG_WARN("%s: %s: request of %d slots exceeds capacity (%d layers x %d experts), "
                               "saturating to full coverage per layer\n",
                        __func__, ggml_backend_dev_name(dv_order[b]), N, (int32_t) keep.size(), n_expert);
                n_slot = n_expert;
                rem    = 0;
            }
            if (n_slot < min_slots) {
                LLAMA_LOG_WARN("%s: %s: %d slots per layer is under 1%% of the layer's experts (floor %d of %d): "
                               "the mount roundtrip may cost more than the hits save\n",
                        __func__, ggml_backend_dev_name(dv_order[b]), n_slot, min_slots, n_expert);
            }
            for (size_t i = 0; i < keep.size(); ++i) {
                widths[keep[i]] = n_slot + ((int32_t) i >= (int32_t) keep.size() - rem ? 1 : 0);
            }
        }
        sel_dev.push_back(dv_order[b]);
        sel_layers.push_back(keep);
        sel_slots.push_back(N);
        sel_scope.push_back(M);
        kept_ils.insert(kept_ils.end(), keep.begin(), keep.end());
    }
    std::sort(kept_ils.begin(), kept_ils.end());
    if (kept_ils.empty()) {
        LLAMA_LOG_WARN("%s: no device got a usable budget, expert pool disabled\n", __func__);
        return;
    }

    st.layers.assign(n_layer_all, llama_expert_pool_layer{});
    st.resident.resize(n_layer_all);
    st.pooled_layers = kept_ils;
    st.pool_flag.assign((size_t) n_layer_all, 0);
    for (int32_t il : kept_ils) {
        st.pool_flag[il] = 1;
    }
    st.layout_whole = layout_whole;

    // one bundle per selected device, in the budget order resolved above
    expert_pool_make_devs(sel_dev, sel_layers, sel_slots, sel_scope);

    if (layout_whole) {
        // identity slot order: slot s holds expert s, so a fully resident layer's pool
        // tensors are the model's own matrices in its own order and no remap is needed;
        // the width-0 ones stay empty (they have no pool tensors either)
        for (int32_t il : kept_ils) {
            if (widths[il] != n_expert) {
                continue;
            }
            std::vector<int32_t> & res = st.resident[il];
            res.resize((size_t) n_expert);
            for (int32_t e = 0; e < n_expert; ++e) {
                res[e] = e;
            }
        }
        st.phase = llama_expert_pool_state::PHASE_BUILT;
        expert_pool_build();
        return;
    }

    // uniform widths + random content (a random+swap pool beats a static seed; the
    // allocation shape costs <7% vs global top-N at the same budget)
    llama_expert_pool_random(n_layer_all, n_expert, widths, st.resident);
    st.phase = llama_expert_pool_state::PHASE_BUILT;
    expert_pool_build();
}


void llama_expert_pool::expert_pool_build() {
    llama_expert_pool_state & st = this->st;
    const int32_t n_expert = model.hparams.n_expert;
    const std::vector<int32_t> & pooled_ils = st.pooled_layers;
    const int32_t n_pooled = (int32_t) pooled_ils.size();

    // a layer whose width reaches the expert count is fully resident, so its slot order
    // is free: put it in the model's own order and the graph runs the plain path over
    // its pool tensors - the resident-only state is static, and paying the mount
    // machinery for it names a tax with nothing left to save
    for (int32_t il : pooled_ils) {
        std::vector<int32_t> & res = st.resident[il];
        if ((int32_t) res.size() != n_expert) {
            continue;
        }
        for (int32_t e = 0; e < n_expert; ++e) {
            res[e] = e;
        }
    }

    // the bundles were created in expert_pool_init (one per serving device, in the model's
    // device order, with the budget each one resolved) - build only allocates and fills.
    // reset the whole array so a zero-slot layer never keeps stale pointers
    st.layers.assign(model.hparams.n_layer_all, llama_expert_pool_layer{});
    // the swap counters are allocated ONCE here, before any worker starts: every bundle
    // owns its own layer columns of this array (shared storage, disjoint columns, no lock)
    if (st.swap_auto) {
        st.act_cnt.assign(st.layers.size() * (size_t) st.n_expert, 0.0f);
    }

    // --- create pool weight tensors ---
    for (size_t ix = 0; ix < pooled_ils.size(); ++ix) {
        const int32_t il = pooled_ils[ix];
        const int32_t di = st.dev_of(il);
        if (di < 0) {
            continue;
        }
        llama_expert_pool_state::pool_dev & dev = *st.devs[di];
        const llama_layer & L = model.layers[il];
        // per-layer slot count = the pool file's line length; zero-slot layers get no
        // pool tensors (the layer falls back to CPU). COMPACT layout: slot s holds
        // resident expert res[s] (no zero padding), non-residents route to -1, so no
        // sentinel slice is needed. ne2 = S keeps the ids compact (the layout A/B
        // measured 2.65 vs 0.35 ms/token; the attributed cause is unverified).
        const int32_t s_il = (int32_t) st.resident[il].size();
        if (s_il <= 0) {
            continue;
        }
        // per-layer sources in llama_expert_pool_kind order
        ggml_tensor * src[PK_N] = {
            L.ffn_gate_up_exps, L.ffn_up_exps,      L.ffn_gate_exps, L.ffn_down_exps,
            L.ffn_gate_up_exps_b, L.ffn_up_exps_b,  L.ffn_gate_exps_b, L.ffn_down_exps_b,
        };
        llama_expert_pool_layer & l = st.layers[il];
        for (int k = 0; k < PK_N; ++k) {
            if (src[k] == nullptr) {
                continue;
            }
            l.orig[k] = src[k];
            // weights 4D, slot-compact biases 2D; both are (expert -> slot) copies,
            // so the per-expert stride is ne[2] and ne[1] respectively
            l.pool[k] = (k < PK_UP_B)
                ? ggml_new_tensor_4d(dev.w_ctx, src[k]->type, src[k]->ne[0], src[k]->ne[1], s_il, 1)
                : ggml_new_tensor_2d(dev.w_ctx, src[k]->type, src[k]->ne[0], s_il);
        }
    }

    // --- direct-mount registration ---
    // registered HERE, not at first fill: sched_reserve() builds probe graphs before
    // the first process_ubatch, and the mount adds nodes, so the probe must see the
    // same topology. the tables are plain pool-buffer tensors (created before the
    // alloc below); their content is written by expert_pool_fill().
    // NOTE: the check below only skips REGISTRATION, never the allocation above.
    // sized to n_layer_all before any mount is written, so a reference taken
    // here stays valid for the whole run
    st.mounts.assign(model.hparams.n_layer_all, llama_expert_pool_mount{});
    st.direct_mount = true;
    for (int32_t il : pooled_ils) {
        const int32_t s_il = (int32_t) st.resident[il].size();
        if (s_il <= 0) {
            continue;
        }
        const llama_expert_pool_layer & l = st.layers[il];
        if (l.pool[PK_DOWN] == nullptr) {
            continue;
        }
        // a layer that is fully resident in expert order (whole-layer layout) keeps its
        // pool tensors in the model's own layout: the graph runs the plain path over
        // them, so it needs no mount chain, no remap and no merge
        bool identity = s_il == n_expert;
        for (int32_t s = 0; identity && s < n_expert; ++s) {
            identity = st.resident[il][s] == s;
        }
        llama_expert_pool_mount m;
        m.active    = true;
        m.full      = identity;
        m.w_up      = l.pool[PK_UP];
        m.w_gate    = l.pool[PK_GATE];
        m.w_down    = l.pool[PK_DOWN];
        m.w_gate_up = l.pool[PK_GATE_UP];
        m.w_down_s  = model.layers[il].ffn_down_exps_s;
        m.w_gate_up_b = l.pool[PK_GATE_UP_B];
        m.w_up_b    = l.pool[PK_UP_B];
        m.w_gate_b  = l.pool[PK_GATE_B];
        m.w_down_b  = l.pool[PK_DOWN_B];
        st.set_mount(il, m);
    }
    // step-boundary anchors: the first/last layer that actually has a mount (a 0-slot
    // layer is never registered, so the pooled_layers edges would freeze the swap)
    st.first_active_ilx = -1;
    st.last_active_ilx  = -1;
    for (int32_t ilx = 0; ilx < (int32_t) pooled_ils.size(); ++ilx) {
        if (st.mount(pooled_ils[ilx]).active) {
            if (st.first_active_ilx < 0) {
                st.first_active_ilx = ilx;
            }
            st.last_active_ilx = ilx;
        }
    }

    // --- mount routing tables, one set per serving device: every gather runs on the
    // GPU segment; ids_cpu and the scale values reach the CPU segment as 32B inputs ---
    if (st.direct_mount) {
        // one pool-side copy of the top-k rows per pool layer (see the field comment):
        // the graph writes it every ubatch, the pool reads it after the graph
        st.ids_shadow.assign(model.hparams.n_layer_all, nullptr);
        for (int32_t il : pooled_ils) {
            const int32_t di = st.dev_of(il);
            if (di < 0) {
                continue;
            }
            st.ids_shadow[il] = ggml_new_tensor_1d(st.devs[di]->tab_ctx, GGML_TYPE_I32,
                    (int64_t) model.hparams.n_expert_used(il) * cparams.n_ubatch);
            char nm[64];
            snprintf(nm, sizeof(nm), "mnt_ids_shadow_%d", il);
            ggml_set_name(st.ids_shadow[il], nm);
        }
    }
    for (int32_t il : pooled_ils) {
        if (!st.direct_mount) {
            break;
        }
        const llama_expert_pool_mount & mreg = st.mount(il);
        if (!mreg.active) {
            continue;
        }
        const int32_t di = st.dev_of(il);
        if (di < 0) {
            continue;
        }
        llama_expert_pool_state::pool_dev & d = *st.devs[di];
        llama_expert_pool_mount m = mreg;
        const llama_layer & L = model.layers[il];
        const size_t  i32sz        = ggml_type_size(GGML_TYPE_I32);
        const int64_t n_layers_all = (int64_t) model.hparams.n_layer_all;
        m.remap     = ggml_view_2d(d.tab_ctx, d.tab_all, 1, n_expert, i32sz,
                                   il * n_expert * i32sz);
        m.remap_inv = ggml_view_2d(d.tab_ctx, d.tab_all, 1, n_expert, i32sz,
                                   (n_layers_all + il) * n_expert * i32sz);
        // CPU-side view of the host table (tab_cpu holds only the inv part)
        m.remap_inv_host = ggml_view_2d(d.tab_cpu_ctx, d.tab_cpu, 1, n_expert, i32sz,
                                       il * n_expert * i32sz);
        if (L.ffn_down_exps_s != nullptr) {
            m.scale = ggml_new_tensor_1d(d.tab_ctx, GGML_TYPE_F32, n_expert);
        }
        if (L.ffn_up_exps_s != nullptr) {
            m.scale_up = ggml_new_tensor_1d(d.tab_ctx, GGML_TYPE_F32, n_expert);
        }
        if (L.ffn_gate_exps_s != nullptr) {
            m.scale_gate = ggml_new_tensor_1d(d.tab_ctx, GGML_TYPE_F32, n_expert);
        }
        char nm[64];
        snprintf(nm, sizeof(nm), "mnt_remap_%d", il);
        ggml_set_name(m.remap, nm);
        snprintf(nm, sizeof(nm), "mnt_remap_inv_host_%d", il);
        ggml_set_name(m.remap_inv_host, nm);
        snprintf(nm, sizeof(nm), "mnt_remap_inv_%d", il);
        ggml_set_name(m.remap_inv, nm);
        if (m.scale != nullptr) {
            snprintf(nm, sizeof(nm), "mnt_scale_%d", il);
            ggml_set_name(m.scale, nm);
        }
        if (m.scale_up != nullptr) {
            snprintf(nm, sizeof(nm), "mnt_scale_up_%d", il);
            ggml_set_name(m.scale_up, nm);
        }
        if (m.scale_gate != nullptr) {
            snprintf(nm, sizeof(nm), "mnt_scale_gate_%d", il);
            ggml_set_name(m.scale_gate, nm);
        }
        st.set_mount(il, m);
    }
    // --- allocate: each bundle's tables on its own device, the host halves on the CPU ---
    if (st.direct_mount) {
        for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
            llama_expert_pool_state::pool_dev & d = *dp;
            d.tab_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.tab_ctx, d.buft));
            if (!d.tab_buf) {
                LLAMA_LOG_ERROR("%s: mount table allocation failed on %s, direct mount disabled\n",
                        __func__, ggml_backend_buft_name(d.buft));
                st.direct_mount = false;
                break;
            }
            // CPU-side table: allocate with the CPU buft (host memory)
            d.tab_cpu_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.tab_cpu_ctx,
                    ggml_backend_cpu_buffer_type()));
            if (!d.tab_cpu_buf) {
                LLAMA_LOG_ERROR("%s: CPU mount table allocation failed, direct mount disabled\n", __func__);
                st.direct_mount = false;
                break;
            }
        }
        if (!st.direct_mount) {
            // de-register the mounts so the graph builder never sees active=true with null tables
            for (int32_t il : pooled_ils) {
                llama_expert_pool_mount m = st.mount(il);
                m.active = false;
                st.set_mount(il, m);
            }
        }
    }

    // --- allocate each bundle's weight tensors on its own device (CPU fallback per bundle) ---
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        // the pool uploads weight bytes exactly as they sit in host memory, so a backend
        // that reorders expert weights on the device side would read every uploaded slice
        // wrong. SYCL does that lazily and in place (opt_for_reorder_id, K-quants on Intel
        // GPUs) with no hook to re-apply, and there is no neutral way to ask for it -
        // refuse the whole pool
        ggml_backend_dev_t pool_dev = ggml_backend_buft_get_device(d.buft);
        if (pool_dev != nullptr &&
                std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(pool_dev))) == "SYCL") {
            LLAMA_LOG_WARN("%s: %s reorders expert weights in place on first use; the pool uploads raw "
                           "weights and would read them wrong - pool disabled\n",
                    __func__, ggml_backend_dev_name(pool_dev));
            expert_pool_release(); // frees the bundles
            st.reset();
            return;
        }
        d.w_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.w_ctx, d.buft));
        if (!d.w_buf) {
            LLAMA_LOG_WARN("%s: pool allocation on %s failed, falling back to CPU\n",
                    __func__, ggml_backend_buft_name(d.buft));
            d.buft = ggml_backend_cpu_buffer_type();
            d.backend = nullptr;
            for (size_t i = 0; i < backend_buft.size() && i < backend_ptrs.size(); ++i) {
                if (backend_buft[i] == d.buft) {
                    d.backend = backend_ptrs[i];
                    break;
                }
            }
            d.w_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.w_ctx, d.buft));
        }
        if (!d.w_buf) {
            LLAMA_LOG_ERROR("%s: expert pool allocation failed, pool disabled\n", __func__);
            expert_pool_release(); // frees the bundles
            st.reset();
            return;
        }
        LLAMA_LOG_INFO("%s: pool allocated on %s (host=%d, %zu bytes)\n",
                __func__, ggml_backend_buft_name(d.buft),
                (int) ggml_backend_buffer_is_host(d.w_buf.get()), ggml_backend_buffer_get_size(d.w_buf.get()));
        // mark the pool tensors as weights so the scheduler keeps the warm-path mat_mul_id
        // (and its outputs) on the pool device
        ggml_backend_buffer_set_usage(d.w_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // zero the pool buffer ONCE: fresh backend buffers are not guaranteed zeroed
        ggml_backend_buffer_clear(d.w_buf.get(), 0);
    }

    int32_t tot_slots = 0;
    for (int32_t il : pooled_ils) {
        tot_slots += (int32_t) st.resident[il].size();
    }
    LLAMA_LOG_INFO("%s: expert pool enabled: %d pooled layers, %d slots total (%d/layer if uniform)\n",
            __func__, n_pooled, tot_slots, n_pooled > 0 ? tot_slots / n_pooled : 0);
    {
        size_t pool_bytes = 0;
        for (int32_t il : pooled_ils) {
            const llama_expert_pool_layer & l = st.layers[il];
            for (int k = 0; k < PK_UP_B; ++k) { // weights only (biases excluded, as before)
                if (l.pool[k]) { pool_bytes += ggml_nbytes(l.pool[k]); }
            }
        }
        LLAMA_LOG_INFO("%s: pool weight bytes = %.2f GB (%.1f MiB per expert)\n",
                __func__, pool_bytes / 1e9,
                tot_slots > 0 ? pool_bytes / (double) tot_slots / 1.0e6 : 0.0);
    }

    // the weight/table fill is deferred to expert_pool_fill(), after graph_reserve: with
    // no_alloc model tensors the data is valid only once the scheduler placed them

    // the tensors exist now: the fill may write them. a re-layout runs this again, so the
    // phase belongs here - a caller that also set it would clobber the fill's FILLED
    st.phase = llama_expert_pool_state::PHASE_BUILT;
}


void llama_expert_pool::expert_pool_fill() {
    llama_expert_pool_state & st = this->st;
    if (st.phase != llama_expert_pool_state::PHASE_BUILT || st.layers.empty()) {
        return;
    }
    const int32_t n_expert = model.hparams.n_expert;

    // --- fill weights + routing tables (model tensor data is valid now) ---
    for (size_t pi = 0; pi < st.pooled_layers.size(); ++pi) {
        const int32_t il = st.pooled_layers[pi];
        const llama_layer & L = model.layers[il];
        const std::vector<int32_t> & res = st.resident[il];

        // tables: write every element (sentinel -1), then residents override (same
        // content as init; keeps them correct if fill() ever refreshes the set)
        const llama_expert_pool_mount & m = st.mount(il);
        const bool has_mount = st.direct_mount && m.active &&
                               m.remap != nullptr && m.remap_inv_host != nullptr;
        if (has_mount) {
            // I32 tables: get_rows outputs the table type on every backend, so no cast;
            // both halves are flushed once below (tab_build/tab_publish).
            // flat scale tables: expert e -> its scale (indexed by clean ids, never
            // changes on swap)
            auto fill_scale = [&](ggml_tensor * tab, ggml_tensor * src) {
                if (tab == nullptr || src == nullptr) {
                    return;
                }
                GGML_ASSERT(src->type == GGML_TYPE_F32 && src->ne[0] == n_expert);
                std::vector<float> vals(n_expert, 0.0f);
                const float * src_s = (const float *) src->data;
                for (int32_t e = 0; e < n_expert; ++e) {
                    vals[e] = src_s[e];
                }
                ggml_backend_tensor_set(tab, vals.data(), 0, n_expert * sizeof(float));
            };
            fill_scale(m.scale,      L.ffn_down_exps_s);
            fill_scale(m.scale_up,   L.ffn_up_exps_s);
            fill_scale(m.scale_gate, L.ffn_gate_exps_s);
        }

        const llama_expert_pool_layer & l = st.layers[il];
        if (m.full) {
            // identity slots: the pool tensors already are the model's matrices in the
            // model's order, so one whole-tensor copy per kind replaces the per-slot
            // assembly (and there is nothing to swap later either)
            for (int k = 0; k < PK_N; ++k) {
                if (l.orig[k] != nullptr && l.pool[k] != nullptr) {
                    ggml_backend_tensor_set(l.pool[k], l.orig[k]->data, 0, ggml_nbytes(l.orig[k]));
                }
            }
        } else {
            // the whole expert payload moves: weights and per-expert biases are
            // (expert -> slot) copies, the scales are a static expert-order table
            auto copy_slots = [&](ggml_tensor * src, ggml_tensor * pw, int k) {
                if (src == nullptr || pw == nullptr) {
                    return;
                }
                const size_t sz = llama_expert_pool_stride(src, k);
                for (int32_t s = 0; s < (int32_t) res.size(); ++s) {
                    const int32_t e = res[s];
                    if (e < 0 || e >= n_expert) {
                        continue;
                    }
                    ggml_backend_tensor_set(pw, (const char *) src->data + e * llama_expert_pool_stride(src, k),
                            s * llama_expert_pool_stride(pw, k), sz);
                }
            };
            for (int k = 0; k < PK_N; ++k) {
                copy_slots(l.orig[k], l.pool[k], k);
            }
        }
    }
    st.phase = llama_expert_pool_state::PHASE_FILLED;

    // start the swap-copy worker: all later H2D copies run off the inference thread and
    // off the main graph stream; the step boundary publishes completed fills only
    if (st.swap_auto && !st.pooled_layers.empty()) {
        llama_expert_pool_start_worker(st);
    }

    // initial table mirror: the resident sets land on the device in ONE publish before
    // the first decode (no worker running yet)
    llama_expert_pool_tab_build(st);
    llama_expert_pool_tab_publish(st);

    if (st.direct_mount) {
        int32_t n_full = 0;
        for (int32_t il : st.pooled_layers) {
            if (st.mount(il).full) {
                n_full += 1;
            }
        }
        LLAMA_LOG_INFO("%s: pool active: %d layers fully resident in expert order (plain path), %d mounted (dual chain)\n",
                __func__, n_full, (int32_t) st.pooled_layers.size() - n_full);
    }
}


void llama_expert_pool::expert_pool_finalize(int32_t id_slot) {
    llama_expert_pool_state & st = this->st;
    if (st.seg.hit + st.seg.miss == 0) {
        return;
    }
    // a slot's segment window = the steps it took part in. the row caliber inside
    // the window is the pool's (every slot's columns feed it), so overlapping slots
    // see the same rows and differ only in which steps they cover.
    llama_expert_pool_state::slot_window * win = nullptr;
    if (id_slot >= 0 && id_slot < LLAMA_MAX_SEQ && st.slot_windows[id_slot].open.load(std::memory_order_relaxed)) {
        win = &st.slot_windows[id_slot];
    }
    const uint64_t win_hit  = win != nullptr ? win->hit  : st.seg.hit;
    const uint64_t win_miss = win != nullptr ? win->miss : st.seg.miss;
    if (win != nullptr) {
        const int64_t pairs = win->pairs.load(std::memory_order_relaxed);
        LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO,
                "%s: pool hit rate %.1f%% (%llu/%llu rows, %d steps, slot %d), swap avg %.1f pairs/step\n",
                __func__, 100.0 * win_hit / (double) (win_hit + win_miss),
                (unsigned long long) win_hit, (unsigned long long) (win_hit + win_miss),
                win->steps, id_slot, win->steps > 0 ? (double) pairs / (double) win->steps : 0.0);
        // close the window: this slot's next step opens a fresh one
        win->open.store(false, std::memory_order_relaxed);
        win->pairs.store(0, std::memory_order_relaxed);
        win->swap_base = 0;
        win->steps = 0;
        win->hit = 0;
        win->miss = 0;
    } else {
        LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO, "%s: pool hit rate %.1f%% (%llu/%llu rows, generation segment)\n", __func__,
                100.0 * st.seg.hit / (double) (st.seg.hit + st.seg.miss),
                (unsigned long long) st.seg.hit,
                (unsigned long long) (st.seg.hit + st.seg.miss));
    }
    st.seg = {};

    // the swap average: per window for a slot report, per segment otherwise. each pool
    // reports its OWN average (one worker per device), so the quotient stays readable
    // whatever the other pools settled; the device name is appended only when there is
    // more than one bundle, which keeps the single-device line text unchanged
    if (win == nullptr) {
        const bool multi = st.devs.size() > 1;
        for (size_t di = 0; di < st.devs.size(); ++di) {
            llama_expert_pool_state::pool_dev & d = *st.devs[di];
            if (d.swap_sum > 0 && d.swap_steps > 0) {
                if (multi) {
                    LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO,
                            "%s: swap avg %.1f expert slots/step (segment, %d steps, %s)\n",
                            __func__, (float) d.swap_sum / (float) d.swap_steps, d.swap_steps,
                            ggml_backend_buft_name(d.buft));
                } else {
                    LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO,
                            "%s: swap avg %.1f expert slots/step (segment, %d steps)\n",
                            __func__, (float) d.swap_sum / (float) d.swap_steps, d.swap_steps);
                }
            }
            d.swap_sum   = 0;
            d.swap_steps = 0;
        }
    }
}

// re-lay the pool in the other layout: the whole-layer target puts the budget into full
// shallow layers, the partial one spreads it uniformly with the remainder on the deep
// layers. then the worker is stopped (its join is the barrier) and the sched synchronized
// before the old tensors are freed. the new tensors are built and filled like at init, at
// the same total size - the budget is the invariant, only the widths change.
void llama_expert_pool::expert_pool_relayout(bool whole) {
    llama_expert_pool_state & st = this->st;
    if (st.phase == llama_expert_pool_state::PHASE_NONE || st.pooled_layers.empty()) {
        return;
    }
    const int32_t n_expert = model.hparams.n_expert;

    // one exchange spans two settles: stop the workers first (their joins are the barrier)
    st.stop_worker();

    // the layer set and the budgets stay; only the widths change, and with them the tensor
    // shapes, so EVERY pool is re-laid with its OWN budget and the bundles are re-created
    std::vector<ggml_backend_dev_t>   sel_dev;
    std::vector<std::vector<int32_t>> sel_layers;
    const std::vector<int32_t>        sel_slots = st.dev_slots;
    const std::vector<int32_t>        sel_scope = st.dev_scope;
    for (const std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        sel_dev.push_back(ggml_backend_buft_get_device(dp->buft));
        sel_layers.push_back(dp->layers);
    }

    std::vector<std::vector<int32_t>> resident; // indexed by LAYER
    resident.resize(model.hparams.n_layer_all);
    for (size_t si = 0; si < sel_dev.size(); ++si) {
        const int32_t N = si < sel_slots.size() ? sel_slots[si] : 0;
        std::vector<int32_t> & ls = sel_layers[si];
        if (N <= 0 || ls.empty()) {
            continue;
        }
        const int32_t P = (int32_t) ls.size();
        if (whole) {
            // this pool's shallowest (budget / n_expert) layers go fully resident in the
            // model's own order (a full-width layer is laid out in expert order); its other
            // layers go to width 0
            const int32_t n_full = std::min<int32_t>(P, N / n_expert);
            if (n_full <= 0) {
                LLAMA_LOG_WARN("%s: %s: budget %d is below one full layer (%d experts), its pool stays as it is\n",
                        __func__, ggml_backend_buft_name(st.devs[si]->buft), N, n_expert);
                continue;
            }
            for (int32_t i = 0; i < n_full; ++i) {
                std::vector<int32_t> & res = resident[ls[i]];
                res.resize((size_t) n_expert);
                for (int32_t e = 0; e < n_expert; ++e) {
                    res[e] = e;
                }
            }
        } else {
            // uniform widths, the remainder on the DEEP layers of this pool: at the coverage
            // ratios where the remainder is worth deciding (below ~13% of the experts per
            // layer, see the routing baseline ref) the marginal expert sits in the deeper
            // layers, so they take the spare
            const int32_t wu  = N / P;
            const int32_t rem = N % P;
            for (int32_t i = 0; i < P; ++i) {
                const int32_t wi = wu + (i >= P - rem ? 1 : 0);
                const float * row = st.act_cnt.data() + (size_t) ls[i] * n_expert;
                std::vector<std::pair<float, int32_t>> top;
                top.reserve((size_t) n_expert);
                for (int32_t e = 0; e < n_expert; ++e) {
                    if (row[e] > 0.0f) {
                        top.emplace_back(row[e], e);
                    }
                }
                std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) {
                    return a.first > b.first || (a.first == b.first && a.second < b.second);
                });
                const int32_t take = std::min<int32_t>(wi, (int32_t) top.size());
                for (int32_t k = 0; k < take; ++k) {
                    resident[ls[i]].push_back(top[k].second);
                }
            }
        }
    }

    ctx.synchronize(); // the publish pipeline may still have async copies in flight
    // the cached graphs and the scheduler still reference the pool tensors freed below; both
    // callers reserve right after this (sched_need_reserve, set at the end of this function),
    // and sched_reserve() drops the cached graphs and builds a fresh scheduler itself
    release_devs();
    st.resident.assign(model.hparams.n_layer_all, {});
    for (int32_t il : st.pooled_layers) {
        st.resident[il] = std::move(resident[il]);
    }
    st.layout_whole = whole;
    expert_pool_make_devs(sel_dev, sel_layers, sel_slots, sel_scope);
    expert_pool_build();
    expert_pool_fill(); // restarts the swap workers
    ctx.sched_need_reserve = true;
    int32_t tot_slots = 0;
    for (const int32_t v : sel_slots) {
        tot_slots += v;
    }
    LLAMA_LOG_INFO("%s: pool re-laid out (%s layout, %d layers, %d device(s), %d slots)\n",
            __func__, whole ? "whole-layer" : "partial", (int) st.pooled_layers.size(),
            (int) sel_dev.size(), tot_slots);
}

// the layout follows the batch regime (GGML_EXPPOOL_LAYOUT=auto): at or above the
// offload threshold the whole-layer layout serves the batch, below it the partial one.
// only a batch that lands on the other side re-lays the pool; the same-side batches just
// run. the caller passes the tokens this batch will actually hand to one graph, so the
// decision follows the same threshold the scheduler uses to move a miss chain off the CPU.
void llama_expert_pool::expert_pool_cross(int32_t n_tokens) {
    llama_expert_pool_state & st = this->st;
    // only a batch the pool has actually served decides the regime: the tiny decodes the
    // context runs while reserving must not re-lay it before it has served anything
    if (!llama_expert_pool_layout_auto() || st.phase != llama_expert_pool_state::PHASE_FILLED ||
            st.pooled_layers.empty()) {
        return;
    }
    const bool want_whole = n_tokens >= llama_expert_pool_offload_min_batch();
    if (want_whole == st.layout_whole) {
        // the regime's layout is in place: a later crossing may try again
        st.layout_tried = -1;
        return;
    }
    if ((int8_t) want_whole == st.layout_tried) {
        // this regime's target was refused once; retrying every batch would cost a
        // worker join per step for nothing
        return;
    }
    st.layout_tried = (int8_t) want_whole;
    LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO, "%s: batch of %d tokens crosses the %d-token threshold: re-laying the pool\n",
            __func__, n_tokens, llama_expert_pool_offload_min_batch());
    expert_pool_relayout(want_whole);
}


// release the per-device bundles: weight context/buffer, tables and mirrors. Buffers
// first, then the contexts that own the tensor metadata.
void llama_expert_pool::release_devs() {
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        d.w_buf.reset();
        if (d.w_ctx != nullptr) {
            ggml_free(d.w_ctx);
            d.w_ctx = nullptr;
        }
        d.tab_buf.reset();
        d.tab_cpu_buf.reset();
        d.tab_all = nullptr;
        d.tab_cpu = nullptr;
        for (auto & m : d.mirror) {
            m.clear();
        }
        d.mirror_seq[0] = d.mirror_seq[1] = d.mirror_seq[2] = -1;
        d.mirror_ready.store(-1);
        d.mirror_pub.store(-1);
        d.pub_seq.store(-1);
        if (d.tab_ctx != nullptr) {
            ggml_free(d.tab_ctx);
            d.tab_ctx = nullptr;
        }
        if (d.tab_cpu_ctx != nullptr) {
            ggml_free(d.tab_cpu_ctx);
            d.tab_cpu_ctx = nullptr;
        }
    }
    st.devs.clear();
    st.layer_dev.clear();
}

void llama_expert_pool::expert_pool_release() {
    llama_expert_pool_state & st = this->st;

    // only a tool built on common that never calls finalize (llama-batched, ...) still has
    // an unclosed segment here; a tool that reported per request (server, and the tools
    // embedding it: cli / completion) already drained st.seg there. report the unclosed
    // segment's rows and its swap average together - or not at all
    const bool unclosed = st.seg.hit + st.seg.miss > 0;
    if (unclosed) {
        LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO, "%s: pool hit rate %.1f%% (%llu/%llu rows, generation segment)\n", __func__,
                100.0 * st.seg.hit / (double) (st.seg.hit + st.seg.miss),
                (unsigned long long) st.seg.hit,
                (unsigned long long) (st.seg.hit + st.seg.miss));
        st.seg = {};
    }
    if (unclosed) {
        // every pool's own tally (the workers are still running here - the stop below is
        // what makes this the last read); the device name only in a multi-bundle run, so
        // the single-device line text stays unchanged
        const bool multi = st.devs.size() > 1;
        for (size_t di = 0; di < st.devs.size(); ++di) {
            llama_expert_pool_state::pool_dev & d = *st.devs[di];
            if (d.swap_sum > 0 && d.swap_steps > 0) {
                if (multi) {
                    LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO, "%s: swap avg %.1f expert slots/step (segment, %d steps, %s)\n", __func__,
                            (float) d.swap_sum / (float) d.swap_steps, d.swap_steps, ggml_backend_buft_name(d.buft));
                } else {
                    LLAMA_LOG_INFO_V(LLAMA_LOG_VERBOSITY_INFO, "%s: swap avg %.1f expert slots/step (segment, %d steps)\n", __func__,
                            (float) d.swap_sum / (float) d.swap_steps, d.swap_steps);
                }
            }
            d.swap_sum   = 0;
            d.swap_steps = 0;
        }
    }

    // stop the swap worker first: its sync tensor_set calls reference the pool tensors
    // freed below
    st.stop_worker();

    release_devs();
}
