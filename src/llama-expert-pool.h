#pragma once

#include "llama.h"
#include "llama-cparams.h"  // LLAMA_MAX_SEQ
#include "ggml.h"
#include "ggml-backend.h"

#include "ggml-cpp.h"   // ggml_backend_buffer_ptr (RAII buffer holder)

#include <array>
#include <atomic>
#include <memory>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <utility>

// the MoE small-batch threshold: the same env the CUDA backend and the sched
// read. every batch-size gate in the pool derives from this - never a literal.
inline int32_t llama_expert_pool_offload_min_batch() {
    static const int32_t v = getenv("GGML_OP_OFFLOAD_MIN_BATCH") != nullptr
        ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
    return v;
}

// miss method (llama_expert_pool_params::miss_method): how the non-resident chain
// runs below the offload threshold. at and above it the scheduler's lane takes the
// chain, so the method only picks the CPU-side form there.
enum llama_expert_pool_miss_method {
    LLAMA_EXPERT_POOL_MISS_CPU_SERIAL   = 0, // CPU chain as one split (delivery)
    LLAMA_EXPERT_POOL_MISS_CPU_PARALLEL = 1, // mount chain submitted ahead of the CPU chain
};

// minimum slots per pooled layer: below this width a mounted layer pays the
// per-layer roundtrip tax for near-zero hits, so the pool warns. the budget is
// still laid out as asked - re-cutting the layer set is the placement logic's
// job (--fit), not the pool's. 1% of the expert count, which brackets the
// measured net-zero widths.
inline int32_t llama_expert_pool_min_slots(int32_t n_expert) {
    return (n_expert + 99) / 100;
}

// whole-layer layout (experiment, GGML_EXPPOOL_LAYOUT=whole): every pooled layer is
// either fully resident in the model's own expert order (identity slots) or not
// pooled at all - a fully resident layer then runs the plain path over the pool's
// matrices, so the batch regime that cannot use routing locality pays no dual-chain
// tax at all. the layer set and the budget keep their -nep meaning (the budget still
// picks how many layers can be covered), so the resident bytes match the budget.
inline bool llama_expert_pool_layout_whole() {
    static const bool v = [] {
        const char * s = getenv("GGML_EXPPOOL_LAYOUT");
        return s != nullptr && std::string(s) == "whole";
    }();
    return v;
}

// the layout follows the batch regime (GGML_EXPPOOL_LAYOUT=auto): a batch at or above
// the offload threshold runs the whole-layer layout, below it the partial one, and only
// a batch that crosses the threshold re-lays the pool. the static values above/below
// stay for the A/B arms.
inline bool llama_expert_pool_layout_auto() {
    static const bool v = [] {
        const char * s = getenv("GGML_EXPPOOL_LAYOUT");
        return s != nullptr && std::string(s) == "auto";
    }();
    return v;
}

// ---------------------------------------------------------------
// direct mount (main-graph execution): per-layer tensors that let build_moe_ffn
// run a second, device-resident chain over the pool weights inside the MAIN
// graph. the -1 skip ids split the columns by construction (resident -> pool
// slot, non-resident -> -1), so one add merges the two chains.
// the tables are I32: ggml_get_rows keeps the table type on every backend, so
// the id gathering needs no cast.
struct llama_context;
struct llama_model;

struct llama_expert_pool_mount {
    bool active = false;
    bool full   = false; // fully resident in the model's own order (identity slots): the
                         // graph runs the plain path over these tensors - no mount chain,
                         // no remap table, no merge, and the ids stay the top-k ids
    ggml_tensor * w_gate_up = nullptr; // [n_ff*2, n_embd, S] or null (pool copy)
    ggml_tensor * w_up      = nullptr; // [n_ff, n_embd, S] or null (pool copy)
    ggml_tensor * w_gate    = nullptr;
    ggml_tensor * w_down    = nullptr;
    ggml_tensor * w_down_s  = nullptr; // per-expert down scale source (staged into `scale`)
    ggml_tensor * w_gate_up_b = nullptr; // the model's fused per-expert bias [n_ff*2, n_expert],
    ggml_tensor * w_up_b      = nullptr; // the separate ones [n_ff, n_expert] / [n_embd, n_expert]:
    ggml_tensor * w_gate_b    = nullptr; // slot-compact pool copies [.., S], swapped with the
    ggml_tensor * w_down_b    = nullptr; // weights, indexed by the chain's own masked ids
    ggml_tensor * remap     = nullptr; // I32 [1, n_expert] on the pool device: resident -> slot, else -1
    ggml_tensor * remap_inv_host = nullptr; // I32 [1, n_expert] on the CPU: resident -> -1, else expert id
                                            // (read when the miss mmids stay on the CPU)
    ggml_tensor * remap_inv = nullptr; // I32 [1, n_expert] on the pool device: the inv half, gathered
                                      // device-side when the miss mmids run there
    ggml_tensor * scale     = nullptr; // F32 [1, n_expert] on the pool device: per-expert down scale
    ggml_tensor * scale_up   = nullptr; // F32 [1, n_expert]: up scale, same form
    ggml_tensor * scale_gate = nullptr; // F32 [1, n_expert]: gate scale, same
};

// ---------------------------------------------------------------
// model-level runtime state (direct-mount mode): per pooled layer the
// device-resident pool weight tensors (compact: slot s holds the s-th resident
// expert) plus the routing tables. content = random, refreshed per step.

// per-layer expert tensors the pool copies and the swap moves. the kind order
// is the array index used everywhere (extend here and in the per-layer source
// table of expert_pool_build when a tensor lands). the weights AND the
// per-expert biases are slot-compact (slot s holds the s-th resident expert)
// and move together - the slot is one expert. the per-expert scales are a 1D
// scalar table: expert order, written once, never swapped (slot-ordering the
// scale would need a second mask-free ids table for a 4-byte payload).
enum llama_expert_pool_kind {
    PK_GATE_UP = 0, // fused gate+up [n_ff*2, n_embd, n_expert]; pool [.., S]
    PK_UP,          // separate up [n_ff, n_embd, n_expert]
    PK_GATE,        // separate gate [n_ff, n_embd, n_expert]
    PK_DOWN,        // down [n_embd, n_ff, n_expert]
    PK_GATE_UP_B,   // per-expert fused bias [n_ff*2, n_expert]; pool [.., S],
    PK_UP_B,        // per-expert bias [n_ff, n_expert]      slot-compact like
    PK_GATE_B,      // per-expert bias [n_ff, n_expert]      the weights: the
    PK_DOWN_B,      // per-expert bias [n_embd, n_expert]    chain's own ids index it
    PK_N,
};

// per-expert byte stride: ne[2] for the 4D weight copies, ne[1] for the
// slot-compact 2D bias copies (nb[2] of a 2D tensor is the whole tensor size)
inline size_t llama_expert_pool_stride(const ggml_tensor * t, int k) {
    return k < PK_UP_B ? t->nb[2] : t->nb[1];
}

struct llama_expert_pool_layer {
    ggml_tensor * orig[PK_N] = {}; // model weight tensor (host); null = not present
    ggml_tensor * pool[PK_N] = {}; // pool copy (pool device); null = not pooled
};

// per-layer hit/miss rows (snapshot semantics: a read clears) + one segment total
struct llama_expert_pool_counts {
    uint64_t hit  = 0;
    uint64_t miss = 0;
};

struct llama_expert_pool_state {
    // per-layer expert tensors, indexed by layer id (slot s = s-th resident expert)
    std::vector<llama_expert_pool_layer> layers;

    // registry, indexed by layer id (setup writes; graph/observer/worker read).
    // sized to n_layer_all before the mounts are written, so references stay valid
    std::vector<llama_expert_pool_mount> mounts;
    void set_mount(int il, const llama_expert_pool_mount & m);
    const llama_expert_pool_mount & mount(int il) const;  // read-only (no growth)

    // resident expert lists, indexed by layer (diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // pooled layer indices (filled at init, consumed by the delayed fill). every eligible
    // layer is a pool layer: the layout only picks its width ({0, .., n_expert}), so a
    // layer stays observed (and its counters stay warm) even while nothing of it is
    // resident
    std::vector<int32_t> pooled_layers;
    // pool membership by layer (the graph asks before handing its rows over)
    std::vector<uint8_t> pool_flag;
    // current layout of the pool: true when the pooled layers sit fully resident in
    // expert order (the whole-layer layout), false for the slot cache (partial)
    bool layout_whole = false;
    // the direction the last crossing asked for (-1 none): a refused target is not
    // retried until the regime crosses again - the counts only improve as rows arrive,
    // and re-attempting every batch costs a worker join per step
    int8_t layout_tried = -1;

    // split-head registry, rebuilt per graph build (the builder registers each
    // pooled layer's clean topk ids and the miss chain's first node)
    struct split_head_ref {
        int32_t il   = -1;
        int32_t ilx  = -1;
        bool    lane = false;  // true: read the clean rows from the device
    };
    mutable std::unordered_map<const ggml_tensor *, split_head_ref> split_head_refs;
    mutable int32_t                          bind_last_il = -1;
    mutable std::vector<const ggml_tensor *> ids_clean;   // [layer] clean topk ids (a view
                                                          // of the pool-side copy below)
    mutable std::vector<int32_t>             ids_n_used;  // [layer]
    // [layer] pool-side copy of the layer's top-k rows: the graph writes it every ubatch
    // and the pool reads it after the graph, so it must NOT live in the graph arena -
    // the allocator reuses a short-lived tensor's memory as soon as its last consumer
    // ran, and the values would be gone by the read
    std::vector<ggml_tensor *> ids_shadow;
    mutable std::vector<int32_t>             ids_buf;     // host scratch for a device read
    // route-observer step flag (the wrap is detected by the layer index)
    int32_t rtlog_prev_il = -1;

    bool direct_mount = false;                             // two-chain form active
    ggml_backend_buffer_type_t pool_buft = nullptr;        // pool buft (device)

    // step boundary, single-layer-safe: set at the last active-mount layer of a
    // step, consumed at the first one of the next (anchors the swap publish)
    bool   step_done = false;

    // swap (on by default with -nep): decaying activation count per (pooled
    // layer, expert). the refresh converges the resident set to the most used
    // experts, at most swap_per_step pairs per settled step (a burst fuse - the
    // steady-state rate sits well below it with the default half-life)
    bool swap_auto = false;
    int32_t swap_per_step = 40;            // max expert pairs swapped in per settled
                                           // step, all layers (<= 0 freezes the set)
    int32_t n_expert = 0;                  // experts per layer (set at init)
    // step counter (only the worker owns the counters below)
    int32_t hook_step = 0;

    // routing rows from the observer to the swap worker: one block per
    // (step, layer) carrying the layer's full activation ids
    struct route_block {
        int32_t step = 0;                  // decode step this row belongs to
        int32_t il   = -1;                 // the layer (-1 = step marker)
        int32_t n_tok = 1;                 // token columns (decay: increments are 1/n_tok)
        std::vector<int32_t> ids;          // full activation expert ids
    };
    std::mutex      route_mtx;
    std::condition_variable route_cv;
    std::deque<route_block> route_q;
    bool route_stop = false;

    // worker-owned swap state: the hook only pushes rows and publishes the
    // mirror the worker hands over - it never reads these
    int32_t settled_steps = 0;             // decode steps accounted so far
    // decaying counter: per settled step every count is multiplied by
    // swap_lambda and the step's rows land with per-token normalized increments
    // (1/n_tok per activation), so a batch of n columns is one step of evidence
    float   swap_lambda   = 0.0f;          // per-step decay, lambda = 2^(-1/H)
    int32_t swap_decay_hl = 96;            // decay half-life in steps (default 96)
    std::vector<float> act_cnt;            // [layers * n_expert] decayed counts (keyed by
                                           // layer, so a layout switch keeps them)
    int32_t swap_sum = 0;                  // exchanges accumulated this period (slot-less report)
    int32_t swap_steps = 0;                // settled steps of that period (same reset as swap_sum)
    // monotonic ctx-level swap total: the per-slot windows snapshot it (never reset)
    std::atomic<int64_t> swap_total{0};
    // per-slot segment window: the set of settled steps the slot took part in.
    // the pool learns each ubatch's sequences (llama_expert_pool_set_step_slots),
    // so the decision counters stay slot-agnostic - only the report is attributed,
    // and the row caliber inside a window is the pool's (all slots' columns).
    struct slot_window {
        std::atomic<int64_t> pairs{0};    // swap pairs of the window's steps (worker writes)
        std::atomic<bool>    open{false}; // written at the report, read by the worker
        int64_t  swap_base = 0;           // swap_total when the window opened
        int32_t  steps     = 0;           // steps the slot took part in
        uint64_t hit       = 0;           // pool-caliber rows over those steps
        uint64_t miss      = 0;
    };
    std::array<slot_window, LLAMA_MAX_SEQ> slot_windows;
    // sequences of the ubatch being built (the report thread fills it per ubatch,
    // the observer reads it), and the per-step slot sets the worker consumes
    std::vector<int32_t> step_slots;
    std::deque<std::pair<int32_t, std::vector<int32_t>>> step_slots_hist;

    // segment totals of the inference-side counters (the per-layer array is
    // cleared by each read)
    llama_expert_pool_counts seg;

    // swap worker: consumes the route rows, runs the top-k refresh, performs
    // the H2D weight copies off the inference thread and rebuilds the mirror.
    // the table tensors are only written by the hook at a step boundary
    std::thread cp_worker;
    // mirror handshake: the worker writes a slot only when it is neither the
    // ready one nor the published one, so an async set's source is never
    // overwritten inside the two-step distance
    std::atomic<int32_t> mirror_ready{-1};
    std::atomic<int32_t> mirror_pub{-1};

    // sequence handshake of the two-phase swap: a pending fill runs only once
    // the mirror that unmapped its slot has reached the tables - a slot's
    // content may not change while an expert is still mapped onto it
    int32_t seq_ctr = 0;                    // worker-owned mirror counter
    int32_t mirror_seq[3] = { -1, -1, -1 }; // worker-owned, per mirror slot
    std::atomic<int32_t> pub_seq{-1};       // observer-owned: sequence published
    struct pending_fill {
        int32_t il   = -1;
        int32_t e    = -1; // expert the fill writes into the slot
        int32_t slot = -1;
        int32_t seq  = 0;  // mirror sequence that unmapped the slot (0 = not built yet)
        float   cnt  = 0.0f;
    };
    std::vector<pending_fill> pend_fill;    // worker-owned fill queue

    // NONE -> BUILT (build() ran) -> FILLED (weights and tables copied, first compute)
    enum phase_t : uint8_t { PHASE_NONE = 0, PHASE_BUILT, PHASE_FILLED };
    phase_t phase = PHASE_NONE;
    int32_t last_active_ilx = -1;          // pooled index of the last active-mount layer
    int32_t first_active_ilx = -1;         // ... and the first (the two step anchors)
    // per-pooled-layer hit/miss (in the direct mount e < 0 means the device
    // chain computed the row); read and cleared via llama_expert_pool_get_stats
    std::vector<llama_expert_pool_counts> stat; // [pooled layers]

    // backend owning the pool buft (the hook publishes the tables on its main stream)
    ggml_backend_t pool_backend = nullptr;

    // merged mount tables: both halves live in tab_all (remap block, then inv
    // block) in the mirror's own layout, so publishing is one stream-ordered
    // copy and each layer's tables are views. tab_cpu keeps the CPU-side copy
    // of the inv half for the host-segment reads below the offload threshold
    ggml_tensor * tab_all = nullptr;             // [n_expert, 2*n_layers] remap|inv (pool device)
    ggml_tensor * tab_cpu = nullptr;             // [n_expert, n_layers] remap_inv (CPU buft)
    std::vector<int32_t> tab_mirror[3];          // host mirror slots (see mirror_ready/pub)

    void reset();
    // signal + drain the route queue + join the swap worker (safe with no
    // worker running); shared by reset() and the ctx teardown
    void stop_worker();
};

// random resident set (fixed seed, reproducible); `widths` indexed by layer (0 = not pooled)
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert,
                              const std::vector<int32_t> & widths,
                              std::vector<std::vector<int32_t>> & resident);

// split-head observer: the scheduler hands it every split's head; the pool
// reads its statistics from the registered ones, and the route observer
// (llama-ext.h) forwards the same rows - with or without a pool
void llama_expert_pool_bind_ids(const llama_expert_pool_state & st, int32_t il, int32_t n_used, const ggml_tensor * ids_clean);
void llama_expert_pool_bind_split_head(const llama_expert_pool_state & st, int32_t il, const ggml_tensor * head, bool lane);
void llama_expert_pool_observe_split_head(void * user_data, ggml_tensor * head, ggml_backend_t backend);

// ---------------------------------------------------------------
// the pool driver: owns the pool state, the pool weight context/buffers, the mount
// tables and the engine entry points. llama_context holds one instance and hands it the
// handles the engine needs; the engine reaches a few context-owned names (the scheduler,
// the reserve flag, the cached graph result, synchronize()) through `ctx`, which the
// friend declaration in llama-context.h makes accessible.
struct llama_expert_pool {
    llama_expert_pool(
            llama_context & ctx,
            const llama_model & model,
            const llama_cparams & cparams,
            const std::vector<ggml_backend_t> & backend_ptrs,
            const std::vector<ggml_backend_buffer_type_t> & backend_buft);

    llama_expert_pool_state st;

    llama_context &       ctx;
    const llama_model &   model;
    const llama_cparams & cparams;

    const std::vector<ggml_backend_t> &             backend_ptrs;
    const std::vector<ggml_backend_buffer_type_t> & backend_buft;

    // pooled weight context/buffer + mount-table ctx
    // (kept alive as long as the pool; pointers also stored in st)
    ggml_context *          pool_ctx         = nullptr;
    ggml_context *          pool_tab_ctx     = nullptr; // direct-mount remap tables (pool device)
    ggml_context *          pool_tab_cpu_ctx = nullptr; // direct-mount remap_inv table (CPU device)
    ggml_backend_buffer_ptr pool_buf;
    ggml_backend_buffer_ptr mount_tab_buf;     // device buffer holding the remap tables
    ggml_backend_buffer_ptr mount_tab_cpu_buf; // CPU buffer holding the remap_inv mirror

    void expert_pool_init();
    void expert_pool_build();
    void expert_pool_fill();
    void expert_pool_release();
    void expert_pool_finalize(int32_t id_slot);
    void expert_pool_relayout(bool whole);
    void expert_pool_cross(int32_t n_tokens);
};

using llama_expert_pool_ptr = std::unique_ptr<llama_expert_pool>;

// the pool-side copy of a pool layer's top-k rows (null when the layer is not pooled)
inline ggml_tensor * llama_expert_pool_ids_shadow(const llama_expert_pool_state & st, int32_t il) {
    return il >= 0 && (size_t) il < st.ids_shadow.size() ? st.ids_shadow[il] : nullptr;
}

// is this layer a pool layer? (all eligible layers are: the layout only sets their width)
inline bool llama_expert_pool_is_pool_layer(const llama_expert_pool_state & st, int32_t il) {
    return il >= 0 && (size_t) il < st.pool_flag.size() && st.pool_flag[il] != 0;
}

// ubatch-boundary accounting for the pool's layers: the graph hands every pool layer's
// top-k rows over (the mount chains their own contiguous copy, the plain-path layers a
// direct view at one token and a copy beyond that), and this reads them back once per
// ubatch - the counters, the hit/miss statistics and the step boundary all come from
// here. the split-head hook cannot do it: the plain path forces no split, so a fully
// resident or width-0 layer has no observed head.
void llama_expert_pool_account_ids(llama_expert_pool_state & st);

// per-ubatch slot channel: the sequences (slots) that have columns in this
// ubatch. the segment windows are per slot; the decision counters stay ctx-level.
void llama_expert_pool_set_step_slots(llama_expert_pool_state & st, const int32_t * seq_ids, int32_t n_seq_ids);

// swap worker: owns the counters, the refresh decisions and the weight copies
void llama_expert_pool_start_worker(llama_expert_pool_state & st);
// push one step marker: the rows queued before it are a complete step. hook use
// plus the segment-end drain in llama_context::expert_pool_finalize
void llama_expert_pool_push_marker(llama_expert_pool_state & st);
// settle one queued step marker: run the fills whose unmap reached the tables,
// count the rows, run the top-k refresh (one global queue, capped by the
// per-step swap limit, unmapping the victims), rebuild the mirror.
// returns false when no marker is queued
bool llama_expert_pool_worker_settle(llama_expert_pool_state & st);
// rebuild the merged table mirror from resident[] (worker thread only; no
// tensor writes): picks a free slot, writes it, sets mirror_ready, returns the
// new sequence number
int32_t llama_expert_pool_tab_build(llama_expert_pool_state & st);
// publish the ready mirror (hook thread only, step boundary): one set_async to
// tab_all and one sync set to tab_cpu - the single place the tensors are written
void llama_expert_pool_tab_publish(llama_expert_pool_state & st);
