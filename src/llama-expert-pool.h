#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
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

// miss method (llama_expert_pool_params::miss_method); the threshold above
// which the miss chain goes to the pool device applies to every method.
enum llama_expert_pool_miss_method {
    LLAMA_EXPERT_POOL_MISS_CPU_SERIAL   = 0, // CPU chain as one split (delivery)
    LLAMA_EXPERT_POOL_MISS_CPU_PARALLEL = 1, // mount chain submitted ahead of the CPU chain
    LLAMA_EXPERT_POOL_MISS_GPU          = 2, // miss rows on the pool device at every batch size
};

// minimum slots per pooled layer (desert rule): below this width a mounted
// layer pays the per-layer roundtrip tax for near-zero hits, so an unnamed
// layer count trims the layer set instead of spreading the budget thin (an
// explicit count only warns). 1% of the expert count, which brackets the
// measured net-zero widths.
inline int32_t llama_expert_pool_min_slots(int32_t n_expert) {
    return (n_expert + 99) / 100;
}

// ---------------------------------------------------------------
// direct mount (main-graph execution): per-layer tensors that let build_moe_ffn
// run a second, device-resident chain over the pool weights inside the MAIN
// graph. the -1 skip ids split the columns by construction (resident -> pool
// slot, non-resident -> -1), so one add merges the two chains.
// the tables are I32: ggml_get_rows keeps the table type on every backend, so
// the id gathering needs no cast.
struct llama_expert_pool_mount {
    bool active = false;
    ggml_tensor * w_gate_up = nullptr; // [n_ff*2, n_embd, S] or null (pool copy)
    ggml_tensor * w_up      = nullptr; // [n_ff, n_embd, S] or null (pool copy)
    ggml_tensor * w_gate    = nullptr;
    ggml_tensor * w_down    = nullptr;
    ggml_tensor * w_down_s  = nullptr; // per-expert down scale source (staged into `scale`)
    ggml_tensor * w_up_b      = nullptr; // pool-compacted up bias [n_ff, S]
    ggml_tensor * w_gate_b    = nullptr; // pool-compacted gate bias [n_ff, S]
    ggml_tensor * w_down_b    = nullptr; // pool-compacted down bias [n_embd, S] (slot ids index it)
    ggml_tensor * remap     = nullptr; // I32 [1, n_expert] on the pool device: resident -> slot, else -1
    ggml_tensor * remap_inv_host = nullptr; // I32 [1, n_expert] on the CPU: resident -> -1, else expert id
                                            // (read when the miss mmids stay on the CPU)
    ggml_tensor * remap_inv = nullptr; // I32 [1, n_expert] on the pool device: the inv half, gathered
                                      // device-side when the miss mmids run there
    ggml_tensor * scale     = nullptr; // F32 [1, n_expert] on the pool device: per-expert down scale
    ggml_tensor * scale_up   = nullptr; // F32 [1, n_expert]: up scale, factored like scale (pre-activation)
    ggml_tensor * scale_gate = nullptr; // F32 [1, n_expert]: gate scale, same
};

// ---------------------------------------------------------------
// model-level runtime state (direct-mount mode): per pooled layer the
// device-resident pool weight tensors (compact: slot s holds the s-th resident
// expert) plus the routing tables. content = csv seed (--expert-pool-init,
// debug) or random, refreshed per step.

// per-layer expert tensors the pool copies and the swap moves. the kind order
// is the array index used everywhere (extend here and in the per-layer source
// table of expert_pool_build when a tensor lands)
enum llama_expert_pool_kind {
    PK_GATE_UP = 0, // fused gate+up [n_ff*2, n_embd, n_expert]; pool [.., S]
    PK_UP,          // separate up [n_ff, n_embd, n_expert]
    PK_GATE,        // separate gate [n_ff, n_embd, n_expert]
    PK_DOWN,        // down [n_embd, n_ff, n_expert]
    PK_UP_B,        // per-expert bias [n_ff, n_expert]; pool [n_ff, S]
    PK_GATE_B,      // per-expert bias [n_ff, n_expert]
    PK_DOWN_B,      // per-expert bias [n_embd, n_expert]
    PK_N,
};

// per-expert byte stride: ne[2] for the 4D weight copies, ne[1] for the
// compact 2D bias copies (nb[2] of a 2D tensor is the whole tensor size)
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
    // setup passes whole mounts by value - the vector may reallocate while it
    // grows, so never hold a reference into it
    std::vector<llama_expert_pool_mount> mounts;
    void set_mount(int il, const llama_expert_pool_mount & m);
    const llama_expert_pool_mount & mount(int il) const;  // read-only (no growth)

    // resident expert lists, indexed by layer (diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // pooled layer indices (filled at init, consumed by the delayed fill)
    std::vector<int32_t> pooled_layers;

    // split-head registry, rebuilt per graph build (the builder registers each
    // pooled layer's clean topk ids and the miss chain's first node)
    struct split_head_ref {
        int32_t il   = -1;
        int32_t ilx  = -1;
        bool    lane = false;  // true: read the clean rows from the device
    };
    mutable std::unordered_map<const ggml_tensor *, split_head_ref> split_head_refs;
    mutable int32_t                          bind_last_il = -1;
    mutable std::vector<const ggml_tensor *> ids_clean;   // [layer] clean topk ids
    mutable std::vector<int32_t>             ids_n_used;  // [layer]
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
                                           // step, all layers (negative = unlimited)
    int32_t n_expert = 0;                  // experts per layer (set at init)
    // step counter (only the worker owns the counters below)
    int32_t hook_step = 0;
    // once per (step, layer): the observer reports every mmid split head of a
    // layer and the ids are identical across them
    int32_t last_ilx = -1;                 // last layer handled this step

    // routing rows from the observer to the swap worker: one block per
    // (step, layer) carrying the layer's full activation ids
    struct route_block {
        int32_t step = 0;                  // decode step this row belongs to
        int32_t ilx  = -1;                 // pooled layer index (-1 = step marker)
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
    std::vector<float> act_cnt;            // [pooled layers * n_expert] decayed counts
    int32_t swap_sum = 0;                  // exchanges accumulated this period

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

// csv seed, one line per layer: "il,e1,e2,...". the line length sets that
// layer's slot count; layers missing from the file get zero slots (full CPU
// compute). returns false only if the file cannot be opened.
bool llama_expert_pool_parse_init(const std::string & path, int32_t n_layer,
                                  int32_t n_expert,
                                  std::vector<std::vector<int32_t>> & resident);

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
