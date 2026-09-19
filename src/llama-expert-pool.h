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

// single source for the MoE small-batch threshold: env
// GGML_OP_OFFLOAD_MIN_BATCH (same variable and default as the CUDA backend
// and the sched). every batch-size gate in the pool mechanism derives from
// this - never a literal.
inline int32_t llama_expert_pool_offload_min_batch() {
    static const int32_t v = getenv("GGML_OP_OFFLOAD_MIN_BATCH") != nullptr
        ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
    return v;
}

// minimum slots per pooled layer (desert rule): below this width a mounted
// layer pays the per-layer roundtrip tax for near-zero hits, so the budget
// form that did not name the layer count trims the layer set instead of
// spreading the budget thin, while an explicit layer count only warns. 1% of
// the layer's expert count, rounded up (128 -> 2, 256 -> 3, 512 -> 6), which
// brackets the measured net-zero widths on all four models.
inline int32_t llama_expert_pool_min_slots(int32_t n_expert) {
    return (n_expert + 99) / 100;
}

// the CPU MoE delegate is one process-wide slot, refcounted across pools.
// acquiring registers the hook, destroying releases it, so a pool never has to
// remember whether it holds the registration
void llama_expert_pool_delegate_register();
void llama_expert_pool_delegate_unregister();

struct llama_expert_pool_delegate_ref {
    llama_expert_pool_delegate_ref() = default;
    ~llama_expert_pool_delegate_ref() { release(); }

    llama_expert_pool_delegate_ref(const llama_expert_pool_delegate_ref &) = delete;
    llama_expert_pool_delegate_ref & operator=(const llama_expert_pool_delegate_ref &) = delete;

    void acquire() { if (!held) { held = true; llama_expert_pool_delegate_register();   } }
    void release() { if ( held) { held = false; llama_expert_pool_delegate_unregister(); } }

    bool held = false;
};

// ---------------------------------------------------------------
// direct mount (main-graph execution): per-layer tensors that let
// build_moe_ffn run a second, GPU-resident chain over the pool
// weights inside the MAIN graph. the -1 skip ids zero the matching
// column, so the two chains split the columns by construction:
// remap (device, for the GPU chain) sends non-resident experts to -1,
// remap_inv (host, for the CPU chain) sends resident experts to -1.
// a single add merges both chains.
// NOTE: remap/remap_inv are I32: ggml_get_rows supports I32 tables natively
// (the output type follows the table, ggml.c) on every backend, so the ids
// gathering needs no cast. the F32 REPEAT gate on CUDA only matters for the
// scale tables, which are F32.
struct llama_expert_pool_mount {
    bool active = false;
    ggml_tensor * w_gate_up = nullptr; // [n_ff*2, n_embd, S] or null (pool copy)
    ggml_tensor * w_up      = nullptr; // [n_ff, n_embd, S] or null (pool copy)
    ggml_tensor * w_gate    = nullptr;
    ggml_tensor * w_down    = nullptr;
    ggml_tensor * w_down_s  = nullptr; // per-expert down scale source (values
                                       // are staged into `scale` at fill time)
    ggml_tensor * w_up_b      = nullptr; // pool-compacted up bias [n_ff, S]
    ggml_tensor * w_gate_b    = nullptr; // pool-compacted gate bias [n_ff, S]
    ggml_tensor * w_down_b    = nullptr; // pool-compacted down bias [n_embd, S]
                                         // (slot ids index it; the old full-size
                                         // orig misindexed hits, only -1 was safe)
    ggml_tensor * remap     = nullptr; // I32 [1, n_expert] on the pool device:
                                       // resident -> pool slot, non-resident -> -1
    ggml_tensor * remap_inv_host = nullptr; // I32 [1, n_expert] on the CPU device:
                                           // resident -> -1, non-resident -> expert id
                                           // (CPU-segment get_rows host mirror; the
                                           // GPU side has NO inv table - only remap)
    ggml_tensor * scale     = nullptr; // F32 [1, n_expert] on the pool device:
                                       // per-expert down scale (null = no scale)
    ggml_tensor * scale_up   = nullptr; // F32 [1, n_expert]: up scale, factored
                                       // like scale (pre-activation mul)
    ggml_tensor * scale_gate = nullptr; // F32 [1, n_expert]: gate scale, same
};

// ---------------------------------------------------------------
// model-level runtime state of the expert pool (direct-mount mode)
//
// holds, per pooled layer, GPU-resident pool weight tensors (compact
// layout: slot s holds the s-th resident expert, no zero padding) plus
// the routing tables that split the two chains of the direct mount.
// the pool starts from a csv seed (--expert-pool-init, debug) or
// random; the top-k refresh updates the content per step.

// per-layer expert tensors that the pool copies and the swap moves. the kind
// order is fixed: it is the array index used everywhere (extend here plus in
// the per-layer source table of expert_pool_build when a tensor lands)
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

// per-expert byte stride of a pool or source tensor: the expert dimension is
// ne[2] for the 4D weight copies, ne[1] for the compact 2D bias copies (the
// nb[2] of a 2D tensor is the whole tensor size, not the expert stride)
inline size_t llama_expert_pool_stride(const ggml_tensor * t, int k) {
    return k < PK_UP_B ? t->nb[2] : t->nb[1];
}

struct llama_expert_pool_layer {
    ggml_tensor * orig[PK_N] = {}; // model weight tensor (host); null = not present
    ggml_tensor * pool[PK_N] = {}; // pool copy (pool device); null = not pooled
};

// hit/miss row counters: per pooled layer (snapshot semantics, a read clears)
// and one segment total (printed by expert_pool_finalize)
struct llama_expert_pool_counts {
    uint64_t hit  = 0;
    uint64_t miss = 0;
    // route rows the hook could not attribute (no readable original top-k), counted loudly instead of degenerating to miss-only counting
    uint64_t rows_skipped = 0;
};

struct llama_expert_pool_state {
    // per-layer expert tensors, indexed by layer id (entries stay empty for
    // layers the pool does not own). compact layout: pool slot s holds the
    // s-th resident expert (S slots, ne2 = S, no zero padding)
    std::vector<llama_expert_pool_layer> layers;

    // direct-mount registry, indexed by layer id (setup writes; the graph,
    // the hook and the worker read). setup passes whole mounts by value - the
    // vector may reallocate while it grows, so never hold a reference into it
    std::vector<llama_expert_pool_mount> mounts;
    void set_mount(int il, const llama_expert_pool_mount & m);
    const llama_expert_pool_mount & mount(int il) const;  // read-only (no growth)

    // resident expert lists, indexed by layer (for diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // pooled layer indices (filled at init, consumed by the delayed fill)
    std::vector<int32_t> pooled_layers;
    // host expert tensor -> (layer, index into pooled_layers), registered while
    // the pool is built: the CPU hook then identifies its node with one lookup
    struct tensor_ref {
        int32_t il;
        int32_t ilx;
    };
    std::unordered_map<const ggml_tensor *, tensor_ref> tensor_refs;

    // set once the pool weights/tables have been copied (idempotent fill):
    // part of the phase below

    // the CPU MoE delegate registration belongs to this state: the ref
    // registers on acquire() and unregisters in its destructor, so teardown
    // order releases it and no flag tracks whether we hold it
    llama_expert_pool_delegate_ref delegate_ref;

    // direct mount: a second GPU-resident
    // expert chain runs inside the main graph; the -1 skip ids zero the
    // non-resident columns on the GPU chain (and the resident columns on the
    // CPU chain via the inverse table), so no delegate hook is needed
    bool direct_mount = false;
    ggml_backend_buffer_type_t pool_buft = nullptr;        // pool buft (device)

    // decode-step boundary detector (single-layer-safe): set when the last
    // active-mount layer of a step is served, consumed at the first
    // active-mount layer of the next step; anchors the swap publish and the
    // step flag forwarded to the route observer (llama-ext.h)
    bool   step_done = false;

    // swap (on by default with -nep): decaying activation count of expert
    // activations, one entry per (pooled layer, expert); the top-k refresh
    // converges the resident set to the k most used experts (k = slot count),
    // at most swap_per_step expert pairs per settled step (a burst fuse - with
    // the default half-life the steady-state rate sits well below it)
    bool swap_auto = false;
    int32_t swap_per_step = 40;            // max expert pairs swapped in per
                                           // settled step, across all pooled
                                           // layers (negative = unlimited)
    int32_t n_expert = 0;                  // experts per layer (set at init)
    // hook-side step counter (the hook only pushes routing rows; the swap
    // worker owns the counters below)
    int32_t hook_step = 0;
    // per-step per-layer gate: the hook fires once per MUL_MAT_ID node
    // (2-3 per layer per step) and the ids are identical across a layer's
    // nodes, so the row push and the hit/miss counting run once per layer
    int32_t last_ilx = -1;                 // last layer handled this step

    // routing rows from the delegate hook to the swap worker. one
    // block per (step, layer): the FULL activation list of the layer -
    // resident entries are recovered from the original top-k ids at the -1
    // positions of the inverse-remapped ids the CPU chain received, so the
    // counter counts BOTH sides (miss + hit) and the eviction side is no
    // longer blind. the row carries its own mapping facts (the -1 pattern),
    // so the worker needs no table-version to attribute it.
    struct route_block {
        int32_t step = 0;                  // decode step this row belongs to
        int32_t ilx  = -1;                 // pooled layer index (-1 = step marker)
        int32_t n_tok = 1;                 // token columns of the batch (decay mode: increments are 1/n_tok)
        std::vector<int32_t> ids;          // FULL activation expert ids (pool hits recovered from the original top-k row)
    };
    std::mutex      route_mtx;
    std::condition_variable route_cv;
    std::deque<route_block> route_q;
    bool route_stop = false;

    // worker-owned swap state (only the swap worker thread touches these):
    // the activation counters and the refresh decisions. the hook
    // NEVER reads them - it only pushes route rows and, at each step
    // boundary, publishes the mirror the worker hands over.
    int32_t settled_steps = 0;             // decode steps accounted so far
    // decaying activation counter (--expert-pool-swap-decay H): per settled step
    // every count is multiplied by swap_lambda and the step's rows land with
    // per-token normalized increments (1 / n_tok per activation), so a batch of
    // n token columns contributes one step's worth of evidence instead of n.
    // float by design: the value is a weight, not a count.
    float   swap_lambda   = 0.0f;          // per-step decay factor, lambda = 2^(-1/H)
    int32_t swap_decay_hl = 96;            // decay half-life in decode steps (default 96)
    std::vector<float> act_cnt;            // [pooled layers * n_expert] decayed activation counts
    int32_t swap_sum = 0;                  // exchanges accumulated this period

    // segment totals of the inference-side counters; independent of the
    // per-layer stat array (which is cleared by each read)
    llama_expert_pool_counts seg;

    // swap worker thread: consumes the route rows, attributes
    // counts, runs the top-k refresh, performs the H2D weight copies synchronously (off the
    // inference thread, off the main graph stream), and rebuilds the merged
    // table mirror. the tables themselves are only written by the hook at a
    // step boundary (tab_publish) - the worker does not touch the table
    // tensors, so no stream/CPU-side ordering is lost.
    std::thread cp_worker;
    // mirror handshake: three mirrored table slots. the worker rebuilds
    // mirror[k] then sets ready=k; the hook publishes the ready mirror (one
    // set_async to tab_all + one sync set to tab_cpu) once per step boundary
    // and records pub=k. the worker only writes slots that are neither ready
    // nor pub, so an async set's source is never overwritten inside the
    // two-step distance.
    std::atomic<int32_t> mirror_ready{-1};
    std::atomic<int32_t> mirror_pub{-1};

    // sequence handshake of the two-phase swap: every built mirror carries a
    // sequence number, and the hook records the sequence it published. a
    // pending slot fill runs only once the mirror that unmapped its slot has
    // reached the tables - a slot's content may not change while an expert is
    // still mapped onto it.
    int32_t seq_ctr = 0;                    // worker-owned mirror counter
    int32_t mirror_seq[3] = { -1, -1, -1 }; // worker-owned, per mirror slot
    std::atomic<int32_t> pub_seq{-1};       // hook-owned: sequence published
    struct pending_fill {
        int32_t il   = -1;
        int32_t e    = -1; // expert the fill writes into the slot
        int32_t slot = -1;
        int32_t seq  = 0;  // mirror sequence that unmapped the slot (0 = not built yet)
        float   cnt  = 0.0f;
    };
    std::vector<pending_fill> pend_fill;    // worker-owned fill queue

    // phase: NONE -> BUILT (expert_pool_build() ran; sched_reserve() re-enters
    // expert_pool_init after a rebuild, and a reset() would wipe the fresh
    // pool) -> FILLED (weights and tables copied, on the first compute).
    // one field, so "built" and "filled" cannot disagree
    enum phase_t : uint8_t { PHASE_NONE = 0, PHASE_BUILT, PHASE_FILLED };
    phase_t phase = PHASE_NONE;
    int32_t last_active_ilx = -1;          // pooled index of the last layer with an active
                                           // mount (the step-boundary anchor - not always
                                           // the last pooled layer: a 0-slot layer has no
                                           // mount and its hook early-returns)
    int32_t first_active_ilx = -1;         // pooled index of the first layer with an active
                                           // mount (start boundary anchor; symmetric)
    // per-pooled-layer hit/miss counters (direct mount: the CPU segment
    // receives remap_inv ids, so e < 0 means the GPU pool chain computed the
    // row and e >= 0 is a CPU miss); read (and cleared) via
    // llama_expert_pool_get_stats
    std::vector<llama_expert_pool_counts> stat; // [pooled layers]

    // backend owning the pool buft (the hook publishes the tables on its
    // main stream, see tab_publish below)
    ggml_backend_t pool_backend = nullptr;

    // merged mount tables: the remap half lives in tab_all on the pool
    // device (read by the GPU chain's get_rows), the remap_inv half in
    // tab_cpu on the CPU device (read by the miss chain's get_rows). both
    // are [n_expert, n_layers] I32; each layer's remap/remap_inv_host are
    // views sliced from them. the host mirrors rebuilt by the worker and
    // published by the hook (step-granular swap update).
    ggml_tensor * tab_all = nullptr;             // [n_expert, n_layers] remap (pool device)
    // CPU-hosted copy of the inverse table: the miss chain's get_rows reads
    // remap_inv_host from HERE (host memory), so its ids are not tied to the
    // pool segment's 32B output slot. [n_expert, n_layers] layout,
    // filled in the same tab_build/tab_publish flush as tab_all.
    ggml_tensor * tab_cpu = nullptr;             // [n_expert, n_layers] remap_inv (CPU buft)
    // host mirror slots of the merged table (three, see mirror_ready/pub):
    // the worker writes a slot only when it is neither the ready one nor the
    // published one, so an async set's source buffer stays untouched for at
    // least one publish cycle.
    std::vector<int32_t> tab_mirror[3];

    void reset();
    // signal + drain the route queue + join the swap worker (also safe when
    // no worker is running); shared by reset() and the ctx teardown
    void stop_worker();
};

// seed the pool from a csv file, one line per layer: "il,e1,e2,...". the
// per-layer slot count = number of entries on that layer's line (the file
// itself determines the distribution; nothing is padded with random experts).
// layers missing from the file get zero slots (the layer falls back to full
// CPU compute). returns false only if the file cannot be opened.
bool llama_expert_pool_parse_init(const std::string & path, int32_t n_layer,
                                  int32_t n_expert,
                                  std::vector<std::vector<int32_t>> & resident);

// random resident set per layer (fixed seed, reproducible); `widths` is
// indexed by layer number (0 = layer not pooled)
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert,
                              const std::vector<int32_t> & widths,
                              std::vector<std::vector<int32_t>> & resident);

// moe delegate hook: called by the CPU MUL_MAT_ID kernel (ith==0); feeds the
// activation counter and the hit/miss counters, and fans the served ids out to the route
// observer (llama-ext.h). returns a null skip table: no rows are skipped,
// column zeroing is done by the -1 ids natively.
void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud);

// swap worker: the worker owns the activation counters, the top-k refresh
// decisions and the weight copies; the hook only pushes
// routing rows and publishes ready mirrors at step boundaries.
void llama_expert_pool_start_worker(llama_expert_pool_state & st);
// push one step marker: the rows queued before it are now a complete step
// and the worker may settle them. hook use, plus the segment-end drain in
// llama_context::expert_pool_finalize (which has no decode in flight).
void llama_expert_pool_push_marker(llama_expert_pool_state & st);
// settle one queued step marker: the rows pushed before it are the previous
// step's FULL activation counts (resident + non-resident); the fills whose
// unmap reached the tables are executed first, then the rows are counted into
// the counter, then the top-k refresh runs (all layers' pairs in one global
// queue, capped by the per-step swap limit, unmapping the victims) and the
// ready mirror is rebuilt.
// returns false when no marker is queued.
bool llama_expert_pool_worker_settle(llama_expert_pool_state & st);
// rebuild the merged table mirror from resident[] (worker thread only: picks
// a free slot, writes it, sets mirror_ready and returns the new sequence
// number). no tensor writes here.
int32_t llama_expert_pool_tab_build(llama_expert_pool_state & st);
// publish the ready mirror (hook thread only, at a step boundary): one
// set_async to tab_all and one sync set to tab_cpu. the single place the
// table tensors are written, so the old stream/CPU ordering is preserved.
void llama_expert_pool_tab_publish(llama_expert_pool_state & st);

// per-context pool plumbing: the CPU MoE delegate is one process-wide slot,
// refcounted across pools; the active pool of the current thread is marked
// around llama_context::graph_compute (the hook resolves its state from it)
llama_expert_pool_state * llama_expert_pool_set_current(llama_expert_pool_state * st);

// scoped binding for that mark: the hook must be told which pool a graph
// belongs to while the compute runs, and every exit path has to restore the
// previous value - an object does that by construction, a hand-paired
// set/restore does not.
struct llama_expert_pool_bind {
    explicit llama_expert_pool_bind(llama_expert_pool_state * st) : prev(llama_expert_pool_set_current(st)) {}
    ~llama_expert_pool_bind() { llama_expert_pool_set_current(prev); }

    llama_expert_pool_bind(const llama_expert_pool_bind &) = delete;
    llama_expert_pool_bind & operator=(const llama_expert_pool_bind &) = delete;

    llama_expert_pool_state * prev;
};
