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

// single source for the minimum slots per pooled layer (desert rule): below
// this width a mounted layer pays the per-layer roundtrip tax for near-zero
// hits, so the pool trims the layer set instead of spreading the budget thin.
// default = 1% of the layer's expert count, rounded up (128 -> 2, 256 -> 3,
// 512 -> 6), which brackets the measured net-zero widths on all four models;
// GGML_EXPPOOL_MIN_SLOTS overrides with an absolute slot count (1 = no
// minimum). used by the init allocation (desert trim).
inline int32_t llama_expert_pool_min_slots(int32_t n_expert) {
    static const int32_t v = getenv("GGML_EXPPOOL_MIN_SLOTS") != nullptr
        ? atoi(getenv("GGML_EXPPOOL_MIN_SLOTS")) : 0;
    return v > 0 ? v : (n_expert + 99) / 100;
}

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
// random; the marginal exchange refreshes the content per step.

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

struct llama_expert_pool_layer {
    ggml_tensor * orig[PK_N] = {}; // model weight tensor (host); null = not present
    ggml_tensor * pool[PK_N] = {}; // pool copy (pool device); null = not pooled
};

// hit/miss row counters: per pooled layer (snapshot semantics, a read clears)
// and one segment total (printed by expert_pool_finalize)
struct llama_expert_pool_counts {
    uint64_t hit  = 0;
    uint64_t miss = 0;
};

struct llama_expert_pool_state {
    bool enabled = false;

    // per-layer expert tensors, indexed by layer id (entries stay empty for
    // layers the pool does not own). compact layout: pool slot s holds the
    // s-th resident expert (S slots, ne2 = S, no zero padding)
    std::vector<llama_expert_pool_layer> layers;

    // direct-mount registry, indexed by layer id (setup writes; the graph,
    // the hook and the worker read)
    std::vector<llama_expert_pool_mount> mounts;
    void register_mount(int il, const llama_expert_pool_mount & m);
    llama_expert_pool_mount & mount(int il);              // setup (auto-grows)
    const llama_expert_pool_mount & mount(int il) const;  // read-only (no growth)

    // resident expert lists, indexed by layer (for diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // pooled layer indices (filled at init, consumed by the delayed fill)
    std::vector<int32_t> pooled_layers;

    // set once the pool weights/tables have been copied (idempotent fill)
    bool fill_done = false;

    // the CPU MoE delegate registration belongs to this state (one process-wide
    // slot, refcounted: register once per pool, drop it in the ctx dtor)
    bool delegate_registered = false;

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

    // stage 3 swap (on by default with -nep): sliding decode window count of
    // expert activations, one entry per (pooled layer, expert); the rate-
    // gated top-k refresh converges the resident set to the window's k most
    // used experts (k = slot count), at most swap_per_step expert pairs per
    // settled step
    bool swap_auto = false;
    int32_t swap_W = 512;                  // window length in decode steps
    int32_t swap_per_step = 10;            // max expert pairs swapped in per
                                           // settled step, across all pooled
                                           // layers (negative = unlimited);
                                           // the swap rate control of the
                                           // window top-k refresh (the window
                                           // warm-up fill is fast by design;
                                           // this caps the tail)
    int32_t n_expert = 0;                  // experts per layer (set at init)
    // hook-side step counter (the hook only pushes routing rows; the swap
    // worker owns the window below)
    int32_t hook_step = 0;
    // per-step per-layer gate: the hook fires once per MUL_MAT_ID node
    // (2-3 per layer per step) and the ids are identical across a layer's
    // nodes, so the row push and the hit/miss counting run once per layer
    int32_t last_ilx = -1;                 // last layer handled this step

    // routing rows from the delegate hook to the swap worker. one
    // block per (step, layer): the FULL activation list of the layer -
    // resident entries are recovered from the original top-k ids at the -1
    // positions of the inverse-remapped ids the CPU chain received, so the
    // window counts BOTH sides (miss + hit) and the eviction side is no
    // longer blind. the row carries its own mapping facts (the -1 pattern),
    // so the worker needs no table-version to attribute it.
    struct route_block {
        int32_t step = 0;                  // decode step this row belongs to
        int32_t ilx  = -1;                 // pooled layer index (-1 = step marker)
        std::vector<int32_t> ids;          // FULL activation expert ids (pool hits recovered from the original top-k row)
    };
    std::mutex      route_mtx;
    std::condition_variable route_cv;
    std::deque<route_block> route_q;
    bool route_stop = false;

    // worker-owned swap state (only the swap worker thread touches these):
    // window counts, history, and the marginal exchange decisions. the hook
    // NEVER reads them - it only pushes route rows and, at each step
    // boundary, publishes the mirror the worker hands over.
    int32_t win_step = 0;                  // decode steps accounted in the window
    std::vector<int32_t> win_cnt;          // [pooled layers * n_expert] FULL counts
    std::vector<std::vector<int32_t>> win_hist; // [W] flat (ilx, e) pairs per step
    int32_t swap_sum = 0;                  // exchanges accumulated this period

    // segment totals of the inference-side counters; independent of the
    // per-layer stat array (which is cleared by each read)
    llama_expert_pool_counts win;

    // swap worker thread: consumes the route rows, attributes
    // counts, runs the marginal exchange (one pair per pooled layer per step
    // settled), performs the H2D weight copies synchronously (off the
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
    // two-step distance (see the old ping-pong comment below).
    std::atomic<int32_t> mirror_ready{-1};
    std::atomic<int32_t> mirror_pub{-1};

    // built flag: expert_pool_build() has run (sched_reserve() re-enters
    // expert_pool_init after a rebuild, and a reset() would wipe the fresh
    // pool)
    bool pool_ready = false;               // pool allocation finished
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
// swap window / hit-miss counters and fans the served ids out to the route
// observer (llama-ext.h). returns a null skip table: no rows are skipped,
// column zeroing is done by the -1 ids natively.
void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud);

// stage 3 swap worker: the worker owns the window counters, the
// marginal exchange decisions and the weight copies; the hook only pushes
// routing rows and publishes ready mirrors at step boundaries.
void llama_expert_pool_start_worker(llama_expert_pool_state & st);
// push one step marker: the rows queued before it are now a complete step
// and the worker may settle them. hook use, plus the segment-end drain in
// llama_context::expert_pool_finalize (which has no decode in flight).
void llama_expert_pool_push_marker(llama_expert_pool_state & st);
// settle one queued step marker: the rows pushed before it are the previous
// step's FULL activation counts (resident + non-resident); they are counted
// into the window, then the marginal exchange runs (one pair per pooled
// layer per settled step, capped by the per-step swap limit), and the
// ready mirror is rebuilt.
// returns false when no marker is queued.
bool llama_expert_pool_worker_settle(llama_expert_pool_state & st);
// rebuild the merged table mirror from resident[] (worker thread only: picks
// a free slot, writes it, sets mirror_ready). no tensor writes here.
void llama_expert_pool_tab_build(llama_expert_pool_state & st);
// publish the ready mirror (hook thread only, at a step boundary): one
// set_async to tab_all and one sync set to tab_cpu. the single place the
// table tensors are written, so the old stream/CPU ordering is preserved.
void llama_expert_pool_tab_publish(llama_expert_pool_state & st);

// per-context pool plumbing: the CPU MoE delegate is one process-wide slot,
// refcounted across pools; the active pool of the current thread is marked
// around llama_context::graph_compute (the hook resolves its state from it)
llama_expert_pool_state * llama_expert_pool_set_current(llama_expert_pool_state * st);
void llama_expert_pool_delegate_register();
void llama_expert_pool_delegate_unregister();
