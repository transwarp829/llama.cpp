#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <utility>

// ---------------------------------------------------------------
// model-level runtime state of the expert pool (direct-mount mode)
//
// holds, per pooled layer, GPU-resident pool weight tensors (compact
// layout: slot s holds the s-th resident expert, no zero padding) plus
// the routing tables that split the two chains of the direct mount.
// the pool starts from a csv seed (GGML_EXPPOOL_INIT_CSV, debug) or
// random; the marginal exchange refreshes the content per step and the
// segment-end reallocation refits the slot widths (see 阶段3-设计.md).
struct llama_expert_pool_state {
    bool enabled = false;

    // the original weight tensors, indexed by layer (null = not present); used by
    // the graph to pair a mul_mat_id weight with its pool copy
    std::vector<ggml_tensor *> orig_gate_up;
    std::vector<ggml_tensor *> orig_up;
    std::vector<ggml_tensor *> orig_gate;
    std::vector<ggml_tensor *> orig_down;

    // pool weight tensors, indexed by layer; null = not pooled / not present.
    // compact layout: slot s holds the s-th resident expert (S slots per
    // layer, ne2 = S, no zero padding)
    std::vector<ggml_tensor *> w_pool_gate_up; // fused [n_ff*2, n_embd, S]
    std::vector<ggml_tensor *> w_pool_up;      // separate [n_ff, n_embd, S]
    std::vector<ggml_tensor *> w_pool_gate;    // separate [n_ff, n_embd, S]
    std::vector<ggml_tensor *> w_pool_down;    // [n_embd, n_ff, S]

    // resident expert lists, indexed by layer (for diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // pooled layer indices (filled at init, consumed by the delayed fill)
    std::vector<int32_t> pooled_layers;

    // set once the pool weights/tables have been copied (idempotent fill)
    bool fill_done = false;

    // direct mount (GGML_EXPPOOL_MOUNT=0 disables it): a second GPU-resident
    // expert chain runs inside the main graph; PR #26631 -1 ids zero the
    // non-resident columns on the GPU chain (and the resident columns on the
    // CPU chain via the inverse table), so no delegate hook is needed
    bool direct_mount = false;
    bool rtlog_only = false;     // GGML_EXPPOOL_ROUTING_LOG with no pool: the
                                 // hook only feeds the routing log (no slots,
                                 // no delegation); set in expert_pool_init()
    ggml_backend_buffer_type_t pool_buft = nullptr;        // pool buft (device)

    // runtime routing log (GGML_EXPPOOL_ROUTING_LOG=<path>; ONLY for analysis,
    // writes the ids seen by the CPU mul_mat_id kernel: "step,layer,id0,id1,..".
    // with direct mount the ids are the inverse-remap values (resident = -1,
    // non-resident = expert id), so hit ratio = share of -1 entries.
    FILE * rt_log = nullptr;               // opened lazily on first begin
    bool   rt_log_tried = false;           // env already checked (avoid re-getenv)
    uint64_t log_step = 0;                 // current decode step (incremented at ilx==0)
    int32_t logged_il = -1;                // last logged layer id (dedup per step)
    bool   rt_step_done = false;           // last pooled layer logged since the last
                                           // step advance (single-layer-safe step detect)

    // stage 3 swap (on by default with -nep): sliding decode window count of
    // expert activations, one entry per (pooled layer, expert); the marginal
    // exchange refreshes the resident set one pair per layer per step
    bool swap_auto = false;
    int32_t swap_W = 512;                  // window length in decode steps
    int32_t swap_sigma = 3;                // marginal-exchange confidence (z sigma)
    int32_t n_expert = 0;                  // experts per layer (set at init)
    int32_t win_step = 0;                  // decode steps accounted in the window
    std::vector<int32_t> win_cnt;          // [pooled layers * n_expert]
    std::vector<std::vector<int32_t>> win_hist; // [W] flat (ilx, e) pairs per step

    // built flag: expert_pool_build() has run (sched_reserve() re-enters
    // expert_pool_init after a rebuild, and a reset() would wipe the fresh
    // pool)
    bool pool_ready = false;               // pool allocation finished
    int32_t budget_slots = 0;              // -nep total slot budget
    int32_t last_active_ilx = -1;          // pooled index of the last layer with an active
                                           // mount (the step-boundary anchor - not always
                                           // the last pooled layer: a 0-slot layer has no
                                           // mount and its hook early-returns)
    int32_t first_active_ilx = -1;         // pooled index of the first layer with an active
                                           // mount (start boundary anchor; symmetric)
    // cumulative routing counter (infinite window): the segment-end width
    // reallocation reads the global top-N pairs from this table; never
    // decayed, so the prefill's small under-count is negligible noise.
    std::vector<int32_t> seg_cnt;          // [pooled layers * n_expert]
    // per-pooled-layer hit/miss counters (direct mount: the CPU segment
    // receives remap_cpu ids, so e < 0 means the GPU pool chain computed the
    // row and e >= 0 is a CPU miss); read (and cleared) via
    // llama_expert_pool_get_stats
    std::vector<uint64_t> stat_hit;        // [pooled layers]
    std::vector<uint64_t> stat_miss;       // [pooled layers]
    // swap-window aggregates (cleared by each swap; independent of the
    // per-layer snapshot counters above, which a frequent stats reader
    // resets)
    uint64_t win_hit  = 0;
    uint64_t win_miss = 0;

    // marginal-swap exchange pipeline. decisions are made at the step
    // boundary (win_cnt statistics); the weight copies run on a DEDICATED
    // worker thread (sync tensor_set), fully decoupled from the main-graph
    // stream and from inference - the step only pushes requests and, at the
    // next boundary, commits the ones the worker reported done. this keeps
    // the swap-in cost off the main graph (no FIFO with the GPU segment)
    // and guarantees one in-flight copy per slot (the slot stays -1 until
    // the worker completes; a re-exchange of the same slot waits for the
    // commit, which only happens after the copy finished).
    ggml_backend_t pool_backend = nullptr;   // backend owning the pool buft
    struct pending_exchange {
        int32_t il;      // pooled layer index
        int32_t e;       // incoming expert id
        int32_t slot;    // pool slot it occupies
    };
    // swap summary aggregation (periodic INFO line, like print_timings)
    int32_t swap_sum = 0;                    // exchanges accumulated this period

    // asynchronous copy worker (stage 3, swap tax -> 0):
    //  - the step's run_swap() only decides and queues requests;
    //  - the worker performs the H2D copies (ggml_backend_tensor_set, sync on
    //    the worker's own context) and reports done;
    //  - the next step boundary publishes completed fills (double sync point,
    //    no torn slots: a slot is unmaped in the tables when its fill is
    //    queued and only remapped after the copy completed).
    std::thread cp_worker;
    std::mutex cp_mtx;
    std::condition_variable cp_cv;
    std::deque<pending_exchange> cp_todo;    // requests, worker pops
    std::deque<pending_exchange> cp_done;    // completed, step consumes
    bool cp_stop = false;                    // worker shutdown flag
    std::vector<int32_t> cp_inflight;        // [pooled layers] slot of an in-flight fill
                                             // (or -1); protects against double-fill

    // merged mount tables (9/1): all layers' remap/remap_cpu live in ONE
    // contiguous [2*n_expert, n_layers] I32 tensor; each layer's views are
    // sliced from it. the host mirror is rebuilt from resident[] once per
    // step and flushed with a single tensor_set (step-granular swap update).
    ggml_tensor * tab_all = nullptr;             // [2*n_expert, n_layers]
    std::vector<int32_t> tab_mirror;             // [2*n_expert * n_layers]

    void reset();
};

// seed the pool from a csv file, one line per layer: "il,e1,e2,...". the
// per-layer slot count = number of entries on that layer's line (the file
// itself determines the distribution; nothing is padded with random experts).
// layers missing from the file get zero slots (the layer falls back to full
// CPU compute). returns false only if the file cannot be opened.
bool llama_expert_pool_parse_init(const std::string & path, int32_t n_layer,
                                  int32_t n_expert,
                                  std::vector<std::vector<int32_t>> & resident);

// random resident set per layer (fixed seed, reproducible)
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert, int32_t n_slot,
                              std::vector<std::vector<int32_t>> & resident);

// moe routing-log hook: called by the CPU MUL_MAT_ID kernel (ith==0), feeds
// GGML_EXPPOOL_ROUTING_LOG (see llama-expert-pool.h). returns a null skip
// table: no rows are skipped, column zeroing is done by the -1 ids natively.
void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud);

// stage 3: one marginal exchange per pooled layer per step. called at a
// decode step boundary from the delegate hook (swap is on by default with -nep).
void llama_expert_pool_run_swap(llama_expert_pool_state & st);
void llama_expert_pool_start_worker(llama_expert_pool_state & st);
// merged mount-table refresh: rebuild the host mirror from resident[] and
// flush it with a single tensor_set (step-granular update).
void llama_expert_pool_tab_sync(llama_expert_pool_state & st);

// segment-end width allocation: the global top-N (layer, expert) pairs of
// the cumulative activation counts (N = budget slots) decide the per-layer
// slot widths (1-2-slot layers fall back to 0: the desert rule). returns
// false when the counts are too sparse to rank (the caller keeps the
// current layout).
bool llama_expert_pool_alloc_from_counts(
        const std::vector<int32_t> & counts, int32_t P, int32_t n_expert,
        int32_t budget, std::vector<std::vector<int32_t>> & resident);

// ---------------------------------------------------------------
// direct mount (main-graph execution): per-layer tensors that let
// build_moe_ffn run a second, GPU-resident chain over the pool
// weights inside the MAIN graph. PR #26631 -1 ids zero the matching
// column, so the two chains split the columns by construction:
// remap (device, for the GPU chain) sends non-resident experts to -1,
// remap_cpu (host, for the CPU chain) sends resident experts to -1.
// a single add merges both chains.
// NOTE: remap/remap_cpu are I32: ggml_get_rows supports I32 tables natively
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
    ggml_tensor * w_down_b  = nullptr; // per-expert down bias (add_id -1 makes
                                       // it a no-op for skipped columns)
    ggml_tensor * remap     = nullptr; // I32 [1, n_expert] on the pool device:
                                       // resident expert e -> slot s, non-resident -> -1
    ggml_tensor * remap_cpu = nullptr; // I32 [1, n_expert] on the pool device:
                                       // resident expert e -> -1, non-resident -> e
    ggml_tensor * scale     = nullptr; // F32 [1, n_expert] on the pool device:
                                       // per-expert down scale (null = no scale)
};

void llama_expert_pool_register_mount(int il, const llama_expert_pool_mount & mount);
llama_expert_pool_mount & llama_expert_pool_get_mount(int il);
void llama_expert_pool_clear_mount();
