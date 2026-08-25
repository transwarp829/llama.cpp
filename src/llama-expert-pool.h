#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

// GPU-resident expert pool (stage 2 of the expert cache, milestone 1).
//
// holds a fixed number of expert slots per MoE layer. the slot table maps
// expert id -> slot (or -1 when not resident). the swap engine computes the
// delta between the current resident set and a requested top-K, then applies
// it as evict -> copy -> commit. between evict and commit the slot is empty,
// so the hot path (which consults the slot table) never reads a half-written
// entry; a miss simply falls back to the cold CPU path.
//
// milestone 1: data structures + swap-set algebra + cost gate only.
// GPU buffer allocation and H2D copies are milestone 3 (the copy is
// abstracted here as a commit callback).

struct llama_expert_pool {
    bool enabled = false;
    int32_t n_layer = 0;     // pooled layers (CPU-resident MoE layers)
    int32_t n_expert = 0;    // experts per layer
    int32_t n_slot = 0;      // slots per layer (S = budget / n_layer)
    int64_t expert_size = 0; // bytes per expert (9.45 MB for DSV IQ3_S)

    // expert id -> slot, -1 = not resident; [n_layer][n_expert]
    std::vector<std::vector<int32_t>> expert_to_slot;
    // slot -> expert id, -1 = empty; [n_layer][n_slot]
    std::vector<std::vector<int32_t>> slot_expert;
    // number of resident experts per layer (diagnostics)
    std::vector<int32_t> n_resident;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slot, int64_t expert_size);

    int32_t slot_of(int32_t il, int32_t expert) const;
    int32_t expert_in_slot(int32_t il, int32_t slot) const;
    bool is_resident(int32_t il, int32_t expert) const;

    // delta set algebra between the current resident set and a target top-K.
    // returns per-layer evict/fill list; filling into the slots of evicted
    // entries first (stable reuse) to minimize H2D churn for the same slot.
    struct swap_plan {
        std::vector<int32_t> evict_il;  // parallel arrays of (il, expert) to evict
        std::vector<int32_t> evict_ex;
        std::vector<int32_t> fill_il;   // parallel arrays of (il, expert) to fill
        std::vector<int32_t> fill_ex;
        int32_t delta = 0;              // total experts changed
    };

    swap_plan plan_swap(const std::vector<std::vector<int32_t>> & topk) const;

    // cost gate (synchronous-swap conservative accounting):
    //   (c_cand - h_inc) * CPU_MS > delta * SWAP_MS / L
    // all inputs are per-token averages; returns true when the repin pays off.
    static bool gate_should_swap(float h_inc, float c_cand, int32_t delta,
                                 float cpu_ms, float swap_ms, float L);

    // phase 1 of a swap: evict all entries (slot table entries become empty).
    void begin_swap(const swap_plan & plan);

    // phase 3 of a swap: commit one filled slot (call after the copy landed).
    void commit_slot(int32_t il, int32_t slot, int32_t expert);

    // diagnostics
    int32_t total_resident() const;
};

// ---------------------------------------------------------------
// model-level runtime state of the expert pool (stage 2, milestone 3/4)
//
// holds, per pooled layer, GPU-resident pool weight tensor(s) plus the
// host-side mapping tables consumed by the graph:
//   - remap_tab     I32 [n_expert]: expert -> pool slot (or 0 for non-resident)
//   - mask_tab      F32 [n_expert]: 1.0 resident, 0.0 non-resident (warm-path mask)
//   - cold_mask_tab I8  [n_expert]: 0 resident, 1 non-resident (cold-op mask)
// the tables are plain ggml tensors in a host buffer context so the graph
// can read them via get_rows / as the cold op's src[3].
// the pool is initialized once (from --expert-pool-init or random) and
// never refreshed in the static v1; swap logic is not wired in yet.
struct llama_expert_pool_state {
    bool enabled = false;
    int32_t n_slot = 0;   // slots per pooled layer

    // original weight tensors, indexed by layer (null = not present); used by
    // the graph to pair a mul_mat_id weight with its pool copy
    std::vector<ggml_tensor *> orig_gate_up;
    std::vector<ggml_tensor *> orig_up;
    std::vector<ggml_tensor *> orig_gate;
    std::vector<ggml_tensor *> orig_down;

    // pool weight tensors, indexed by layer; null = not pooled / not present
    std::vector<ggml_tensor *> w_pool_gate_up; // fused [n_ff*2, n_embd, S]
    std::vector<ggml_tensor *> w_pool_up;      // separate [n_ff, n_embd, S]
    std::vector<ggml_tensor *> w_pool_gate;    // separate [n_ff, n_embd, S]
    std::vector<ggml_tensor *> w_pool_down;    // [n_embd, n_ff, S]

    // mapping tables, indexed by layer
    std::vector<ggml_tensor *> remap_tab;      // I32 [1, n_expert] (row = 1 int)
    std::vector<ggml_tensor *> mask_tab;       // F32 [1, n_expert]
    std::vector<ggml_tensor *> cold_mask_tab;  // I8  [n_expert]

    // resident expert lists, indexed by layer (for diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // pooled layer indices (filled at init, consumed by the delayed fill)
    std::vector<int32_t> pooled_layers;

    // set once the pool weights/tables have been copied (idempotent fill)
    bool fill_done = false;

    // CPU-side expert->slot table per pooled layer (int32, -1 = miss); used by the
    // moe delegate hook inside the CPU MUL_MAT_ID kernel
    std::vector<std::vector<int32_t>> slots;

    // moe delegate runtime
    bool delegate_ok = false;
    ggml_backend_buffer_type_t pool_buft = nullptr;        // pool buft (device)
    ggml_backend_t gpu_backend = nullptr;                  // target device[[truncated]
    ggml_context *  mg_ctx    = nullptr;               // mini-graph ctx (no_alloc)
    ggml_backend_buffer_t dg_buf = nullptr;            // GPU scratch: cur/out/ids
    ggml_tensor * t_cur = nullptr;                     // [n_embd] F32
    ggml_tensor * t_out = nullptr;                     // [n_ff, 8] F32
    ggml_tensor * t_ids = nullptr;                     // [8] I32
    ggml_tensor * mg_w  = nullptr;                     // pool weight 4D (set per call)
    int32_t n_used = 0;                                // top-k (hparams.n_expert_used)
    std::vector<int32_t> hit_slots;                    // hit slots per ids row (size = n_used)
    std::vector<int32_t> hit_cols;                     // original ids column per hit
    int32_t n_hit = 0;
    size_t  sz_out_row = 0;                            // n_ff * sizeof(float)
    // per-layer delegate statistics (indexed by ilx = position in pooled_layers;
    // accumulated since last read, see llama_expert_pool_get_stats; the delegate
    // runs on the ith==0 thread only)
    struct layer_delegate_stats {
        uint64_t submits = 0;    // mini-graph submissions for this layer
        uint64_t hit_rows = 0;   // rows computed by the GPU delegate
        uint64_t miss_rows = 0;  // rows computed by the CPU kernel
        uint64_t getset_us = 0;  // cur copy H2D preparation time
        uint64_t ids_us = 0;     // ids scratch prepare + upload time
        uint64_t comp_us = 0;    // graph compute submission time
        uint64_t sync_us = 0;    // end() wait for GPU completion
        uint64_t get_us = 0;     // end() D2H copy of hit rows
    };
    std::vector<layer_delegate_stats> s_layer;
    ggml_backend_buffer_t host_buf = nullptr;          // pinned host mirrors (fast H2D/D2H)
    void * host_cur = nullptr;                         // pinned mirror of t_cur
    void * host_ids = nullptr;                         // pinned mirror of t_ids
    bool async_d2h = false;                            // stream-ordered D2H in begin (GGML_EXPPOOL_ASYNC_D2H=1)

    // per-layer cached mini graphs (built once at fill time; each call only
    // refills t_cur/t_ids and recomputes). each entry owns the llm_graph
    // context/result that built it (the chain is built by calling
    // build_moe_ffn(chain_only) with pool weights, so the model's own
    // activation variants are reused verbatim).
    struct mini_graph_entry {
        ggml_cgraph * g = nullptr;         // the built whole-chain graph
        ggml_tensor * out_down = nullptr;  // down output (chain result)
        void * gres = nullptr;             // llm_graph_result* (owned)
    };
    std::vector<std::vector<mini_graph_entry>> mini;
    std::vector<ggml_backend_buffer_t> layer_bufs; // per-layer chain scratch (freed in reset)
    std::vector<ggml_tensor *> layer_out_down;     // per-layer chain result tensors

    // fused up+gate submission state: the first begin of a layer submits the
    // combined graph; the second begin only scans ids and returns the skip table.
    // end() uses this map to find (il, which) for a given dst tensor.
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, int32_t>> dst_ilx_which;
    ggml_tensor * t_out_down = nullptr;    // [n_embd, n_used] F32 down hit output (GPU) — chain result
    void * host_out_down = nullptr;        // pinned mirror of t_out_down
    bool mini_submitted = false;           // combined graph already submitted this step
    int32_t mini_submitted_il = -1;        // which layer submitted it
    ggml_backend_event_t ev = nullptr;     // per-submit completion event (end() waits only its own graph)

    // runtime routing log (GGML_EXPPOOL_ROUTING_LOG=<path>; ONLY for analysis,
    // writes the raw ids seen by the delegate: "step,layer,id0,id1,..." — the
    // same format as the CB-on capture, but from the REAL NO_CB execution).
    FILE * rt_log = nullptr;               // opened lazily on first begin
    bool   rt_log_tried = false;           // env already checked (avoid re-getenv)
    uint64_t log_step = 0;                 // current decode step (incremented at ilx==0)
    int32_t logged_il = -1;                // last logged layer id (dedup per step)

    void reset();
};

// seed the pool from a csv file, one line per layer: "il,e1,e2,...".
// layers missing from the file / with fewer entries are filled randomly.
bool llama_expert_pool_parse_init(const std::string & path, int32_t n_layer,
                                  int32_t n_expert, int32_t n_slot,
                                  std::vector<std::vector<int32_t>> & resident);

// random resident set per layer (fixed seed, reproducible)
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert, int32_t n_slot,
                              std::vector<std::vector<int32_t>> & resident);

struct llama_chain_params {
    int      type_op  = -1;      // llm_ffn_op_type as int (-1 = not registered)
    bool     norm_w   = false;
    float    w_scale  = 1.0f;
    uint32_t gating_op = 0;
};

// registered by build_moe_ffn at the real call site (il >= 0, pool active):
// exact per-layer chain parameters of THIS model. pool mini graphs look them
// up and call build_moe_ffn(chain_only) with the same values, so every
// arch/variant is handled by the same code without per-model replication.
void llama_expert_pool_register_chain(int il, int type_op, bool norm_w, float w_scale, uint32_t gating_op);
const llama_chain_params & llama_expert_pool_get_chain(int il);
void llama_expert_pool_clear_chain();

// moe delegate: called by the CPU MUL_MAT_ID kernel for cache-hit experts (see ggml-cpu.h)
void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud);
void llama_expert_pool_delegate_end(ggml_tensor * dst, void * ud);
