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
// holds, per pooled layer, GPU-resident pool weight tensors (identity
// layout: slot e = expert e, non-resident slots stay zero) plus the
// routing tables that split the two chains of the direct mount.
// the pool is initialized once (from --expert-pool-init or random) and
// never refreshed in the static v1; swap logic is not wired in yet.
struct llama_expert_pool_state {
    bool enabled = false;

    // the original weight tensors, indexed by layer (null = not present); used by
    // the graph to pair a mul_mat_id weight with its pool copy
    std::vector<ggml_tensor *> orig_gate_up;
    std::vector<ggml_tensor *> orig_up;
    std::vector<ggml_tensor *> orig_gate;
    std::vector<ggml_tensor *> orig_down;

    // pool weight tensors, indexed by layer; null = not pooled / not present.
    // identity layout: slot e holds expert e (non-resident slots stay zero)
    std::vector<ggml_tensor *> w_pool_gate_up; // fused [n_ff*2, n_embd, n_expert]
    std::vector<ggml_tensor *> w_pool_up;      // separate [n_ff, n_embd, n_expert]
    std::vector<ggml_tensor *> w_pool_gate;    // separate [n_ff, n_embd, n_expert]
    std::vector<ggml_tensor *> w_pool_down;    // [n_embd, n_ff, n_expert]

    // resident expert lists, indexed by layer (for diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // pooled layer indices (filled at init, consumed by the delayed fill)
    std::vector<int32_t> pooled_layers;

    // set once the pool weights/tables have been copied (idempotent fill)
    bool fill_done = false;

    // main-graph mount mode (GGML_EXPPOOL_MOUNT=1): a second GPU-resident
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

// moe routing-log hook: called by the CPU MUL_MAT_ID kernel (ith==0), feeds
// GGML_EXPPOOL_ROUTING_LOG (see llama-expert-pool.h). returns a null skip
// table: no rows are skipped, column zeroing is done by the -1 ids natively.
void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud);

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
    ggml_tensor * w_gate_up = nullptr; // [n_ff*2, n_embd, n_expert] or null
    ggml_tensor * w_up      = nullptr; // [n_ff, n_embd, n_expert] or null
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
