#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

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
