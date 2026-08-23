#pragma once

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// window-based expert routing statistics (stage 1 of the expert cache)
//
// three layers:
//   - llama_expert_window: per (seq, layer) sliding-window frequency counter
//   - llama_context_expert_cache: per-seq collection, owned by llama_context
//   - llama_model_expert_cache: model-wide aggregator, owned by llama_model
//
// note: stage 1 only collects statistics. slot pool, LUT remap and the cost
// gate are stage 2/3 and live in this file's future extensions.

struct llama_expert_window {
    int64_t  n_expert = 0;   // number of experts (e.g. 256)
    int32_t  n_used   = 0;   // experts used per token (top-k)
    int32_t  cap      = 0;   // ring capacity = W * n_used (W fixed 512)

    std::vector<int32_t> counts;  // [n_expert] window frequency
    std::vector<int32_t> ring;    // [cap] ring buffer of expert ids in window
    int32_t  ring_head = 0;       // oldest entry
    int32_t  ring_len  = 0;       // current length; < cap == min(W, ctx)
    int64_t  total_tokens = 0;    // tokens fed into the window (diagnostics)

    void init(int64_t n_expert, int32_t n_used, int32_t cap);

    // consume n_tokens routing rows (each row = n_used ids), row-major
    void update(const int32_t * ids, int64_t n_tokens);

    // top-k expert ids by window frequency
    std::vector<int32_t> top_k(int32_t k) const;
};

struct llama_model_expert_cache;

// context-level: per-seq windows (multi-slot aware from the start)
struct llama_context_expert_cache {
    bool   enabled = false;
    int32_t n_layer   = 0;
    int32_t n_seq_max = 0;
    int32_t W = 512;
    int32_t K = 32;                 // flush cadence in tokens (repin interval)
    int32_t tokens_since_flush = 0;

    // [seq_id][il] -> window (pre-allocated to n_seq_max x n_layer)
    std::vector<std::vector<llama_expert_window>> seqs;

    // stage-1 verification dump (env LLAMA_EXPERT_CACHE_DUMP=path, same style
    // as STEP_PROFILE_OUT). writes top-k of every (seq, layer) at flush cadence.
    FILE * dump_f = nullptr;

    void init(int32_t n_seq_max, int32_t n_layer);

    // feed one routing choice for layer il (n_tokens rows of n_used ids)
    // window is lazily initialized with the observed expert count on first use
    void record(llama_seq_id seq, int32_t il, const int32_t * ids, int64_t n_tokens,
                int32_t n_used, int64_t n_expert);

    // aggregate all seq windows into the model-level aggregator
    void flush_to_aggregator(llama_model_expert_cache & agg);

    // open/close the verification dump file (env driven)
    void dump_open();
    void dump_close();

    // write the current top-k of all (seq, layer) as CSV rows:
    //   seq,il,total_tokens,expert_id1,...expert_idk,count1,...countk
    // total_tokens = window's cumulative token count; lets the offline
    // simulator place its snapshots at exactly the same window state.
    void dump_step(int32_t k);
};

// model-level: aggregator (stage 1: global frequency view only)
struct llama_model_expert_cache {
    bool enabled = false;

    // [il][expert] summed across all active seq windows (stage 2 uses this
    // for the global hot-pool slot snapshot; stage 3 adds the cost gate)
    std::vector<std::vector<int32_t>> per_layer_counts;

    void ensure(int32_t n_layer, int64_t n_expert);
    void aggregate(const llama_context_expert_cache & ctx_cache);
};
