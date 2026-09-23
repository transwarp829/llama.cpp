#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

//
// expert pool: configuration (fork-private)
//

// The pool's knobs (llama_context_params.expert_pool; a null pointer there means
// "no pool", the default). Start from llama_expert_pool_default_params() - a
// zeroed struct means a frozen pool, not a default one.
struct llama_expert_pool_params {
    int32_t slots;           // slot budget (0 = disabled)
    int32_t layers;          // spread the budget over the deepest N layers (0 = all eligible)
    int32_t swap_cap;        // max expert pairs swapped in per decode step (<= 0 freezes the set)
    int32_t swap_decay;      // decaying activation counter: half-life in decode steps (0 = library default)
    int32_t miss_method;     // how a miss runs: 0 = cpu-serial, 1 = cpu-parallel
    const char * init_file;  // csv file to seed the pool content, null = random
};

// swap_cap 40, swap_decay 96, miss_method cpu-serial, everything else off; the
// struct must outlive the context's first reserve, like the other pointer args.
LLAMA_API struct llama_expert_pool_params llama_expert_pool_default_params(void);

//
// expert pool: routing observer (fork-private experimental API)
//

// The split-head observer fans out the CLEAN topk rows of one (step, layer), one call
// per split head (one-token batches only); pass cb = nullptr to unregister.
//   il            - layer number the row belongs to
//   ids           - the layer's clean topk ids (the pool's -1 encodings stay out
//                   of the stream)
//   n_ids         - ids per token
//   first_of_step - 1 when this call begins a new decode step
typedef void (*llama_expert_pool_route_fn)(
        void * user_data, int32_t il, const int32_t * ids, int32_t n_ids, int32_t first_of_step);

LLAMA_API void llama_expert_pool_set_route_observer(llama_expert_pool_route_fn cb, void * user_data);

//
// expert pool: delegate statistics and segment accounting (fork-private)
//

// per pooled-layer counters fed by the split-head observer. the call returns the
// totals since the previous call (call once after each llama_decode), fills up to
// max_layers entries in pooled-layer order and resets them.
struct llama_expert_pool_layer_stats {
    int32_t  layer;          // actual model layer number
    uint64_t hit_rows;       // rows computed by the GPU pool chain
    uint64_t miss_rows;      // rows computed by the CPU kernel
};

LLAMA_API uint32_t llama_expert_pool_get_stats(struct llama_context * ctx,
        struct llama_expert_pool_layer_stats * out, uint32_t max_layers);

// end of a generation segment: print the accumulated hit rate at info verbosity and
// reset the segment counters (call once after the decode loop). id_slot >= 0 reports
// that sequence slot's own segment window (the steps it took part in) instead; -1
// reports the pool's segment since the previous call.
LLAMA_API void llama_expert_pool_finalize(struct llama_context * ctx, int32_t id_slot);

//
// verbosity-explicit logging (fork-private)
//

// bypasses the ggml-level -> verbosity remap in the common default callback;
// `verbosity` numbers as common LOG_LEVEL_* (3 = info). with no callback registered
// llama_log_verbose falls back to llama_log_internal.
typedef enum llama_log_verbosity {
    LLAMA_LOG_VERBOSITY_ERROR = 1,
    LLAMA_LOG_VERBOSITY_WARN  = 2,
    LLAMA_LOG_VERBOSITY_INFO  = 3,
    LLAMA_LOG_VERBOSITY_TRACE = 4,
    LLAMA_LOG_VERBOSITY_DEBUG = 5,
} llama_log_verbosity;

// same signature as ggml_log_callback with an explicit verbosity prepended
typedef void (*llama_log_verbosity_callback)(int verbosity, enum ggml_log_level level, const char * text, void * user_data);

// register the verbosity-explicit callback (global, not thread safe, like llama_log_set)
LLAMA_API void llama_log_set_verbosity(llama_log_verbosity_callback callback, void * user_data);
