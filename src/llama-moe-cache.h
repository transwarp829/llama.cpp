#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // scope: the per-device --moe-cache-layers counts (one entry per device of `backends`, 0 = all eligible)
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const std::vector<int32_t> & scope);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;

// ---------------------------------------------------------------------------
// expert pool (fork-private): the dual-chain form of the same idea (docs/expert-pool.md).
// ---------------------------------------------------------------------------

#include "llama.h"
#include "llama-cparams.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "ggml-cpp.h"   // ggml_backend_buffer_ptr (RAII buffer holder)

#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

// the MoE small-batch threshold, the same env the CUDA backend and the sched read
inline int32_t llama_expert_pool_offload_min_batch() {
    static const int32_t v = getenv("GGML_OP_OFFLOAD_MIN_BATCH") != nullptr
        ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
    return v;
}

// how a miss runs below the offload threshold; at and above it the lane takes the chain, and the
// two CPU forms only pick the submission order there  (the third form, device, is the MoE cache)
enum llama_expert_pool_miss_method {
    LLAMA_EXPERT_POOL_MISS_CPU_SERIAL   = 0, // CPU chain as one split
    LLAMA_EXPERT_POOL_MISS_CPU_PARALLEL = 1, // hit chain submitted ahead of the CPU chain
    LLAMA_EXPERT_POOL_MISS_GPU          = 2, // the cold experts fill the same-level cache and are computed
                                             // on the device in the same chain (the MoE cache path; no
                                             // dual chain, no extra staging buffer)
};

// LLAMA_EXPERT_POOL_LAYOUT: "auto" (the default) follows the batch regime, "whole"/"partial" pin a form,
// anything else pins partial. the whole-layer form keeps the layer set and the -nep budget meaning
enum llama_expert_pool_layout {
    LLAMA_EXPERT_POOL_LAYOUT_AUTO    = 0,
    LLAMA_EXPERT_POOL_LAYOUT_PARTIAL = 1,
    LLAMA_EXPERT_POOL_LAYOUT_WHOLE   = 2,
};

// what one miss method means to the run. the three values are peers: each names where a miss is
// computed, and with it the chain the layer's experts run in
struct llama_expert_pool_miss_strategy {
    bool pool_active     = true;  // the dual chain: the pool's own tables, worker and tables
    bool cache_required  = false; // the same-level cache takes the misses (the device form)
    bool layer_parallel  = false; // the hit chain is handed to the device ahead of the CPU chain
};

inline llama_expert_pool_miss_strategy llama_expert_pool_miss_strategy_of(int32_t method) {
    llama_expert_pool_miss_strategy s;
    if (method == LLAMA_EXPERT_POOL_MISS_GPU) {
        s.pool_active    = false;
        s.cache_required = true;
    } else {
        s.layer_parallel = method == LLAMA_EXPERT_POOL_MISS_CPU_PARALLEL;
    }
    return s;
}

inline const char * llama_expert_pool_miss_method_name(int32_t method) {
    if (method == LLAMA_EXPERT_POOL_MISS_CPU_PARALLEL) {
        return "cpu-parallel";
    }
    return method == LLAMA_EXPERT_POOL_MISS_GPU ? "gpu" : "cpu-serial";
}

inline llama_expert_pool_layout llama_expert_pool_layout_pinned() {
    static const llama_expert_pool_layout v = [] {
        const char * s = getenv("LLAMA_EXPERT_POOL_LAYOUT");
        if (s == nullptr || std::string(s) == "auto") {
            return LLAMA_EXPERT_POOL_LAYOUT_AUTO;
        }
        return std::string(s) == "whole" ? LLAMA_EXPERT_POOL_LAYOUT_WHOLE : LLAMA_EXPERT_POOL_LAYOUT_PARTIAL;
    }();
    return v;
}

// ---------------------------------------------------------------
// the hit chain: per-layer tensors that run the resident chain inside the main graph. the -1 skip ids
// split the columns by construction, so one add merges the two chains; the tables are I32
// the tables are I32 because ggml_get_rows keeps the table type on every backend.
struct llama_context;
struct llama_model;


// ---------------------------------------------------------------
// per pooled layer: the device-resident pool tensors (slot s = s-th resident expert) and the routing tables

// the tensors the pool copies and the swap moves; the enum index is the array index used everywhere.
// the weights and the per-expert biases are slot-compact and move together, the scales stay in expert order
enum llama_expert_pool_kind {
    PK_GATE_UP = 0, // fused gate+up [n_ff*2, n_embd, n_expert]; pool [.., S]
    PK_UP,          // separate up [n_ff, n_embd, n_expert]
    PK_GATE,        // separate gate [n_ff, n_embd, n_expert]
    PK_DOWN,        // down [n_embd, n_ff, n_expert]
    PK_GATE_UP_B,   // per-expert fused bias [n_ff*2, n_expert]; pool [.., S],
    PK_UP_B,        // per-expert bias [n_ff, n_expert]      slot-compact like
    PK_GATE_B,      // per-expert bias [n_ff, n_expert]      the weights
    PK_DOWN_B,      // per-expert bias [n_embd, n_expert]
    PK_N,
};

// the per-expert byte stride: ne[2] for the 4D weight copies, ne[1] for the 2D bias copies
inline size_t llama_expert_pool_stride(const ggml_tensor * t, int k) {
    return k < PK_UP_B ? t->nb[2] : t->nb[1];
}

struct llama_expert_pool_layer {
    int32_t ig = -1;               // index into groups (-1 = the pool does not serve this layer)
    ggml_tensor * orig[PK_N] = {}; // model weight tensor (host); null = not present
    ggml_tensor * pool[PK_N] = {}; // pool copy (pool device); null = not pooled
    ggml_tensor * remap     = nullptr; // I32 [1, n_expert] on the pool device: resident -> slot, else -1
    ggml_tensor * cpu_map   = nullptr; // I32 [1, n_expert] host: e for the experts the CPU computes, -1 for the device's
    ggml_tensor * remap_inv = nullptr; // I32 [1, n_expert] on the pool device: the inv half, gathered there
    ggml_tensor * scale     = nullptr; // F32 [1, n_expert] on the pool device: per-expert down scale
    ggml_tensor * scale_up   = nullptr; // F32 [1, n_expert]: up scale
    ggml_tensor * scale_gate = nullptr; // F32 [1, n_expert]: gate scale
    // the layer's form: no chain, the dual chain over the group bank, or the plain path (a complete
    // identity run reads the layer's own view of the run, in the model's own order)
    enum form_t : uint8_t { FORM_NONE = 0, FORM_CHAIN, FORM_PLAIN };
    form_t form = FORM_NONE;
    int32_t base    = -1;          // the identity run base while this layer sits in the model's own order (-1 = no)
};

// per-layer id hit/miss counts (a read clears) + one segment total
struct llama_expert_pool_counts {
    uint64_t hit  = 0;
    uint64_t miss = 0;
};

struct llama_expert_pool_state {
    ~llama_expert_pool_state();   // pool_dev is opaque here
    std::vector<llama_expert_pool_layer> layers;
    // slot bookkeeping: slot_of[(layer, expert)] -> group slot (-1 = not resident), key_of[slot]
    // -> (layer, expert) (-1 = free). `resident` is the policy's view, the slots the allocator's
    std::vector<int32_t> slot_of;                    // [n_layer_all * n_expert]
    std::vector<std::vector<int32_t>> key_of;        // [group][slot]
    // the layer's slot capacity (its width in the partial mode), layout independent
    std::vector<int32_t> width_max;
    // one group per (device, layout group): the banks hold the slots, the plan maps each member's run into them
    // the bank groups (defined in llama-moe-cache.cpp)
    struct group;
    std::vector<group> groups;

    // resident expert lists, indexed by layer (diagnostics/serialization)
    std::vector<std::vector<int32_t>> resident;

    // every eligible layer is a pool layer: the layout only picks its width ({0, .., n_expert})
    std::vector<int32_t> pooled_layers;
    // true when the pooled layers sit fully resident in expert order (whole-layer layout)
    bool layout_whole = false;

    // split-head registry, rebuilt per graph build (the clean topk ids and the miss chain's head)
    struct split_head_ref {
        int32_t il    = -1;
        bool    lane  = false;  // true: read the clean ids from the device
        bool    mount = false;  // true: the layer's resident matmul (the fill wait lands here)
    };
    mutable std::unordered_map<const ggml_tensor *, split_head_ref> split_head_refs;
    mutable int32_t                          bind_last_il = -1;
    // per layer: layer points seen, and layer points that read and folded their ids
    mutable std::vector<int32_t> lp_calls;
    mutable std::vector<int32_t> lp_folds;
    struct ids_rec {
        const ggml_tensor * clean  = nullptr; // the layer's clean topk ids (a view of the ubatch)
        int32_t             n_used = 0;
    };
    mutable std::vector<ids_rec> ids;  // [layer]
    // [layer] pool-side copy of the layer's top-k ids, written by the graph every ubatch: it must NOT live
    // in the graph group, which frees a tensor's memory once its last consumer ran
    std::vector<ggml_tensor *> ids_shadow;
    mutable std::vector<int32_t>             ids_buf;     // host scratch for a device read
    // the scheduler, for the host copies it stages for the CPU split's inputs (early CPU inputs)
    ggml_backend_sched_t sched = nullptr;

    bool direct_mount = false;                             // two-chain form active

    // swap (on by default with -nep): ids_cnt_per_token, a decaying per-token count of the ids that
    // select each (pooled layer, expert); the layer-point refresh converges the resident set
    bool plan_active = false;
    int32_t plan_per_layer = 1;             // swap on/off (> 0 swaps; <= 0 freezes the set)
    int32_t n_expert = 0;                  // experts per layer (set at init)
    // ubatch counter (the decode path advances it; the markers carry it to the workers)
    int32_t hook_step = 0;
    // a batch at or above the offload threshold has been served (the setup decodes must not decide the regime)
    bool prefill_seen = false;

    // the marker queue, its mutex/condvar and the worker thread live per bundle (pool_dev)

    // the counters are shared but the columns are disjoint: a worker decays and increments only its own bundle's layers
    float   cnt_lambda   = 0.0f;          // per-tick decay, lambda = 2^(-1/H)
    float   plan_margin  = 0.02f;         // a swap needs the candidate's rate to beat the victim's by this share; code-set, no CLI flag
    int32_t cnt_decay_hl = 4096;          // decay half-life in layer ticks
    // layer-point bookkeeping: the fold runs at the layer point, the aging once per ubatch
    int32_t lp_il   = -1;                  // last layer the observer folded evidence for
    int32_t lp_step = -1;                  // the step that layer belonged to
    int32_t lp_lane = 0;                   // the layer point's head was a staged-copy lane
    int64_t lp_n_ids = 0;                  // ids the layer point just folded (0 = it did not)
    int32_t lp_n_tok  = 0;                 // tokens those ids cover
    int     lp_counted = 0;                // the layer point classified this ubatch (feeds `any`)
    mutable std::vector<int32_t> cnt_done;  // [layer] = ubatch+1 once the layer point counted
    // layers whose table slice changed since the last flush: one write per half, not per entry
    std::vector<int32_t> slot_map_publish_q;
    std::vector<int32_t> slot_map_row;       // both halves of one layer, adjacent
    // ids the ubatch being served has used: the eviction table never takes an expert it just selected
    std::vector<uint8_t> ub_use;
    // scratch of one plan call: keys already consumed by the point's matching
    std::vector<uint8_t> plan_taken;
    // per layer: the inverse half as this ubatch's graph saw it (the hit rate classifies against that, not
    // the live table, which a later promotion in the ubatch has already moved)
    std::vector<std::vector<int32_t>> miss_snap;
    std::vector<float> ids_cnt_per_token;  // [layers * n_expert] decayed ids per token, keyed by
                                           // layer so a layout switch keeps them
    std::vector<float> ids_cnt_tot;        // [layers] that layer's counter sum: the rank compares
                                           // c[e] / ids_cnt_tot[il], i.e. rates and not accumulated mass

    llama_expert_pool_counts seg;        // the ubatch <= 8 counters
    llama_expert_pool_counts seg_large;  // the ubatch  > 8 counters

    // NONE -> BUILT (build() ran) -> FILLED (weights and tables copied, first compute)
    enum phase_t : uint8_t { PHASE_NONE = 0, PHASE_BUILT, PHASE_FILLED };
    phase_t phase = PHASE_NONE;

    // per device pool bundle: one entry per device that serves pooled layers.
    // every bundle owns its tables and its worker state; the tables keep the absolute layer-column layout
    // (only this device's layers are written), so the per-layer views and their offsets are unchanged
    // the per-device bundle (defined in llama-moe-cache.cpp)
    struct pool_dev;

    // the pooled layers by serving device; layer_dev[il] = index into devs (-1 = not pooled), in --device order
    std::vector<std::unique_ptr<pool_dev>> devs;
    std::vector<int32_t>  layer_dev;
    // the budgets the bundles were built with: dev_slots[i] belongs to devs[i] (a relayout
    // re-lays every pool with its own budget)
    std::vector<int32_t>  dev_slots;

    int32_t dev_of(int32_t il) const {
        return il >= 0 && (size_t) il < layer_dev.size() ? layer_dev[il] : -1;
    }

    void reset();
};

// ---------------------------------------------------------------
// group slot helpers shared by the plan and the swap: key_of[slot] < 0 = free (-1) or retired (-2)
int32_t llama_expert_group_cap(const llama_expert_pool_state & st, size_t ai);

inline int32_t llama_expert_group_free(const llama_expert_pool_state & st, size_t ai) {
    int32_t n = 0;
    for (int32_t k : st.key_of[ai]) {
        if (k == -1) {
            n += 1;
        }
    }
    return n;
}

inline int32_t llama_expert_group_alloc(const llama_expert_pool_state & st, size_t ai) {
    const std::vector<int32_t> & keys = st.key_of[ai];
    for (size_t s = 0; s < keys.size(); ++s) {
        if (keys[s] == -1) {
            return (int32_t) s;
        }
    }
    return -1;
}

// split-head observer: the scheduler hands it every split's head, the pool reads the registered ones
void llama_expert_pool_bind_ids(const llama_expert_pool_state & st, int32_t il, int32_t n_used, const ggml_tensor * ids_clean);
void llama_expert_pool_bind_split_head(const llama_expert_pool_state & st, int32_t il, const ggml_tensor * head, bool lane);
// the layer's resident matmul: the fills the layer launched must be in before this split runs
void llama_expert_pool_bind_mount_head(const llama_expert_pool_state & st, int32_t il, const ggml_tensor * head);
void llama_expert_pool_observe_split_head(void * user_data, ggml_tensor * head, ggml_backend_t backend);

// ---------------------------------------------------------------
// the pool driver: owns the state, the weight context/buffers, the hit-chain tables and the entry
// points. llama_context holds one and hands it the handles the engine needs
struct llama_expert_pool {
    llama_expert_pool(
            llama_context & ctx,
            const llama_model & model,
            const llama_cparams & cparams,
            const std::vector<ggml_backend_t> & backend_ptrs,
            const std::vector<ggml_backend_buffer_type_t> & backend_buft);

    llama_expert_pool_state st;

    // the mechanism's device path: the miss method that computes a miss on the device in the same chain
    // (upstream's own arrangement). null when this run takes one of the pool's own
    llama_moe_cache_ptr cache;
    llama_moe_cache * moe_cache() const { return cache.get(); }

    llama_context &       ctx;
    const llama_model &   model;
    const llama_cparams & cparams;

    const std::vector<ggml_backend_t> &             backend_ptrs;
    const std::vector<ggml_backend_buffer_type_t> & backend_buft;

    void expert_pool_init();
    void expert_pool_build();
    // the build's three phases: the weight groups and their banks, the tables and the weight buffers
    void pool_build_groups();
    bool pool_alloc_storage();
    // the plan's views and hit-chain operands; re-run by every relayout
    void expert_pool_build_views();
    // the view pass in three steps: the cleared views and slot spaces, the per-layer operands, the forms
    void pool_reset_views();
    void pool_build_layer_views();
    void pool_set_forms();
    // create one bundle per selected serving device, in order, with the given layers and budgets; the one
    // place bundles are created. the relayout frees them first, because each form builds its own views
    void expert_pool_make_devs(const std::vector<ggml_backend_dev_t> &   dv_order,
                               const std::vector<std::vector<int32_t>> & layers_by_dev,
                               const std::vector<int32_t> &              slots_by_dev);
    void expert_pool_fill();
    // release the per-device bundles (weight context/buffer, tables)
    void release_devs();
    void expert_pool_release();
    // report and reset the pool's segment; the slot id is kept for the API
    void expert_pool_finalize();
    void expert_pool_relayout(bool whole);
    void expert_pool_cross(int32_t n_tokens);

    // device memory held by the pool (weight copies + tables) per buffer type; read by memory_breakdown
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
};

using llama_expert_pool_ptr = std::unique_ptr<llama_expert_pool>;

// the pool-side copy of a pool layer's top-k ids (null when the layer is not pooled)
inline ggml_tensor * llama_expert_pool_ids_shadow(const llama_expert_pool_state & st, int32_t il) {
    return il >= 0 && (size_t) il < st.ids_shadow.size() ? st.ids_shadow[il] : nullptr;
}


// is this layer a pool layer? (all eligible layers are: the layout only sets their width)
// the operand a chain reads for one kind: the layer's own run view (plain) or the group bank (chain)
ggml_tensor * llama_expert_pool_operand(const llama_expert_pool_state & st, int32_t il, int k, bool full);

inline bool llama_expert_pool_is_pool_layer(const llama_expert_pool_state & st, int32_t il) {
    return il >= 0 && (size_t) il < st.layers.size() && st.layers[il].ig >= 0;
}

// the CPU part's table of a layer (null when the layer is not pooled)
inline ggml_tensor * llama_expert_pool_cpu_map(const llama_expert_pool_state & st, int32_t il) {
    return il >= 0 && (size_t) il < st.layers.size() ? st.layers[il].cpu_map : nullptr;
}

// does any slot of this layer hold an expert? the chain form starts empty in the partial layout and the
// plan fills it, so the graph takes the plain path until the first slots land
bool llama_expert_pool_has_resident(const llama_expert_pool_state & st, int32_t il);

// ubatch-boundary accounting: reads back every pool layer's top-k ids once per ubatch - the counters
// and the hit/miss statistics all come from here. a fully resident or width-0 layer has no split head
void llama_expert_pool_account_ids(llama_expert_pool_state & st);
