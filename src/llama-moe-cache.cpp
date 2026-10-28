#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// the counters both halves report: hits and misses count the requests (the ids of a slot map copy),
// fills counts the experts uploaded, bytes the uploaded bytes
struct moe_cache_stats {
    size_t hits   = 0;
    size_t misses = 0;
    size_t fills  = 0;
    size_t bytes  = 0;
};

// the ubatch size picks the bucket: the cache and the pool both report these two
inline bool moe_cache_stats_is_small(int64_t n_ids, int64_t n_expert_used) {
    return n_ids <= 8*n_expert_used;
}

inline moe_cache_stats & moe_cache_stats_bucket(moe_cache_stats & small, moe_cache_stats & large, int64_t n_ids, int64_t n_expert_used) {
    return moe_cache_stats_is_small(n_ids, n_expert_used) ? small : large;
}

// the one report line for these counters
inline void log_moe_cache_stats(const char * mech, const char * bucket, const moe_cache_stats & st) {
    const size_t n = st.hits + st.misses;
    if (n == 0) {
        return;
    }
    LLAMA_LOG_INFO("%s: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, fills = %zu, uploaded = %.2f MiB\n",
            mech, bucket, st.hits, st.misses, 100.0*st.hits/n, st.fills, st.bytes/1024.0/1024.0);
}

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    // n_hit_ids counts the requests (the ids), not the experts: the dedup below only serves the fills
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit_ids) {
        fills.clear();
        n_hit_ids = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // classify the requests first: the fills below take their slots right after, so a
        // request counted later would read its own fill as a hit
        for (size_t i = 0; i < n_ids; ++i) {
            if (slots[ids[i]] >= 0) {
                n_hit_ids ++;
            }
        }

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

// one bank of cap + 1 slots for one expert tensor (the extra slot is the MMQ past-read guard); a
// slot-compact per-expert bias is 2D, a weight keeps its own two dimensions
static ggml_tensor * moe_cache_new_bank(ggml_context * ctx, const ggml_tensor * t, int32_t cap, bool slot_compact) {
    return slot_compact ? ggml_new_tensor_2d(ctx, t->type, t->ne[0], cap + 1)
                        : ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], cap + 1);
}

// the layers of one device that share a bank set, and the banks themselves
struct moe_cache_group {
    int32_t                    id = -1;  // device
    std::vector<ggml_tensor *> ref;      // expert tensors of the first member
    std::vector<int32_t>       layers;   // members
    std::vector<ggml_tensor *> banks;    // device storage of all slots, one per expert tensor
};

// the handles of one device's cache: the weight context, its buffer and the device they live on
struct moe_cache_device {
    ggml_backend_t             backend = nullptr;
    ggml_backend_buffer_type_t buft    = nullptr;
    ggml_context_ptr           ctx;   // the banks and their views
    ggml_backend_buffer_ptr    buf;   // the banks themselves
};

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

// the bytes a bank of n_slots slots takes on a buffer type: the one rule both halves size their banks
// with. the weights are 4D (a slot's stride is nb[2]), the pool's slot-compact per-expert biases are 2D
// (nb[1]); the extra slot is the MMQ past-read guard both carry
static size_t moe_cache_bank_bytes(ggml_backend_buffer_type_t bt,
                                   const std::vector<ggml_tensor *> & weights,
                                   const std::vector<ggml_tensor *> & biases,
                                   int32_t n_slots) {
    const size_t alignment = ggml_backend_buft_get_alignment(bt);
    size_t res = 0;
    for (const ggml_tensor * t : weights) {
        res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
    }
    for (const ggml_tensor * t : biases) {
        if (t != nullptr) {
            res += GGML_PAD(t->nb[1]*(n_slots + 1), alignment);
        }
    }
    return res;
}

// how many slots a budget of that many bytes holds, bounded by max_slots
static int32_t moe_cache_slots_in(ggml_backend_buffer_type_t bt,
                                  const std::vector<ggml_tensor *> & weights,
                                  const std::vector<ggml_tensor *> & biases,
                                  size_t bytes, int32_t max_slots) {
    int32_t n = 0;
    while (n < max_slots && moe_cache_bank_bytes(bt, weights, biases, n + 1) <= bytes) {
        n += 1;
    }
    return n;
}

// the device's first non-host buffer type: where its banks live. a device without one keeps the CPU
// buft, the degenerate configuration (nothing to cache on it)
static ggml_backend_buffer_type_t pool_dev_buft(const std::vector<ggml_backend_buffer_type_t> & bufts,
                                                ggml_backend_dev_t dv) {
    for (const ggml_backend_buffer_type_t cand : bufts) {
        if (cand != nullptr && !ggml_backend_buft_is_host(cand) && ggml_backend_buft_get_device(cand) == dv) {
            return cand;
        }
    }
    return ggml_backend_cpu_buffer_type();
}

// the backend handle of a buffer type: the context's paired lists are the only source
static ggml_backend_t pool_backend_of(const std::vector<ggml_backend_t> & backends,
                                      const std::vector<ggml_backend_buffer_type_t> & bufts,
                                      ggml_backend_buffer_type_t bt) {
    for (size_t i = 0; i < bufts.size() && i < backends.size(); ++i) {
        if (bufts[i] == bt) {
            return backends[i];
        }
    }
    return nullptr;
}

// the share of the budget one device gets: the tensor split when it is given, else its free memory; a
// device that serves nothing keeps a zero share, and when no device reports memory the shares are equal
static std::vector<double> moe_cache_device_split(const llama_model & model,
                                                  const std::vector<ggml_backend_dev_t> & devices,
                                                  const std::vector<char> & serves) {
    const float * tensor_split = model.tensor_split();
    const bool split_by_free = tensor_split == nullptr ||
        std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
    std::vector<double> share(devices.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i >= serves.size() || serves[i] == 0) {
            continue;
        }
        if (split_by_free) {
            size_t free_mem  = 0;
            size_t total_mem = 0;
            ggml_backend_dev_memory(devices[i], &free_mem, &total_mem);
            share[i] = (double) free_mem;
        } else {
            const auto it = std::find_if(model.devices.begin(), model.devices.end(),
                    [&](const llama_device & ld) { return ld.dev == devices[i]; });
            if (it != model.devices.end()) {
                share[i] = (double) tensor_split[it - model.devices.begin()];
            }
        }
        sum += share[i];
    }
    if (sum == 0.0) {
        for (size_t i = 0; i < devices.size(); ++i) {
            share[i] = i < serves.size() && serves[i] != 0 ? 1.0 : 0.0;
            sum += share[i];
        }
    }
    return share;
}

// one eligible layer and the matrices it would store
struct moe_cache_layer {
    int32_t                    il      = -1;
    ggml_backend_dev_t         dev     = nullptr;
    std::vector<ggml_tensor *> experts;
};

// the layers of the model in [0, il_end) the cache may serve, ascending: every present expert matrix of
// the layer must sit in host memory as a weight, and the layer's FFN must be on a device the cache can
// hold tables for (a CPU layer keeps the plain path, a meta device cannot). this is the ONE place the
// layer set is resolved - the MoE cache's groups and the pool's bundles both read it
static std::vector<moe_cache_layer> moe_cache_eligible_layers(const llama_model & model, int32_t il_end,
                                                              bool warn, const char * who) {
    std::vector<moe_cache_layer> res;
    for (int32_t il = 0; il < il_end; ++il) {
        const llama_layer & L = model.layers[il];
        const ggml_backend_dev_t dev = model.dev_layer(il);
        const auto dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_META) {
            if (warn) {
                LLAMA_LOG_WARN("%s: layer %d is placed on a meta device (tensor parallel): "
                               "it cannot be cached, layer skipped\n", who, il);
            }
            continue;
        }
        std::vector<ggml_tensor *> experts = llama_moe_cache_layer_experts(L);
        if (experts.empty()) {
            continue;
        }
        bool any_host   = false;
        bool all_host   = true;
        bool any_repack = false; // a CPU-device buffer that is not host visible: its bytes are not the model's
        for (const ggml_tensor * t : experts) {
            const bool host = llama_moe_cache_is_host_weight(t);
            any_host = any_host || host;
            all_host = all_host && host;
            if (!host && t->buffer != nullptr &&
                    ggml_backend_dev_type(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer))) ==
                        GGML_BACKEND_DEVICE_TYPE_CPU) {
                any_repack = true;
            }
        }
        if (!all_host) {
            if (warn && any_host) {
                LLAMA_LOG_WARN("%s: layer %d has mixed CPU/GPU expert matrices, it cannot be cached "
                               "(keep the whole layer on one device)\n", who, il);
            } else if (warn && any_repack) {
                // repacked expert weights live in a layout that is not the model's, and the cache copies raw host bytes
                // TODO: upload from the GGUF offsets (pread) instead
                LLAMA_LOG_WARN("%s: layer %d expert matrices sit in a repacked CPU buffer: that layout cannot be "
                               "read, layer skipped\n", who, il);
            }
            continue;
        }
        res.push_back({ il, dev, std::move(experts) });
    }
    return res;
}

// the per-device scope of --moe-cache-layers: the deepest n of a device's eligible layers (0 = all of them)
template <typename T>
static std::vector<T> moe_cache_deepest(const std::vector<T> & layers, int32_t n) {
    if (n <= 0 || (int32_t) layers.size() <= n) {
        return layers;
    }
    return std::vector<T>(layers.end() - n, layers.end());
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device : moe_cache_device {
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget
        size_t buf_size = 0;   // the banks' size
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group : moe_cache_group {
        size_t host_bytes = 0;   // host experts of its members
        int32_t n_slots = 0;     // slots it holds, shared by its members
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    // the counters both halves report (moe_cache_stats)
    using stats = moe_cache_stats;

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends,
         const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const std::vector<int32_t> & scope) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // the layer set: ONE resolution, the same one the pool reads, then this device's
        // --moe-cache-layers scope
        const std::vector<moe_cache_layer> elig = moe_cache_eligible_layers(model, (int32_t) model.layers.size(),
                /*warn=*/false, __func__);
        for (size_t i = 0; i < devices.size(); ++i) {
            const ggml_backend_dev_t dev = ggml_backend_get_device(devices[i].backend);
            std::vector<moe_cache_layer> mine;
            for (const moe_cache_layer & e : elig) {
                if (e.dev == dev) {
                    mine.push_back(e);
                }
            }
            const int32_t n_scope = i < scope.size() ? scope[i] : 0;
            for (const moe_cache_layer & e : moe_cache_deepest(mine, n_scope)) {
                auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) {
                        return g.id == (int32_t) i && llama_moe_cache_same_layout(g.ref, e.experts); });
                if (it == groups.end()) {
                    groups.emplace_back();
                    it = groups.end() - 1;
                    it->id  = (int32_t) i;
                    it->ref = e.experts;
                }
                it->layers.push_back(e.il);
                for (const ggml_tensor * t : e.experts) {
                    it->host_bytes         += ggml_nbytes(t);
                    devices[i].host_bytes  += ggml_nbytes(t);
                }
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // the budget is split among the devices with host experts like the layers, by the tensor split or
        // by default by free memory; one rule, shared with the pool
        std::vector<ggml_backend_dev_t> split_devs;
        std::vector<char>               split_serves;
        for (device & d : devices) {
            split_devs.push_back(ggml_backend_get_device(d.backend));
            split_serves.push_back(d.host_bytes > 0 ? 1 : 0);
        }
        const std::vector<double> split_share = moe_cache_device_split(model, split_devs, split_serves);
        double split_sum = 0.0;
        for (size_t i = 0; i < devices.size(); ++i) {
            devices[i].split = i < split_share.size() ? split_share[i] : 0.0;
            split_sum += devices[i].split;
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            g.n_slots = moe_cache_slots_in(d.buft, g.ref, {}, budget, max_slots);
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = moe_cache_new_bank(ctx, t, g.n_slots, false);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += moe_cache_bank_bytes(devices[g.id].buft, g.ref, {}, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit_ids = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit_ids)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit_ids;
        st.misses += ids.size() - n_hit_ids;
        st.fills  += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        log_moe_cache_stats("llama_moe_cache", "ubatch <= 8", stats_small);
        log_moe_cache_stats("llama_moe_cache", "ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends,
        const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const std::vector<int32_t> & scope) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, scope)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

// ---------------------------------------------------------------------------
// expert pool (fork-private): implementation.
// ---------------------------------------------------------------------------

#include "llama-impl.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-graph.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>


llama_expert_pool_params llama_expert_pool_default_params() {
    llama_expert_pool_params p = {};
    p.fill_cap   = 1;  // plan on/off: > 0 fills, <= 0 freezes the resident set
    // decaying counter half-life, in layer visits (a layer ticks when its evidence folds). 128 measured
    // better than the old 4096 on both models (2026-10-23, 600-token decode: gemma4 tg 75.5 vs 71.7 and
    // the fewest fills, DSV4F hit rate 55.11% vs 53.39%); the old value was an early, loose measurement
    p.cnt_decay = 128;
    // the CPU chain runs under the device chain: it is as fast as the serial form and it is reproducible,
    // while the serial form's output depends on when the fills land (measured 2026-10-23, gemma4)
    p.miss_method = 1;  // cpu-parallel
    return p;
}



// one device's pool bundle: the cache's own handles, this device's tables and its copy counters
struct llama_expert_pool_state::pool_dev : moe_cache_device {
        ggml_backend_buffer_type_t buft    = nullptr;
        ggml_backend_t             backend = nullptr;
        ggml_context_ptr            ctx;          // pool weight tensors (host-upload copies)
        ggml_backend_buffer_ptr    buf;       // the banks of this device
        ggml_context_ptr            slot_map_ctx;
        ggml_context_ptr            ctx_host;     // the cpu_map tables' host half
        ggml_backend_buffer_ptr    slot_map_buf;      // device buffer holding the remap tables
        ggml_backend_buffer_ptr    buf_host;  // CPU buffer holding the cpu_map tables
        ggml_tensor *              slot_map_dev = nullptr; // [n_expert, 2*n_layer_all] remap|inv
        std::vector<int32_t>       layers;       // pooled layers served here, ascending

        moe_cache_stats         stats;               // this bundle's fills and bytes, summed by the report
};

llama_expert_pool_state::~llama_expert_pool_state() = default;

// one bank set of one device: the cache's own group plus the pool's index over it
struct llama_expert_pool_state::group : moe_cache_group {
        int32_t id = -1;          // device
        std::vector<int32_t> kinds;        // PK kind ids present in every member
        // layout key: the members' expert tensors in `kinds` order (the MoE cache's own layout test)
        std::vector<ggml_tensor *> ref;
        std::vector<int32_t> layers;       // members, in pooled order
        std::vector<ggml_tensor *> banks;   // one per kind
};

int32_t llama_expert_group_cap(const llama_expert_pool_state & st, size_t ai) {
    int32_t cap = 0;
    for (int32_t il : st.groups[ai].layers) {
        cap += (size_t) il < st.width_max.size() ? st.width_max[il] : 0;
    }
    return cap;
}

void llama_expert_pool_state::reset() {
    phase = PHASE_NONE;
    direct_mount = false;
    layers.clear();
    resident.clear();
    pooled_layers.clear();
    slot_of.clear();
    key_of.clear();
    width_max.clear();
    groups.clear();

    plan_active = false;
    hook_step = 0;
    prefill_seen = false;
    ids_cnt_per_token.clear();
    ids_cnt_tot.clear();
    seg = {};
    for (std::unique_ptr<pool_dev> & dp : devs) {
        pool_dev & d = *dp;
        // the last placement's copies are in the compute stream: settle them before their entries are forgotten
        if (d.backend != nullptr) {
            ggml_backend_synchronize(d.backend);
        }
        d.stats = {};
    }
}

// -----------------------------------------------------------------------------
// split-head observation: the scheduler reports every split's head right before it runs
// -----------------------------------------------------------------------------

// match a split head: registered tensors first, then a foreign expert mmid (layer from the weight name)
static bool split_head_match(llama_expert_pool_state & st, ggml_tensor * head,
                             int32_t & il, bool & lane, bool & mount) {
    il = -1;
    lane = false;
    mount = false;
    const auto it = st.split_head_refs.find(head);
    if (it != st.split_head_refs.end()) {
        il    = it->second.il;
        lane  = it->second.lane;
        mount = it->second.mount;
        return true;
    }
    if (head->op != GGML_OP_MUL_MAT_ID || head->src[0] == nullptr || head->src[2] == nullptr ||
        head->src[2]->type != GGML_TYPE_I32 || head->src[0]->name == nullptr) {
        return false;
    }
    // a "BACKEND#" prefix marks a staged copy (above the threshold only)
    if (sscanf(head->src[0]->name, "blk.%d.", &il) != 1 &&
        sscanf(head->src[0]->name, "%*[^#]#blk.%d.", &il) != 1) {
        il = -1;
    }
    return il >= 0;
}


// per-layer plan (defined below the table writers it calls)
static int32_t pool_plan_layer(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d, int32_t il);

// the layer point classifies its own layer (defined next to the ubatch boundary, which still covers the layers no point reached)
static void account_layer_ids(llama_expert_pool_state & st, int32_t il,
                               int32_t n_used, int32_t n_tok, bool pool_form);
static int32_t bind_ilx(const llama_expert_pool_state & st, int32_t il);

// mark one layer's slot map as needing a publish (inference thread only, at a layer point).
// the two entry points name what changed; both enqueue the layer's slice for the next publish
static void slot_map_mark(llama_expert_pool_state & st, int32_t il) {
    if (il >= 0 && (size_t) il < st.rank.size()) {
        st.rank[(size_t) il].dirty = true; // its residency moved, so its ends did too
    }
    for (const int32_t v : st.slot_map_publish_q) {
        if (v == il) {
            return;
        }
    }
    st.slot_map_publish_q.push_back(il);
}

// publish every marked layer's slice: the slot map mirrors slot_of, so the whole slice is recomputed
// and written in one async call per half. not a graph input, which would force a scheduler copy plus a
// same-step update
static void slot_map_publish(llama_expert_pool_state & st) {
    if (st.slot_map_publish_q.empty()) {
        return;
    }
    if (st.slot_map_row.size() != (size_t) 2 * st.n_expert) {
        st.slot_map_row.assign((size_t) 2 * st.n_expert, 0);
    }
    for (const int32_t il : st.slot_map_publish_q) {
        // the DEVICE index, not the group index: a group index is per (device, layout group) and reaches
        // past the bundle list as soon as a device holds more than one group
        const int32_t di = st.dev_of(il);
        if (di < 0 || (size_t) di >= st.devs.size()) {
            continue;
        }
        llama_expert_pool_state::pool_dev & d = *st.devs[di];
        if (d.slot_map_dev == nullptr) {
            continue;
        }
        const int32_t n_expert = st.n_expert;
        for (int32_t e = 0; e < n_expert; ++e) {
            const int32_t s = st.slot_of[(size_t) il * n_expert + e];
            st.slot_map_row[(size_t) e] = s;
            st.slot_map_row[(size_t) n_expert + e] = s < 0 ? e : -1;
        }
        // the two halves sit next to each other: one stream-ordered write per publish
        ggml_backend_tensor_set_async(d.backend, d.slot_map_dev, st.slot_map_row.data(),
                (size_t) il * 2 * n_expert * sizeof(int32_t), (size_t) 2 * n_expert * sizeof(int32_t));
        if (ggml_tensor * cpu_map = llama_expert_pool_cpu_map(st, il)) {
            ggml_backend_tensor_set(cpu_map, st.slot_map_row.data() + n_expert, 0,
                    (size_t) n_expert * sizeof(int32_t));
        }
    }
    st.slot_map_publish_q.clear();
}

// the clean top-k ids of one pooled layer and their token count (null = nothing readable here). the
// callers read the pointer, the layer point and the ubatch boundary both
static const ggml_tensor * pool_layer_ids(const llama_expert_pool_state & st, int32_t il,
                                          int32_t & n_used, int32_t & n_tok) {
    n_used = 0;
    n_tok  = 0;
    const ggml_tensor * ids = 0 <= il && (size_t) il < st.ids.size() ? st.ids[il].clean : nullptr;
    const int32_t nu = 0 <= il && (size_t) il < st.ids.size() ? st.ids[il].n_used : 0;
    if (ids == nullptr || nu <= 0 || ids->type != GGML_TYPE_I32 || ids->ne[0] <= 0) {
        return nullptr;
    }
    const int64_t n_ids = ggml_nelements(ids);
    if (n_ids <= 0 || n_ids % nu != 0) {
        return nullptr;
    }
    n_used = nu;
    n_tok  = (int32_t) (n_ids / nu);
    return ids;
}

// count one layer's realized ids into its decayed counters: per token, so a batch of any size weighs
// the same, and the count itself decays by one tick per visit, so a counter's level is the layer's
// per-token rate. `mark_ub` also protects them from this ubatch's eviction
static void pool_count_ids(llama_expert_pool_state & st, int32_t il, const int32_t * ids_host, int64_t n_ids,
                          int32_t n_tok, bool mark_ub) {
    const int32_t n_expert = st.n_expert;
    float * c = st.ids_cnt_per_token.data() + (size_t) il * (size_t) n_expert;
    float   tot = 0.0f;
    if ((size_t) il < st.rank.size()) {
        st.rank[(size_t) il].dirty = true; // the counters moved, so its ends did too
    }
    for (int32_t e = 0; e < n_expert; ++e) {
        c[e] *= st.cnt_lambda;
        tot  += c[e];
    }
    const float inc = 1.0f / (float) (n_tok > 0 ? n_tok : 1);
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t e = ids_host[i];
        if (e >= 0 && e < n_expert) {
            c[e] += inc;
            tot  += inc;
            if (mark_ub && !st.ub_use.empty()) {
                st.ub_use[(size_t) il * (size_t) n_expert + (size_t) e] = 1; // in use by this ubatch
            }
        }
    }
    if ((size_t) il < st.ids_cnt_tot.size()) {
        st.ids_cnt_tot[(size_t) il] = tot; // the layer's mass, the divisor that turns a counter into a rate
    }
}

// layer point: fold this layer's id evidence into the counters, one fold per pooled layer per ubatch.
// the aging is not here (llama_expert_pool_account_ids covers the whole-layer layout too)
static void counter_layer_point(llama_expert_pool_state & st, int32_t il) {
    st.lp_n_ids = 0; // only the successful fold below stamps it (a stale value must not leak)
    st.lp_n_tok  = 0;
    if (il < 0 || (size_t) il >= st.layers.size() || st.layers[il].ig < 0) {
        return; // not a pooled layer: no evidence
    }
    // the clean top-k ids, dense [n_used*n_tok, 1]. no wait needed, the sched staged this split's inputs
    // before the head callback
    int32_t n_used = 0;
    int32_t n_tok  = 0;
    const ggml_tensor * ids = pool_layer_ids(st, il, n_used, n_tok);
    if (ids == nullptr || !ggml_is_contiguous(ids)) {
        return; // the flat read below needs the rows packed
    }
    const int64_t n_ids = ggml_nelements(ids);
    const ggml_backend_buffer_t ids_buf = ids->view_src != nullptr ? ids->view_src->buffer : ids->buffer;
    const int32_t * ids_host = nullptr;
    if (st.sched != nullptr) {
        const ggml_tensor * staged = ggml_backend_sched_get_early_input(st.sched, ids);
        if (staged != nullptr && staged->data != nullptr && ggml_nbytes(staged) == ggml_nbytes(ids)) {
            ids_host = (const int32_t *) staged->data;
        }
    }
    if (ids_host == nullptr && ids_buf != nullptr && ids->data != nullptr && ggml_backend_buffer_is_host(ids_buf)) {
        ids_host = (const int32_t *) ids->data;
    }
    if (ids_host == nullptr && ids_buf != nullptr) {
        st.ids_buf.resize((size_t) n_ids);
        ggml_backend_tensor_get(ids, st.ids_buf.data(), 0, ggml_nbytes(ids));
        ids_host = st.ids_buf.data();
    }
    if (ids_host == nullptr) {
        return; // unreadable ids are counted at the ubatch boundary instead
    }
    // keep the ids where the layer point's classification (and the route dump) can reach them:
    // the ubatch boundary then needs no second D2H + synchronize for a layer the point covered
    if (ids_host != (const int32_t *) st.ids_buf.data()) {
        st.ids_buf.assign(ids_host, ids_host + n_ids);
    }
    st.lp_n_ids = n_ids;
    st.lp_n_tok = n_tok;
    pool_count_ids(st, il, ids_host, n_ids, n_tok, /*mark_ub=*/true);
    if ((size_t) il < st.lp_folds.size()) {
        st.lp_folds[il] += 1;
    }
    if ((size_t) il < st.cnt_done.size()) {
        st.cnt_done[il] = st.hook_step + 1;
    }
}

// does any slot of this layer hold an expert? the chain form starts empty in the partial layout and the
// plan fills it, so the graph takes the plain path until the first slots land
bool llama_expert_pool_has_resident(const llama_expert_pool_state & st, int32_t il) {
    if (il < 0 || (size_t) il >= st.layers.size() || st.n_expert <= 0) {
        return false;
    }
    const size_t base = (size_t) il * (size_t) st.n_expert;
    for (int32_t e = 0; e < st.n_expert; ++e) {
        if (base + (size_t) e < st.slot_of.size() && st.slot_of[base + (size_t) e] >= 0) {
            return true;
        }
    }
    return false;
}

// the operand a hit chain reads: the group bank for a partial layer (global slot numbers), the layer's
// own run view for a full one
ggml_tensor * llama_expert_pool_operand(const llama_expert_pool_state & st, int32_t il, int k, bool full) {
    const llama_expert_pool_state::group & a = st.groups[st.layers[il].ig];
    for (size_t ki = 0; ki < a.kinds.size(); ++ki) {
        if (a.kinds[ki] == k) {
            return full ? st.layers[il].pool[k] : a.banks[ki];
        }
    }
    return nullptr;
}

// set a layer's form: the plain path once its identity run is complete, the chain form otherwise
static void pool_set_form(llama_expert_pool_state & st, int32_t il, bool full) {
    st.layers[il].form = full ? llama_expert_pool_layer::FORM_PLAIN : llama_expert_pool_layer::FORM_CHAIN;
}

// the experts of one layer that sit on their identity slot: read off the placement, never maintained
static int32_t pool_identity_count(const llama_expert_pool_state & st, int32_t il) {
    const int32_t base = st.layers[il].base;
    if (base < 0) {
        return 0;
    }
    const int32_t n_expert = st.n_expert;
    int32_t n = 0;
    for (int32_t e = 0; e < n_expert; ++e) {
        n += st.slot_of[(size_t) (il * n_expert + e)] == base + e ? 1 : 0;
    }
    return n;
}

// a layer's form from the placement: the plain path once its identity run is complete
static void pool_form_from_run(llama_expert_pool_state & st, int32_t il) {
    pool_set_form(st, il, st.layers[il].base >= 0 && pool_identity_count(st, il) == st.n_expert);
}

// the form of every pooled layer: the whole form keeps a layer in it only when it stated a run for it
static void pool_refresh_forms(llama_expert_pool_state & st) {
    for (const int32_t il : st.pooled_layers) {
        const bool in_form = st.layers[il].ig >= 0 &&
                             (!st.layout_whole || (int32_t) st.resident[il].size() == st.n_expert);
        if (!in_form) {
            st.layers[il].form = llama_expert_pool_layer::FORM_NONE;
            continue;
        }
        pool_form_from_run(st, il);
    }
}



void llama_expert_pool_observe_split_head(void * user_data, ggml_tensor * head, ggml_backend_t backend) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) user_data;
    if (head == nullptr) {
        return;
    }
    int32_t il  = -1;
    bool lane = false;
    bool mount = false;
    if (!split_head_match(st, head, il, lane, mount)) {
        return;
    }
    // the layer's resident matmul: the copies the plan issued are ahead of it in the compute stream
    if (mount) {
        return;
    }
    // layer point: the counters update at the layer's first matched head of the ubatch
    if (st.plan_active && st.phase == llama_expert_pool_state::PHASE_FILLED && !st.ids_cnt_per_token.empty() &&
            il >= 0 && (il != st.lp_il || st.hook_step != st.lp_step)) {
        st.lp_il   = il;
        st.lp_step = st.hook_step;
        if ((size_t) il < st.lp_calls.size()) {
            st.lp_calls[il] += 1;
        }
        if (st.cnt_done.size() < st.layers.size()) {
            st.cnt_done.resize(st.layers.size(), -1);
        }
        st.lp_lane = lane; // a staged copy (the lane regime) is never read at the layer point
        counter_layer_point(st, il);
        // the classification snapshot: the placement this ubatch's plan found, taken before it runs.
        // the plan fills what this ubatch needs, so a snapshot after it would score every id a hit and
        // the rate would stop being the cache's caliber
        ggml_tensor * cpu_map = llama_expert_pool_cpu_map(st, il);
        if ((size_t) il < st.miss_snap.size() && cpu_map != nullptr && cpu_map->data != nullptr) {
            std::vector<int32_t> & s = st.miss_snap[il];
            if (s.size() != (size_t) st.n_expert) {
                s.assign((size_t) st.n_expert, 0);
            }
            const int32_t * src = (const int32_t *) cpu_map->data;
            std::copy(src, src + st.n_expert, s.begin());
        }
        // this layer's plan: the fold is done, now rank and exchange. the tables are written right here - the layer point is the only writer.
        // a batch the lane serves (prefill sizes) has no pool copy to hide, so its placement stays frozen:
        // the counts above are still recorded, but no expert is admitted or evicted
        const int32_t di = st.dev_of(il);
        if (di >= 0 && (size_t) di < st.devs.size() && st.lp_n_tok < llama_expert_pool_offload_min_batch()) {
            pool_plan_layer(st, *st.devs[di], il);
        }
        // classify this layer's realized ids here: the snapshot above is the table state its lookups saw
        if (st.lp_n_ids > 0 && st.lp_n_tok > 0) {
            st.lp_counted = 1; // the fold counted this layer this ubatch (drives the boundary's aging)
            if (st.lp_n_tok < llama_expert_pool_offload_min_batch()) {
                const int32_t n_used = (size_t) il < st.ids.size() ? st.ids[il].n_used : 0;
                account_layer_ids(st, il, n_used, st.lp_n_tok, /*pool_form=*/true);
            }
        }
    }
}

// %il ascends within one build: a decreasing index means a new graph, and no pointer survives its graph
static void bind_new_build(const llama_expert_pool_state & st, int32_t il) {
    if (il < st.bind_last_il) {
        st.split_head_refs.clear();
        st.ids.clear();
    }
    st.bind_last_il = il;
}

static int32_t bind_ilx(const llama_expert_pool_state & st, int32_t il) {
    for (size_t i = 0; i < st.pooled_layers.size(); ++i) {
        if (st.pooled_layers[i] == il) {
            return (int32_t) i;
        }
    }
    return -1;
}

void llama_expert_pool_bind_ids(const llama_expert_pool_state & st, int32_t il, int32_t n_used, const ggml_tensor * ids_clean) {
    if (il < 0 || ids_clean == nullptr) {
        return;
    }
    bind_new_build(st, il);
    if ((size_t) il >= st.ids.size()) {
        st.ids.resize((size_t) il + 1);
    }
    if ((size_t) il >= st.cnt_done.size()) {
        st.cnt_done.resize((size_t) il + 1, -1);
    }
    st.ids[il].clean   = ids_clean;
    st.ids[il].n_used  = n_used;
}

void llama_expert_pool_bind_split_head(const llama_expert_pool_state & st, int32_t il, const ggml_tensor * head, bool lane) {
    if (il < 0 || head == nullptr) {
        return;
    }
    bind_new_build(st, il);
    const int32_t ilx = bind_ilx(st, il);
    if (ilx < 0) {
        return;
    }
    st.split_head_refs[head] = { il, lane };
}

// the layer's resident matmul is the end of its window: the fills the layer launched hide in the CPU
// segment above it, and whatever is still out is on the stream's critical path either way, so the
// wait belongs right here - once per layer, where the bytes are about to be read
void llama_expert_pool_bind_mount_head(const llama_expert_pool_state & st, int32_t il, const ggml_tensor * head) {
    if (il < 0 || head == nullptr) {
        return;
    }
    bind_new_build(st, il);
    if (bind_ilx(st, il) < 0) {
        return;
    }
    st.split_head_refs[head] = { il, false, true };
}

// ubatch-boundary accounting: reads back every pool layer's top-k ids once per ubatch and feeds the
// counters, the per-layer hit/miss and the ubatch marker
// one layer's realized hit/miss counts for one ubatch, classified against the inverse half of the
// snapshot its lookups saw; called from the layer point and from the boundary for the layers it missed
static void account_layer_ids(llama_expert_pool_state & st, int32_t il,
                               int32_t n_used, int32_t n_tok, bool pool_form) {
    const int32_t n_expert = st.n_expert;
    const int64_t n_ids   = (int64_t) n_used * (int64_t) n_tok;
    // a width-0 layer has no residents, so it is not part of the hit rate; a fully resident plain layer is
    // (its columns are all hits, which is what the whole-layer layout's report has always shown)
    const bool have_tab = (size_t) il < st.miss_snap.size() &&
                          st.miss_snap[il].size() == (size_t) n_expert;
    // the layer is in the pool as long as the form carries it: nothing here reads a resident list (the plan
    // owns the residency), only whether the layer got a group in the storage pass
    const bool in_pool = llama_expert_pool_is_pool_layer(st, il) &&
                         (size_t) il < st.layers.size() && st.layers[il].ig >= 0;
    // a plain layer is fully resident in expert order: every realized expert is a hit by construction, so it
    // needs no snapshot (it has no split head to take one at)
    const bool plain = st.layers[il].form == llama_expert_pool_layer::FORM_PLAIN;
    if (!(pool_form && in_pool && st.layers[il].form != llama_expert_pool_layer::FORM_NONE && (plain || have_tab))) {
        return;
    }
    const int32_t * inv = have_tab ? st.miss_snap[il].data() : nullptr;
    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t e = st.ids_buf[(size_t) i];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        if (plain || inv[e] == -1) {
            n_hit ++;
        } else {
            n_miss ++;
        }
    }
    // per-layer realized ids (LLAMA_EXPERT_POOL_MISS_LOG=<path>): this ubatch's hit/miss for this layer
    static FILE * ml_fp = []() -> FILE * {
        const char * p = getenv("LLAMA_EXPERT_POOL_MISS_LOG");
        if (p == nullptr || p[0] == '\0') {
            return nullptr;
        }
        FILE * f = fopen(p, "w");
        if (f != nullptr) {
            setvbuf(f, NULL, _IOFBF, 1 << 20);
            fprintf(f, "# v1 step,layer,n_tok,n_used,hit,miss\n");
        }
        return f;
    }();
    if (ml_fp != nullptr) {
        fprintf(ml_fp, "%d,%d,%d,%d,%llu,%llu\n", st.hook_step, il, n_tok, n_used,
                (unsigned long long) n_hit, (unsigned long long) n_miss);
        // flush per row: the log is a live instrument, a reader tails it while the run goes
        fflush(ml_fp);
    }
    llama_expert_pool_counts & seg = moe_cache_stats_is_small(n_tok, n_used) ? st.seg : st.seg_large;
    seg.hit           += n_hit;
    seg.miss          += n_miss;
}

void llama_expert_pool_account_ids(llama_expert_pool_state & st) {
    if (!st.direct_mount || st.pooled_layers.empty() || st.phase != llama_expert_pool_state::PHASE_FILLED) {
        return;
    }
    // large batches are served by the scheduler's lane, not the pool: their ids are not pool hits
    int32_t n_tok_max = 0;
    for (size_t ilx = 0; ilx < st.pooled_layers.size(); ++ilx) {
        int32_t n_used = 0;
        int32_t n_tok  = 0;
        if (pool_layer_ids(st, st.pooled_layers[ilx], n_used, n_tok) == nullptr) {
            continue;
        }
        n_tok_max = n_tok > n_tok_max ? n_tok : n_tok_max;
    }
    const bool pool_form = n_tok_max < llama_expert_pool_offload_min_batch();
    if (!pool_form) {
        // a batch the lane serves: the prefill regime has begun, so the batch sizes may decide the layout from here on
        st.prefill_seen = true;
    }

    bool any = false;
    // the layer points classified at least one layer: the boundary must still age the counters
    if (st.lp_counted) {
        any = true;
        st.lp_counted = 0;
    }
    for (size_t ilx = 0; ilx < st.pooled_layers.size(); ++ilx) {
        const int32_t il = st.pooled_layers[ilx];
        int32_t n_used = 0;
        int32_t n_tok  = 0;
        const ggml_tensor * ids = pool_layer_ids(st, il, n_used, n_tok);
        if (ids == nullptr) {
            continue;
        }
        // the layer point folded AND classified this layer while the graph ran (the common path):
        // the point already read, dumped and classified this layer, so the boundary only covers what it could not
        if ((size_t) il < st.cnt_done.size() && st.cnt_done[il] == st.hook_step + 1) {
            continue;
        }
        const int64_t n_ids = ggml_nelements(ids);
        st.ids_buf.resize((size_t) n_ids);
        ggml_backend_tensor_get(ids, st.ids_buf.data(), 0, ggml_nbytes(ids));
        if (st.plan_active) {
            pool_count_ids(st, il, st.ids_buf.data(), n_ids, n_tok, /*mark_ub=*/false);
        }
        account_layer_ids(st, il, n_used, n_tok, pool_form);
        any = true;
    }
    if (any) {
        // the ubatch is closed: its in-use experts no longer protect anything. every layer's victim
        // side changes with that, so every cached end has to be re-derived - once per ubatch, not per
        // layer point, which is what keeps the cache cheap
        if (!st.ub_use.empty()) {
            std::fill(st.ub_use.begin(), st.ub_use.end(), 0);
            for (llama_expert_pool_state::layer_rank & rk : st.rank) {
                rk.dirty = true;
            }
        }
        // one accounted ubatch is one counted unit, planned or not: the dump and any future reader need it to advance even in the frozen arm
        st.hook_step += 1;
    }
}

// ---------------------------------------------------------------
// the hit-chain tables: the initial install and the per-entry writes

namespace {

// copy one expert slice (n=1) or a run of n consecutive experts into n consecutive slots of one pool
// tensor. the weights and the per-expert biases move together: one expert, one slot
void upload_run(ggml_backend_t backend, ggml_tensor * src, ggml_tensor * pw, int32_t e0, int32_t slot, int32_t n, int k) {
    if (src == nullptr || pw == nullptr || n <= 0) {
        return;
    }
    const size_t stride = llama_expert_pool_stride(src, k);
    // async when the backend can: the batch is joined once, not per run
    ggml_backend_tensor_set_async(backend, pw, (const char *) src->data + (size_t) e0 * stride,
            (size_t) slot * stride, (size_t) n * stride);
}

} // namespace

// install the initial tables (before any graph has run, so the write races with nothing)
// the tables mirror slot_of: the install is the publish applied to every layer
static void slot_map_install(llama_expert_pool_state & st) {
    for (size_t il = 0; il < st.layers.size(); ++il) {
        slot_map_mark(st, (int32_t) il);
    }
    slot_map_publish(st);
}



// one walk answers both sides of a pair: `sel` ranks the non-residents (candidates, hottest first),
// the other side the residents the served ubatch is not using (victims, coldest first). the plan asks
// for a few of each per layer point, so a walk per query is the cheap side; ties follow the smaller key.
// the candidate side takes every layer: a fill for a layer that already ran this ubatch serves the next
// ubatch of the same layer, which is the same kind of reuse a fill for a later layer serves
// one layer's two ends of the rank. the counters only move at a fold, the residency only at a move,
// and the plan only consumes keys - each of those marks the layer, so a cached pair answers every
// query in between
static void pool_rank_layer(llama_expert_pool_state & st, int32_t il) {
    llama_expert_pool_state::layer_rank & rk = st.rank[(size_t) il];
    rk.cand      = -1;
    rk.cand_rate = 0.0f;
    rk.vict      = -1;
    rk.vict_rate = 0.0f;
    rk.dirty     = false;
    if (st.layers[il].form != llama_expert_pool_layer::FORM_CHAIN) {
        return; // not in the index
    }
    const int32_t n_expert = st.n_expert;
    // the group is one rank space, so the counters are compared as rates: a layer folds evidence only on
    // the ubatches that reach it, so its mass carries its visit count as a factor
    const float inv = (size_t) il < st.ids_cnt_tot.size() && st.ids_cnt_tot[(size_t) il] > 0.0f
                      ? 1.0f / st.ids_cnt_tot[(size_t) il] : 0.0f;
    const float * cl = st.ids_cnt_per_token.data() + (size_t) il * (size_t) n_expert;
    for (int32_t e = 0; e < n_expert; ++e) {
        const int32_t key = il * n_expert + e;
        const float   r   = cl[e] * inv;
        if (st.slot_of[(size_t) key] >= 0) {
            if (!st.ub_use.empty() && st.ub_use[(size_t) key] != 0) {
                continue; // in use by the ubatch being served
            }
            if (rk.vict < 0 || r < rk.vict_rate || (r == rk.vict_rate && key < rk.vict)) {
                rk.vict      = key;
                rk.vict_rate = r;
            }
        } else {
            if (!st.plan_taken.empty() && st.plan_taken[(size_t) key] != 0) {
                continue; // already consumed by this plan
            }
            if (rk.cand < 0 || r > rk.cand_rate || (r == rk.cand_rate && key < rk.cand)) {
                rk.cand      = key;
                rk.cand_rate = r;
            }
        }
    }
}

// the two sides of a pair, merged off the cached ends: the hottest candidate and the coldest victim of
// the group. a free slot is a victim of heat 0, so it comes before any resident and evicts nothing; its
// index rides out in `slot_free`, and `key_out` stays -1 for it. ties follow the smaller key, like the
// single scans this replaces
static bool pool_rank_pair(llama_expert_pool_state & st, size_t ai,
                           int32_t & key_in, float & cnt_in, int32_t & key_out, float & cnt_out,
                           int32_t & slot_free) {
    key_in    = -1;
    cnt_in    = 0.0f;
    key_out   = -1;
    cnt_out   = 0.0f;
    slot_free = -1;
    if (st.ids_cnt_per_token.empty() || ai >= st.groups.size() || st.plan_taken.empty() || st.rank.empty()) {
        return false;
    }
    for (int32_t il : st.groups[ai].layers) {
        if (st.layers[il].form != llama_expert_pool_layer::FORM_CHAIN) {
            continue; // not in the index
        }
        if (st.rank[(size_t) il].dirty) {
            pool_rank_layer(st, il);
        }
        const llama_expert_pool_state::layer_rank & rk = st.rank[(size_t) il];
        if (rk.cand >= 0 && (key_in < 0 || rk.cand_rate > cnt_in || (rk.cand_rate == cnt_in && rk.cand < key_in))) {
            key_in  = rk.cand;
            cnt_in  = rk.cand_rate;
        }
        if (rk.vict >= 0 && (key_out < 0 || rk.vict_rate < cnt_out || (rk.vict_rate == cnt_out && rk.vict < key_out))) {
            key_out = rk.vict;
            cnt_out = rk.vict_rate;
        }
    }
    if (key_in < 0 || cnt_in <= 0.0f) {
        return false; // a spent counter is no candidate
    }
    slot_free = llama_expert_group_alloc(st, ai);
    if (slot_free >= 0) {
        key_out = -1;   // nothing is evicted
        cnt_out = 0.0f; // a free slot reads as heat 0
        return true;
    }
    return key_out >= 0;
}

static void pool_admit(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d, int32_t key, int32_t slot);

// the ranking decision: one policy, no branch. free slots take the hottest unconsumed candidates
// (nothing is evicted), then the rank pairs the next hottest candidate against the next coldest
// unconsumed resident, and only while the candidate beats it by the margin: a swap for a hair of rate
// pays a copy and buys nothing
static int32_t pool_plan_rank(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d,
                              int32_t il, int32_t ai) {
    const int32_t n_expert = st.n_expert;
    int32_t n_fill = 0;
    auto admit = [&](int32_t key, int32_t slot) {
        pool_admit(st, d, key, slot);
        n_fill += 1;
    };

    // one rank for free slots and residents alike: a free slot is a victim of heat 0, so it is taken
    // before any resident can be evicted, and the margin follows from the same comparison
    for (;;) {
        int32_t key_in   = -1;
        float   cnt_in   = 0.0f;
        int32_t key_out  = -1;
        float   cnt_out  = 0.0f;
        int32_t slot_free = -1;
        if (!pool_rank_pair(st, ai, key_in, cnt_in, key_out, cnt_out, slot_free)) {
            break;
        }
        if (!(cnt_in - cnt_out > st.plan_margin * cnt_in)) {
            break;
        }
        int32_t slot = slot_free;
        if (key_out >= 0) {
            st.plan_taken[(size_t) key_out] = 1; // evicted: it must not come back as a candidate
            st.rank_consumed.push_back(key_out / n_expert);
            slot = st.slot_of[key_out];
            st.slot_of[key_out] = -1;
            slot_map_mark(st, key_out / n_expert);
        }
        admit(key_in, slot);
    }
    slot_map_publish(st);
    return n_fill;
}

// the per-layer plan, run at the owning layer's point (inference thread). the two tables are read
// rank by rank: the hottest unconsumed candidate takes free slots first, then pairs with the coldest
// unconsumed resident, and only while the candidate is hotter. both rank over the shared group
// experts the served ubatch is using stay out of the eviction table (ub_use)
// place one key in a slot: copy its bytes and install the entry - the one writer of the placement.
// the copy goes out here, on the inference thread and in the compute stream; the slot's resident
// matmul is submitted later on the same stream, so it reads the bytes without an event or a wait.
// the key carries its own layer, because the ranked candidates may come from any layer of the group
static void pool_admit(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d, int32_t key, int32_t slot) {
    const int32_t il = key / st.n_expert;
    const int32_t e  = key % st.n_expert;
    const int32_t ai = st.layers[il].ig;
    if (ai < 0 || slot < 0 || (size_t) slot >= st.key_of[ai].size()) {
        return;
    }
    llama_expert_pool_layer & l = st.layers[il];
    for (size_t ki = 0; ki < st.groups[ai].kinds.size(); ++ki) {
        const int k = st.groups[ai].kinds[ki];
        upload_run(d.backend, l.orig[k], st.groups[ai].banks[ki], e, slot, 1, k);
        d.stats.bytes += llama_expert_pool_stride(l.orig[k], k);
    }
    st.slot_of[(size_t) key] = slot;
    st.key_of[ai][slot] = key;
    slot_map_mark(st, il);
    if (!st.plan_taken.empty()) {
        st.plan_taken[(size_t) key] = 1; // consumed: neither table may pick it again
        st.rank_consumed.push_back(il);
    }
    d.stats.fills += 1;
}

static int32_t pool_plan_layer(llama_expert_pool_state & st, llama_expert_pool_state::pool_dev & d, int32_t il) {
    const int32_t n_expert = st.n_expert;
    if (st.plan_per_layer <= 0 || il < 0 || (size_t) il >= st.layers.size() || st.layers[il].ig < 0) {
        return 0; // swapping frozen, or not a pooled layer
    }
    if (st.ids_cnt_per_token.empty() || st.plan_taken.empty()) {
        return 0;
    }
    const int32_t ai = st.layers[il].ig;
    // the consumed set is about to empty, so the ends derived under it are stale - but only for the
    // layers a key was taken from: other layers read the same counters either way
    for (const int32_t lt : st.rank_consumed) {
        if (lt >= 0 && (size_t) lt < st.rank.size()) {
            st.rank[(size_t) lt].dirty = true;
        }
    }
    st.rank_consumed.clear();
    std::fill(st.plan_taken.begin(), st.plan_taken.end(), 0);

    int32_t n_fill = 0;
    auto admit = [&](int32_t key, int32_t slot) {
        pool_admit(st, d, key, slot);
        n_fill += 1;
    };

    // a registered layer fills its identity run: every expert off its own slot is admitted there, and
    // whoever holds that slot is evicted - the registration outranks the heat pairing
    if (st.layers[il].base >= 0) {
        const int32_t base = st.layers[il].base;
        if (pool_identity_count(st, il) < n_expert) {
            for (int32_t e = 0; e < n_expert; ++e) {
                const int32_t key  = il * n_expert + e;
                const int32_t held = st.key_of[ai][base + e];
                if (held == key) {
                    continue; // already in place
                }
                if (held >= 0) {
                    st.slot_of[(size_t) held] = -1;
                    slot_map_mark(st, held / n_expert);
                }
                admit(key, base + e);
            }
            pool_refresh_forms(st); // the run this pass completed, or one it broke
        }
        slot_map_publish(st);
        return n_fill;
    }
    if (st.layout_whole) {
        return 0; // the whole form leaves the unregistered layers empty
    }

    return pool_plan_rank(st, d, il, ai);
}

// ---------------------------------------------------------------
// the pool driver: llama_context holds one instance. the entry points were moved verbatim from
// llama-context.cpp; the context-owned names they reach go through `ctx`

llama_expert_pool::llama_expert_pool(
        llama_context & ctx,
        const llama_model & model,
        const llama_cparams & cparams,
        const std::vector<ggml_backend_t> & backend_ptrs,
        const std::vector<ggml_backend_buffer_type_t> & backend_buft) :
    ctx(ctx), model(model), cparams(cparams), backend_ptrs(backend_ptrs), backend_buft(backend_buft) {
    // the mechanism's device path. it is built here, inside the mechanism, so that one budget configures
    // one object and the miss method only says how a miss is computed
    const llama_expert_pool_miss_strategy s = llama_expert_pool_miss_strategy_of(cparams.expert_pool_miss_method);
    if (s.cache_required && cparams.moe_cache_size > 0) {
        cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size,
                cparams.moe_cache_layers);
    } else if (s.cache_required) {
        LLAMA_LOG_WARN("%s: the gpu miss method runs the MoE cache: set --moe-cache-mib\n", __func__);
    }
}

// create one bundle per selected serving device, with the given layers and budgets; the single place
// bundles are created. the relayout frees them first (each form builds its own views)
void llama_expert_pool::expert_pool_make_devs(
        const std::vector<ggml_backend_dev_t> &   dv_order,
        const std::vector<std::vector<int32_t>> & layers_by_dev,
        const std::vector<int32_t> &              slots_by_dev) {
    llama_expert_pool_state & st = this->st;
    const int32_t n_expert    = (int32_t) model.hparams.n_expert;
    const int32_t n_layer_all = (int32_t) model.hparams.n_layer_all;

    st.layer_dev.assign(n_layer_all, -1);
    st.devs.clear();
    st.dev_slots.clear();
    for (size_t si = 0; si < dv_order.size(); ++si) {
        const ggml_backend_dev_t layer_dev = dv_order[si];
        // the device's first non-host buft; a device without one gets the CPU buft, which is the degenerate (nothing to cache) configuration
        ggml_backend_buffer_type_t bt = pool_dev_buft(backend_buft, layer_dev);
        std::unique_ptr<llama_expert_pool_state::pool_dev> d(new llama_expert_pool_state::pool_dev);
        d->buft = bt;
        d->backend = pool_backend_of(backend_ptrs, backend_buft, bt);
        d->ctx.reset(ggml_init({ 4u*1024u*1024u, nullptr, true })); // no_alloc
        d->slot_map_ctx.reset(ggml_init({ 4u*1024u*1024u, nullptr, true }));
        d->ctx_host.reset(ggml_init({ 64u*1024u, nullptr, true }));
        // the device table: ONE [n_expert, 2*n_layer_all] I32 tensor with both halves; each layer gets a view
        d->slot_map_dev = ggml_new_tensor_2d(d->slot_map_ctx.get(), GGML_TYPE_I32, n_expert, 2 * (int64_t) n_layer_all);
        ggml_set_name(d->slot_map_dev, "hit_slot_map_dev");
        st.dev_slots.push_back(si < slots_by_dev.size() ? slots_by_dev[si] : 0);
        st.devs.push_back(std::move(d));
        for (int32_t il : layers_by_dev[si]) {
            st.layer_dev[il] = (int32_t) si;
            st.devs.back()->layers.push_back(il);
        }
    }
    int32_t n_layers_pooled = 0;
    for (const std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        n_layers_pooled += (int32_t) dp->layers.size();
    }
    LLAMA_LOG_INFO("%s: pool serves %d layer(s) on %d device bundle(s)\n",
            __func__, n_layers_pooled, (int32_t) st.devs.size());
    for (size_t bi = 0; bi < st.devs.size(); ++bi) {
        LLAMA_LOG_INFO("%s: bundle %zu: %s, %zu layer(s), %d slots (--moe-cache-mib)\n",
                __func__, bi, ggml_backend_buft_name(st.devs[bi]->buft), st.devs[bi]->layers.size(),
                st.dev_slots[bi]);
    }
}

// how many slots of one device each of its layers may hold: sizing only, the plan owns the placement
static void pool_spread_widths(const std::vector<int32_t> & layers, int32_t slots, int32_t n_expert,
                               std::vector<int32_t> & widths) {
    const int32_t P   = (int32_t) layers.size();
    int32_t       cap = slots / P;
    const int32_t rem = cap > n_expert ? 0 : slots % P;
    if (cap > n_expert) {
        cap = n_expert;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        widths[layers[i]] = cap + ((int32_t) i >= P - rem ? 1 : 0);
    }
}

// the whole form's residency: the shallowest (budget / n_expert) layers of each device hold every expert
// at its own slot, in the model's order - those run the plain path and need no remap
static void pool_set_full_resident(std::vector<std::vector<int32_t>> & resident,
                                   const std::vector<std::vector<int32_t>> & layers_by_dev,
                                   const std::vector<int32_t> & slots_by_dev, int32_t n_expert) {
    for (size_t b = 0; b < layers_by_dev.size(); ++b) {
        const int32_t N = b < slots_by_dev.size() ? slots_by_dev[b] : 0;
        const int32_t n_full = std::min<int32_t>((int32_t) layers_by_dev[b].size(), N / n_expert);
        for (int32_t i = 0; i < n_full; ++i) {
            std::vector<int32_t> & res = resident[layers_by_dev[b][i]];
            res.resize((size_t) n_expert);
            for (int32_t e = 0; e < n_expert; ++e) {
                res[e] = e;
            }
        }
    }
}

// the budget each serving device gets, in bytes: one entry per device, or upstream's single budget split
// among the devices like the layers (--tensor-split when it is given, else free memory). false = the
// entries do not fit the devices, so the pool must stay off
static bool pool_device_bytes(const llama_model & model, const llama_cparams & cparams, const char * func,
                              const std::vector<ggml_backend_dev_t> & dv_order, std::vector<size_t> & out) {
    const int32_t n_dv = (int32_t) dv_order.size();
    out.assign((size_t) n_dv, 0);
    const std::vector<size_t> & budgets = cparams.moe_cache_size_dev;
    if (budgets.size() > (size_t) n_dv) {
        LLAMA_LOG_ERROR("%s: %zu budget entries but the pool serves %d device(s); pool disabled\n",
                func, budgets.size(), n_dv);
        return false;
    }
    if (budgets.size() == 1) {
        // one budget: split it among the devices like the layers, by the same rule the MoE cache uses
        const std::vector<char>   serves(n_dv, 1);
        const std::vector<double> share = moe_cache_device_split(model, dv_order, serves);
        double sum = 0.0;
        for (const double s : share) {
            sum += s;
        }
        for (int32_t b = 0; b < n_dv; ++b) {
            out[b] = (size_t) ((double) budgets[0]*share[b]/sum);
        }
    } else {
        for (size_t b = 0; b < budgets.size(); ++b) {
            out[b] = budgets[b];
        }
    }
    return true;
}

// one device that ends up with a pool: its layer set and the slots its budget buys
struct pool_selection {
    ggml_backend_dev_t   dev    = nullptr;
    std::vector<int32_t> layers;
    int32_t              slots  = 0;
};

// turn the per-device budgets into the pools that will exist: keep each device's layers (its deepest M),
// size its slots from the budget, and record the per-layer capacity the groups are sized from
static std::vector<pool_selection> pool_select_devices(const llama_model & model, const llama_cparams & cparams,
        const char * func, const std::vector<ggml_backend_buffer_type_t> & backend_buft,
        const std::vector<ggml_backend_dev_t> & dv_order,
        const std::vector<std::vector<int32_t>> & dv_layers,
        const std::vector<size_t> & dv_bytes, bool init_whole, int32_t n_expert,
        std::vector<int32_t> & caps) {
    std::vector<pool_selection> out;
    const int32_t n_dv = (int32_t) dv_order.size();
    for (int32_t b = 0; b < n_dv; ++b) {
        std::vector<int32_t> keep = dv_layers[b];   // ascending
        const int32_t M = b < (int32_t) cparams.moe_cache_layers.size() ? cparams.moe_cache_layers[b] : 0;
        if (dv_bytes[b] == 0) {
            LLAMA_LOG_INFO("%s: %s: no budget entry, its %d eligible MoE layer(s) stay at width 0\n",
                    func, ggml_backend_dev_name(dv_order[b]), (int32_t) keep.size());
            continue;
        }
        if (M > 0 && (int32_t) keep.size() > M) {
            const int32_t n_eligible = (int32_t) keep.size();
            keep = moe_cache_deepest(keep, M); // the same scope rule the MoE cache's path applies
            LLAMA_LOG_INFO("%s: %s: pooling the deepest %d of %d eligible MoE layers (layer %d..%d)\n",
                    func, ggml_backend_dev_name(dv_order[b]), (int32_t) keep.size(), n_eligible,
                    keep.front(), keep.back());
        }
        if (keep.empty()) {
            continue;
        }
        // the budget is in bytes and the plan works in slots: the shared capacity rule decides how many fit
        const llama_layer & Lref = model.layers[keep.front()];
        const std::vector<ggml_tensor *> ref_bias = {
            Lref.ffn_gate_up_exps_b, Lref.ffn_up_exps_b, Lref.ffn_gate_exps_b, Lref.ffn_down_exps_b,
        };
        const int32_t N = moe_cache_slots_in(pool_dev_buft(backend_buft, dv_order[b]), llama_moe_cache_layer_experts(Lref),
                ref_bias, dv_bytes[b], (int32_t) keep.size()*n_expert);
        if (N <= 0) {
            LLAMA_LOG_WARN("%s: %s: a budget of %zu MiB is below one expert, that pool stays off\n",
                    func, ggml_backend_dev_name(dv_order[b]), dv_bytes[b]/(1024*1024));
            continue;
        }
        // the capacity attribution: how many slots of this device each of its layers may hold. sizing only -
        // nothing decides a placement with it (the plan owns the residency); the groups are sized once
        pool_spread_widths(keep, N, n_expert, caps);
        if (init_whole) {
            const int32_t n_full = std::min<int32_t>((int32_t) keep.size(), N / n_expert);
            if (n_full <= 0) {
                LLAMA_LOG_WARN("%s: %s: budget %d is below one full layer (%d experts): the pool starts empty, "
                               "the partial layout would spread it\n",
                        func, ggml_backend_dev_name(dv_order[b]), N, n_expert);
                continue;
            }
            // the full layers own a full run each; only the rest splits the remaining slots
            std::vector<int32_t> rest(keep.begin() + n_full, keep.end());
            for (int32_t i = 0; i < n_full; ++i) {
                caps[keep[i]] = n_expert;
            }
            for (int32_t il : rest) {
                caps[il] = 0;
            }
            if (!rest.empty()) {
                pool_spread_widths(rest, N - n_full * n_expert, n_expert, caps);
            }
            LLAMA_LOG_INFO("%s: %s: whole-layer layout, %d full layer(s) (layer %d..%d), %d slots unused\n",
                    func, ggml_backend_dev_name(dv_order[b]), n_full, keep.front(), keep[n_full - 1],
                    N - n_full * n_expert);
        }
        out.push_back({ dv_order[b], std::move(keep), N });
    }
    return out;
}

// expert pool initialization: find the host-resident MoE layers, resolve the per-device budgets
void llama_expert_pool::expert_pool_init() {
    // already built (a reserve after a rebuild re-enters here): a reset() would wipe the fresh pool
    if (this->st.phase != llama_expert_pool_state::PHASE_NONE) {
        return;
    }
    llama_expert_pool_state & st = this->st;
    st.reset();

    if (cparams.expert_pool <= 0) {
        return;
    }
    // one bundle per serving device, each with its own weight buffer and tables
    if (model.n_devices() > 1) {
        LLAMA_LOG_WARN("%s: %d devices: the pool runs one bundle per serving device, but this "
                       "path is NOT verified on this build (single-GPU host)\n",
                __func__, model.n_devices());
    }
    const int32_t n_layer = model.hparams.n_layer();
    const int32_t n_layer_all = (int32_t) model.hparams.n_layer_all;
    const int32_t n_expert = model.hparams.n_expert;
    if (n_expert <= 0) {
        return;
    }
    // the method governs the range below the offload threshold only: at/above it the scheduler's lane runs the miss chain on the pool device
    LLAMA_LOG_INFO("%s: expert pool: miss method %s, offload threshold %d\n", __func__,
            llama_expert_pool_miss_method_name(cparams.expert_pool_miss_method),
            llama_expert_pool_offload_min_batch());
    // unified memory: nothing to cache, the same memory already holds the experts. informational
    // GGML_CUDA_ENABLE_UNIFIED_MEMORY is the same case but the device still reports itself as discrete, so it is not detected here.
    for (int32_t il = 0; il < n_layer_all; ++il) {
        if (ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            LLAMA_LOG_WARN("%s: expert pool is used on an integrated/unified-memory GPU (layer %d) - "
                           "consider disabling the pool and running without CPU offload for better performance\n",
                           __func__, il);
            break;
        }
    }
    const int32_t il_end   = n_layer;
    st.n_expert = n_expert;
    // swap <= 0 freezes the resident set (the static A/B control arm). LLAMA_EXPERT_POOL_FILL_CAP is the
    // debug override that reaches it: the frozen arm is the only bit-identity oracle the pool has
    st.plan_per_layer = cparams.expert_pool_fill_cap;
    if (const char * s = getenv("LLAMA_EXPERT_POOL_FILL_CAP")) {
        st.plan_per_layer = atoi(s);
    }
    st.plan_active     = st.plan_per_layer > 0;
    // debug override for the sweep, like the half-life: the margin itself is code-set
    if (const char * s = getenv("LLAMA_EXPERT_POOL_MARGIN")) {
        st.plan_margin = (float) atof(s);
    }
    LLAMA_LOG_INFO("%s: expert pool: pairing margin %.3f of the candidate's rate\n", __func__, (double) st.plan_margin);
    // decaying activation counter: lambda = 2^(-1/H) per layer visit; a layer ticks when its own evidence
    // is folded, so an unvisited layer (one the batch does not reach) keeps its counters as they are
    st.cnt_decay_hl = cparams.expert_pool_cnt_decay > 0 ? cparams.expert_pool_cnt_decay : st.cnt_decay_hl;
    // debug override for the sweep: the half-life is code-set, this reaches it without a rebuild
    if (const char * s = getenv("LLAMA_EXPERT_POOL_CNT_DECAY")) {
        const int32_t v = atoi(s);
        if (v > 0) {
            st.cnt_decay_hl = v;
        }
    }
    st.cnt_lambda   = std::pow(2.0f, -1.0f / (float) st.cnt_decay_hl);
    LLAMA_LOG_INFO("%s: expert pool: decaying counter, half-life %d layer visits (per-visit lambda %.6f)\n",
            __func__, st.cnt_decay_hl, st.cnt_lambda);

    // --- find the pooled layers: the ONE layer-set resolution (shared with the MoE cache), then the pool's
    // own gates. the pool serves the model's own layers, not the NextN blocks: its chain reads the layer's
    // FFN input after a device segment, which those blocks do not have
    std::vector<int32_t> pooled_ils;
    // pooling stays opt-in: without a host-placement override there is no chain to build
    if (model.get_params().tensor_buft_overrides != nullptr) {
        for (const moe_cache_layer & e : moe_cache_eligible_layers(model, il_end, /*warn=*/true, __func__)) {
            pooled_ils.push_back(e.il);
        }
    }
    if (pooled_ils.empty()) {
        LLAMA_LOG_WARN("%s: no CPU-resident MoE layers found, expert pool disabled\n", __func__);
        return;
    }

    // --- per device budgets -------------------------------------------------------------
    // group the eligible layers by their serving device, in the model's device order (= --device order).
    // the budget entries resolve against that order. every pool gets its OWN layer set and capacity:
    // nothing is shared between pools except the rate knobs and the batch-regime threshold.
    std::vector<ggml_backend_dev_t>   dv_order;
    std::vector<std::vector<int32_t>> dv_layers;
    for (const llama_device & md : model.devices) {
        if (std::find(dv_order.begin(), dv_order.end(), md.dev) != dv_order.end()) {
            continue; // a device listed twice serves one pool
        }
        std::vector<int32_t> mine;
        for (const int32_t il : pooled_ils) {
            if (model.dev_layer(il) == md.dev) {
                mine.push_back(il);
            }
        }
        if (mine.empty()) {
            continue;
        }
        dv_order.push_back(md.dev);
        dv_layers.push_back(std::move(mine));
    }

    // the budget is --moe-cache-mib: one entry per device in --device order, in bytes
    std::vector<size_t> dv_bytes;
    if (!pool_device_bytes(model, cparams, __func__, dv_order, dv_bytes)) {
        st.reset();
        return;
    }

    // per device: keep (its layers, its deepest M), then its own capacity
    const auto    layout_pin   = llama_expert_pool_layout_pinned();
    // auto decides by the batch regime, and the first phase is almost always a prefill
    const bool    init_whole   = layout_pin != LLAMA_EXPERT_POOL_LAYOUT_PARTIAL;
    std::vector<int32_t> caps(n_layer_all, 0);   // the per-layer capacity the groups are sized from
    const std::vector<pool_selection> sel = pool_select_devices(model, cparams, __func__, backend_buft,
            dv_order, dv_layers, dv_bytes, init_whole, n_expert, caps);
    std::vector<int32_t> kept_ils;
    for (const pool_selection & p : sel) {
        kept_ils.insert(kept_ils.end(), p.layers.begin(), p.layers.end());
    }
    std::sort(kept_ils.begin(), kept_ils.end());
    if (kept_ils.empty()) {
        LLAMA_LOG_WARN("%s: no device got a usable budget, expert pool disabled\n", __func__);
        return;
    }

    st.layers.assign(n_layer_all, llama_expert_pool_layer{});
    st.resident.resize(n_layer_all);
    st.width_max = caps;
    st.pooled_layers = kept_ils;

    st.layout_whole = init_whole;

    // one bundle per selected device, in the budget order resolved above
    std::vector<ggml_backend_dev_t>   sel_dev;
    std::vector<std::vector<int32_t>> sel_layers;
    std::vector<int32_t>              sel_slots;
    for (const pool_selection & p : sel) {
        sel_dev.push_back(p.dev);
        sel_layers.push_back(p.layers);
        sel_slots.push_back(p.slots);
    }

    expert_pool_make_devs(sel_dev, sel_layers, sel_slots);

    if (init_whole) {
        // identity slot order: slot s holds expert s, so no remap is needed
        pool_set_full_resident(st.resident, sel_layers, sel_slots, n_expert);
        st.phase = llama_expert_pool_state::PHASE_BUILT;
        expert_pool_build();
        return;
    }

    // the partial form keeps every pooled layer and starts with nothing resident: the counters arrive with
    // the first ubatch and the plan fills the pool from them, so there is no initial placement to write
    st.phase = llama_expert_pool_state::PHASE_BUILT;
    expert_pool_build();
}

// the initial placement: the one placement that is filled directly. it goes through the same admission
// path the plan uses - copy the experts, install the entries, publish. every later re-layout places
// nothing - the plan owns it
static void pool_place_initial(llama_expert_pool_state & st) {
    for (int32_t il : st.pooled_layers) {
        const int32_t ai = st.layers[il].ig;
        if (ai < 0 || (size_t) il >= st.width_max.size() || st.width_max[il] <= 0) {
            continue;
        }
        std::vector<int32_t> res = st.resident[il];
        std::sort(res.begin(), res.end()); // the same ascending order the old pass used to assign
        for (int32_t e : res) {
            const int32_t slot = st.layers[il].base >= 0
                ? st.layers[il].base + e : llama_expert_group_alloc(st, ai);
            if (slot < 0) {
                break;
            }
            // the same admission the plan uses: one writer for the placement, one writer for its count
            const int32_t di = st.dev_of(il);
            if (di < 0 || (size_t) di >= st.devs.size()) {
                continue;
            }
            pool_admit(st, *st.devs[di], il * st.n_expert + e, slot);
        }
    }
    slot_map_install(st);
    pool_refresh_forms(st); // the runs the placement completed open the plain path
}

void llama_expert_pool::expert_pool_build() {
    llama_expert_pool_state & st = this->st;
    const int32_t n_expert = model.hparams.n_expert;
    const std::vector<int32_t> & pooled_ils = st.pooled_layers;

    // the bundles were created in expert_pool_init: build only allocates and fills
    // reset the whole array so a zero-slot layer never keeps stale pointers
    st.layers.assign(model.hparams.n_layer_all, llama_expert_pool_layer{});
    st.lp_calls.assign(st.layers.size(), 0);
    st.lp_folds.assign(st.layers.size(), 0);
    // the counters and the plan's scratch are allocated once here: shared storage, disjoint columns
    if (st.plan_active) {
        st.ids_cnt_per_token.assign(st.layers.size() * (size_t) st.n_expert, 0.0f);
        st.ids_cnt_tot.assign(st.layers.size(), 0.0f);
        st.ub_use.assign(st.layers.size() * (size_t) st.n_expert, 0);
        st.plan_taken.assign(st.layers.size() * (size_t) st.n_expert, 0);
        st.rank.assign(st.layers.size(), {});
        st.miss_snap.assign(st.layers.size(), std::vector<int32_t>());
    }

    // one group per (device, layout group), and one bank per kind
    pool_build_groups();

    // --- hit_chain registry ---
    // sized to n_layer_all before any hit_chain is written, so a reference stays valid for the whole run
    st.direct_mount = true;
    {
        // one pool-side copy of the top-k ids per pool layer (see the field comment):
        // the graph writes it every ubatch, the pool reads it after the graph
        st.ids_shadow.assign(model.hparams.n_layer_all, nullptr);
        for (int32_t il : pooled_ils) {
            const int32_t di = st.dev_of(il);
            if (di < 0) {
                continue;
            }
            st.ids_shadow[il] = ggml_new_tensor_1d(st.devs[di]->slot_map_ctx.get(), GGML_TYPE_I32,
                    (int64_t) model.hparams.n_expert_used(il) * cparams.n_ubatch);
            char nm[64];
            snprintf(nm, sizeof(nm), "mnt_ids_shadow_%d", il);
            ggml_set_name(st.ids_shadow[il], nm);
        }
    }
    for (int32_t il : pooled_ils) {
        if (!st.direct_mount) {
            break;
        }
        // the tables exist for every layer the storage pass gave a group, whether or not the plan makes a
        // hit_chain of it
        if (il < 0 || (size_t) il >= st.layers.size() || st.layers[il].ig < 0) {
            continue;
        }
        const int32_t di = st.dev_of(il);
        if (di < 0) {
            continue;
        }
        llama_expert_pool_state::pool_dev & d = *st.devs[di];
        // a fresh hit_chain: the table and scale views (here) plus the plan fields (build_views)
        llama_expert_pool_layer & l = st.layers[il];
        const llama_layer & L = model.layers[il];
        const size_t  i32sz        = ggml_type_size(GGML_TYPE_I32);
        const int64_t n_layers_all = (int64_t) model.hparams.n_layer_all;
        l.remap     = ggml_view_2d(d.slot_map_ctx.get(), d.slot_map_dev, 1, n_expert, i32sz,
                                   il * 2 * n_expert * i32sz);
        l.remap_inv = ggml_view_2d(d.slot_map_ctx.get(), d.slot_map_dev, 1, n_expert, i32sz,
                                   (il * 2 + 1) * n_expert * i32sz);
        // the CPU part's table: I32 [1, n_expert] in host memory, one tensor per layer (upstream's cpu_map shape).
        // the lane reads it in the graph, so it must not be a view of a shared slab
        l.cpu_map = ggml_new_tensor_2d(d.ctx_host.get(), GGML_TYPE_I32, 1, n_expert);
        if (L.ffn_down_exps_s != nullptr) {
            l.scale = ggml_new_tensor_1d(d.slot_map_ctx.get(), GGML_TYPE_F32, n_expert);
        }
        if (L.ffn_up_exps_s != nullptr) {
            l.scale_up = ggml_new_tensor_1d(d.slot_map_ctx.get(), GGML_TYPE_F32, n_expert);
        }
        if (L.ffn_gate_exps_s != nullptr) {
            l.scale_gate = ggml_new_tensor_1d(d.slot_map_ctx.get(), GGML_TYPE_F32, n_expert);
        }
        char nm[64];
        snprintf(nm, sizeof(nm), "mnt_remap_%d", il);
        ggml_set_name(l.remap, nm);
        snprintf(nm, sizeof(nm), "mnt_remap_inv_%d", il);
        ggml_set_name(l.remap_inv, nm);
        snprintf(nm, sizeof(nm), "cpu_map_%d", il);
        ggml_set_name(l.cpu_map, nm);
        if (l.scale != nullptr) {
            snprintf(nm, sizeof(nm), "mnt_scale_%d", il);
            ggml_set_name(l.scale, nm);
        }
        if (l.scale_up != nullptr) {
            snprintf(nm, sizeof(nm), "mnt_scale_up_%d", il);
            ggml_set_name(l.scale_up, nm);
        }
        if (l.scale_gate != nullptr) {
            snprintf(nm, sizeof(nm), "mnt_scale_gate_%d", il);
            ggml_set_name(l.scale_gate, nm);
        }
    }
    // the tables and the weight buffers, on each bundle's own device
    if (!pool_alloc_storage()) {
        expert_pool_release(); // frees the bundles
        st.reset();
        return;
    }

    // deferred to expert_pool_fill(), after graph_reserve: with no_alloc tensors the data is valid only
    // once the scheduler placed them

    // the storage exists: the plan's views and hit_chains may be written now
    expert_pool_build_views();

    // the initial placement: the one placement that is filled directly, and filled synchronously so the
    // first decode does not start on an empty pool. every later re-layout places nothing - the plan owns it
    if (!model.no_alloc()) { // the memory-fit pass has no weights to place
        pool_place_initial(st);
    }
}

// the weight storage: one group per (device, layout group), then one bank per kind sized by the members
void llama_expert_pool::pool_build_groups() {
    llama_expert_pool_state & st = this->st;
    const std::vector<int32_t> & pooled_ils = st.pooled_layers;

    // --- pool weight storage: one group per (device, layout group) ---
    // the group holds the slots of its members; the plans lay the members' runs out inside it (the
    // capacity is the sum of the members' caps, so a relayout keeps the group)
    st.groups.clear();
    for (int32_t il : pooled_ils) {
        const int32_t di = st.dev_of(il);
        if (di < 0) {
            continue;
        }
        const llama_layer & L = model.layers[il];
        // the expert weights come from the same enumeration the MoE cache reads; the per-expert biases are pool-only (they move with their weights)
        ggml_tensor * src[PK_N] = {
            nullptr, nullptr, nullptr, nullptr,
            L.ffn_gate_up_exps_b, L.ffn_up_exps_b, L.ffn_gate_exps_b, L.ffn_down_exps_b,
        };
        for (ggml_tensor * t : llama_moe_cache_layer_experts(L)) {
            const int k = t == L.ffn_gate_up_exps ? PK_GATE_UP
                        : t == L.ffn_up_exps      ? PK_UP
                        : t == L.ffn_gate_exps    ? PK_GATE
                        :                           PK_DOWN;
            src[k] = t;
        }
        std::vector<int32_t> kinds;
        std::vector<ggml_tensor *> ref;
        for (int k = 0; k < PK_N; ++k) {
            if (src[k] != nullptr) {
                kinds.push_back(k);
                ref.push_back(src[k]);
            }
        }
        if (kinds.empty()) {
            continue;
        }
        // same expert tensor set, same slot payload, one group; the device is part of the key
        llama_expert_pool_state::group * a = nullptr;
        for (llama_expert_pool_state::group & cand : st.groups) {
            if (cand.id == di && cand.kinds == kinds && llama_moe_cache_same_layout(cand.ref, ref)) {
                a = &cand;
                break;
            }
        }
        if (a == nullptr) {
            st.groups.emplace_back();
            a = &st.groups.back();
            a->id    = di;
            a->kinds = kinds;
            a->ref   = ref;
        }
        a->layers.push_back(il);
        st.layers[il].ig = (int32_t) (a - st.groups.data());
    }

    // one bank per kind, sized by the members' capacities
    for (size_t ai = 0; ai < st.groups.size(); ++ai) {
        llama_expert_pool_state::group & a = st.groups[ai];
        llama_expert_pool_state::pool_dev & dev = *st.devs[a.id];
        const int32_t cap = llama_expert_group_cap(st, ai);
        for (size_t ki = 0; ki < a.kinds.size(); ++ki) {
            ggml_tensor * bank = moe_cache_new_bank(dev.ctx.get(), a.ref[ki], cap, a.kinds[ki] >= PK_UP_B);
            ggml_format_name(bank, "group%zu_k%d", ai, (int) a.kinds[ki]);
            a.banks.push_back(bank);
        }
    }

}

// the tables and the weight buffers of every bundle; false = the pool must be released
bool llama_expert_pool::pool_alloc_storage() {
    llama_expert_pool_state & st = this->st;
    const std::vector<int32_t> & pooled_ils = st.pooled_layers;

    // the memory-fit pass runs on a model that carries no weight data: the pool states its shape there
    // and lays out dummy buffers, like the MoE cache does for the same pass
    if (model.no_alloc()) {
        for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
            llama_expert_pool_state::pool_dev & d = *dp;
            d.slot_map_buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
            d.buf_host.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 0));
            d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(d.slot_map_ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.slot_map_ctx.get(), t)) {
                t->buffer = d.slot_map_buf.get();
            }
            for (ggml_tensor * t = ggml_get_first_tensor(d.ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx_host.get(), t)) {
                t->buffer = d.buf_host.get();
            }
            for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                t->buffer = d.buf.get();
            }
        }
        return true;
    }

    // --- allocate: each bundle's tables on its own device, the host halves on the CPU ---
    if (st.direct_mount) {
        for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
            llama_expert_pool_state::pool_dev & d = *dp;
            d.slot_map_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.slot_map_ctx.get(), d.buft));
            if (!d.slot_map_buf) {
                LLAMA_LOG_ERROR("%s: hit_chain table allocation failed on %s, hit chain disabled\n",
                        __func__, ggml_backend_buft_name(d.buft));
                st.direct_mount = false;
                break;
            }
            // CPU-side table: allocate with the CPU buft (host memory); -1 = no expert on the CPU part
            d.buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx_host.get(),
                    ggml_backend_cpu_buffer_type()));
            if (d.buf_host) {
                ggml_backend_buffer_clear(d.buf_host.get(), 0xff);
            }
            if (!d.buf_host) {
                LLAMA_LOG_ERROR("%s: CPU hit_chain table allocation failed, hit chain disabled\n", __func__);
                st.direct_mount = false;
                break;
            }
        }
        if (!st.direct_mount) {
            // de-register the hit_chains so the graph builder never sees active=true with null tables
            for (int32_t il : pooled_ils) {
                st.layers[il].form = llama_expert_pool_layer::FORM_NONE;
            }
        }
    }

    // --- allocate each bundle's weight tensors on its own device (CPU fallback per bundle) ---
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        // a backend that reorders expert weights on the device side would read every uploaded slice wrong
        // (SYCL does that lazily, in place), so the pool is refused there
        ggml_backend_dev_t pool_dev = ggml_backend_buft_get_device(d.buft);
        if (pool_dev != nullptr &&
                std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(pool_dev))) == "SYCL") {
            LLAMA_LOG_WARN("%s: %s reorders expert weights in place on first use; the pool uploads raw "
                           "weights and would read them wrong - pool disabled\n",
                    __func__, ggml_backend_dev_name(pool_dev));
            return false;
        }
        d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
        if (!d.buf) {
            LLAMA_LOG_WARN("%s: pool allocation on %s failed, falling back to CPU\n",
                    __func__, ggml_backend_buft_name(d.buft));
            d.buft = ggml_backend_cpu_buffer_type();
            d.backend = pool_backend_of(backend_ptrs, backend_buft, d.buft);
            d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
        }
        if (!d.buf) {
            LLAMA_LOG_ERROR("%s: expert pool allocation failed, pool disabled\n", __func__);
            return false;
        }
        LLAMA_LOG_INFO("%s: pool allocated on %s (host=%d, %zu bytes)\n",
                __func__, ggml_backend_buft_name(d.buft),
                (int) ggml_backend_buffer_is_host(d.buf.get()), ggml_backend_buffer_get_size(d.buf.get()));
        // mark the pool tensors as weights so the scheduler keeps the warm-path mat_mul_id (and its outputs) on the pool device
        ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // zero the pool buffer ONCE: fresh backend buffers are not guaranteed zeroed
        ggml_backend_buffer_clear(d.buf.get(), 0);
    }

    return true;
}

// the plan's views and hit-chain operands; every relayout re-runs this
void llama_expert_pool::expert_pool_build_views() {
    llama_expert_pool_state & st = this->st;

    // a fresh view pass: clear the tensor views, keep the placement (a form change is a policy
    // statement - the residency survives it; only the identity base is re-derived below)
    pool_reset_views();

    // the plan assigns slots: a full layer owns the run at its own index (the plain path reads it with
    // the clean ids), the others draw single slots
    pool_build_layer_views();

    // the form of every pooled layer, from the placement
    pool_set_forms();

    // the plan is in place: report it (the group may be wider than the plan's slots)
    int32_t tot_slots  = 0;
    size_t  pool_bytes = 0;
    for (int32_t il : st.pooled_layers) {
        const int32_t nx = (int32_t) st.resident[il].size();
        tot_slots += nx;
        const int32_t ai = st.layers[il].ig;
        if (ai < 0 || nx <= 0) {
            continue;
        }
        for (size_t ki = 0; ki < st.groups[ai].kinds.size(); ++ki) {
            if (st.groups[ai].kinds[ki] < PK_UP_B) { // weights only (biases excluded, as before)
                pool_bytes += (size_t) nx * st.groups[ai].banks[ki]->nb[2];
            }
        }
    }
    const int32_t n_pooled = (int32_t) st.pooled_layers.size();
    LLAMA_LOG_INFO("%s: expert pool enabled: %d pooled layers, %d slots total (%d/layer if uniform)\n",
            __func__, n_pooled, tot_slots, n_pooled > 0 ? tot_slots / n_pooled : 0);
    LLAMA_LOG_INFO("%s: pool weight bytes = %.2f GB (%.1f MiB per expert)\n",
            __func__, pool_bytes / 1e9, tot_slots > 0 ? pool_bytes / (double) tot_slots / 1.0e6 : 0.0);

    // the tensors exist now: the fill may write them
    st.phase = llama_expert_pool_state::PHASE_BUILT;
}

// clear the tensor views and re-derive the group slot spaces; the placement itself survives a re-layout
void llama_expert_pool::pool_reset_views() {
    llama_expert_pool_state & st = this->st;
    const int32_t n_expert = model.hparams.n_expert;

    for (llama_expert_pool_layer & L : st.layers) {
        // the routing and scale tables are per-layer VIEWS of the bundle's tables: their contents follow the
        // plan, the tensors outlive a re-layout, so the view reset keeps them (a rebuild recreates them)
        ggml_tensor * remap          = L.remap;
        ggml_tensor * remap_inv      = L.remap_inv;
        ggml_tensor * cpu_map        = L.cpu_map;
        ggml_tensor * scale          = L.scale;
        ggml_tensor * scale_up       = L.scale_up;
        ggml_tensor * scale_gate     = L.scale_gate;
        const int32_t ig = L.ig;
        L = llama_expert_pool_layer{};
        L.ig             = ig;
        L.remap          = remap;
        L.remap_inv      = remap_inv;
        L.cpu_map        = cpu_map;
        L.scale          = scale;
        L.scale_up       = scale_up;
        L.scale_gate     = scale_gate;
    }
    st.slot_of.resize((size_t) model.hparams.n_layer_all * n_expert, -1);
    st.key_of.resize(st.groups.size());
    for (size_t ai = 0; ai < st.groups.size(); ++ai) {
        std::vector<int32_t> & keys = st.key_of[ai];
        // the group's slot space is the budget's, not the policy's: it never shrinks under a re-layout
        keys.resize((size_t) llama_expert_group_cap(st, ai), -1);
        for (int32_t & k : keys) {
            if (k == -2) {
                k = -1; // a reservation whose copy the re-layout ended: the slot is free again
            }
        }
    }
}

// the per-layer operands: every pooled layer hands the plan the model's tensors to copy out of, and a
// full layer additionally gets a view at its own run of slots
void llama_expert_pool::pool_build_layer_views() {
    llama_expert_pool_state & st = this->st;
    const int32_t n_expert = model.hparams.n_expert;

    for (size_t ai = 0; ai < st.groups.size(); ++ai) {
        llama_expert_pool_state::group & a = st.groups[ai];
        llama_expert_pool_state::pool_dev & dev = *st.devs[a.id];
        int32_t n_isl = 0; // the full layers this group has placed so far
        for (int32_t il : a.layers) {
            // the identity placement is re-derived by this pass: the counter restarts from what it writes
            llama_expert_pool_layer & l = st.layers[il];
            l.base = -1;
            const std::vector<int32_t> & res = st.resident[il];
            const llama_layer & L = model.layers[il];
            // the model's own tensors of this layer; a kind the model does not carry stays null
            ggml_tensor * src[PK_N] = {
                L.ffn_gate_up_exps, L.ffn_up_exps,      L.ffn_gate_exps, L.ffn_down_exps,
                L.ffn_gate_up_exps_b, L.ffn_up_exps_b,  L.ffn_gate_exps_b, L.ffn_down_exps_b,
            };
            // every branch hands these to the plan, which copies out of them when it fills the layer
            for (int k = 0; k < PK_N; ++k) {
                l.orig[k] = src[k];
            }
            if (res.empty()) {
                continue; // nothing resident yet (the partial form before its first fill)
            }
            bool identity = (int32_t) res.size() == n_expert;
            for (int32_t s = 0; identity && s < n_expert; ++s) {
                identity = res[s] == s;
            }
            if (!identity) {
                continue; // partial layer: the operand is the group bank, the tables carry the slot numbers
            }
            // the i-th full layer of this group owns the i-th run of n_expert slots: the base is a
            // formula, not a search - the placement itself is claimed by the plan, layer by layer
            const int32_t base = n_isl++ * n_expert;
            l.base = base;
            // the full layer's operand: a view at its slots, in the model's own order
            for (size_t ki = 0; ki < a.kinds.size(); ++ki) {
                const int k = a.kinds[ki];
                ggml_tensor * bank = a.banks[ki];
                ggml_tensor * view = (k < PK_UP_B)
                    ? ggml_view_3d(dev.ctx.get(), bank, bank->ne[0], bank->ne[1], n_expert,
                                   bank->nb[1], bank->nb[2], (size_t) base * bank->nb[2])
                    : ggml_view_2d(dev.ctx.get(), bank, bank->ne[0], n_expert, bank->nb[1],
                                   (size_t) base * bank->nb[1]);
                ggml_format_name(view, "pool_full_%s_%d", ggml_type_name(bank->type), il);
                l.pool[k] = view;
            }
        }
    }
}

// the form of every pooled layer, from the placement: the plain path once its identity run is complete
void llama_expert_pool::pool_set_forms() {
    // the form says who is in the pool: the partial form keeps every pooled layer (its residency starts
    // empty and the plan fills it), the whole form only the layers whose identity run is stated - those
    // run the plain path, so "in the form" and "has a chain" differ there
    pool_refresh_forms(st);
}
void llama_expert_pool::expert_pool_fill() {
    llama_expert_pool_state & st = this->st;
    if (st.phase != llama_expert_pool_state::PHASE_BUILT || st.layers.empty()) {
        return;
    }
    if (model.no_alloc()) {
        return; // the memory-fit pass: no weights to read and no table to publish
    }
    const int32_t n_expert = model.hparams.n_expert;

    // the flat scale tables: expert e -> its scale (indexed by the clean ids, never changes on swap). the
    // I32 routing tables are the placement's and are written by slot_map_install below; the weight copies
    // are the plan's job
    auto fill_scale = [&](ggml_tensor * tab, ggml_tensor * src) {
        if (tab == nullptr || src == nullptr) {
            return;
        }
        GGML_ASSERT(src->type == GGML_TYPE_F32 && src->ne[0] == n_expert);
        ggml_backend_tensor_set(tab, src->data, 0, n_expert * sizeof(float));
    };
    if (st.direct_mount) {
        for (int32_t il : st.pooled_layers) {
            const llama_expert_pool_layer & l = st.layers[il];
            if (l.form == llama_expert_pool_layer::FORM_NONE || l.remap == nullptr || l.cpu_map == nullptr) {
                continue;
            }
            const llama_layer & L = model.layers[il];
            fill_scale(l.scale,      L.ffn_down_exps_s);
            fill_scale(l.scale_up,   L.ffn_up_exps_s);
            fill_scale(l.scale_gate, L.ffn_gate_exps_s);
        }
    }
    st.phase = llama_expert_pool_state::PHASE_FILLED;

    // the resident sets land in both tables before the first decode
    slot_map_install(st);

    if (st.direct_mount) {
        int32_t n_full = 0;
        for (int32_t il : st.pooled_layers) {
            if (st.layers[il].form == llama_expert_pool_layer::FORM_PLAIN) {
                n_full += 1;
            }
        }
        LLAMA_LOG_INFO("%s: pool active: %d layers fully resident in expert order (plain path), %d on the hit chain (dual chain)\n",
                __func__, n_full, (int32_t) st.pooled_layers.size() - n_full);
    }
}


// report and reset the pool's segment counters: the standard hit-rate line in the server log
static void pool_report(llama_expert_pool_state & st) {
    moe_cache_stats bundles = {}; // the bundle counters are the report's only source for fills and bytes
    for (const std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        bundles.fills += dp->stats.fills;
        bundles.bytes += dp->stats.bytes;
        dp->stats = {};
    }
    const moe_cache_stats small = { (size_t) st.seg.hit,       (size_t) st.seg.miss,       bundles.fills, bundles.bytes };
    const moe_cache_stats large = { (size_t) st.seg_large.hit, (size_t) st.seg_large.miss, 0,             0             };
    log_moe_cache_stats("llama_expert_pool", "ubatch <= 8", small);
    log_moe_cache_stats("llama_expert_pool", "ubatch  > 8", large);
    {
        // a layer point that could not read its ids leaves the counters stale: say so, once per report
        std::string row;
        bool stale = false;
        for (size_t i = 0; i < st.lp_calls.size(); ++i) {
            const int32_t c = st.lp_calls[i];
            const int32_t f = i < st.lp_folds.size() ? st.lp_folds[i] : 0;
            if (c != f) {
                stale = true;
                row += std::to_string(i) + ":" + std::to_string(c) + "/" + std::to_string(f) + " ";
            }
        }
        if (stale) {
            LLAMA_LOG_WARN("llama_expert_pool: layer point without ids (calls/counts) %s\n", row.c_str());
        }
        std::fill(st.lp_calls.begin(), st.lp_calls.end(), 0);
        std::fill(st.lp_folds.begin(), st.lp_folds.end(), 0);
    }
    st.seg       = {};
    st.seg_large = {};
}

void llama_expert_pool::expert_pool_finalize() {
    llama_expert_pool_state & st = this->st;
    if (st.seg.hit + st.seg.miss + st.seg_large.hit + st.seg_large.miss == 0) {
        return;
    }
    pool_report(st);
}

// re-lay the pool in the other layout: the stream is settled, the views are rebuilt, and the placement
// the new view sees is the plan's business from there on
void llama_expert_pool::expert_pool_relayout(bool whole) {
    llama_expert_pool_state & st = this->st;
    if (st.phase == llama_expert_pool_state::PHASE_NONE || st.pooled_layers.empty()) {
        return;
    }
    // LLAMA_EXPERT_POOL_CROSS_LOG: one line per exchange, with the phases it spends the time in
    const int64_t t0 = ggml_time_us();

    // the layer set and the budgets stay, only the form changes, so every pool is re-laid with its own budget
    std::vector<ggml_backend_dev_t>   sel_dev;
    std::vector<std::vector<int32_t>> sel_layers;
    const std::vector<int32_t>        sel_slots = st.dev_slots;
    for (const std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        sel_dev.push_back(ggml_backend_buft_get_device(dp->buft));
        sel_layers.push_back(dp->layers);
    }

    std::vector<std::vector<int32_t>> resident; // indexed by LAYER
    resident.resize(model.hparams.n_layer_all);
    // the whole form states its full layers; the partial form states nothing per layer: its residency starts
    // where the last form left it (empty for a pinned partial run) and the plan converges it from the counters
    if (whole) {
        pool_set_full_resident(resident, sel_layers, sel_slots, model.hparams.n_expert);
    }

    // the plan below issues the copies on this thread: settle the stream before the views are rebuilt
    ctx.synchronize();
    const int64_t t1 = ggml_time_us();
    // the target lists are the policy, the placement is the residency: a form change keeps every
    // resident where it sits (a slot number carries no meaning below the whole form) and the ordinary
    // per-layer plan converges from there - the registered layers claim their runs, the partial ones
    // exchange. nothing is wiped and nothing is moved to make room
    st.resident.assign(model.hparams.n_layer_all, {});
    for (int32_t il : st.pooled_layers) {
        st.resident[il] = std::move(resident[il]);
    }
    st.layout_whole = whole;
    expert_pool_build_views();
    const int64_t t2 = ggml_time_us();
    // the whole form claims its runs through the ordinary admission path (the plan, one layer point at
    // a time): nothing is placed, wiped or published here. the copies go out on this thread, in the
    // compute stream, and the next ubatch reads them
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        for (const int32_t il : d.layers) {
            pool_plan_layer(st, d, il);
        }
    }
    for (int32_t il : st.pooled_layers) {
        slot_map_mark(st, il);
    }
    slot_map_publish(st);
    const int64_t t3 = ggml_time_us();
    // the scheduler is not rebuilt here: the reserve taken at load covers both forms (measured on a
    // crossing-heavy server run; if a config ever allocates short the failure is an error, not corruption)
    int32_t tot_slots = 0;
    for (const int32_t v : sel_slots) {
        tot_slots += v;
    }
    if (getenv("LLAMA_EXPERT_POOL_CROSS_LOG") != nullptr) {
        LLAMA_LOG_INFO("%s: exchange: sync %.1f, views %.1f, plan %.1f, total %.1f ms\n",
                __func__, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0, (t3 - t2) / 1000.0, (t3 - t0) / 1000.0);
    }
    LLAMA_LOG_INFO("%s: pool re-laid out (%s layout, %d layers, %d device(s), %d slots)\n",
            __func__, whole ? "whole-layer" : "partial", (int) st.pooled_layers.size(),
            (int) sel_dev.size(), tot_slots);
}

// the layout follows the batch regime: whole-layer at or above the offload threshold, partial below
// the batch that lands on the other side re-lays the pool; the caller passes the tokens this batch
// hands to one graph, so the decision follows the scheduler's own threshold
void llama_expert_pool::expert_pool_cross(int32_t n_tokens) {
    llama_expert_pool_state & st = this->st;
    // the first down-cross waits for a prefill: the setup decodes arrive below the threshold
    if (llama_expert_pool_layout_pinned() != LLAMA_EXPERT_POOL_LAYOUT_AUTO || st.phase != llama_expert_pool_state::PHASE_FILLED ||
            st.pooled_layers.empty() || !st.prefill_seen) {
        return;
    }
    const bool want_whole = n_tokens >= llama_expert_pool_offload_min_batch();
    if (want_whole == st.layout_whole) {
        // the regime's layout is in place
        return;
    }
    // a crossing always lands: the budgets are fixed at init, so a relayout cannot refuse its target and there is nothing to retry
    LLAMA_LOG_INFO("%s: batch of %d tokens crosses the %d-token threshold: re-laying the pool\n",
            __func__, n_tokens, llama_expert_pool_offload_min_batch());
    expert_pool_relayout(want_whole);
}


// release the per-device bundles: weight context/buffer and tables. Buffers first, then the contexts that own the tensor metadata.
void llama_expert_pool::release_devs() {
    for (std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        llama_expert_pool_state::pool_dev & d = *dp;
        d.buf.reset();
        d.ctx.reset();
        d.slot_map_buf.reset();
        d.buf_host.reset();
        d.slot_map_dev = nullptr;
        // the copies run in the compute stream: the caller synchronizes before the buffers go away
        d.slot_map_ctx.reset();
        d.ctx_host.reset();
    }
    st.devs.clear();
    st.layer_dev.clear();
}

void llama_expert_pool::expert_pool_release() {
    llama_expert_pool_state & st = this->st;

    // an unclosed segment (a caller that never finalizes): report it here
    if (st.seg.hit + st.seg.miss + st.seg_large.hit + st.seg_large.miss > 0) {
        pool_report(st);
    }

    // the copies the plan issued are in the compute stream: settle it before the buffers are freed
    for (const std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        if (dp->backend != nullptr) {
            ggml_backend_synchronize(dp->backend);
        }
    }

    release_devs();
}

// the bytes a context's tensors would take in a buffer of this type: ggml's own size for that list
static size_t pool_ctx_bytes(ggml_backend_buffer_type_t buft, ggml_context * ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        tensors.push_back(t);
    }
    return ggml_backend_buft_get_alloc_size_n(buft, tensors.data(), (int) tensors.size());
}

std::map<ggml_backend_buffer_type_t, size_t> llama_expert_pool::memory_breakdown() const {
    const llama_expert_pool_state & st = this->st;
    std::map<ggml_backend_buffer_type_t, size_t> res;
    // the memory-fit pass allocates dummy buffers, so the real sizes are summed from the tensors there
    const bool count_tensors = model.no_alloc();
    for (const std::unique_ptr<llama_expert_pool_state::pool_dev> & dp : st.devs) {
        const llama_expert_pool_state::pool_dev & d = *dp;
        if (count_tensors) {
            const ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
            res[d.buft] += pool_ctx_bytes(d.buft, d.ctx.get());
            res[d.buft] += pool_ctx_bytes(d.buft, d.slot_map_ctx.get());
            res[buft_host] += pool_ctx_bytes(buft_host, d.ctx_host.get());
            continue;
        }
        const std::pair<ggml_backend_buffer_t, ggml_backend_buffer_type_t> bufs[3] = {
            { d.buf.get(),        d.buft                        },
            { d.slot_map_buf.get(),      d.buft                        },
            { d.buf_host.get(),  ggml_backend_cpu_buffer_type() },
        };
        for (const auto & [buf, buft] : bufs) {
            if (buf == nullptr || buft == nullptr) {
                continue;
            }
            res[buft] += ggml_backend_buffer_get_size(buf);
        }
    }
    return res;
}