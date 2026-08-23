#include "llama-expert-cache.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <numeric>

void llama_expert_window::init(int64_t n_expert, int32_t n_used, int32_t cap) {
    this->n_expert = n_expert;
    this->n_used   = n_used;
    this->cap      = cap;

    counts.assign(n_expert, 0);
    ring.assign(cap > 0 ? cap : 1, 0);
    ring_head = 0;
    ring_len  = 0;
    total_tokens = 0;
}

void llama_expert_window::update(const int32_t * ids, int64_t n_tokens) {
    if (cap <= 0 || ids == nullptr) {
        return;
    }
    total_tokens += n_tokens;
    for (int64_t t = 0; t < n_tokens; ++t) {
        const int32_t * row = ids + t * n_used;
        for (int32_t j = 0; j < n_used; ++j) {
            const int32_t e = row[j];
            if (e < 0 || e >= n_expert) {
                // routing id out of range - skip entry (defensive)
                continue;
            }
            counts[e]++;
            ring[(ring_head + ring_len) % cap] = e;
            if (ring_len >= cap) {
                const int32_t old = ring[ring_head];
                counts[old]--;
                ring_head = (ring_head + 1) % cap;
            } else {
                ring_len++;
            }
        }
    }
}

std::vector<int32_t> llama_expert_window::top_k(int32_t k) const {
    std::vector<int32_t> idx(n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
            [&](int32_t a, int32_t b) { return counts[a] > counts[b]; });
    if ((int32_t) idx.size() > k) {
        idx.resize(k);
    }
    return idx;
}

void llama_context_expert_cache::init(int32_t n_seq_max, int32_t n_layer) {
    this->n_seq_max = n_seq_max;
    this->n_layer   = n_layer;
    seqs.assign(n_seq_max, std::vector<llama_expert_window>(n_layer));
}

void llama_context_expert_cache::record(llama_seq_id seq, int32_t il, const int32_t * ids, int64_t n_tokens,
                                        int32_t n_used, int64_t n_expert) {
    if (!enabled || seq < 0 || seq >= n_seq_max || il < 0 || il >= n_layer) {
        return;
    }
    llama_expert_window & w = seqs[seq][il];
    if (w.cap == 0) {
        // first record for this layer - use observed layout (model agnostic)
        w.init(n_expert, n_used, W * n_used);
    }
    w.update(ids, n_tokens);
}

void llama_context_expert_cache::flush_to_aggregator(llama_model_expert_cache & agg) {
    if (!enabled || !agg.enabled) {
        return;
    }
    agg.aggregate(*this);
    dump_step(19);  // stage-1 verification dump at flush cadence (top-19)
    tokens_since_flush = 0;
}

void llama_context_expert_cache::dump_open() {
    const char * path = getenv("LLAMA_EXPERT_CACHE_DUMP");
    if (path == nullptr || !enabled) {
        return;
    }
    dump_f = fopen(path, "w");
    if (dump_f != nullptr) {
        fprintf(dump_f, "seq,il,expert_ids,counts\n");
        fflush(dump_f);
    }
}

void llama_context_expert_cache::dump_close() {
    if (dump_f != nullptr) {
        fclose(dump_f);
        dump_f = nullptr;
    }
}

void llama_context_expert_cache::dump_step(int32_t k) {
    if (dump_f == nullptr) {
        return;
    }
    for (int32_t s = 0; s < n_seq_max; ++s) {
        for (int32_t il = 0; il < n_layer; ++il) {
            const llama_expert_window & w = seqs[s][il];
            if (w.cap == 0) {
                continue;
            }
            const std::vector<int32_t> top = w.top_k(k);
            fprintf(dump_f, "%d,%d,%" PRId64, s, il, w.total_tokens);
            for (const int32_t e : top) {
                fprintf(dump_f, ",%d,%d", e, w.counts[e]);
            }
            fprintf(dump_f, "\n");
        }
    }
    fflush(dump_f);
}

void llama_model_expert_cache::ensure(int32_t n_layer, int64_t n_expert) {
    if ((int32_t) per_layer_counts.size() != n_layer) {
        per_layer_counts.assign(n_layer, std::vector<int32_t>(n_expert, 0));
    }
}

void llama_model_expert_cache::aggregate(const llama_context_expert_cache & ctx_cache) {
    ensure(ctx_cache.n_layer, 256);
    for (int32_t s = 0; s < ctx_cache.n_seq_max; ++s) {
        for (int32_t il = 0; il < ctx_cache.n_layer; ++il) {
            const llama_expert_window & w = ctx_cache.seqs[s][il];
            if (w.cap == 0) {
                continue;
            }
            std::vector<int32_t> & dst = per_layer_counts[il];
            if ((int32_t) dst.size() != w.n_expert) {
                dst.assign(w.n_expert, 0);
            }
            for (int64_t e = 0; e < w.n_expert; ++e) {
                dst[e] += w.counts[e];
            }
        }
    }
}
