#include "llama-expert-pool.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>

void llama_expert_pool::init(int32_t n_layer, int32_t n_expert, int32_t n_slot, int64_t expert_size) {
    enabled   = true;
    this->n_layer   = n_layer;
    this->n_expert  = n_expert;
    this->n_slot    = n_slot;
    this->expert_size = expert_size;

    expert_to_slot.assign(n_layer, std::vector<int32_t>(n_expert, -1));
    slot_expert.assign(n_layer, std::vector<int32_t>(n_slot, -1));
    n_resident.assign(n_layer, 0);
}

int32_t llama_expert_pool::slot_of(int32_t il, int32_t expert) const {
    if (!enabled || il < 0 || il >= n_layer || expert < 0 || expert >= n_expert) {
        return -1;
    }
    return expert_to_slot[il][expert];
}

int32_t llama_expert_pool::expert_in_slot(int32_t il, int32_t slot) const {
    if (!enabled || il < 0 || il >= n_layer || slot < 0 || slot >= n_slot) {
        return -1;
    }
    return slot_expert[il][slot];
}

bool llama_expert_pool::is_resident(int32_t il, int32_t expert) const {
    return slot_of(il, expert) >= 0;
}

llama_expert_pool::swap_plan llama_expert_pool::plan_swap(const std::vector<std::vector<int32_t>> & topk) const {
    swap_plan plan;

    for (int32_t il = 0; il < n_layer; ++il) {
        if (il >= (int32_t) topk.size()) {
            continue;
        }
        const std::vector<int32_t> & want = topk[il];

        // evict residents that are no longer wanted
        for (int32_t s = 0; s < n_slot; ++s) {
            const int32_t e = slot_expert[il][s];
            if (e < 0) {
                continue;
            }
            const bool still_wanted = std::find(want.begin(), want.end(), e) != want.end();
            if (!still_wanted) {
                plan.evict_il.push_back(il);
                plan.evict_ex.push_back(e);
            }
        }

        // fill missing experts, reusing slots freed by eviction first.
        // a slot is free when empty or when its resident is being evicted
        // (stable reuse keeps the same slots across successive repins).
        std::vector<int32_t> free_slots;
        for (int32_t s = 0; s < n_slot; ++s) {
            const int32_t e = slot_expert[il][s];
            const bool to_evict = e >= 0 && std::find(want.begin(), want.end(), e) == want.end();
            if (e < 0 || to_evict) {
                free_slots.push_back(s);
            }
        }
        for (int32_t e : want) {
            if (expert_to_slot[il][e] >= 0) {
                continue; // already resident (check the entry, not the slot values)
            }
            if (free_slots.empty()) {
                break; // no room (should not happen: |want| <= n_slot)
            }
            const int32_t s = free_slots.back();
            free_slots.pop_back();
            plan.fill_il.push_back(il);
            plan.fill_ex.push_back(e);
        }
    }

    // swap cost = H2D copies only (evictions are LUT updates, the CPU keeps
    // the original weights), so the delta is the fill set size.
    plan.delta = (int32_t) plan.fill_il.size();
    return plan;
}

bool llama_expert_pool::gate_should_swap(float h_inc, float c_cand, int32_t delta,
                                         float cpu_ms, float swap_ms, float L) {
    if (delta <= 0) {
        return false;
    }
    const float benefit = (c_cand - h_inc) * cpu_ms;
    const float cost    = delta * swap_ms / L;
    return benefit > cost;
}

void llama_expert_pool::begin_swap(const swap_plan & plan) {
    // evict: unmap every expert scheduled for eviction
    for (size_t i = 0; i < plan.evict_il.size(); ++i) {
        const int32_t il = plan.evict_il[i];
        const int32_t e  = plan.evict_ex[i];
        const int32_t s  = expert_to_slot[il][e];
        if (s >= 0) {
            expert_to_slot[il][e] = -1;
            slot_expert[il][s]    = -1;
            if (n_resident[il] > 0) {
                --n_resident[il];
            }
        }
    }
}

void llama_expert_pool::commit_slot(int32_t il, int32_t slot, int32_t expert) {
    if (!enabled || il < 0 || il >= n_layer || slot < 0 || slot >= n_slot ||
        expert < 0 || expert >= n_expert) {
        return;
    }
    // slot must be empty; if it still holds another expert, unmap that one first
    const int32_t old = slot_expert[il][slot];
    if (old >= 0) {
        expert_to_slot[il][old] = -1;
    } else {
        ++n_resident[il];
    }
    slot_expert[il][slot] = expert;
    expert_to_slot[il][expert] = slot;
}

int32_t llama_expert_pool::total_resident() const {
    int32_t total = 0;
    for (int32_t r : n_resident) {
        total += r;
    }
    return total;
}

#include <random>
#include <fstream>
#include <sstream>

void llama_expert_pool_state::reset() {
    enabled = false;
    n_slot = 0;
    w_pool_gate_up.clear();
    w_pool_up.clear();
    w_pool_gate.clear();
    w_pool_down.clear();
    remap_tab.clear();
    mask_tab.clear();
    cold_mask_tab.clear();
    resident.clear();
    pooled_layers.clear();
    slots.clear();
    delegate_ok = false;
    gpu_backend = nullptr;
    dg_buf = nullptr;
    t_cur = nullptr;
    t_out = nullptr;
    t_ids = nullptr;
    n_used = 0;
    n_hit = 0;
    hit_slots.clear();
    hit_cols.clear();
}

// parse seed csv: one line per layer "il,e1,e2,...". layers missing from the
// file (or with fewer than n_slot entries) are filled randomly; extra entries
// are truncated. returns false on parse error (bad line / bad il).
bool llama_expert_pool_parse_init(const std::string & path, int32_t n_layer,
                                  int32_t n_expert, int32_t n_slot,
                                  std::vector<std::vector<int32_t>> & resident) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    resident.assign(n_layer, std::vector<int32_t>(n_slot, -1));
    std::mt19937 rng(0);
    for (int32_t il = 0; il < n_layer; ++il) {
        for (int32_t s = 0; s < n_slot; ++s) {
            resident[il][s] = (int32_t) (rng() % n_expert);
        }
    }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        // skip comment/empty lines
        if (tok.empty() || tok[0] == '#') {
            continue;
        }
        const int32_t il = std::atoi(tok.c_str());
        if (il < 0 || il >= n_layer) {
            continue;
        }
        std::vector<int32_t> experts;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) {
                continue;
            }
            int32_t e = std::atoi(tok.c_str());
            if (e < 0 || e >= n_expert) {
                continue;
            }
            experts.push_back(e);
        }
        if (experts.empty()) {
            continue;
        }
        // use up to n_slot entries, dedup
        std::fill(resident[il].begin(), resident[il].end(), -1);
        std::vector<int32_t> dedup;
        for (int32_t e : experts) {
            if (std::find(dedup.begin(), dedup.end(), e) == dedup.end()) {
                dedup.push_back(e);
            }
            if ((int32_t) dedup.size() == n_slot) {
                break;
            }
        }
        const int32_t m = (int32_t) dedup.size();
        for (int32_t s = 0; s < m; ++s) {
            resident[il][s] = dedup[s];
        }
        while ((int32_t) dedup.size() < n_slot) {
            const int32_t e = (int32_t) (rng() % n_expert);
            if (std::find(dedup.begin(), dedup.end(), e) == dedup.end()) {
                dedup.push_back(e);
            }
        }
        const int32_t k = (int32_t) dedup.size();
        for (int32_t s = 0; s < k; ++s) {
            resident[il][s] = dedup[s];
        }
    }
    return true;
}

// random resident set per layer (fixed seed for reproducibility)
void llama_expert_pool_random(int32_t n_layer, int32_t n_expert, int32_t n_slot,
                              std::vector<std::vector<int32_t>> & resident) {
    resident.assign(n_layer, std::vector<int32_t>(n_slot, -1));
    std::mt19937 rng(0);
    for (int32_t il = 0; il < n_layer; ++il) {
        for (int32_t s = 0; s < n_slot; ++s) {
            resident[il][s] = (int32_t) (rng() % n_expert);
        }
    }
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// moe delegate: take over cache-hit expert rows inside the CPU MUL_MAT_ID kernel
//
// begin(): called by the CPU kernel (ith==0) before row grouping. Scans the ids,
// collects hit (slot >= 0) rows, submits ONE batched matvec on the pool weights
// to the GPU (async), returns true so the kernel skips the mask==0 rows.
// end(): called after the local row loop (ith==0); waits for the GPU, copies the
// hit results back into the CPU dst rows.
// -----------------------------------------------------------------------------

static inline ggml_tensor * pool_w_for(const llama_expert_pool_state & st, int32_t il, int which) {
    switch (which) {
        case 0: return st.w_pool_gate_up[il];
        case 1: return st.w_pool_up[il];
        case 2: return st.w_pool_gate[il];
        default: return st.w_pool_down[il];
    }
}

void llama_expert_pool_delegate_begin(
        ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids, ggml_tensor * dst,
        const int32_t ** skip_out, void * ud) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) ud;
    *skip_out = nullptr;
    const int64_t dbg_t0 = ggml_time_us();
    if (!st.delegate_ok || st.pooled_layers.empty()) {
        return;
    }

    // find the pooled layer/matrix this node corresponds to (ilx = index into
    // pooled_layers; il = actual layer number)
    int32_t il = -1;
    int32_t ilx = -1;
    int which = -1;
    for (size_t ix = 0; ix < st.pooled_layers.size(); ++ix) {
        const int32_t i = st.pooled_layers[ix];
        if      (st.orig_gate_up[i] == src0) { il = i; ilx = (int32_t) ix; which = 0; break; }
        else if (st.orig_up[i]       == src0) { il = i; ilx = (int32_t) ix; which = 1; break; }
        else if (st.orig_gate[i]     == src0) { il = i; ilx = (int32_t) ix; which = 2; break; }
        else if (st.orig_down[i]     == src0) { il = i; ilx = (int32_t) ix; which = 3; break; }
    }
    if (il < 0) {
        return;
    }
    if (which == 3) {
        // down uses the per-column expert input layout; not delegated in V2
        return;
    }
    // remember (ilx, which) for end() to map a dst back to its output region
    st.dst_ilx_which[dst] = std::make_pair(ilx, which);

    ggml_tensor * pw = pool_w_for(st, il, which);
    if (pw == nullptr) {
        return;
    }

    // scan ids -> hit slots
    st.n_hit = 0;
    if (st.slots[il].empty()) {
        return;
    }
    if ((int) st.hit_slots.size() < (int) ids->ne[0]) {
        st.hit_slots.resize(ids->ne[0]);
        st.hit_cols.resize(ids->ne[0]);
    }
    for (int id = 0; id < (int) ids->ne[0]; ++id) {
        const int32_t e = *((const int32_t *) ((const char *) ids->data + id*ids->nb[0]));
        if (e < 0 || e >= (int32_t) st.slots[il].size()) {
            return;
        }
        const int32_t s = st.slots[il][e];
        if (s >= 0) {
            st.hit_slots[st.n_hit] = s;
            st.hit_cols[st.n_hit]  = id;
            st.n_hit += 1;
        }
    }
    if (st.n_hit == 0) {
        // no hit: the CPU kernel computes all rows. record the miss count
        // exactly once per layer (both gate/up nodes call begin; the second
        // begin of the layer must not double-count).
        if (st.mini_submitted_il != il) {
            if ((size_t) ilx < st.s_layer.size()) {
                st.s_layer[ilx].miss_rows += (uint64_t) ids->ne[0];
            }
            st.mini_submitted_il = il;
        }
        return; // nothing to delegate
    }
    if (st.host_buf == nullptr) {
        return; // pinned mirrors unavailable; fall back to full CPU compute
    }

    // cur -> GPU scratch (generic: src1 may be 2D/3D, single token on decode)
    const int64_t dbg_t1 = ggml_time_us();
    ggml_backend_tensor_get(src1, st.host_cur, 0, ggml_nbytes(src1));
    ggml_backend_tensor_set(st.t_cur, st.host_cur, 0, ggml_nbytes(st.t_cur));
    const int64_t dbg_t2 = ggml_time_us();
    // hit slots -> ids scratch (padding zeros; extra columns are ignored via n_hit)
    memset(st.host_ids, 0, (size_t) st.n_used * sizeof(int32_t));
    memcpy(st.host_ids, st.hit_slots.data(), (size_t) st.n_hit * sizeof(int32_t));
    ggml_backend_tensor_set(st.t_ids, st.host_ids, 0, (size_t) st.n_used * sizeof(int32_t));
    const int64_t dbg_t3 = ggml_time_us();

    const llama_expert_pool_state::mini_graph_entry & m = st.mini[ilx][0];
    if (m.g == nullptr) {
        st.n_hit = 0;
        return;
    }
    // a new layer starts a new submit round: reset the combined-graph flag
    // BEFORE deciding whether this layer has to submit (need_submit below
    // reads it; using the previous layer's leftover value would skip the
    // submission and write stale GPU data back to this layer's dst).
    if (st.mini_submitted_il != il) {
        st.mini_submitted = false;
    }

    // combined up+gate graph: submit it once per layer (first begin of the layer
    // submits, the second begin only scans and returns the skip table). fused
    // gate/up (single output) or single-matrix models submit on every begin.
    const bool combined = m.out_up != nullptr && m.out_gate != nullptr;
    bool need_submit = !combined || !st.mini_submitted;
    if (need_submit) {
        const int64_t dbg_t1 = ggml_time_us();
        ggml_backend_tensor_get(src1, st.host_cur, 0, ggml_nbytes(src1));
        ggml_backend_tensor_set(st.t_cur, st.host_cur, 0, ggml_nbytes(st.t_cur));
        const int64_t dbg_t2 = ggml_time_us();
        // hit slots -> ids scratch (padding zeros; extra columns are ignored via n_hit)
        memset(st.host_ids, 0, (size_t) st.n_used * sizeof(int32_t));
        memcpy(st.host_ids, st.hit_slots.data(), (size_t) st.n_hit * sizeof(int32_t));
        ggml_backend_tensor_set(st.t_ids, st.host_ids, 0, (size_t) st.n_used * sizeof(int32_t));
        const int64_t dbg_t3 = ggml_time_us();

        const ggml_status gstatus = ggml_backend_graph_compute_async(st.gpu_backend, m.g);
        if (gstatus != GGML_STATUS_SUCCESS) {
            st.n_hit = 0;
            return;
        }
        st.mini_submitted = true;
        st.mini_submitted_il = il;

        // accumulate per-layer statistics (delegate runs on the ith==0 thread only)
        if ((size_t) ilx < st.s_layer.size()) {
            st.s_layer[ilx].submits   += 1;
            st.s_layer[ilx].miss_rows += (uint64_t) (ids->ne[0] - st.n_hit);
            st.s_layer[ilx].getset_us += (uint64_t) (dbg_t2 - dbg_t1);
            st.s_layer[ilx].ids_us    += (uint64_t) (dbg_t3 - dbg_t2);
            st.s_layer[ilx].comp_us   += (uint64_t) (ggml_time_us() - dbg_t3);
        }
    }

    // give the CPU kernel the slot table: experts with slot >= 0 are cache hits
    *skip_out = st.slots[il].data();
}

void llama_expert_pool_delegate_end(ggml_tensor * dst, void * ud) {
    llama_expert_pool_state & st = *(llama_expert_pool_state *) ud;
    if (st.n_hit == 0 || !st.delegate_ok) {
        return;
    }
    // map this dst back to (ilx, which); unknown dst -> nothing to write back
    auto it = st.dst_ilx_which.find(dst);
    if (it == st.dst_ilx_which.end()) {
        st.n_hit = 0;
        return;
    }
    const int32_t ilx = it->second.first;
    const int32_t which = it->second.second;
    const llama_expert_pool_state::mini_graph_entry & m = st.mini[ilx][0];
    const ggml_tensor * out_t = (which == 2) ? m.out_gate : m.out_up;
    if (out_t == nullptr) {
        st.n_hit = 0;
        return;
    }

    const int64_t dbg_t0 = ggml_time_us();
    const size_t row_bytes = st.sz_out_row;
    ggml_backend_synchronize(st.gpu_backend);
    const int64_t dbg_t1 = ggml_time_us();
    ggml_backend_tensor_get(out_t, st.host_out, 0, (size_t) st.n_hit * row_bytes);
    for (int i = 0; i < st.n_hit; ++i) {
        float * dst_col = (float *) ((char *) dst->data + st.hit_cols[i]*dst->nb[1]);
        memcpy(dst_col, (float *) st.host_out + (size_t) i * (row_bytes / sizeof(float)), row_bytes);
    }

    // accumulate per-layer statistics (delegate runs on the ith==0 thread only)
    if ((size_t) ilx < st.s_layer.size()) {
        st.s_layer[ilx].hit_rows += (uint64_t) st.n_hit;
        st.s_layer[ilx].sync_us  += (uint64_t) (dbg_t1 - dbg_t0);
        st.s_layer[ilx].get_us   += (uint64_t) (ggml_time_us() - dbg_t1);
    }
    st.n_hit = 0;
}
