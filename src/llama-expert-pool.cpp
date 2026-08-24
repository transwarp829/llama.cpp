#include "llama-expert-pool.h"

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
