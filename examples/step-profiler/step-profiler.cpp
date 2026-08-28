#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// llama-step-profiler
//
// measures per-step (per-token) decode timing using the scheduler eval callback
// wrtes:
//   <prefix>.timing.csv   raw view, one row per (step, layer): the 20-column
//                         execution-order layout, plus a per-step summary row
//                         (layer = -1) carrying wall/gap/cb and totals
//   <prefix>.summary.csv  aggregate view: run totals row (step = -1, layer = -1)
//                         and per-layer means across decode tokens
//   <prefix>.routing.csv  per step x per layer: activated expert ids
//   <prefix>.delegate.csv per step: delegate submits/hit_rows totals
//
// prefix: env STEP_PROFILE_OUT, default "step-profile"
//
// usage: llama-step-profiler -m model.gguf -p "prompt" -n 100 [-t N] [-ngl N] [--cpu-moe]
//
// note: the callback forces the per-node compute path, so timings are contaminated
// (CUDA graphs disabled). routing data is timing-independent and still valid.

enum step_cat {
    CAT_ATTN,
    CAT_EXPERT,
    CAT_ROUTER,
    CAT_SHARED,
    CAT_NORM,
    CAT_OTHER,
    CAT_COUNT,
};

// column order matches one decode step's execution order (timing.csv + summary.csv)
static const char * const csv_header =
    "step,layer,wall_ms,attn_ms,router_ms,prep_getset_us,prep_ids_us,prep_comp_us,"
    "end_sync_us,end_get_us,moe_ms,shared_ms,norm_ms,other_ms,submits,hit_rows,"
    "miss_rows,gap_ms,cb_ms\n";

struct step_record {
    double wall_ms = 0.0;       // total time of the llama_decode call
    double sum_nodes_ms = 0.0;  // sum of all node durations
    double cb_ms = 0.0;         // time spent inside the callback itself
    double cat_ms[CAT_COUNT] = {0.0};
    int64_t n_nodes = 0;
    int64_t n_mm_id = 0;
    // per-layer per-category node time: layer_cat[cat][layer]
    std::array<std::vector<double>, CAT_COUNT> layer_cat;
};

struct profiler_data {
    int64_t n_layers = 0;
    std::vector<step_record> steps;
    std::vector<std::vector<llama_expert_pool_layer_stats>> step_delegate; // per step, per pooled layer
    int64_t t_node_start_us = 0;
    std::ofstream * f_routing = nullptr;  // routing trace CSV (step,layer,expert_ids)
    int last_routing_step = -1;           // dedupe: last captured (step, layer)
    int last_routing_layer = -1;
};

// layer index from node name like "ffn_moe_up-3", -1 if none
static int node_layer(const char * name) {
    const char * p = strrchr(name, '-');
    if (p == nullptr || p[1] == '\0') {
        return -1;
    }
    for (const char * q = p + 1; *q != '\0'; ++q) {
        if (*q < '0' || *q > '9') {
            return -1;
        }
    }
    return atoi(p + 1);
}

// classify a node into a timing bucket
// - MUL_MAT_ID is always the routed expert matmul
// - attention ops are named "attn_*"
// - other MoE ops (router, top-k, weights) are named "ffn_moe_*"
// - shared/dense FFN ops are named "ffn_*" (not "ffn_moe_*")
static int classify(const ggml_tensor * t) {
    const char * name = t->name;
    if (t->op == GGML_OP_MUL_MAT_ID) {
        return CAT_EXPERT;
    }
    if (strstr(name, "attn") != nullptr) {
        return CAT_ATTN;
    }
    if (strncmp(name, "ffn_moe", 7) == 0) {
        return CAT_ROUTER;
    }
    if (strstr(name, "norm") != nullptr) {
        return CAT_NORM;
    }
    if (strncmp(name, "ffn_", 4) == 0) {
        return CAT_SHARED;
    }
    return CAT_OTHER;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (profiler_data *) user_data;
    const int64_t t0 = ggml_time_us();

    if (ask) {
        data->t_node_start_us = t0;
    } else {
        step_record & s = data->steps.back();
        const double dur_ms = (t0 - data->t_node_start_us) / 1000.0;

        s.sum_nodes_ms += dur_ms;
        s.n_nodes++;

        const int cat = classify(t);
        s.cat_ms[cat] += dur_ms;

        const int layer = node_layer(t->name);
        if (layer >= 0 && layer < (int) data->n_layers) {
            if (s.layer_cat[cat].size() < (size_t) data->n_layers) {
                s.layer_cat[cat].resize(data->n_layers, 0.0);
            }
            s.layer_cat[cat][layer] += dur_ms;
            if (cat == CAT_EXPERT) {
                s.n_mm_id++;
            }
        }

        // capture activated expert ids from MUL_MAT_ID inputs (once per step/layer)
        // ggml_mul_mat_id(ctx, as, b, ids) -> src[2] = selected expert indices
        if (t->op == GGML_OP_MUL_MAT_ID && data->f_routing != nullptr && t->src[2] != nullptr) {
            const int cur_step = (int) data->steps.size() - 1;
            if (layer >= 0 && layer < (int) data->n_layers &&
                (cur_step != data->last_routing_step || layer != data->last_routing_layer)) {
                const ggml_tensor * ids = t->src[2];
                const int64_t n = ids->ne[0] * ids->ne[1]; // [n_expert_used, n_tokens]
                std::vector<int32_t> buf(n > 0 ? n : 1, 0);
                ggml_backend_tensor_get(ids, buf.data(), 0, n * sizeof(int32_t));
                auto & fout = *data->f_routing;
                fout << cur_step << "," << layer;
                for (int64_t i = 0; i < n; ++i) {
                    fout << "," << buf[i];
                }
                fout << "\n";
                fout.flush();
                data->last_routing_step = cur_step;
                data->last_routing_layer = layer;
            }
        }
    }

    data->steps.back().cb_ms += (ggml_time_us() - t0) / 1000.0;
    return true;
}

// write one timing row. layer < 0 = step summary row (wall/gap/cb + totals),
// layer >= 0 = per-layer row. the delegate stats come from `dl` (nullptr for
// non-pooled layers).
static void write_timing_row(std::ofstream & fout, int step, int layer, const step_record & s,
                             const llama_expert_pool_layer_stats * dl) {
    const double gap_ms = std::max(0.0, s.wall_ms - s.sum_nodes_ms - s.cb_ms);
    fout << step << "," << layer << ",";
    if (layer < 0) {
        fout << s.wall_ms << ","
             << s.cat_ms[CAT_ATTN] << "," << s.cat_ms[CAT_ROUTER] << ","
             << (dl ? std::to_string(dl->prep_getset_us) : "") << ","
             << (dl ? std::to_string(dl->prep_ids_us) : "") << ","
             << (dl ? std::to_string(dl->prep_comp_us) : "") << ","
             << (dl ? std::to_string(dl->end_sync_us) : "") << ","
             << (dl ? std::to_string(dl->end_get_us) : "") << ","
             << s.cat_ms[CAT_EXPERT] << "," << s.cat_ms[CAT_SHARED] << ","
             << s.cat_ms[CAT_NORM] << "," << s.cat_ms[CAT_OTHER] << ","
             << (dl ? std::to_string(dl->submits) : "0") << ","
             << (dl ? std::to_string(dl->hit_rows) : "0") << ","
             << (dl ? std::to_string(dl->miss_rows) : "0") << ","
             << gap_ms << "," << s.cb_ms << "\n";
    } else {
        const auto & lc = s.layer_cat;
        const double attn   = layer < (int) lc[CAT_ATTN].size()   ? lc[CAT_ATTN][layer]   : 0.0;
        const double router = layer < (int) lc[CAT_ROUTER].size() ? lc[CAT_ROUTER][layer] : 0.0;
        const double moe    = layer < (int) lc[CAT_EXPERT].size() ? lc[CAT_EXPERT][layer] : 0.0;
        const double shared = layer < (int) lc[CAT_SHARED].size() ? lc[CAT_SHARED][layer] : 0.0;
        const double norm   = layer < (int) lc[CAT_NORM].size()   ? lc[CAT_NORM][layer]   : 0.0;
        const double other  = layer < (int) lc[CAT_OTHER].size()  ? lc[CAT_OTHER][layer]  : 0.0;
        fout << ","
             << attn << "," << router << ","
             << (dl ? std::to_string(dl->prep_getset_us) : "") << ","
             << (dl ? std::to_string(dl->prep_ids_us) : "") << ","
             << (dl ? std::to_string(dl->prep_comp_us) : "") << ","
             << (dl ? std::to_string(dl->end_sync_us) : "") << ","
             << (dl ? std::to_string(dl->end_get_us) : "") << ","
             << moe << "," << shared << "," << norm << "," << other << ","
             << (dl ? std::to_string(dl->submits) : "0") << ","
             << (dl ? std::to_string(dl->hit_rows) : "0") << ","
             << (dl ? std::to_string(dl->miss_rows) : "0")
             << ",,\n"; // gap/cb are step-level only
    }
    fout.flush();
}

// find the delegate stats for a model layer within this step's vector
static const llama_expert_pool_layer_stats * find_layer_stats(const std::vector<llama_expert_pool_layer_stats> & v, int layer) {
    for (const auto & d : v) {
        if (d.layer == layer) {
            return &d;
        }
    }
    return nullptr;
}

static void print_summary(const profiler_data & data) {
    // aggregate over decode steps (skip step 0 = prompt)
    if (data.steps.size() < 2) {
        return;
    }
    double wall = 0.0, gap = 0.0, attn = 0.0, expert = 0.0, cb = 0.0;
    const size_t n = data.steps.size() - 1;
    for (size_t i = 1; i < data.steps.size(); ++i) {
        const step_record & s = data.steps[i];
        wall   += s.wall_ms;
        gap    += std::max(0.0, s.wall_ms - s.sum_nodes_ms - s.cb_ms);
        attn   += s.cat_ms[CAT_ATTN];
        expert += s.cat_ms[CAT_EXPERT];
        cb     += s.cb_ms;
    }
    LOG_INF("profiler: decode steps: %zu\n", n);
    LOG_INF("profiler: mean wall   = %.3f ms/step (%.2f t/s)\n", wall / n, 1000.0 * n / wall);
    LOG_INF("profiler: mean attn   = %.3f ms/step\n", attn / n);
    LOG_INF("profiler: mean expert = %.3f ms/step\n", expert / n);
    LOG_INF("profiler: mean gap    = %.3f ms/step (sched/sync overhead)\n", gap / n);
    LOG_INF("profiler: mean cb     = %.3f ms/step (instrumentation overhead)\n", cb / n);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    profiler_data data;

    const char * no_cb = getenv("STEP_PROFILE_NO_CB");
    if (no_cb == nullptr) {
        params.cb_eval = cb_eval;
        params.cb_eval_user_data = &data;
        LOG_INF("%s: cb profiling enabled (set STEP_PROFILE_NO_CB=1 for wall-only timing)\n", __func__);
    } else {
        LOG_INF("%s: cb profiling disabled (wall-only)\n", __func__);
    }
    params.warmup = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    data.n_layers = llama_model_n_layer(model);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::string prompt = params.prompt;
    // -f/--file: read the prompt file (common's prompt_file field is parsed but
    // not consumed by any common function in this tree; read it here). only
    // used when -p/--prompt was not given on the command line.
    if (prompt.empty() && !params.prompt_file.empty()) {
        std::ifstream fin(params.prompt_file);
        if (!fin) {
            LOG_ERR("%s: failed to open prompt file %s\n", __func__, params.prompt_file.c_str());
            return 1;
        }
        prompt.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        LOG_INF("%s: prompt file loaded (%zu chars)\n", __func__, prompt.size());
    }
    // chat-template mode (like llama-cli -cnv): STEP_PROFILE_CHAT_FILE
    // = text file used as the user message; the model's own chat template
    // (from GGUF metadata) is applied before tokenizing.
    const char * chat_file = getenv("STEP_PROFILE_CHAT_FILE");
    if (chat_file != nullptr && chat_file[0] != '\0') {
        std::ifstream fin(chat_file);
        if (!fin) {
            LOG_ERR("%s: failed to open chat file %s\n", __func__, chat_file);
            return 1;
        }
        std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
        common_chat_templates_inputs inputs;
        inputs.messages = {{"user", content}};
        common_chat_params cp = common_chat_templates_apply(tmpls.get(), inputs);
        prompt = cp.prompt;
        LOG_INF("%s: chat template applied (%d chars), prompt %zu chars\n", __func__, (int) content.size(), prompt.size());
    }
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, false, true);

    if (tokens.empty()) {
        LOG_ERR("%s: there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return 1;
    }

    const int n_predict = params.n_predict >= 0 ? params.n_predict : 100;

    const char * prefix = getenv("STEP_PROFILE_OUT");
    if (prefix == nullptr || prefix[0] == '\0') {
        prefix = "step-profile";
    }

    std::ofstream f_timing(std::string(prefix) + ".timing.csv");
    if (!f_timing) {
        LOG_ERR("failed to open %s\n", (std::string(prefix) + ".timing.csv").c_str());
        return 1;
    }
    f_timing << csv_header;

    std::ofstream f_summary(std::string(prefix) + ".summary.csv");
    if (!f_summary) {
        LOG_ERR("failed to open %s\n", (std::string(prefix) + ".summary.csv").c_str());
        return 1;
    }
    f_summary << csv_header;

    std::ofstream f_deleg(std::string(prefix) + ".delegate.csv");
    if (!f_deleg) {
        LOG_ERR("failed to open %s\n", (std::string(prefix) + ".delegate.csv").c_str());
        return 1;
    }
    f_deleg << "step,submits,hit_rows,miss_rows\n";

    std::ofstream f_rout(std::string(prefix) + ".routing.csv");
    if (!f_rout) {
        LOG_ERR("failed to open %s\n", (std::string(prefix) + ".routing.csv").c_str());
        return 1;
    }
    f_rout << "step,layer,expert_ids\n";
    f_rout.flush();
    data.f_routing = &f_rout;

    auto * smpl = common_sampler_init(model, params.sampling);

    // prompt step
    {
        data.steps.emplace_back();
        data.steps.back().layer_cat[CAT_ATTN].resize(data.n_layers, 0.0);
        data.steps.back().layer_cat[CAT_EXPERT].resize(data.n_layers, 0.0);

        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            LOG_ERR("%s: failed to eval prompt\n", __func__);
            return 1;
        }
        data.steps.back().wall_ms = (ggml_time_us() - t0) / 1000.0;
        data.step_delegate.emplace_back();
        data.step_delegate.back().resize(data.n_layers);
        const uint32_t n_dl = llama_expert_pool_get_stats(ctx, data.step_delegate.back().data(), (uint32_t) data.n_layers);
        data.step_delegate.back().resize(n_dl);
        write_timing_row(f_timing, 0, -1, data.steps.back(), nullptr);
        for (int l = 0; l < (int) data.n_layers; ++l) {
            write_timing_row(f_timing, 0, l, data.steps.back(),
                    find_layer_stats(data.step_delegate.back(), l));
        }
        uint64_t ss = 0, sh = 0, sm = 0;
        for (const auto & d : data.step_delegate.back()) {
            ss += d.submits; sh += d.hit_rows; sm += d.miss_rows;
        }
        f_deleg << 0 << "," << ss << "," << sh << "," << sm << "\n";
    }

    // decode steps
    int step = 1;
    llama_token cur = tokens.back();
    for (int i = 0; i < n_predict; ++i) {
        data.steps.emplace_back();
        data.steps.back().layer_cat[CAT_ATTN].resize(data.n_layers, 0.0);
        data.steps.back().layer_cat[CAT_EXPERT].resize(data.n_layers, 0.0);

        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) {
            LOG_ERR("%s: failed to eval\n", __func__);
            break;
        }
        data.steps.back().wall_ms = (ggml_time_us() - t0) / 1000.0;

        // expert-pool delegate statistics for this step (snapshot-and-reset)
        data.step_delegate.emplace_back();
        data.step_delegate.back().resize(data.n_layers);
        const uint32_t n_dl = llama_expert_pool_get_stats(ctx, data.step_delegate.back().data(), (uint32_t) data.n_layers);
        data.step_delegate.back().resize(n_dl);

        write_timing_row(f_timing, step, -1, data.steps.back(), nullptr);
        for (int l = 0; l < (int) data.n_layers; ++l) {
            write_timing_row(f_timing, step, l, data.steps.back(),
                    find_layer_stats(data.step_delegate.back(), l));
        }
        uint64_t ss = 0, sh = 0, sm = 0;
        for (const auto & d : data.step_delegate.back()) {
            ss += d.submits; sh += d.hit_rows; sm += d.miss_rows;
        }
        f_deleg << step << "," << ss << "," << sh << "," << sm << "\n";

        cur = common_sampler_sample(smpl, ctx, -1);
        printf("%s", common_token_to_piece(ctx, cur).c_str());
        fflush(stdout);
        if (llama_vocab_is_eog(vocab, cur)) {
            break;
        }
        step++;
    }

    // end of the generation segment: print the accumulated pool hit rate
    llama_expert_pool_finalize(ctx);

    // aggregate view: run totals + per-layer means (decode steps only)
    if (data.steps.size() > 1) {
        const double nd = (double) (data.steps.size() - 1);
        step_record run;
        for (size_t i = 1; i < data.steps.size(); ++i) {
            const step_record & s = data.steps[i];
            run.wall_ms += s.wall_ms;
            run.sum_nodes_ms += s.sum_nodes_ms;
            run.cb_ms += s.cb_ms;
            for (int c = 0; c < CAT_COUNT; ++c) {
                run.cat_ms[c] += s.cat_ms[c];
            }
        }
        run.wall_ms /= nd; run.sum_nodes_ms /= nd; run.cb_ms /= nd;
        for (int c = 0; c < CAT_COUNT; ++c) {
            run.cat_ms[c] /= nd;
        }
        uint64_t r_submits = 0, r_hit = 0, r_miss = 0, r_gs = 0, r_ids = 0, r_comp = 0, r_sync = 0, r_get = 0;
        std::vector<uint64_t> l_submits(data.n_layers, 0), l_hit(data.n_layers, 0), l_miss(data.n_layers, 0),
                              l_gs(data.n_layers, 0), l_ids(data.n_layers, 0), l_comp(data.n_layers, 0),
                              l_sync(data.n_layers, 0), l_get(data.n_layers, 0);
        for (size_t i = 1; i < data.steps.size(); ++i) {
            for (const auto & d : data.step_delegate[i]) {
                r_submits += d.submits; r_hit += d.hit_rows; r_miss += d.miss_rows;
                r_gs += d.prep_getset_us; r_ids += d.prep_ids_us; r_comp += d.prep_comp_us;
                r_sync += d.end_sync_us; r_get += d.end_get_us;
                if (d.layer >= 0 && d.layer < (int) data.n_layers) {
                    l_submits[d.layer] += d.submits; l_hit[d.layer] += d.hit_rows; l_miss[d.layer] += d.miss_rows;
                    l_gs[d.layer] += d.prep_getset_us; l_ids[d.layer] += d.prep_ids_us; l_comp[d.layer] += d.prep_comp_us;
                    l_sync[d.layer] += d.end_sync_us; l_get[d.layer] += d.end_get_us;
                }
            }
        }
        {
            const double subs = r_submits / nd, hit = r_hit / nd, miss = r_miss / nd;
            const double gs = r_gs / nd, ids = r_ids / nd, comp = r_comp / nd, sync = r_sync / nd, get = r_get / nd;
            f_summary << "-1,-1," << run.wall_ms << "," << run.cat_ms[CAT_ATTN] << "," << run.cat_ms[CAT_ROUTER] << ","
                      << gs << "," << ids << "," << comp << "," << sync << "," << get << ","
                      << run.cat_ms[CAT_EXPERT] << "," << run.cat_ms[CAT_SHARED] << "," << run.cat_ms[CAT_NORM] << ","
                      << run.cat_ms[CAT_OTHER] << "," << subs << "," << hit << "," << miss << ","
                      << std::max(0.0, run.wall_ms - run.sum_nodes_ms - run.cb_ms) << "," << run.cb_ms << "\n";
        }
        for (int l = 0; l < (int) data.n_layers; ++l) {
            f_summary << "-1," << l << ","
                      << run.cat_ms[CAT_ATTN] << "," << run.cat_ms[CAT_ROUTER] << ","
                      << (l_gs[l] / nd) << "," << (l_ids[l] / nd) << "," << (l_comp[l] / nd) << ","
                      << (l_sync[l] / nd) << "," << (l_get[l] / nd) << ","
                      << run.cat_ms[CAT_EXPERT] << "," << run.cat_ms[CAT_SHARED] << "," << run.cat_ms[CAT_NORM] << ","
                      << run.cat_ms[CAT_OTHER] << ","
                      << (l_submits[l] / nd) << "," << (l_hit[l] / nd) << "," << (l_miss[l] / nd) << ",,\n";
        }
        f_summary.flush();
    }

    f_timing.close();
    f_summary.close();
    f_deleg.close();
    f_rout.close();
    common_sampler_free(smpl);

    print_summary(data);

    llama_backend_free();

    return 0;
}
