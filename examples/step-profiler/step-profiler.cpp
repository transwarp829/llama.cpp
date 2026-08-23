#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
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
// writes results to CSV files:
//   <prefix>.summary.csv          - per step: wall, attention, expert, router, gap, ...
//   <prefix>.expert-per-layer.csv - per step per layer: routed expert matmul time
//   <prefix>.attn-per-layer.csv   - per step per layer: attention time
//   <prefix>.routing.csv          - per step per layer: activated expert ids (for cache sizing)
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

static const char * const cat_names[CAT_COUNT] = {
    "attn_ms",
    "expert_ms",
    "router_ms",
    "shared_ms",
    "norm_ms",
    "other_ms",
};

struct step_record {
    double wall_ms = 0.0;       // total time of the llama_decode call
    double sum_nodes_ms = 0.0;  // sum of all node durations
    double cb_ms = 0.0;         // time spent inside the callback itself
    double cat_ms[CAT_COUNT] = {0.0};
    int64_t n_nodes = 0;
    int64_t n_mm_id = 0;
    std::vector<double> layer_expert_ms;
    std::vector<double> layer_attn_ms;
};

struct profiler_data {
    int64_t n_layers = 0;
    std::vector<step_record> steps;
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
        if (cat == CAT_EXPERT) {
            s.n_mm_id++;
            if (layer >= 0 && layer < (int) data->n_layers) {
                s.layer_expert_ms[layer] += dur_ms;
            }
        } else if (cat == CAT_ATTN) {
            if (layer >= 0 && layer < (int) data->n_layers) {
                s.layer_attn_ms[layer] += dur_ms;
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

static void write_row(std::ofstream & fout, int step, const char * kind, const step_record & s) {
    const double gap_ms = std::max(0.0, s.wall_ms - s.sum_nodes_ms - s.cb_ms);
    fout << step << "," << kind << "," << s.wall_ms << "," << s.sum_nodes_ms << "," << gap_ms << "," << s.cb_ms;
    for (int c = 0; c < CAT_COUNT; ++c) {
        fout << "," << s.cat_ms[c];
    }
    fout << "," << s.n_nodes << "," << s.n_mm_id << "\n";
    fout.flush();
}

static void write_layer_matrix(const std::string & fname, const profiler_data & data, const std::vector<double> step_record::* member) {
    std::ofstream fout(fname);
    if (!fout) {
        LOG_ERR("failed to open %s\n", fname.c_str());
        return;
    }
    fout << "step";
    for (int l = 0; l < (int) data.n_layers; ++l) {
        fout << ",layer_" << l;
    }
    fout << "\n";
    for (size_t i = 0; i < data.steps.size(); ++i) {
        const auto & v = data.steps[i].*member;
        fout << i;
        for (int l = 0; l < (int) data.n_layers; ++l) {
            fout << "," << v[l];
        }
        fout << "\n";
    }
    fout.close();
    LOG_INF("profiler: wrote %s\n", fname.c_str());
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

    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &data;
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
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s: there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return 1;
    }

    const int n_predict = params.n_predict >= 0 ? params.n_predict : 100;

    const char * prefix = getenv("STEP_PROFILE_OUT");
    if (prefix == nullptr || prefix[0] == '\0') {
        prefix = "step-profile";
    }

    const std::string f_summary = std::string(prefix) + ".summary.csv";
    const std::string f_expert  = std::string(prefix) + ".expert-per-layer.csv";
    const std::string f_attn    = std::string(prefix) + ".attn-per-layer.csv";
    const std::string f_routing = std::string(prefix) + ".routing.csv";

    std::ofstream fout(f_summary);
    if (!fout) {
        LOG_ERR("failed to open %s\n", f_summary.c_str());
        return 1;
    }

    fout << "step,kind,wall_ms,sum_nodes_ms,gap_ms,cb_ms";
    for (int c = 0; c < CAT_COUNT; ++c) {
        fout << "," << cat_names[c];
    }
    fout << ",n_nodes,n_mm_id\n";

    std::ofstream f_rout(f_routing);
    if (!f_rout) {
        LOG_ERR("failed to open %s\n", f_routing.c_str());
        return 1;
    }
    f_rout << "step,layer,expert_ids\n";
    f_rout.flush();
    data.f_routing = &f_rout;

    auto * smpl = common_sampler_init(model, params.sampling);

    // prompt step
    {
        data.steps.emplace_back();
        data.steps.back().layer_expert_ms.resize(data.n_layers);
        data.steps.back().layer_attn_ms.resize(data.n_layers);

        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            LOG_ERR("%s: failed to eval prompt\n", __func__);
            return 1;
        }
        data.steps.back().wall_ms = (ggml_time_us() - t0) / 1000.0;
        write_row(fout, 0, "prompt", data.steps.back());
    }

    // decode steps
    int step = 1;
    llama_token cur = tokens.back();
    for (int i = 0; i < n_predict; ++i) {
        data.steps.emplace_back();
        data.steps.back().layer_expert_ms.resize(data.n_layers);
        data.steps.back().layer_attn_ms.resize(data.n_layers);

        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) {
            LOG_ERR("%s: failed to eval\n", __func__);
            break;
        }
        data.steps.back().wall_ms = (ggml_time_us() - t0) / 1000.0;
        write_row(fout, step, "decode", data.steps.back());
        step++;

        cur = common_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, cur)) {
            break;
        }
    }

    fout.close();
    common_sampler_free(smpl);

    write_layer_matrix(f_expert, data, &step_record::layer_expert_ms);
    write_layer_matrix(f_attn, data, &step_record::layer_attn_ms);

    print_summary(data);

    llama_backend_free();

    return 0;
}
