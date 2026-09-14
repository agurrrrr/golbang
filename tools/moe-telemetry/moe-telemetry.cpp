// moe-telemetry: per-layer MoE expert routing frequency collector for DSV4.1.
//
// Issue #227 (DSV41-OPT-C). It attaches a `cb_eval` callback to the llama.cpp
// backend scheduler and reads the `ffn_moe_topk-<layer>` tensor (the selected
// expert ids, [n_expert_used, n_tokens], I32) after every graph node is
// computed. That gives the exact expert routing the model used, so we can
// measure how skewed the per-layer expert distribution is and how many experts
// a GPU-resident "hot set" would have to hold to cover a given share of the
// routing.
//
// Build: see build.sh (links the llama.cpp-ds41 tree, HIP or CUDA build).
// Run:   GOLBANG_MOE_TELEMETRY_OUT=/path/out moe-telemetry -m model.gguf ...
//
// Notes:
// - We do not modify the runtime. cb_eval is a stock llama.h context parameter
//   (see examples/eval-callback). Expert-level placement is *not* possible
//   through -ot/--override-tensor; this tool measures the data that any such
//   placement layer would need.
// - Prefill and decode selections are counted separately so the decode hot set
//   can be compared with the prefill hot set.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

struct telemetry {
    int n_layer = 0;
    int n_expert = 0;

    // counts[phase][layer][expert]; phase 0 = prefill, 1 = decode
    std::vector<std::vector<std::vector<uint64_t>>> counts;

    // currently active phase, set around each llama_decode
    int phase = 0;

    uint64_t n_calls = 0;
    uint64_t n_selected = 0;

    std::mutex mtx;
};

static telemetry g_tel;

static const char * k_topk_prefix = "ffn_moe_topk";
static const size_t k_topk_prefix_len = 12;

// ask=true: scheduler wants to know whether we want this tensor's data.
// ask=false: the data is ready; read it.
static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * tel = (telemetry *) user_data;

    if (ask) {
        // only the selected-expert tensor; skip everything else
        return std::strncmp(t->name, k_topk_prefix, k_topk_prefix_len) == 0;
    }

    if (t->type != GGML_TYPE_I32) {
        return true;
    }

    // name is "ffn_moe_topk-<layer>"
    int il = -1;
    const char * dash = std::strrchr(t->name, '-');
    if (dash != nullptr) {
        il = std::atoi(dash + 1);
    }
    if (il < 0 || il >= tel->n_layer) {
        return true;
    }

    const int64_t n_used = t->ne[0]; // n_expert_used (top-k)
    const int64_t n_tok  = t->ne[1]; // tokens in this ubatch
    if (n_used <= 0 || n_tok <= 0) {
        return true;
    }

    // `selected_experts` is a strided view of the full argsort tensor
    // ([n_expert, n_tokens]): nb[1] is the row stride, not n_used*4. Read a
    // linear span and index by the real row stride, otherwise a multi-token
    // prefill silently reads each row's full 384-entry argsort instead of the
    // top-k prefix.
    const size_t row_stride = t->nb[1] > 0 ? t->nb[1] : (size_t) n_used * sizeof(int32_t);
    const size_t span = (size_t) (n_tok - 1) * row_stride + (size_t) n_used * sizeof(int32_t);
    std::vector<uint8_t> raw(span);
    ggml_backend_tensor_get(t, raw.data(), 0, span);

    if (std::getenv("GOLBANG_MOE_TELEMETRY_DEBUG") != nullptr) {
        static int dbg_n = 0;
        if (dbg_n < 6) {
            std::fprintf(stderr, "[dbg] %s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%s first=[",
                         t->name, (long long) t->ne[0], (long long) t->ne[1],
                         (long long) t->ne[2], (long long) t->ne[3],
                         t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_type_name(t->type));
            const int nprint = std::min<int64_t>(8, n_used);
            const int32_t * r0 = reinterpret_cast<const int32_t *>(raw.data());
            for (int j = 0; j < nprint; ++j) {
                std::fprintf(stderr, "%s%d", j ? " " : "", r0[j]);
            }
            std::fprintf(stderr, "] tok1=[");
            if (n_tok > 1) {
                const int32_t * r1 = reinterpret_cast<const int32_t *>(raw.data() + row_stride);
                for (int j = 0; j < nprint; ++j) {
                    std::fprintf(stderr, "%s%d", j ? " " : "", r1[j]);
                }
            }
            std::fprintf(stderr, "]\n");
            dbg_n++;
        }
    }

    std::lock_guard<std::mutex> lock(tel->mtx);
    telemetry & t2 = *tel;
    const int phase = t2.phase;
    for (int64_t tok = 0; tok < n_tok; ++tok) {
        const int32_t * row = reinterpret_cast<const int32_t *>(raw.data() + (size_t) tok * row_stride);
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e = row[u];
            if (e < 0 || e >= t2.n_expert) {
                continue;
            }
            t2.counts[phase][il][e]++;
            t2.n_selected++;
        }
    }
    t2.n_calls++;
    return true;
}

static uint64_t layer_total(const std::vector<uint64_t> & v) {
    uint64_t s = 0;
    for (uint64_t x : v) {
        s += x;
    }
    return s;
}

// sum of the top-k expert counts in one layer
static uint64_t topk_sum(std::vector<uint64_t> v, int k) {
    if (v.empty()) {
        return 0;
    }
    if (k > (int) v.size()) {
        k = (int) v.size();
    }
    std::partial_sort(v.begin(), v.begin() + k, v.end(), std::greater<uint64_t>());
    uint64_t s = 0;
    for (int i = 0; i < k; ++i) {
        s += v[i];
    }
    return s;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * out_env = std::getenv("GOLBANG_MOE_TELEMETRY_OUT");
    const std::string out_prefix = out_env ? std::string(out_env) : std::string("moe-telemetry");

    if (params.prompt.empty() && !params.prompt_file.empty()) {
        std::ifstream in(params.prompt_file);
        if (in) {
            std::string line;
            while (std::getline(in, line)) {
                params.prompt += line;
                params.prompt += '\n';
            }
        } else {
            LOG_ERR("%s : cannot open prompt file %s\n", __func__, params.prompt_file.c_str());
            return 1;
        }
    }
    if (params.prompt.empty()) {
        params.prompt =
            "The history of computing spans mechanical calculators, punched cards, "
            "vacuum tubes, transistors, and integrated circuits. Each transition "
            "changed not only the hardware but the languages and abstractions that "
            "programmers used to describe computation.";
    }
    if (params.n_predict < 0) {
        params.n_predict = 64;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &g_tel;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    g_tel.n_layer = llama_model_n_layer(model);

    g_tel.n_expert = 384; // DSV4.1 default
    {
        char buf[64] = { 0 };
        if (llama_model_meta_val_str(model, "deepseek41.expert_count", buf, sizeof(buf)) > 0) {
            const int v = std::atoi(buf);
            if (v > 0) {
                g_tel.n_expert = v;
            }
        }
    }
    g_tel.counts.assign(2, std::vector<std::vector<uint64_t>>(
                                g_tel.n_layer, std::vector<uint64_t>(g_tel.n_expert, 0)));

    LOG_INF("%s : n_layer = %d, n_expert = %d, prompt tokens will be tokenized\n",
            __func__, g_tel.n_layer, g_tel.n_expert);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s : no input tokens\n", __func__);
        return 1;
    }

    // cap the prefill length so a run stays bounded; the routing distribution
    // is what we want, not the whole document
    size_t max_prompt_tokens = 2048;
    {
        const char * env = std::getenv("GOLBANG_MOE_TELEMETRY_MAX_TOKENS");
        if (env != nullptr) {
            const long v = std::atol(env);
            if (v > 0) {
                max_prompt_tokens = (size_t) v;
            }
        }
    }
    if (tokens.size() > max_prompt_tokens) {
        tokens.resize(max_prompt_tokens);
    }

    LOG_INF("%s : prompt tokens = %zu, generating %d tokens, n_batch = %d\n",
            __func__, tokens.size(), params.n_predict, params.n_batch);

    // prefill in n_batch-sized chunks (llama_batch_get_one assigns positions
    // sequentially for a single sequence, so repeated calls continue)
    g_tel.phase = 0;
    const auto t_prefill0 = std::chrono::steady_clock::now();
    {
        const int n_batch = params.n_batch > 0 ? params.n_batch : (int) tokens.size();
        size_t off = 0;
        while (off < tokens.size()) {
            const int chunk = (int) std::min<size_t>((size_t) n_batch, tokens.size() - off);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + off, chunk))) {
                LOG_ERR("%s : prefill decode failed at offset %zu\n", __func__, off);
                return 1;
            }
            off += (size_t) chunk;
        }
    }
    const auto t_prefill1 = std::chrono::steady_clock::now();

    // decode with the common sampler chain (temperature/top-k/top-p from the
    // CLI), so the generated text stays diverse and the routing stats are not
    // an artifact of greedy repetition
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    int n_generated = 0;
    g_tel.phase = 1;
    const auto t_decode0 = std::chrono::steady_clock::now();
    for (int i = 0; i < params.n_predict; ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        common_sampler_accept(smpl, id, true);

        llama_token tok = id;
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("%s : decode failed at token %d\n", __func__, i);
            break;
        }
        n_generated++;
    }
    common_sampler_free(smpl);
    const auto t_decode1 = std::chrono::steady_clock::now();

    const double prefill_s = std::chrono::duration<double>(t_prefill1 - t_prefill0).count();
    const double decode_s  = std::chrono::duration<double>(t_decode1 - t_decode0).count();
    LOG_INF("%s : prefill %zu tok in %.2fs (%.2f t/s) | decode %d tok in %.2fs (%.2f t/s)"
            " [telemetry reads one tensor per layer per step and adds sync overhead]\n",
            __func__, tokens.size(), prefill_s,
            prefill_s > 0 ? tokens.size() / prefill_s : 0.0,
            n_generated, decode_s, decode_s > 0 ? n_generated / decode_s : 0.0);

    LOG_INF("%s : generated %d decode tokens, %llu callbacks, %llu selections\n",
            __func__, n_generated, (unsigned long long) g_tel.n_calls,
            (unsigned long long) g_tel.n_selected);

    // ---- report ----
    const int K_REPORT[] = { 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384 };

    FILE * fsum = std::fopen((out_prefix + "-summary.txt").c_str(), "w");
    FILE * ftsv = std::fopen((out_prefix + "-counts.tsv").c_str(), "w");

    auto emit = [&](FILE * f, const char * fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(stdout, fmt, ap);
        va_end(ap);
        va_start(ap, fmt);
        if (f) {
            std::vfprintf(f, fmt, ap);
        }
        va_end(ap);
    };

    // global coverage: average over layers of top-k share (decode phase = 1)
    for (int phase = 0; phase < 2; ++phase) {
        const char * pname = phase == 0 ? "prefill" : "decode";
        emit(fsum, "\n=== phase=%s ===\n", pname);
        double g_cov[sizeof(K_REPORT) / sizeof(K_REPORT[0])] = { 0 };
        int n_nonzero = 0;
        for (int il = 0; il < g_tel.n_layer; ++il) {
            const uint64_t tot = layer_total(g_tel.counts[phase][il]);
            if (tot == 0) {
                continue;
            }
            n_nonzero++;
            emit(fsum, "layer %2d total=%llu", il, (unsigned long long) tot);
            // top-8 experts for eyeballing
            std::vector<uint64_t> v = g_tel.counts[phase][il];
            std::vector<int> idx(v.size());
            for (size_t j = 0; j < idx.size(); ++j) idx[j] = (int) j;
            const int n_show = std::min<int>(8, (int) idx.size());
            std::partial_sort(idx.begin(), idx.begin() + n_show, idx.end(),
                              [&](int a, int b) { return v[a] > v[b]; });
            emit(fsum, " top%d=[", n_show);
            for (int j = 0; j < n_show; ++j) {
                emit(fsum, "%s%d:%llu", j ? " " : "", idx[j], (unsigned long long) v[idx[j]]);
            }
            emit(fsum, "]");
            for (size_t ki = 0; ki < sizeof(K_REPORT) / sizeof(K_REPORT[0]); ++ki) {
                const int k = K_REPORT[ki];
                const double cov = (double) topk_sum(v, k) / (double) tot;
                g_cov[ki] += cov;
                emit(fsum, " k%d=%.3f", k, cov);
            }
            emit(fsum, "\n");
        }
        if (n_nonzero == 0) {
            continue;
        }
        emit(fsum, "\n%s mean top-k coverage over %d layers:\n", pname, n_nonzero);
        emit(stdout, "\n=== phase=%s mean top-k coverage over %d layers ===\n", pname, n_nonzero);
        for (size_t ki = 0; ki < sizeof(K_REPORT) / sizeof(K_REPORT[0]); ++ki) {
            emit(fsum, "  k=%-3d  %.4f\n", K_REPORT[ki], g_cov[ki] / n_nonzero);
            emit(stdout, "  k=%-3d  %.4f\n", K_REPORT[ki], g_cov[ki] / n_nonzero);
        }
    }

    if (ftsv) {
        std::fprintf(ftsv, "layer\texpert\tcount_prefill\tcount_decode\n");
        for (int il = 0; il < g_tel.n_layer; ++il) {
            std::vector<int> idx(g_tel.n_expert);
            for (int e = 0; e < g_tel.n_expert; ++e) idx[e] = e;
            const auto & cp = g_tel.counts[0][il];
            const auto & cd = g_tel.counts[1][il];
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                const uint64_t sa = cp[a] + cd[a];
                const uint64_t sb = cp[b] + cd[b];
                return sa > sb;
            });
            for (int e = 0; e < g_tel.n_expert; ++e) {
                const int ex = idx[e];
                if (cp[ex] == 0 && cd[ex] == 0) {
                    continue;
                }
                std::fprintf(ftsv, "%d\t%d\t%llu\t%llu\n", il, ex,
                             (unsigned long long) cp[ex], (unsigned long long) cd[ex]);
            }
        }
    }

    if (fsum) std::fclose(fsum);
    if (ftsv) std::fclose(ftsv);

    LOG_INF("%s : wrote %s-summary.txt and %s-counts.tsv\n", __func__,
            out_prefix.c_str(), out_prefix.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
