// Microbenchmark: Q2_K expert kernel candidate for AVX2 (issue #229, DSV41-OPT-A2).
//
// Candidate "panel" kernel, in the spirit of the upstream iqp (IQ panel) path:
//   - decode 8 source rows at a time into an int8 panel at load time
//   - pre-multiply the per-16 integer sub-block scale sc into the weights, so the
//     inner integer dot needs no per-sub-block scale multiply (possible for Q2_K
//     because q in [0,3] and sc in [0,15] => q*sc <= 45, fits int8)
//   - run an integer gemv/gemm against q8_K activations, applying only one fp32
//     factor (d * a.d) per super-block per row
//
// Baseline is the generic packed path: ggml_vec_dot_q2_K_q8_K, called per
// (row, column) pair exactly like the nrc==1 mul_mat fallback.
//
// The layout of the panel weights is the same interleaving as iqp:
//   w[sb*128 + g*32 + row*4 + k] = q * sc
// and the matching activation byte is a->qs[sb*16 + g*4 + k].
//
// Result: numerically exact (max_rel ~2e-6 over full K) but slower than the
// generic path at every batch size, so the candidate is not adopted. See
// docs/bench/dsv41-q2k-panel-negative.md.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

extern "C" void ggml_vec_dot_q2_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx,
                                       const void * vy, size_t by, int nrc);

#define P_SB   16
#define P_NSB  (QK_K / P_SB)   // 16 sub-blocks per super-block
#define P_ROWS 8

struct q2k_panel_x8 {
    float  dfac[P_ROWS];
    float  dmin[P_ROWS];
    int8_t w[QK_K * P_ROWS];      // interleaved, pre-multiplied q*sc
    int8_t mins[P_NSB * P_ROWS];  // per sub-block min (scales >> 4)
};

static int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// ---- panel construction -----------------------------------------------------
static void build_panel(const uint8_t * const rows[P_ROWS], q2k_panel_x8 * out) {
    const block_q2_K * blk[P_ROWS];
    for (int r = 0; r < P_ROWS; ++r) blk[r] = (const block_q2_K *) rows[r];

    for (int r = 0; r < P_ROWS; ++r) {
        out->dfac[r] = GGML_FP16_TO_FP32(blk[r]->d);
        out->dmin[r] = GGML_FP16_TO_FP32(blk[r]->dmin);
    }

    for (int p = 0; p < QK_K; ++p) {
        const int sb     = p >> 4;
        const int rem    = p & 15;
        const int g      = rem >> 2;
        const int k      = rem & 3;
        const int n      = p >> 7;
        const int within = p & 127;
        const int j      = within >> 5;
        const int ww     = within & 31;

        for (int r = 0; r < P_ROWS; ++r) {
            const int     q  = (blk[r]->qs[n * 32 + ww] >> (2 * j)) & 3;
            const uint8_t sc = blk[r]->scales[sb] & 0xF;
            out->w[sb * 128 + g * 32 + r * 4 + k] = (int8_t) (q * sc);
        }
    }

    for (int sb = 0; sb < P_NSB; ++sb) {
        for (int r = 0; r < P_ROWS; ++r) {
            out->mins[sb * P_ROWS + r] = (int8_t) (blk[r]->scales[sb] >> 4);
        }
    }
}

// ---- AVX2 panel kernel ------------------------------------------------------
#if defined(__AVX2__)
static inline __m256i p_dot4(__m256i acc, __m256i xv, __m256i yb) {
    const __m256i dot  = _mm256_maddubs_epi16(xv, yb);   // xv unsigned, yb signed
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(ones, dot));
}

static inline __m256i p_load_y(const int8_t * qs) {
    const __m128i y = _mm_loadu_si128((const __m128i *) qs);
    return _mm256_broadcastsi128_si256(y);
}

static inline __m256i p_acc_block(const q2k_panel_x8 * b, const block_q8_K * a) {
    __m256i sumi = _mm256_setzero_si256();
    for (int sb = 0; sb < P_NSB; ++sb) {
        const int8_t * w = b->w + sb * 128;
        const __m256i yv = p_load_y(a->qs + sb * 16);
        __m256i isum = _mm256_setzero_si256();
        isum = p_dot4(isum, _mm256_loadu_si256((const __m256i *) (w +  0)), _mm256_shuffle_epi32(yv, 0x00));
        isum = p_dot4(isum, _mm256_loadu_si256((const __m256i *) (w + 32)), _mm256_shuffle_epi32(yv, 0x55));
        isum = p_dot4(isum, _mm256_loadu_si256((const __m256i *) (w + 64)), _mm256_shuffle_epi32(yv, 0xAA));
        isum = p_dot4(isum, _mm256_loadu_si256((const __m256i *) (w + 96)), _mm256_shuffle_epi32(yv, 0xFF));
        sumi = _mm256_add_epi32(sumi, isum);
    }
    return sumi;
}

// 8 rows x 1 column; out[8]
static void panel_gemv(const q2k_panel_x8 * p, int nb, const block_q8_K * a, float * out) {
    __m256 sumf = _mm256_setzero_ps();
    for (int l = 0; l < nb; ++l) {
        const __m256i isum = p_acc_block(&p[l], &a[l]);

        float mterm[P_ROWS] = { 0 };
        for (int sb = 0; sb < P_NSB; ++sb) {
            const float bs = (float) a[l].bsums[sb];
            for (int r = 0; r < P_ROWS; ++r) {
                mterm[r] += (float) p[l].mins[sb * P_ROWS + r] * bs;
            }
        }

        const __m256 ad = _mm256_set1_ps(a[l].d);
        const __m256 dv = _mm256_mul_ps(_mm256_loadu_ps(p[l].dfac), ad);
        const __m256 mv = _mm256_loadu_ps(mterm);
        const __m256 dm = _mm256_mul_ps(_mm256_loadu_ps(p[l].dmin), ad);
        sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), dv, sumf);
        sumf = _mm256_fnmadd_ps(dm, mv, sumf);
    }
    _mm256_storeu_ps(out, sumf);
}
#else
static void panel_gemv(const q2k_panel_x8 *, int, const block_q8_K *, float *) { }
#endif

// ---- weight/activation generation ------------------------------------------
static void fabricate_block(block_q2_K * x, std::mt19937 & rng) {
    std::uniform_int_distribution<int> b(0, 255);
    for (int i = 0; i < QK_K / 16; ++i) x->scales[i] = (uint8_t) b(rng);
    for (int i = 0; i < QK_K / 4;  ++i) x->qs[i]     = (uint8_t) b(rng);
    x->d    = ggml_fp32_to_fp16(0.001f + (b(rng) % 800) / 1000.0f);
    x->dmin = ggml_fp32_to_fp16(0.0005f + (b(rng) % 400) / 1000.0f);
}

static void fabricate_q8(block_q8_K * y, std::mt19937 & rng) {
    std::uniform_int_distribution<int> q(-127, 127);
    y->d = 0.005f + (rng() % 500) / 100000.0f;
    for (int i = 0; i < QK_K; ++i) y->qs[i] = (int8_t) q(rng);
    for (int sb = 0; sb < QK_K / 16; ++sb) {
        int s = 0;
        for (int i = 0; i < 16; ++i) s += y->qs[sb * 16 + i];
        y->bsums[sb] = (int16_t) s;
    }
}

int main(int argc, char ** argv) {
    ggml_cpu_init();

    const int    K    = 5120;
    const int    M    = 2304;
    const int    n_as = 64;
    const int    n_ids = 6;
    const size_t row_size = ggml_row_size(GGML_TYPE_Q2_K, K);
    const int    nb   = K / QK_K;

    bool do_accuracy = true;
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--no-acc") == 0) do_accuracy = false;

    if (do_accuracy) {
        // ---- single super-block: builder mapping and dot equivalence ----
        {
            std::mt19937 rng(20240914);
            block_q2_K w[P_ROWS];
            for (auto & x : w) fabricate_block(&x, rng);
            block_q8_K a;
            fabricate_q8(&a, rng);

            const uint8_t * rows[P_ROWS];
            for (int r = 0; r < P_ROWS; ++r) rows[r] = (const uint8_t *) &w[r];

            q2k_panel_x8 panel;
            build_panel(rows, &panel);

            const float d    = GGML_FP16_TO_FP32(w[0].d);
            const float dmin = GGML_FP16_TO_FP32(w[0].dmin);
            float deq[QK_K];
            ggml_get_type_traits(GGML_TYPE_Q2_K)->to_float(&w[0], deq, QK_K);
            double bad = 0;
            for (int p = 0; p < QK_K; ++p) {
                const int sb = p >> 4, rem = p & 15, g = rem >> 2, k = rem & 3;
                const int n = p >> 7, within = p & 127, j = within >> 5, ww = within & 31;
                const int qv = (w[0].qs[n * 32 + ww] >> (2 * j)) & 3;
                const int sc = w[0].scales[sb] & 0xF;
                const int mm = w[0].scales[sb] >> 4;
                const float mine = d * (float) (qv * sc) - dmin * (float) mm;
                const int stored = panel.w[sb * 128 + g * 32 + 0 * 4 + k];
                if (std::fabs(mine - deq[p]) > 1e-4f || stored != qv * sc) bad += 1;
            }

            float got[P_ROWS];
            panel_gemv(&panel, 1, &a, got);

            double max_abs = 0, max_rel = 0;
            for (int r = 0; r < P_ROWS; ++r) {
                float ref = 0;
                ggml_vec_dot_q2_K_q8_K(QK_K, &ref, sizeof(float), &w[r], 0, &a, 0, 1);
                const double diff = std::fabs((double) ref - (double) got[r]);
                max_abs = std::max(max_abs, diff);
                max_rel = std::max(max_rel, diff / (std::fabs((double) ref) + 1e-6));
            }
            printf("builder mapping mismatches: %.0f / %d\n", bad, QK_K);
            printf("single-block ACCURACY max_abs=%.3e max_rel=%.3e -> %s\n",
                   max_abs, max_rel, (bad == 0 && max_abs < 5e-2) ? "PASS" : "FAIL");
        }

        // ---- full-row (K=5120, nb=20) equivalence for 8 rows ----
        {
            std::vector<block_q2_K> full(nb * P_ROWS);
            std::vector<block_q8_K> act(nb);
            std::mt19937 rng(4242);
            for (auto & x : full) fabricate_block(&x, rng);
            for (auto & y : act)  fabricate_q8(&y, rng);

            std::vector<q2k_panel_x8> pf(nb); // pf[l]: 8 rows of superblock l
            for (int l = 0; l < nb; ++l) {
                const uint8_t * rr[P_ROWS];
                for (int r = 0; r < P_ROWS; ++r) rr[r] = (const uint8_t *) &full[r * nb + l];
                build_panel(rr, &pf[l]);
            }

            float g[P_ROWS];
            panel_gemv(pf.data(), nb, act.data(), g);
            double fa = 0, fr = 0;
            for (int r = 0; r < P_ROWS; ++r) {
                float ref = 0;
                ggml_vec_dot_q2_K_q8_K(K, &ref, sizeof(float), &full[r * nb], 0, act.data(), 0, 1);
                const double d = std::fabs((double) ref - (double) g[r]);
                fa = std::max(fa, d);
                fr = std::max(fr, d / (std::fabs((double) ref) + 1e-6));
            }
            printf("full-K       ACCURACY max_abs=%.3e max_rel=%.3e -> %s\n\n",
                   fa, fr, (fr < 1e-4) ? "PASS" : "FAIL");
        }
    }

    // ---- full model-shaped workload ----
    std::vector<std::vector<uint8_t>> W(n_as);
    {
        std::mt19937 rng(777);
        for (int e = 0; e < n_as; ++e) {
            W[e].resize(row_size * M);
            for (int m = 0; m < M; ++m) {
                uint8_t * rp = W[e].data() + (size_t) m * row_size;
                for (int l = 0; l < nb; ++l) {
                    fabricate_block((block_q2_K *) (rp + (size_t) l * sizeof(block_q2_K)), rng);
                }
            }
        }
        printf("weights: %zu MiB for %d experts\n", (size_t) (W[0].size() * n_as) >> 20, n_as);
    }

    std::vector<std::vector<q2k_panel_x8>> P(n_as);
    const int64_t t0 = now_us();
    for (int e = 0; e < n_as; ++e) {
        P[e].resize(M / P_ROWS * nb);
        for (int pg = 0; pg < M / P_ROWS; ++pg) {
            const uint8_t * rows[P_ROWS];
            for (int r = 0; r < P_ROWS; ++r)
                rows[r] = W[e].data() + (size_t) (pg * P_ROWS + r) * row_size;
            for (int l = 0; l < nb; ++l) {
                const uint8_t * r2[P_ROWS];
                for (int r = 0; r < P_ROWS; ++r) r2[r] = rows[r] + (size_t) l * sizeof(block_q2_K);
                build_panel(r2, &P[e][pg * nb + l]);
            }
        }
    }
    printf("panel build: %.1f ms total (%.3f us / panel-superblock)\n\n",
           (now_us() - t0) / 1000.0,
           (now_us() - t0) / (double) (n_as * (M / P_ROWS) * nb));

    // n_tok=1: 6 selected experts, 1 column each.
    // n_tok=64: n_ids*64 = 384 assignments over 64 experts -> 6 columns each.
    static volatile double g_sink = 0;
    const int cols[2] = { 1, (n_ids * 64) / n_as };
    const int nex [2] = { n_ids, n_as };

    std::vector<block_q8_K> A[2];
    for (int wi = 0; wi < 2; ++wi) {
        A[wi].resize((size_t) nb * cols[wi]);
        std::mt19937 rng(99 + wi);
        for (auto & y : A[wi]) fabricate_q8(&y, rng);
    }

    for (int wi = 0; wi < 2; ++wi) {
        const int c_e = cols[wi];
        const int c_n = nex[wi];
        const block_q8_K * act = A[wi].data();

        auto run_generic = [&](int reps) -> double {
            float s = 0;
            const int64_t t = now_us();
            for (int rep = 0; rep < reps; ++rep) {
                for (int e = 0; e < c_n; ++e) {
                    const uint8_t * base = W[e].data();
                    for (int m = 0; m < M; ++m) {
                        const void * xr = base + (size_t) m * row_size;
                        for (int cc = 0; cc < c_e; ++cc) {
                            float d;
                            ggml_vec_dot_q2_K_q8_K(K, &d, sizeof(float), xr, 0, act + (size_t) cc * nb, 0, 1);
                            s += d;
                        }
                    }
                }
            }
            g_sink += s;
            return (now_us() - t) / (double) reps / 1000.0;
        };

        std::vector<float> out(P_ROWS);
        auto run_panel = [&](int reps, bool include_build) -> double {
            float s = 0;
            const int64_t t = now_us();
            for (int rep = 0; rep < reps; ++rep) {
                for (int e = 0; e < c_n; ++e) {
                    for (int pg = 0; pg < M / P_ROWS; ++pg) {
                        if (include_build) {
                            const uint8_t * rows[P_ROWS];
                            for (int r = 0; r < P_ROWS; ++r)
                                rows[r] = W[e].data() + (size_t) (pg * P_ROWS + r) * row_size;
                            for (int l = 0; l < nb; ++l) {
                                const uint8_t * r2[P_ROWS];
                                for (int r = 0; r < P_ROWS; ++r)
                                    r2[r] = rows[r] + (size_t) l * sizeof(block_q2_K);
                                build_panel(r2, &P[e][pg * nb + l]);
                            }
                        }
                        for (int cc = 0; cc < c_e; ++cc) {
                            panel_gemv(&P[e][pg * nb], nb, act + (size_t) cc * nb, out.data());
                            for (int r = 0; r < P_ROWS; ++r) s += out[r];
                        }
                    }
                }
            }
            g_sink += s;
            return (now_us() - t) / (double) reps / 1000.0;
        };

        const int reps = wi == 0 ? 20 : 5;
        std::vector<double> bg, bp, bpb;
        for (int tr = 0; tr < 3; ++tr) {
            bg.push_back(run_generic(reps));
            bp.push_back(run_panel(reps, false));
            bpb.push_back(run_panel(reps, true));
        }
        std::sort(bg.begin(), bg.end());
        std::sort(bp.begin(), bp.end());
        std::sort(bpb.begin(), bpb.end());

        const double flops = (double) c_n * M * c_e * K * 2.0;
        printf("n_tok=%-3s experts=%2d cols/expert=%d\n", wi == 0 ? "1" : "64", c_n, c_e);
        printf("  generic            %9.3f ms  %6.1f GFLOP/s\n", bg.front(), flops / bg.front() / 1e6);
        printf("  panel (prebuilt)   %9.3f ms  %6.1f GFLOP/s   speedup=%.2fx\n",
               bp.front(), flops / bp.front() / 1e6, bg.front() / bp.front());
        printf("  panel (incl.build) %9.3f ms  %6.1f GFLOP/s   speedup=%.2fx\n\n",
               bpb.front(), flops / bpb.front() / 1e6, bg.front() / bpb.front());
    }

    return 0;
}
