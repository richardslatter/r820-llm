#!/usr/bin/env python3
"""Small decode-time ops (run from the ik_llama.cpp root, after numa_barrier_patch.py).

- RMS_NORM / FUSED_RMS_NORM: the sum of squares was a scalar loop with a double accumulator,
  ~4 cycles per element. DeepSeek-V4 normalises single rows of up to 20480 floats (the four
  hyper-connection streams), so each of its 84 norms per token took ~55 us on one thread.
  Now 16 floats per iteration: float products, four double accumulators.
- MUL_MULTI_ADD split its work by token, so at batch 1 one thread did all of it (121 nodes per
  token, ~28 us each). With fewer tokens than threads it now splits each row by columns.
- GGML_OP_PROF=3: like =2, plus one line per node (op, name, shapes, wall/slowest us) for the
  40th decode graph, to see where the small ops go.
"""
import sys


def patch(path, pairs):
    s = open(path).read()
    for old, new, count in pairs:
        n = s.count(old)
        if n != count:
            sys.exit(f"{path}: expected {count} of anchor, found {n}:\n{old[:200]}")
        s = s.replace(old, new)
    open(path, "w").write(s)


sumsq_loop = """        ggml_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
            sum += (ggml_float)(x[i00] * x[i00]);
        }
"""
sumsq_call = """        const ggml_float sum = ggml_sumsq_f32(ne00, x);
"""
sumsq_fn = """// sum of x[i]^2: float products, double accumulation (the order differs from a plain loop)
static inline ggml_float ggml_sumsq_f32(int64_t n, const float * x) {
    int64_t i = 0;
    ggml_float sum = 0.0;
#if defined(__AVX__)
    __m256d a0 = _mm256_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= n; i += 16) {
        const __m128 v0 = _mm_loadu_ps(x + i +  0), v1 = _mm_loadu_ps(x + i +  4);
        const __m128 v2 = _mm_loadu_ps(x + i +  8), v3 = _mm_loadu_ps(x + i + 12);
        a0 = _mm256_add_pd(a0, _mm256_cvtps_pd(_mm_mul_ps(v0, v0)));
        a1 = _mm256_add_pd(a1, _mm256_cvtps_pd(_mm_mul_ps(v1, v1)));
        a2 = _mm256_add_pd(a2, _mm256_cvtps_pd(_mm_mul_ps(v2, v2)));
        a3 = _mm256_add_pd(a3, _mm256_cvtps_pd(_mm_mul_ps(v3, v3)));
    }
    a0 = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
    __m128d h = _mm_add_pd(_mm256_castpd256_pd128(a0), _mm256_extractf128_pd(a0, 1));
    h = _mm_add_sd(h, _mm_unpackhi_pd(h, h));
    sum = _mm_cvtsd_f64(h);
#endif
    for (; i < n; ++i) sum += (ggml_float)(x[i] * x[i]);
    return sum;
}

static void ggml_compute_forward_rms_norm_f32(
"""

patch("ggml/src/ggml.c", [
    (sumsq_loop, sumsq_call, 2),
    ("\nstatic void ggml_compute_forward_rms_norm_f32(\n", "\n" + sumsq_fn, 1),
    # per-node trace
    ("""            g_prof_wall[key] += wall; g_prof_max[key] += mx; g_prof_sum[key] += sm/state->shared->n_threads; g_prof_cnt[key]++;
""", """            g_prof_wall[key] += wall; g_prof_max[key] += mx; g_prof_sum[key] += sm/state->shared->n_threads; g_prof_cnt[key]++;
            if (g_prof_on == 3 && g_prof_graphs == 40) {
                const double tpu = (double)(__rdtsc() - g_prof_tsc0)/(double)(ggml_time_us() - g_prof_us0);
                const struct ggml_tensor * s0 = node->src[0];
                fprintf(stderr, "node %4d %-16s %-34.34s %-6s %6lld %6lld %5lld %3lld <- %6lld %6lld %5lld %3lld | %6lld %6lld %8.1f %8.1f %8.1f\\n",
                        node_n, ggml_op_name(node->op), node->name, s0 ? ggml_type_name(s0->type) : "-",
                        (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3],
                        (long long)(s0 ? s0->ne[0] : 0), (long long)(s0 ? s0->ne[1] : 0), (long long)(s0 ? s0->ne[2] : 0), (long long)(s0 ? s0->ne[3] : 0),
                        (long long)(node->src[1] ? node->src[1]->ne[1] : 0), (long long)(node->src[1] ? node->src[1]->ne[2] : 0),
                        wall/tpu, mx/tpu, sm/state->shared->n_threads/tpu);
            }
""", 1),
])

patch("ggml/src/iqk/iqk_cpu_ops.cpp", [
    ("""    int nrows = dst->ne[1];
    int npt   = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(nrows, first + npt);

    int ne01 = src0->ne[1];
    int ne00 = src0->ne[0];
""", """    int nrows = dst->ne[1];
    int npt   = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(nrows, first + npt);

    int ne01 = src0->ne[1];
    int ne00 = src0->ne[0];

    // fewer rows than threads (decode): every thread takes a column slice of every row
    int k0 = 0;
    if (nrows < nth && ne00 >= 16*nth) {
        first = 0; last = nrows;
        k0 = (ne00/16)*ith/nth*16;
        const int k1 = ith == nth - 1 ? ne00 : (ne00/16)*(ith + 1)/nth*16;
        ne00 = k1 - k0;
    }
""", 1),
    ("""            auto  y = (     float *)cy;
            auto x0 = (const float *)c0;
            auto x1 = (const float *)c1;
            auto ids = (const int *)(cids + ir*src3->nb[1]);
""", """            auto  y = (     float *)cy + k0;
            auto x0 = (const float *)c0 + k0;
            auto x1 = (const float *)c1;
            auto ids = (const int *)(cids + ir*src3->nb[1]);
""", 1),
    ("""                c0 += src0->nb[1];
                c1 += src1->nb[1];
                x0 = (const float *)c0;
                x1 = (const float *)c1;
                s  = x1[0] * scales[ids[j]];
""", """                c0 += src0->nb[1];
                c1 += src1->nb[1];
                x0 = (const float *)c0 + k0;
                x1 = (const float *)c1;
                s  = x1[0] * scales[ids[j]];
""", 1),
    ("""        auto  y = (     float *)cy;
        auto x0 = (const float *)c0;
        auto x1 = (const float *)c1;
        for (int k = 0; k < ne00; ++k) y[k] = x0[k] * x1[0];
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0;
""", """        auto  y = (     float *)cy + k0;
        auto x0 = (const float *)c0 + k0;
        auto x1 = (const float *)c1;
        for (int k = 0; k < ne00; ++k) y[k] = x0[k] * x1[0];
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0 + k0;
""", 1),
])
print("decode ops patch applied")
