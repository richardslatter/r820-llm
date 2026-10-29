// Check ggml_mul_mat (and so the iqk kernels it dispatches to) against a double-precision
// reference built from the dequantized weights.
//   usage: mmcheck [threads]
// Prints, per weight type and batch size, the max relative error of each output row.
#include "ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float frand(unsigned * s) {
    *s = *s * 1664525u + 1013904223u;
    return ((*s >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f;
}

static double check(enum ggml_type type, int K, int N, int M, int nth) {
    struct ggml_init_params ip = { (size_t)1 << 30, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    unsigned seed = 1234u + type * 7u + M;
    float * wf = malloc(sizeof(float) * K * N);
    for (int i = 0; i < K * N; ++i) wf[i] = frand(&seed);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, type, K, N);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_quantize_chunk(type, wf, a->data, 0, N, K, NULL, NULL);
    for (int i = 0; i < K * M; ++i) ((float *)b->data)[i] = frand(&seed);

    struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    // reference: dequantize the weights the kernel actually saw
    ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    float * wd = malloc(sizeof(float) * K * N);
    for (int r = 0; r < N; ++r) {
        tt.to_float((const char *)a->data + r * a->nb[1], wd + (size_t)r * K, K);
    }

    double worst = 0;
    for (int m = 0; m < M; ++m) {
        const float * x = (const float *)b->data + (size_t)m * K;
        for (int r = 0; r < N; ++r) {
            double ref = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
                ref += (double)wd[(size_t)r * K + k] * x[k];
                mag += fabs((double)wd[(size_t)r * K + k] * x[k]);
            }
            double got = ((const float *)c->data)[(size_t)m * N + r];
            double err = fabs(got - ref) / (mag + 1e-30);
            if (!(err == err)) err = INFINITY;
            if (err > worst) worst = err;
        }
    }
    free(wf);
    free(wd);
    ggml_free(ctx);
    return worst;
}

// Same check for ggml_mul_mat_id (the MoE expert path): E experts, U used per token, T tokens,
// one shared activation row per token (as for the up/gate projections).
static double check_id(enum ggml_type type, int K, int N, int E, int U, int T, int nth) {
    struct ggml_init_params ip = { (size_t)1 << 31, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    unsigned seed = 99u + type * 13u + T;
    float * wf = malloc(sizeof(float) * K * N * E);
    for (size_t i = 0; i < (size_t)K * N * E; ++i) wf[i] = frand(&seed);

    struct ggml_tensor * as = ggml_new_tensor_3d(ctx, type, K, N, E);
    struct ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, T);
    ggml_quantize_chunk(type, wf, as->data, 0, (int64_t)N * E, K, NULL, NULL);
    for (int i = 0; i < K * T; ++i) ((float *)b->data)[i] = frand(&seed);
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            // distinct experts per token, spread over the whole range
            ((int32_t *)ids->data)[t * U + u] = (t * 7 + u * (E / U) + (seed >> 28)) % E;
        }
    }

    struct ggml_tensor * c = ggml_mul_mat_id(ctx, as, b, ids);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    float * wd = malloc(sizeof(float) * K);
    double worst = 0;
    for (int t = 0; t < T; ++t) {
        const float * x = (const float *)b->data + (size_t)t * K;
        for (int u = 0; u < U; ++u) {
            int e = ((int32_t *)ids->data)[t * U + u];
            for (int r = 0; r < N; ++r) {
                tt.to_float((const char *)as->data + e * as->nb[2] + r * as->nb[1], wd, K);
                double ref = 0, mag = 0;
                for (int k = 0; k < K; ++k) { ref += (double)wd[k] * x[k]; mag += fabs((double)wd[k] * x[k]); }
                double got = *(const float *)((const char *)c->data + t * c->nb[2] + u * c->nb[1] + r * sizeof(float));
                double err = fabs(got - ref) / (mag + 1e-30);
                if (!(err == err)) err = INFINITY;
                if (err > worst) worst = err;
            }
        }
    }
    free(wf);
    free(wd);
    ggml_free(ctx);
    return worst;
}

int main(int argc, char ** argv) {
    int nth = argc > 1 ? atoi(argv[1]) : 4;
    if (getenv("MMCHECK_NUMA")) {
        // exercise the GGML_NUMA_BLOCKED static expert split: it needs chunks_per_expert =
        // min(nth, rows/32) divisible by the node count, so use full-size expert rows
        ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
        const enum ggml_type t[] = { GGML_TYPE_MXFP4, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K };
        const int tk[] = { 1, 8, 33 };
        printf("mul_mat_id, NUMA blocked mode (K=2304, N=2304, E=16, U=6, %d threads)\n", nth);
        for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); ++i) {
            printf("%-10s", ggml_type_name(t[i]));
            for (size_t j = 0; j < sizeof(tk) / sizeof(tk[0]); ++j) {
                double e = check_id(t[i], 2304, 2304, 16, 6, tk[j], nth);
                printf("   T=%-3d %-9.2e%s", tk[j], e, e > 1e-2 ? "!" : " ");
            }
            printf("\n");
            fflush(stdout);
        }
        return 0;
    }
    const enum ggml_type types[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4,
        GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL,
    };
    const int batches[] = { 1, 8, 33, 128, 512 };
    const int K = 2304, N = 64;
    printf("%-10s", "type");
    for (size_t j = 0; j < sizeof(batches) / sizeof(batches[0]); ++j) printf("   M=%-7d", batches[j]);
    printf("\n");
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
        printf("%-10s", ggml_type_name(types[i]));
        for (size_t j = 0; j < sizeof(batches) / sizeof(batches[0]); ++j) {
            double e = check(types[i], K, N, batches[j], nth);
            printf("   %-9.2e%s", e, e > 1e-2 ? "!" : " ");
        }
        printf("\n");
        fflush(stdout);
    }

    const int tokens[] = { 1, 8, 33, 128, 512 };
    printf("\nmul_mat_id (E=32 experts, U=6 used)\n%-10s", "type");
    for (size_t j = 0; j < sizeof(tokens) / sizeof(tokens[0]); ++j) printf("   T=%-7d", tokens[j]);
    printf("\n");
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
        printf("%-10s", ggml_type_name(types[i]));
        for (size_t j = 0; j < sizeof(tokens) / sizeof(tokens[0]); ++j) {
            double e = check_id(types[i], K, N, 32, 6, tokens[j], nth);
            printf("   %-9.2e%s", e, e > 1e-2 ? "!" : " ");
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
