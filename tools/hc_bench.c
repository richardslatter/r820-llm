// hc_bench: DeepSeek-V4's hyper-connection pre-step in isolation: rms_norm of one 20480-float row,
// then a 24 x 20480 f32 matmul, timed through ggml's thread pool (also each op alone).
// usage: GGML_NUMA_BLOCKED=1 ./hc_bench <threads> [iters] [K] [rows] [cold]
// cold=1: weight rows placed by node (as GGML_NUMA_PLACE does) and caches flushed before each run,
// as in the model, where ~12 GB stream through between two uses of the same weight
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e6 + ts.tv_nsec*1e-3;
}

static char * g_flush = NULL;
static size_t g_flush_n = 0;

static double best_of(struct ggml_cgraph * g, struct ggml_cplan * plan, int iters) {
    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
        if (g_flush) {
            for (size_t i = 0; i < g_flush_n; i += 64) g_flush[i]++;
        }
        const double t0 = now_us();
        ggml_graph_compute(g, plan);
        const double dt = now_us() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char ** argv) {
    const int nth   = argc > 1 ? atoi(argv[1]) : 64;
    const int iters = argc > 2 ? atoi(argv[2]) : 500;
    const int K     = argc > 3 ? atoi(argv[3]) : 20480;
    const int R     = argc > 4 ? atoi(argv[4]) : 24;
    const int cold  = argc > 5 ? atoi(argv[5]) : 0;

    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    struct ggml_init_params ip = { (size_t)64 << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, R);
    srand(1);
    for (int i = 0; i < K; ++i) ((float *) x->data)[i] = rand()/(float)RAND_MAX - 0.5f;
    for (int i = 0; i < K*R; ++i) ((float *) w->data)[i] = (rand()/(float)RAND_MAX - 0.5f)*0.01f;

    if (cold) {
        ggml_numa_place_tensor(w);
        g_flush_n = (size_t)256 << 20;
        g_flush = calloc(g_flush_n, 1);
    }

    struct ggml_tensor * n  = ggml_rms_norm(ctx, x, 1e-6f);
    struct ggml_tensor * y  = ggml_mul_mat(ctx, w, n);
    struct ggml_tensor * y2 = ggml_mul_mat(ctx, w, x);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    struct ggml_cgraph * gn = ggml_new_graph(ctx);
    ggml_build_forward_expand(gn, n);
    struct ggml_cgraph * gm = ggml_new_graph(ctx);
    ggml_build_forward_expand(gm, y2);
    struct ggml_cgraph * g0 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g0, ggml_scale(ctx, x, 2.0f));

    struct ggml_cplan plan = ggml_graph_plan(gf, nth);
    struct ggml_cplan pm   = ggml_graph_plan(gm, nth);
    if (pm.work_size > plan.work_size) plan.work_size = pm.work_size;
    plan.work_data = malloc(plan.work_size + 64);

    ggml_graph_compute(gf, &plan);
    // reference
    double ss = 0;
    for (int i = 0; i < K; ++i) ss += (double)((float *) x->data)[i]*((float *) x->data)[i];
    const double sc = 1.0/sqrt(ss/K + 1e-6);
    double max_err = 0, max_ref = 0;
    for (int r = 0; r < R; ++r) {
        double d = 0;
        for (int i = 0; i < K; ++i) d += (double)((float *) w->data)[r*K + i]*((float *) x->data)[i]*sc;
        max_err = fmax(max_err, fabs(d - ((float *) y->data)[r]));
        max_ref = fmax(max_ref, fabs(d));
    }

    const double t_all = best_of(gf, &plan, iters);
    const double t_n   = best_of(gn, &plan, iters);
    const double t_m   = best_of(gm, &plan, iters);
    const double t_0   = best_of(g0, &plan, iters);
    printf("%s threads %d K %d rows %d: max err %.3g (max |ref| %.3g); best us: norm+matmul %.1f, norm %.1f, matmul %.1f, empty %.1f\n",
           cold ? "cold" : "warm", nth, K, R, max_err, max_ref, t_all, t_n, t_m, t_0);
    return 0;
}
