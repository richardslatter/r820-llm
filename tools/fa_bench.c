// fa_bench: DeepSeek-V4 decode attention in isolation (one query row, 64 heads x 512, one shared
// fp16 K/V head, f16 mask with n_vis visible rows, f32 sinks), timed through ggml's thread pool.
// usage: GGML_NUMA_BLOCKED=1 ./fa_bench <threads> <nkv> <n_vis> [iters]
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e6 + ts.tv_nsec*1e-3;
}

int main(int argc, char ** argv) {
    const int nth   = argc > 1 ? atoi(argv[1]) : 64;
    const int nkv   = argc > 2 ? atoi(argv[2]) : 1024;
    const int n_vis = argc > 3 ? atoi(argv[3]) : 640;
    const int iters = argc > 4 ? atoi(argv[4]) : 200;
    const int D = 512, H = 64, NQ = 1, NQP = 16;

    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    struct ggml_init_params ip = { (size_t)256 << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, H, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, nkv, 1, 1);
    struct ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, NQP, 1, 1);
    struct ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

    srand(1);
    float * qd = (float *) q->data;
    for (int i = 0; i < D*NQ*H; ++i) qd[i] = (rand()/(float)RAND_MAX - 0.5f)*0.2f;
    ggml_fp16_t * kd = (ggml_fp16_t *) k->data;
    for (int i = 0; i < D*nkv; ++i) kd[i] = ggml_fp32_to_fp16((rand()/(float)RAND_MAX - 0.5f)*2.0f);
    ggml_fp16_t * md = (ggml_fp16_t *) m->data;
    // visible: n_vis rows spread like the real cache (a raw window, then compressed rows)
    for (int r = 0; r < NQP; ++r) for (int i = 0; i < nkv; ++i) {
        const int vis = r < NQ && (i < n_vis/2 || (i >= nkv/2 && i < nkv/2 + n_vis - n_vis/2));
        md[r*nkv + i] = ggml_fp32_to_fp16(vis ? 0.0f : -INFINITY);
    }
    float * sd = (float *) s->data;
    for (int h = 0; h < H; ++h) sd[h] = (rand()/(float)RAND_MAX - 0.5f);

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, k, m, 1.0f/sqrtf((float)D), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, s);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    struct ggml_cplan plan = ggml_graph_plan(gf, nth);
    plan.work_data = malloc(plan.work_size + 64);

    ggml_graph_compute(gf, &plan);
    // reference: one double-precision softmax per head
    double max_err = 0, max_ref = 0;
    for (int h = 0; h < H; ++h) {
        double M = sd[h], S = 0;
        double * acc = calloc(D, sizeof(double));
        double * sc = malloc(nkv*sizeof(double));
        for (int i = 0; i < nkv; ++i) {
            const float mv = ggml_fp16_to_fp32(md[i]);
            if (isinf(mv)) { sc[i] = -INFINITY; continue; }
            double d = 0;
            for (int j = 0; j < D; ++j) d += (double)qd[h*D + j]*ggml_fp16_to_fp32(kd[i*D + j]);
            sc[i] = d/sqrt((double)D);
            if (sc[i] > M) M = sc[i];
        }
        S = exp(sd[h] - M);
        for (int i = 0; i < nkv; ++i) {
            if (isinf(sc[i])) continue;
            const double p = exp(sc[i] - M);
            S += p;
            for (int j = 0; j < D; ++j) acc[j] += p*ggml_fp16_to_fp32(kd[i*D + j]);
        }
        const float * o = (const float *) out->data + h*D;
        for (int j = 0; j < D; ++j) {
            const double r = acc[j]/S;
            max_err = fmax(max_err, fabs(r - o[j]));
            max_ref = fmax(max_ref, fabs(r));
        }
        free(acc); free(sc);
    }

    double best = 1e30, tot = 0;
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_us();
        ggml_graph_compute(gf, &plan);
        const double dt = now_us() - t0;
        tot += dt; if (dt < best) best = dt;
    }
    // dispatch cost of a graph with one trivial node, for comparison
    struct ggml_tensor * t = ggml_scale(ctx, s, 2.0f);
    struct ggml_cgraph * g0 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g0, t);
    double best0 = 1e30;
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_us();
        ggml_graph_compute(g0, &plan);
        const double dt = now_us() - t0;
        if (dt < best0) best0 = dt;
    }
    printf("threads %d nkv %d visible %d: max err %.3g (max |ref| %.3g), %.1f us avg, %.1f us best (empty graph %.1f us)\n",
           nth, nkv, n_vis, max_err, max_ref, tot/iters, best, best0);
    return 0;
}
