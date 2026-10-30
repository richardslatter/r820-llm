// mm_bench: decode matmuls (one activation row) in isolation: L weights of rows x K of one type,
// each multiplied by the same row in one graph, so the graph launch is shared and the weights
// stream from DRAM (L*size well past the caches) or stay in cache (L*size small).
// usage: GGML_NUMA_BLOCKED=1 ./mm_bench <threads> <type> <K> <rows> [L] [iters] [place]
// place=1: weight rows placed by node, as GGML_NUMA_PLACE does
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e6 + ts.tv_nsec*1e-3;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <threads> <type> <K> <rows> [L] [iters] [place]\n", argv[0]);
        return 1;
    }
    const int nth   = atoi(argv[1]);
    const int K     = atoi(argv[3]);
    const int R     = atoi(argv[4]);
    const int L     = argc > 5 ? atoi(argv[5]) : 64;
    const int iters = argc > 6 ? atoi(argv[6]) : 20;
    const int place = argc > 7 ? atoi(argv[7]) : 1;

    enum ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((enum ggml_type) t);
        if (n && strcmp(n, argv[2]) == 0) type = (enum ggml_type) t;
    }
    if (type == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown type %s\n", argv[2]);
        return 1;
    }

    const size_t wbytes = ggml_row_size(type, K)*R;
    // 4 KB pages, as the model's mmapped weights are: placement cannot move part of a huge page
    const size_t msize = L*(wbytes + 4096) + ((size_t)64 << 20);
    void * mbuf = mmap(NULL, msize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mbuf == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    madvise(mbuf, msize, MADV_NOHUGEPAGE);
    struct ggml_init_params ip = { msize, mbuf, false };
    struct ggml_context * ctx = ggml_init(ip);
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE); // after ggml_init, which resets the NUMA state

    float * src = malloc(sizeof(float)*K*R);
    srand(1);
    for (int i = 0; i < K*R; ++i) src[i] = (rand()/(float)RAND_MAX - 0.5f)*0.01f;

    // MM_NY activation rows (default 1)
    const int ny = getenv("MM_NY") ? atoi(getenv("MM_NY")) : 1;
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, ny);
    for (int i = 0; i < K*ny; ++i) ((float *) x->data)[i] = rand()/(float)RAND_MAX - 0.5f;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4*L + 16, false);
    size_t placed = 0;
    for (int l = 0; l < L; ++l) {
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, R);
        if (type == GGML_TYPE_F32) {
            memcpy(w->data, src, wbytes);
        } else {
            ggml_quantize_chunk(type, src, w->data, 0, R, K, NULL, NULL);
        }
        if (place) placed += ggml_numa_place_tensor(w);
        // MM_NORM=1: an rms_norm writes the activation just before each matmul, as in the model
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, getenv("MM_NORM") ? ggml_rms_norm(ctx, x, 1e-6f) : x));
    }

    struct ggml_cplan plan = ggml_graph_plan(gf, nth);
    plan.work_data = malloc(plan.work_size + 64);

    ggml_graph_compute(gf, &plan);
    double best = 1e30, sum = 0;
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_us();
        ggml_graph_compute(gf, &plan);
        const double dt = now_us() - t0;
        sum += dt;
        if (dt < best) best = dt;
    }
    printf("%-6s K %6d rows %6d (%.2f MB) x %d, %d threads: best %.1f us/matmul (%.1f GB/s), mean %.1f us, placed %.0f%%\n",
           ggml_type_name(type), K, R, wbytes/1e6, L, nth, best/L, wbytes*L/best/1e3, sum/iters/L, 100.0*placed/(wbytes*L));
    return 0;
}
