// Check and time the batched Q6_K kernel against the gemv kernel and a double reference.
//   g++ -O3 -march=native -fopenmp -std=c++17 -I../ik_llama/parts q6_bench.cpp -o q6_bench
//   OMP_PROC_BIND=spread OMP_PLACES=threads ./q6_bench [threads]
// "cached": each thread repeats one small problem (compute bound). "stream": each thread walks
// 64 MB of its own rows, so the weights come from memory as in the model.
#include "iqk_avx1_gemm.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

using namespace avx1_gemv;

static unsigned rng(unsigned & s) { s = s * 1664525u + 1013904223u; return s >> 8; }

static uint16_t to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float from_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t to_fp16(float f) {  // normal range only
    uint32_t u; memcpy(&u, &f, 4);
    const int e = ((u >> 23) & 0xff) - 127 + 15;
    return (uint16_t)(((u >> 16) & 0x8000) | (e << 10) | ((u >> 13) & 0x3ff));
}

struct Problem {
    int n, rows, ny;
    size_t bx, by;
    std::vector<uint8_t> w;
    std::vector<uint8_t> y;
    std::vector<const char *> yr;
    Problem(int n_, int rows_, int ny_, unsigned seed) : n(n_), rows(rows_), ny(ny_) {
        bx = (size_t)(n/256)*sizeof(blk_q6_K);
        by = (size_t)(n/128)*144;
        w.resize(bx*rows);
        for (int r = 0; r < rows; ++r) for (int b = 0; b < n/256; ++b) {
            blk_q6_K * xb = (blk_q6_K *)(w.data() + r*bx) + b;
            for (auto & q : xb->ql) q = rng(seed);
            for (auto & q : xb->qh) q = rng(seed);
            for (auto & s : xb->scales) s = (int8_t)rng(seed);  // full int8 range
            xb->d = to_fp16(0.0005f + (rng(seed) % 1000)*1e-5f);
        }
        y.resize(by*ny);
        std::vector<float> x(n);
        for (int c = 0; c < ny; ++c) {
            for (int i = 0; i < n; ++i) x[i] = ((rng(seed) % 20001) - 10000) * 1e-4f * (1 + (i % 97 == 0) * 20);
            blk_q8_2_x4 * yb = (blk_q8_2_x4 *)(y.data() + c*by);
            for (int b = 0; b < n/32; ++b) {
                float amax = 0;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32*b + j]));
                const uint16_t dh = to_bf16(amax/127);
                const float d = from_bf16(dh), id = d > 0 ? 1/d : 0;
                int sum = 0;
                for (int j = 0; j < 32; ++j) {
                    int q = (int)std::nearbyint(x[32*b + j]*id);
                    q = std::max(-127, std::min(127, q));
                    yb[b/4].qs[32*(b%4) + j] = q;
                    sum += q;
                }
                yb[b/4].d[b%4] = dh;
                int16_t s16 = sum; memcpy(&yb[b/4].d[4 + b%4], &s16, 2);
            }
        }
        for (int c = 0; c < ny; ++c) yr.push_back((const char *)y.data() + c*by);
    }
    double ref(int r, int c) const {
        const blk_q6_K * xr = (const blk_q6_K *)(w.data() + r*bx);
        const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)yr[c];
        double s = 0;
        for (int b = 0; b < n/256; ++b) {
            const blk_q6_K * xb = xr + b;
            const double d = fp16_to_f32(xb->d);
            for (int h = 0; h < 2; ++h) {
                const uint8_t * ql = xb->ql + 64*h; const uint8_t * qh = xb->qh + 32*h; const int8_t * sc = xb->scales + 8*h;
                int q[128];
                for (int l = 0; l < 32; ++l) {
                    q[l +  0] = ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    q[l + 32] = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    q[l + 64] = ((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    q[l + 96] = ((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                }
                const blk_q8_2_x4 & g = yb[2*b + h];
                for (int i = 0; i < 128; ++i) s += d*sc[i/16]*q[i]*(double)from_bf16(g.d[i/32])*g.qs[i];
            }
        }
        return s;
    }
};

static void run_old(const Problem & p, float * out) {
    int left = p.ny, c0 = 0;
    while (left > 0) {
        const int k = std::min(left, 8);
        auto st = [&](int ix, int iy, float v) { out[(c0 + iy)*p.rows + ix] = v; };
        switch (k) {
            case 1: { Rows<1> r; for (int i = 0; i < 1; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<1>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 2: { Rows<2> r; for (int i = 0; i < 2; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<2>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 3: { Rows<3> r; for (int i = 0; i < 3; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<3>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 4: { Rows<4> r; for (int i = 0; i < 4; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<4>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 5: { Rows<5> r; for (int i = 0; i < 5; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<5>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 6: { Rows<6> r; for (int i = 0; i < 6; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<6>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 7: { Rows<7> r; for (int i = 0; i < 7; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<7>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            default: { Rows<8> r; for (int i = 0; i < 8; ++i) r.y[i] = p.yr[c0 + i]; gemv_q6_K<8>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
        }
        left -= k; c0 += k;
    }
}

template <int R, int C>
static void run_new(const Problem & p, float * out) {
    gemm_q6_K<R, C>(p.n, p.w.data(), p.bx, p.yr.data(), p.ny, p.rows, [&](int ix, int iy, float v) { out[iy*p.rows + ix] = v; });
}

typedef void (*runner)(const Problem &, float *);

static double check(const Problem & p, runner f) {
    std::vector<float> out(p.ny*p.rows);
    f(p, out.data());
    double worst = 0;
    for (int c = 0; c < p.ny; ++c) for (int r = 0; r < p.rows; ++r) {
        const double ref = p.ref(r, c);
        const double err = std::fabs(out[c*p.rows + r] - ref)/(std::fabs(ref) + 1e-3);
        if (!(err == err) || err > worst) worst = err != err ? INFINITY : err;
    }
    return worst;
}

// aggregate MAC/s with nth threads, each on its own problem
static double timeit(int n, int rows, int ny, int nth, runner f) {
    double total_macs = 0, secs = 0;
    #pragma omp parallel num_threads(nth) reduction(+:total_macs) reduction(max:secs)
    {
        Problem p(n, rows, ny, 99 + omp_get_thread_num());
        std::vector<float> out(ny*rows);
        f(p, out.data());
        const int iters = std::max(3, (int)(2e8 / ((double)n*rows*ny)));
        #pragma omp barrier
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) f(p, out.data());
        auto t1 = std::chrono::steady_clock::now();
        total_macs = (double)n*rows*ny*iters;
        secs = std::chrono::duration<double>(t1 - t0).count();
    }
    return total_macs/secs*1e-9;
}

int main(int argc, char ** argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 64;
    const int nys[] = {1, 2, 3, 4, 5, 6, 8, 12, 16, 32};
    printf("correctness (max rel err vs double, n=5120 rows=37)\n");
    for (int ny : nys) {
        Problem p(5120, 37, ny, 7 + ny);
        printf("  ny=%2d  old %.2e  new22 %.2e  new23 %.2e  new14 %.2e\n", ny, check(p, run_old), check(p, run_new<2, 2>), check(p, run_new<2, 3>), check(p, run_new<1, 4>));
    }
    // cached: 36 rows per thread; stream: 64 MB of rows per thread
    const int shapes[3][2] = {{5120, 36}, {1024, 160}, {5120, 12800}};
    for (auto & sh : shapes) {
        printf("GMAC/s (GB/s of weights), n=%d rows=%d, %d threads\n", sh[0], sh[1], nth);
        printf("  ny     old            new22          new23          new14\n");
        for (int ny : nys) {
            if (sh[1] > 1000 && ny > 8) continue;
            const double gb = 210.0/256;
            double v[4] = { timeit(sh[0], sh[1], ny, nth, run_old), timeit(sh[0], sh[1], ny, nth, run_new<2, 2>),
                            timeit(sh[0], sh[1], ny, nth, run_new<2, 3>), timeit(sh[0], sh[1], ny, nth, run_new<1, 4>) };
            printf("  %2d", ny);
            for (double x : v) printf("  %6.1f (%5.1f)", x, x/ny*gb);
            printf("\n");
            fflush(stdout);
        }
    }
    return 0;
}
