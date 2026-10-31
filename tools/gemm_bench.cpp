// Check and time the batched MXFP4 kernel against the gemv kernel and a double reference.
//   g++ -O3 -march=native -fopenmp -std=c++17 -I../ik_llama/parts gemm_bench.cpp -o gemm_bench
//   OMP_PROC_BIND=spread OMP_PLACES=threads ./gemm_bench [threads]
#include "iqk_avx1_gemm.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

using namespace avx1_gemv;

static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static unsigned rng(unsigned & s) { s = s * 1664525u + 1013904223u; return s >> 8; }

static uint16_t to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float from_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

struct Problem {
    int n, rows, ny;
    size_t bx, by;
    std::vector<uint8_t> w;
    std::vector<uint8_t> y;
    std::vector<const char *> yr;
    Problem(int n_, int rows_, int ny_, unsigned seed) : n(n_), rows(rows_), ny(ny_) {
        bx = (size_t)(n/32)*17;
        by = (size_t)(n/128)*144;
        w.resize(bx*rows);
        for (auto & b : w) b = rng(seed) & 0xff;
        for (int r = 0; r < rows; ++r) for (int i = 0; i < n/32; ++i) w[r*bx + i*17] = 116 + rng(seed) % 10;
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
                    const int q = (int)std::nearbyint(x[32*b + j]*id);
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
        const uint8_t * xr = w.data() + r*bx;
        const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)yr[c];
        double s = 0;
        for (int b = 0; b < n/32; ++b) {
            const double dx = std::ldexp(0.5, xr[b*17] - 127), dy = from_bf16(yb[b/4].d[b%4]);
            long long is = 0;
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = xr[b*17 + 1 + j];
                is += kv[q & 15]*yb[b/4].qs[32*(b%4) + j] + kv[q >> 4]*yb[b/4].qs[32*(b%4) + 16 + j];
            }
            s += dx*dy*is;
        }
        return s;
    }
};

template <int ny>
static void run_gemv(const Problem & p, float * out) {
    for (int c0 = 0; c0 < p.ny; c0 += ny) {
        Rows<ny> rows;
        for (int iy = 0; iy < ny; ++iy) rows.y[iy] = p.yr[std::min(c0 + iy, p.ny - 1)];
        gemv_mxfp4<ny>(p.n, p.w.data(), p.bx, rows, p.rows, [&](int ix, int iy, float v) { if (c0 + iy < p.ny) out[(c0 + iy)*p.rows + ix] = v; });
    }
}

static void run_old(const Problem & p, float * out) {
    // as mul_mat_NxM: chunks of up to 8 columns
    int left = p.ny, c0 = 0;
    while (left > 0) {
        const int k = std::min(left, 8);
        Problem const * pp = &p;
        auto st = [&](int ix, int iy, float v) { out[(c0 + iy)*pp->rows + ix] = v; };
        switch (k) {
            case 1: { Rows<1> r; for (int i = 0; i < 1; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<1>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 2: { Rows<2> r; for (int i = 0; i < 2; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<2>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 3: { Rows<3> r; for (int i = 0; i < 3; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<3>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 4: { Rows<4> r; for (int i = 0; i < 4; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<4>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 5: { Rows<5> r; for (int i = 0; i < 5; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<5>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 6: { Rows<6> r; for (int i = 0; i < 6; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<6>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            case 7: { Rows<7> r; for (int i = 0; i < 7; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<7>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
            default: { Rows<8> r; for (int i = 0; i < 8; ++i) r.y[i] = p.yr[c0 + i]; gemv_mxfp4<8>(p.n, p.w.data(), p.bx, r, p.rows, st); } break;
        }
        left -= k; c0 += k;
    }
}

template <int R, int C>
static void run_new(const Problem & p, float * out) {
    gemm_mxfp4<R, C>(p.n, p.w.data(), p.bx, p.yr.data(), p.ny, p.rows, [&](int ix, int iy, float v) { out[iy*p.rows + ix] = v; });
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
        const int iters = std::max(4, (int)(2e8 / ((double)n*rows*ny)));
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
    const int nys[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
    printf("correctness (max rel err vs double, n=5120 rows=37)\n");
    for (int ny : nys) {
        Problem p(5120, 37, ny, 7 + ny);
        printf("  ny=%2d  old %.2e  new23 %.2e  new32 %.2e  new22 %.2e\n", ny, check(p, run_old), check(p, run_new<2, 3>), check(p, run_new<3, 2>), check(p, run_new<2, 2>));
    }
    const int shapes[2][2] = {{5120, 36}, {2304, 80}};
    for (auto & sh : shapes) {
        printf("GMAC/s, n=%d rows=%d, 1 thread | %d threads\n", sh[0], sh[1], nth);
        printf("  ny   old1   new23_1  new32_1  new22_1 |  old    new23   new32   new22\n");
        for (int ny : nys) {
            printf("  %2d  %6.2f  %6.2f  %6.2f  %6.2f | %6.1f  %6.1f  %6.1f  %6.1f\n", ny,
                timeit(sh[0], sh[1], ny, 1, run_old), timeit(sh[0], sh[1], ny, 1, run_new<2, 3>),
                timeit(sh[0], sh[1], ny, 1, run_new<3, 2>), timeit(sh[0], sh[1], ny, 1, run_new<2, 2>),
                timeit(sh[0], sh[1], ny, nth, run_old), timeit(sh[0], sh[1], ny, nth, run_new<2, 3>),
                timeit(sh[0], sh[1], ny, nth, run_new<3, 2>), timeit(sh[0], sh[1], ny, nth, run_new<2, 2>));
            fflush(stdout);
        }
    }
    return 0;
}
