// Correctness and bandwidth test for ik_llama/parts/iqk_avx1_gemv.h.
//   g++ -O3 -march=native -std=c++17 -pthread -I../ik_llama/parts gemv_bench.cpp -o gemv_bench
//   ./gemv_bench <q8_0|mxfp4|iq4_nl|q6_K> <ny> <threads per node> <nodes> [MiB per node] [n]
// Each node gets its own weight matrix (first-touched by a thread on that node) and threads pinned to its CPUs.
#include "iqk_avx1_gemv.h"

#include <pthread.h>
#include <sys/mman.h>
#include <sched.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace avx1_gemv;

static const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

enum { T_Q8_0, T_MXFP4, T_IQ4_NL, T_Q6_K };

static size_t row_bytes(int type, int n) {
    switch (type) {
        case T_Q8_0:   return (size_t)(n/32)*34;
        case T_MXFP4:  return (size_t)(n/32)*17;
        case T_IQ4_NL: return (size_t)(n/32)*18;
        default:       return (size_t)(n/256)*210;
    }
}

// no F16C on Sandy Bridge
static uint16_t f32_to_fp16(float f) {
    const uint16_t sign = std::signbit(f) ? 0x8000 : 0;
    const float a = std::fabs(f);
    if (a < 6.103515625e-05f) return sign | (uint16_t)std::lrint(a/5.9604644775390625e-08f);
    int e; const float m = std::frexp(a, &e);
    int E = e + 14, man = (int)std::lrint((2*m - 1)*1024);
    if (man == 1024) { man = 0; ++E; }
    return sign | (uint16_t)(E << 10) | (uint16_t)man;
}
static double fp16_ref(uint16_t h) {
    const int E = (h >> 10) & 31, man = h & 1023;
    const double v = E == 0 ? std::ldexp((double)man, -24) : std::ldexp(1024.0 + man, E - 25);
    return (h & 0x8000) ? -v : v;
}
static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16;
}

// activations -> q8_2_x4 the way ik lays them out
static void quantize_q8_2_x4(const float * x, char * out, int n) {
    const int nb = n/32, nb4 = 4*(nb/4);
    for (int i = 0; i < nb; ++i) {
        float amax = 0; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32*i+j]));
        uint16_t db = f32_to_bf16(amax/127.f);
        float d = bf16_to_f32(db), id = d > 0 ? 1/d : 0;
        int8_t q[32]; int sum = 0;
        for (int j = 0; j < 32; ++j) { q[j] = (int8_t)std::lrint(x[32*i+j]*id); sum += q[j]; }
        if (i < nb4) {
            blk_q8_2_x4 * y4 = (blk_q8_2_x4 *)out + i/4;
            y4->d[i%4] = db; ((int16_t *)y4->d)[i%4 + 4] = (int16_t)sum;
            memcpy(y4->qs + 32*(i%4), q, 32);
        } else {
            blk_q8_2 * y = (blk_q8_2 *)out + i;
            y->d = db; y->s = (uint16_t)(int16_t)sum; memcpy(y->qs, q, 32);
        }
    }
}

static std::vector<int> node_cpus(int node) {
    std::vector<int> cpus;
    char path[128]; snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE * f = fopen(path, "r"); if (!f) return cpus;
    char buf[1024] = {}; if (!fgets(buf, sizeof(buf), f)) buf[0] = 0; fclose(f);
    for (char * tok = strtok(buf, ",\n"); tok; tok = strtok(nullptr, ",\n")) {
        int a, b; if (sscanf(tok, "%d-%d", &a, &b) == 2) { for (int c = a; c <= b; ++c) cpus.push_back(c); }
        else if (sscanf(tok, "%d", &a) == 1) cpus.push_back(a);
    }
    return cpus;
}

static void pin(int cpu) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

static void get_q8(const char * y, int i, float & d, const int8_t *& q) {
    const int nb4 = i/4;
    (void)nb4;
    // works for both layouts because 4*sizeof(q8_2) == sizeof(q8_2_x4)
    if (true) {
        const blk_q8_2_x4 * y4 = (const blk_q8_2_x4 *)y + i/4;
        d = bf16_to_f32(y4->d[i%4]); q = y4->qs + 32*(i%4);
    }
}

static double ref_row(int type, int n, const char * xrow, const char * y) {
    const int nb = n/32, nb4 = 4*(nb/4);
    double acc = 0;
    for (int i = 0; i < nb; ++i) {
        float dy; const int8_t * qy;
        if (i < nb4) get_q8(y, i, dy, qy);
        else { const blk_q8_2 * b = (const blk_q8_2 *)y + i; dy = bf16_to_f32(b->d); qy = b->qs; }
        double s = 0;
        if (type == T_Q8_0) {
            const blk_q8_0 * b = (const blk_q8_0 *)xrow + i;
            double dx = fp16_ref(b->d);
            for (int j = 0; j < 32; ++j) s += b->qs[j]*qy[j];
            acc += dx*dy*s;
        } else if (type == T_IQ4_NL) {
            const blk_iq4_nl * b = (const blk_iq4_nl *)xrow + i;
            double dx = fp16_ref(b->d);
            for (int j = 0; j < 16; ++j) s += kvalues_iq4nl[b->qs[j] & 15]*qy[j] + kvalues_iq4nl[b->qs[j] >> 4]*qy[j+16];
            acc += dx*dy*s;
        } else if (type == T_Q6_K) {
            // weights of 32-block i straight from ggml's dequantize_row_q6_K layout
            const blk_q6_K * b = (const blk_q6_K *)xrow + i/8;
            const int half = (i%8)/4, k = i%4;
            const uint8_t * ql = b->ql + 64*half; const uint8_t * qh = b->qh + 32*half; const int8_t * sc = b->scales + 8*half;
            for (int j = 0; j < 32; ++j) {
                const int l = j;  // position inside the 32-block
                int q;
                switch (k) {
                    case 0: q = (ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4); break;
                    case 1: q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
                    case 2: q = (ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4); break;
                    default: q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); break;
                }
                s += (double)sc[2*k + l/16]*(q - 32)*qy[j];
            }
            acc += fp16_ref(b->d)*dy*s;
        } else {
            const blk_mxfp4 * b = (const blk_mxfp4 *)xrow + i;
            double dx = std::ldexp(1.0, (int)b->e - 128);
            for (int j = 0; j < 16; ++j) s += kvalues_mxfp4[b->qs[j] & 15]*qy[j] + kvalues_mxfp4[b->qs[j] >> 4]*qy[j+16];
            acc += dx*dy*s;
        }
    }
    return acc;
}

template <int ny>
static void run_kernel(int type, int n, const char * x, size_t bx, const char * const * y, float * out, int nrows) {
    Rows<ny> rows; for (int iy = 0; iy < ny; ++iy) rows.y[iy] = y[iy];
    auto store = [out, nrows](int ix, int iy, float v) { out[iy*nrows + ix] = v; };
    switch (type) {
        case T_Q8_0:   gemv_q8_0<ny>(n, x, bx, rows, nrows, store); break;
        case T_MXFP4:  gemv_mxfp4<ny>(n, x, bx, rows, nrows, store); break;
        case T_IQ4_NL: gemv_iq4_nl<ny>(n, x, bx, rows, nrows, store); break;
        default:       gemv_q6_K<ny>(n, x, bx, rows, nrows, store); break;
    }
}

static void dispatch(int ny, int type, int n, const char * x, size_t bx, const char * const * y, float * out, int nrows) {
    switch (ny) {
        case 1: run_kernel<1>(type, n, x, bx, y, out, nrows); break;
        case 2: run_kernel<2>(type, n, x, bx, y, out, nrows); break;
        case 3: run_kernel<3>(type, n, x, bx, y, out, nrows); break;
        case 4: run_kernel<4>(type, n, x, bx, y, out, nrows); break;
        case 8: run_kernel<8>(type, n, x, bx, y, out, nrows); break;
        default: fprintf(stderr, "ny %d not built\n", ny); exit(1);
    }
}

static void fill_rows(int type, char * x, size_t bx, int nrows, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int nb = n/32;
    for (int r = 0; r < nrows; ++r) {
        char * row = x + r*bx;
        if (type == T_Q6_K) {
            for (int i = 0; i < n/256; ++i) {
                blk_q6_K * b = (blk_q6_K *)row + i;
                for (auto & v : b->ql) v = (uint8_t)rng();
                for (auto & v : b->qh) v = (uint8_t)rng();
                for (auto & v : b->scales) v = (int8_t)((int)(rng() % 255) - 127);
                b->d = f32_to_fp16((rng() % 64 == 0) ? 2e-6f : 1e-5f*(1 + rng()%50));
            }
            continue;
        }
        for (int i = 0; i < nb; ++i) {
            if (type == T_IQ4_NL) {
                blk_iq4_nl * b = (blk_iq4_nl *)row + i;
                b->d = f32_to_fp16((rng() % 64 == 0) ? 3e-6f*(1 + rng()%7) : 1e-4f*(1 + rng()%50));
                for (int j = 0; j < 16; ++j) b->qs[j] = (uint8_t)rng();
            } else if (type == T_Q8_0) {
                blk_q8_0 * b = (blk_q8_0 *)row + i;
                // mostly normal scales, some fp16 denormals
                float d = (rng() % 64 == 0) ? 3e-6f*(1 + rng()%7) : 1e-4f*(1 + rng()%50);
                b->d = f32_to_fp16(d);
                for (int j = 0; j < 32; ++j) b->qs[j] = (int8_t)((int)(rng() % 255) - 127);
            } else {
                blk_mxfp4 * b = (blk_mxfp4 *)row + i;
                b->e = 110 + rng() % 20;
                for (int j = 0; j < 16; ++j) b->qs[j] = (uint8_t)rng();
            }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s q8_0|mxfp4 ny threads_per_node nodes [MiB] [n]\n", argv[0]); return 1; }
    const int type = !strcmp(argv[1], "q8_0") ? T_Q8_0 : !strcmp(argv[1], "mxfp4") ? T_MXFP4 :
                     !strcmp(argv[1], "iq4_nl") ? T_IQ4_NL : T_Q6_K;
    const int ny = atoi(argv[2]), tpn = atoi(argv[3]), nnodes = atoi(argv[4]);
    const size_t mib = argc > 5 ? atol(argv[5]) : 1024;
    const int n = argc > 6 ? atoi(argv[6]) : 4096;
    const size_t bx = row_bytes(type, n);
    const int nrows = (int)(mib*1024*1024/bx);

    // activations
    std::vector<std::vector<char>> ybuf(ny, std::vector<char>((n/32)*36 + 64));
    std::vector<const char *> yp(ny);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    for (int iy = 0; iy < ny; ++iy) {
        std::vector<float> f(n); for (auto & v : f) v = nd(rng);
        quantize_q8_2_x4(f.data(), ybuf[iy].data(), n);
        yp[iy] = ybuf[iy].data();
    }

    // correctness on a small matrix, including a tail (n not a multiple of 128)
    for (int nt : {n, type == T_Q6_K ? n + 256 : n + 96}) {
        const size_t bxt = row_bytes(type, nt);
        const int rows = 64;
        std::vector<char> xt(rows*bxt);
        fill_rows(type, xt.data(), bxt, rows, nt, 7);
        std::vector<std::vector<char>> yt(ny, std::vector<char>((nt/32)*36 + 64));
        std::vector<const char *> ytp(ny);
        for (int iy = 0; iy < ny; ++iy) {
            std::vector<float> f(nt); for (auto & v : f) v = nd(rng);
            quantize_q8_2_x4(f.data(), yt[iy].data(), nt); ytp[iy] = yt[iy].data();
        }
        std::vector<float> out(ny*rows);
        dispatch(ny, type, nt, xt.data(), bxt, ytp.data(), out.data(), rows);
        // error relative to the rms output: single outputs can cancel to near zero
        std::vector<double> ref(ny*rows); double rms = 0, maxerr = 0;
        for (int iy = 0; iy < ny; ++iy) for (int r = 0; r < rows; ++r) {
            ref[iy*rows + r] = ref_row(type, nt, xt.data() + r*bxt, ytp[iy]);
            rms += ref[iy*rows + r]*ref[iy*rows + r];
        }
        rms = std::sqrt(rms/(ny*rows));
        for (int k = 0; k < ny*rows; ++k) maxerr = std::max(maxerr, std::fabs(out[k] - ref[k]));
        printf("check n=%d: max err / rms %.2e %s\n", nt, maxerr/rms, maxerr/rms < 1e-5 ? "OK" : "FAIL");
    }

    // per-node weight buffers
    struct Node { char * x; int nrows; std::vector<int> cpus; };
    std::vector<Node> nodes(nnodes);
    for (int nd_ = 0; nd_ < nnodes; ++nd_) {
        nodes[nd_].x = (char *)mmap(nullptr, (size_t)nrows*bx, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        madvise(nodes[nd_].x, (size_t)nrows*bx, MADV_NOHUGEPAGE);
        nodes[nd_].nrows = nrows;
        nodes[nd_].cpus = node_cpus(nd_);
    }
    // fill in parallel on the owning node
    {
        std::vector<std::thread> th;
        for (int nd_ = 0; nd_ < nnodes; ++nd_) th.emplace_back([&, nd_] {
            pin(nodes[nd_].cpus[0]);
            fill_rows(type, nodes[nd_].x, bx, nodes[nd_].nrows, n, 100 + nd_);
        });
        for (auto & t : th) t.join();
    }

    const int reps = 5;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<double> secs(nnodes*tpn);
    std::vector<std::thread> th;
    for (int nd_ = 0; nd_ < nnodes; ++nd_) for (int t = 0; t < tpn; ++t) {
        th.emplace_back([&, nd_, t] {
            pin(nodes[nd_].cpus[t % nodes[nd_].cpus.size()]);
            const int r0 = (int)((int64_t)nodes[nd_].nrows*t/tpn), r1 = (int)((int64_t)nodes[nd_].nrows*(t+1)/tpn);
            std::vector<float> out(ny*(r1 - r0));
            ready++;
            while (!go.load()) {}
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r)
                dispatch(ny, type, n, nodes[nd_].x + r0*bx, bx, yp.data(), out.data(), r1 - r0);
            secs[nd_*tpn + t] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        });
    }
    while (ready.load() < nnodes*tpn) {}
    auto t0 = std::chrono::steady_clock::now();
    go = true;
    for (auto & t : th) t.join();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double bytes = (double)nrows*bx*nnodes*reps;
    printf("%s ny=%d threads/node=%d nodes=%d n=%d: %.1f GB/s total (%.1f GB/s per node)\n",
           argv[1], ny, tpn, nnodes, n, bytes/wall*1e-9, bytes/wall*1e-9/nnodes);
    return 0;
}
