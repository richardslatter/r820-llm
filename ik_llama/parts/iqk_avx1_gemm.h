//
// Batched MXFP4 x Q8_2_X4 for prompt processing on Sandy Bridge (AVX1).
//
// In a prompt batch each expert sees a handful to a few dozen tokens, so the gemv kernels
// re-unpack every weight block for each 8 columns, transpose four block sums per row and
// column, and spill their accumulators. Here two weight rows and all activation columns are
// first unpacked into int8 scratch with the four blocks of a 128-value group interleaved in
// 4-byte lanes: vector k holds values 4k..4k+3 (k < 4) or 16+4(k-4).. (k >= 4) of blocks 0..3.
// pmaddubsw + pmaddwd then leave the dot of block b in lane b, with no transpose, and the
// unpack, the scales and the block-sum correction are shared by all rows and columns.
//
#pragma once

#include "iqk_avx1_gemv.h"

#include <vector>

namespace avx1_gemv {

struct alignas(16) GemmAct { int8_t q[128]; float d[4]; int32_t s12[4]; };
struct alignas(16) GemmW   { uint8_t q[128]; float d[4]; };
static_assert(sizeof(GemmAct) == 160, "GemmAct");
static_assert(sizeof(GemmW) == 144, "GemmW");

static inline void transpose4_epi32(__m128i & r0, __m128i & r1, __m128i & r2, __m128i & r3) {
    const __m128i t0 = _mm_unpacklo_epi32(r0, r1), t1 = _mm_unpacklo_epi32(r2, r3);
    const __m128i t2 = _mm_unpackhi_epi32(r0, r1), t3 = _mm_unpackhi_epi32(r2, r3);
    r0 = _mm_unpacklo_epi64(t0, t1); r1 = _mm_unpackhi_epi64(t0, t1);
    r2 = _mm_unpacklo_epi64(t2, t3); r3 = _mm_unpackhi_epi64(t2, t3);
}

// One Q8_2_X4 activation row -> interleaved quants, fp32 scales and 12 x block sums.
static inline void gemm_prep_act(int ng, const char * y, GemmAct * out) {
    const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)y;
    const __m128i twelve = _mm_set1_epi32(12);
    for (int g = 0; g < ng; ++g) {
        const int8_t * q = yb[g].qs;
        __m128i a0 = _mm_loadu_si128((const __m128i *)(q +  0)), a1 = _mm_loadu_si128((const __m128i *)(q + 32));
        __m128i a2 = _mm_loadu_si128((const __m128i *)(q + 64)), a3 = _mm_loadu_si128((const __m128i *)(q + 96));
        __m128i b0 = _mm_loadu_si128((const __m128i *)(q + 16)), b1 = _mm_loadu_si128((const __m128i *)(q + 48));
        __m128i b2 = _mm_loadu_si128((const __m128i *)(q + 80)), b3 = _mm_loadu_si128((const __m128i *)(q +112));
        transpose4_epi32(a0, a1, a2, a3);
        transpose4_epi32(b0, b1, b2, b3);
        __m128i * o = (__m128i *)out[g].q;
        _mm_store_si128(o + 0, a0); _mm_store_si128(o + 1, a1); _mm_store_si128(o + 2, a2); _mm_store_si128(o + 3, a3);
        _mm_store_si128(o + 4, b0); _mm_store_si128(o + 5, b1); _mm_store_si128(o + 6, b2); _mm_store_si128(o + 7, b3);
        _mm_store_ps(out[g].d, bf16x4_to_f32(yb[g].d));
        const __m128i ys = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(yb[g].d + 4)));
        _mm_store_si128((__m128i *)out[g].s12, _mm_mullo_epi32(ys, twelve));
    }
}

// One MXFP4 row -> interleaved unsigned values (twice E2M1, +12) and fp32 scales (with the 1/2).
static inline void gemm_prep_mxfp4(int ng, const blk_mxfp4 * x, GemmW * out) {
    const __m128i values = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128 half = _mm_set1_ps(0.5f);
    for (int g = 0; g < ng; ++g) {
        const blk_mxfp4 * xb = x + 4*g;
        __m128i q0 = _mm_loadu_si128((const __m128i *)xb[0].qs), q1 = _mm_loadu_si128((const __m128i *)xb[1].qs);
        __m128i q2 = _mm_loadu_si128((const __m128i *)xb[2].qs), q3 = _mm_loadu_si128((const __m128i *)xb[3].qs);
        transpose4_epi32(q0, q1, q2, q3);
        __m128i * o = (__m128i *)out[g].q;
        _mm_store_si128(o + 0, _mm_shuffle_epi8(values, _mm_and_si128(q0, m4)));
        _mm_store_si128(o + 1, _mm_shuffle_epi8(values, _mm_and_si128(q1, m4)));
        _mm_store_si128(o + 2, _mm_shuffle_epi8(values, _mm_and_si128(q2, m4)));
        _mm_store_si128(o + 3, _mm_shuffle_epi8(values, _mm_and_si128(q3, m4)));
        _mm_store_si128(o + 4, _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q0, 4), m4)));
        _mm_store_si128(o + 5, _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q1, 4), m4)));
        _mm_store_si128(o + 6, _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q2, 4), m4)));
        _mm_store_si128(o + 7, _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q3, 4), m4)));
        const uint32_t e4 = (uint32_t)xb[0].e | ((uint32_t)xb[1].e << 8) | ((uint32_t)xb[2].e << 16) | ((uint32_t)xb[3].e << 24);
        const __m128 dx = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(e4)), 23));
        _mm_store_ps(out[g].d, _mm_mul_ps(dx, half));
    }
}

// R rows x C columns over all groups. Four pmaddubsw per int16 chain: 4 x 2 x 24 x 127 < 2^15.
template <int R, int C>
static inline void gemm_tile(int ng, const GemmW * const * w, const GemmAct * const * a, float * out /*[R*C]*/) {
    const __m128i ones = _mm_set1_epi16(1);
    __m128 acc[R][C];
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) acc[r][c] = _mm_setzero_ps();
    for (int g = 0; g < ng; ++g) {
        __m128i s[R][C];
        for (int half = 0; half < 2; ++half) {
            __m128i h[R][C];
            for (int k = 0; k < 4; ++k) {
                const int kk = 4*half + k;
                for (int r = 0; r < R; ++r) {
                    const __m128i wv = _mm_load_si128((const __m128i *)w[r][g].q + kk);
                    for (int c = 0; c < C; ++c) {
                        const __m128i p = _mm_maddubs_epi16(wv, _mm_load_si128((const __m128i *)a[c][g].q + kk));
                        h[r][c] = k ? _mm_add_epi16(h[r][c], p) : p;
                    }
                }
            }
            for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) {
                const __m128i m = _mm_madd_epi16(h[r][c], ones);
                s[r][c] = half ? _mm_add_epi32(s[r][c], m) : m;
            }
        }
        for (int r = 0; r < R; ++r) {
            const __m128 dx = _mm_load_ps(w[r][g].d);
            for (int c = 0; c < C; ++c) {
                const __m128i v = _mm_sub_epi32(s[r][c], _mm_load_si128((const __m128i *)a[c][g].s12));
                const __m128 d = _mm_mul_ps(dx, _mm_load_ps(a[c][g].d));
                acc[r][c] = _mm_add_ps(acc[r][c], _mm_mul_ps(d, _mm_cvtepi32_ps(v)));
            }
        }
    }
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) out[r*C + c] = hsum_ps(acc[r][c]);
}

struct GemmScratch {
    std::vector<GemmAct> act;
    std::vector<GemmW> w;
};
static inline GemmScratch & gemm_scratch() {
    thread_local GemmScratch s;
    return s;
}

template <int R, int C, typename Store>
static inline void gemm_rows(int ng, const GemmW * const * w, const GemmAct * act, int ny, int ix, Store && store) {
    float out[R*C];
    const GemmAct * a[C];
    int iy = 0;
    for (; iy + C <= ny; iy += C) {
        for (int c = 0; c < C; ++c) a[c] = act + (size_t)(iy + c)*ng;
        gemm_tile<R, C>(ng, w, a, out);
        for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) store(ix + r, iy + c, out[r*C + c]);
    }
    if constexpr (C > 1) {
        if (iy < ny) {
            if constexpr (C > 2) { if (ny - iy >= 2) { gemm_rows<R, 2>(ng, w, act + (size_t)iy*ng, ny - iy, ix, [&](int x, int y, float v) { store(x, iy + y, v); }); return; } }
            gemm_rows<R, 1>(ng, w, act + (size_t)iy*ng, ny - iy, ix, [&](int x, int y, float v) { store(x, iy + y, v); });
        }
    }
}

// n must be a multiple of 128 (whole Q8_2_X4 groups).
template <int R = 2, int C = 3, typename Store>
static void gemm_mxfp4(int n, const void * vx, size_t bx, const char * const * y, int ny, int nrc_x, Store && store) {
    const int ng = n / 128;
    auto & sc = gemm_scratch();
    if (sc.act.size() < (size_t)ny*ng) sc.act.resize((size_t)ny*ng);
    if (sc.w.size() < (size_t)R*ng) sc.w.resize((size_t)R*ng);
    for (int iy = 0; iy < ny; ++iy) gemm_prep_act(ng, y[iy], sc.act.data() + (size_t)iy*ng);
    const GemmW * w[R];
    for (int r = 0; r < R; ++r) w[r] = sc.w.data() + (size_t)r*ng;
    int ix = 0;
    for (; ix + R <= nrc_x; ix += R) {
        for (int r = 0; r < R; ++r) gemm_prep_mxfp4(ng, (const blk_mxfp4 *)((const char *)vx + (ix + r)*bx), sc.w.data() + (size_t)r*ng);
        gemm_rows<R, C>(ng, w, sc.act.data(), ny, ix, store);
    }
    for (; ix < nrc_x; ++ix) {
        gemm_prep_mxfp4(ng, (const blk_mxfp4 *)((const char *)vx + ix*bx), sc.w.data());
        gemm_rows<1, C>(ng, w, sc.act.data(), ny, ix, store);
    }
}

//
// Q6_K, same layout: each 128-weight half of a superblock is one group, its four 32-weight runs
// sit in the four lanes, and the two 16-weight sub-blocks of run b are vectors 0..3 and 4..7.
// The quants stay unsigned (0..63). Four pmaddubsw per int16 chain can then exceed 2^15, but
// paddw wraps, and once 32 x (the matching activation pair sums) comes off with psubw the true
// value, sum (q - 32) * y over 8 weights, is at most 8 * 32 * 127 = 32512: exact modulo 2^16.
// pmaddwd applies the sub-block scales, so the multiply count per group is the MXFP4 one.
//
struct alignas(16) GemmActQ6 { int8_t q[128]; int16_t c[16]; float d[4]; };
struct alignas(16) GemmWQ6   { uint8_t q[128]; int16_t sc[16]; float d[4]; };
static_assert(sizeof(GemmActQ6) == 176, "GemmActQ6");
static_assert(sizeof(GemmWQ6) == 176, "GemmWQ6");

static inline void gemm_prep_act_q6(int ng, const char * y, GemmActQ6 * out) {
    const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)y;
    const __m128i k32 = _mm_set1_epi8(32);
    for (int g = 0; g < ng; ++g) {
        const int8_t * q = yb[g].qs;
        __m128i a0 = _mm_loadu_si128((const __m128i *)(q +  0)), a1 = _mm_loadu_si128((const __m128i *)(q + 32));
        __m128i a2 = _mm_loadu_si128((const __m128i *)(q + 64)), a3 = _mm_loadu_si128((const __m128i *)(q + 96));
        __m128i b0 = _mm_loadu_si128((const __m128i *)(q + 16)), b1 = _mm_loadu_si128((const __m128i *)(q + 48));
        __m128i b2 = _mm_loadu_si128((const __m128i *)(q + 80)), b3 = _mm_loadu_si128((const __m128i *)(q +112));
        transpose4_epi32(a0, a1, a2, a3);
        transpose4_epi32(b0, b1, b2, b3);
        __m128i * o = (__m128i *)out[g].q;
        _mm_store_si128(o + 0, a0); _mm_store_si128(o + 1, a1); _mm_store_si128(o + 2, a2); _mm_store_si128(o + 3, a3);
        _mm_store_si128(o + 4, b0); _mm_store_si128(o + 5, b1); _mm_store_si128(o + 6, b2); _mm_store_si128(o + 7, b3);
        // 32 x pair sums, laid out as the products they correct
        const __m128i clo = _mm_add_epi16(_mm_add_epi16(_mm_maddubs_epi16(k32, a0), _mm_maddubs_epi16(k32, a1)),
                                          _mm_add_epi16(_mm_maddubs_epi16(k32, a2), _mm_maddubs_epi16(k32, a3)));
        const __m128i chi = _mm_add_epi16(_mm_add_epi16(_mm_maddubs_epi16(k32, b0), _mm_maddubs_epi16(k32, b1)),
                                          _mm_add_epi16(_mm_maddubs_epi16(k32, b2), _mm_maddubs_epi16(k32, b3)));
        _mm_store_si128((__m128i *)out[g].c + 0, clo);
        _mm_store_si128((__m128i *)out[g].c + 1, chi);
        _mm_store_ps(out[g].d, bf16x4_to_f32(yb[g].d));
    }
}

static inline void gemm_prep_q6_K(int nsb, const blk_q6_K * x, GemmWQ6 * out) {
    const __m128i m4  = _mm_set1_epi8(0x0f);
    const __m128i m30 = _mm_set1_epi8(0x30);
    // int16 lanes 2b, 2b+1 <- scale 2b (low sub-block of run b) or 2b+1 (high)
    const __m128i lo_idx = _mm_setr_epi8(0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13);
    const __m128i hi_idx = _mm_setr_epi8(2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15);
    for (int ib = 0; ib < nsb; ++ib) {
        const blk_q6_K * xb = x + ib;
        const __m128 dx = _mm_set1_ps(fp16_to_f32(xb->d));
        for (int half = 0; half < 2; ++half) {
            const uint8_t * ql = xb->ql + 64*half;
            const uint8_t * qh = xb->qh + 32*half;
            const __m128i l0 = _mm_loadu_si128((const __m128i *)(ql +  0));
            const __m128i l1 = _mm_loadu_si128((const __m128i *)(ql + 16));
            const __m128i l2 = _mm_loadu_si128((const __m128i *)(ql + 32));
            const __m128i l3 = _mm_loadu_si128((const __m128i *)(ql + 48));
            const __m128i h0 = _mm_loadu_si128((const __m128i *)(qh +  0));
            const __m128i h1 = _mm_loadu_si128((const __m128i *)(qh + 16));
            // sub-block j (16 weights) of this half, unsigned
            __m128i x0 = _mm_or_si128(_mm_and_si128(l0, m4), _mm_and_si128(_mm_slli_epi16(h0, 4), m30));
            __m128i x1 = _mm_or_si128(_mm_and_si128(l1, m4), _mm_and_si128(_mm_slli_epi16(h1, 4), m30));
            __m128i x2 = _mm_or_si128(_mm_and_si128(l2, m4), _mm_and_si128(_mm_slli_epi16(h0, 2), m30));
            __m128i x3 = _mm_or_si128(_mm_and_si128(l3, m4), _mm_and_si128(_mm_slli_epi16(h1, 2), m30));
            __m128i x4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l0, 4), m4), _mm_and_si128(h0, m30));
            __m128i x5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l1, 4), m4), _mm_and_si128(h1, m30));
            __m128i x6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l2, 4), m4), _mm_and_si128(_mm_srli_epi16(h0, 2), m30));
            __m128i x7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l3, 4), m4), _mm_and_si128(_mm_srli_epi16(h1, 2), m30));
            transpose4_epi32(x0, x2, x4, x6);
            transpose4_epi32(x1, x3, x5, x7);
            GemmWQ6 & o = out[2*ib + half];
            __m128i * oq = (__m128i *)o.q;
            _mm_store_si128(oq + 0, x0); _mm_store_si128(oq + 1, x2); _mm_store_si128(oq + 2, x4); _mm_store_si128(oq + 3, x6);
            _mm_store_si128(oq + 4, x1); _mm_store_si128(oq + 5, x3); _mm_store_si128(oq + 6, x5); _mm_store_si128(oq + 7, x7);
            const __m128i s16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)(xb->scales + 8*half)));
            _mm_store_si128((__m128i *)o.sc + 0, _mm_shuffle_epi8(s16, lo_idx));
            _mm_store_si128((__m128i *)o.sc + 1, _mm_shuffle_epi8(s16, hi_idx));
            _mm_store_ps(o.d, dx);
        }
    }
}

template <int R, int C>
static inline void gemm_tile_q6(int ng, const GemmWQ6 * const * w, const GemmActQ6 * const * a, float * out /*[R*C]*/) {
    __m128 acc[R][C];
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) acc[r][c] = _mm_setzero_ps();
    for (int g = 0; g < ng; ++g) {
        __m128i s[R][C];
        for (int half = 0; half < 2; ++half) {
            __m128i h[R][C];
            for (int k = 0; k < 4; ++k) {
                const int kk = 4*half + k;
                for (int r = 0; r < R; ++r) {
                    const __m128i wv = _mm_load_si128((const __m128i *)w[r][g].q + kk);
                    for (int c = 0; c < C; ++c) {
                        const __m128i p = _mm_maddubs_epi16(wv, _mm_load_si128((const __m128i *)a[c][g].q + kk));
                        h[r][c] = k ? _mm_add_epi16(h[r][c], p) : p;
                    }
                }
            }
            for (int r = 0; r < R; ++r) {
                const __m128i scv = _mm_load_si128((const __m128i *)w[r][g].sc + half);
                for (int c = 0; c < C; ++c) {
                    const __m128i v = _mm_sub_epi16(h[r][c], _mm_load_si128((const __m128i *)a[c][g].c + half));
                    const __m128i m = _mm_madd_epi16(v, scv);
                    s[r][c] = half ? _mm_add_epi32(s[r][c], m) : m;
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            const __m128 dx = _mm_load_ps(w[r][g].d);
            for (int c = 0; c < C; ++c) {
                const __m128 d = _mm_mul_ps(dx, _mm_load_ps(a[c][g].d));
                acc[r][c] = _mm_add_ps(acc[r][c], _mm_mul_ps(d, _mm_cvtepi32_ps(s[r][c])));
            }
        }
    }
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) out[r*C + c] = hsum_ps(acc[r][c]);
}

struct GemmScratchQ6 {
    std::vector<GemmActQ6> act;
    std::vector<GemmWQ6> w;
};
static inline GemmScratchQ6 & gemm_scratch_q6() {
    thread_local GemmScratchQ6 s;
    return s;
}

template <int R, int C, typename Store>
static inline void gemm_rows_q6(int ng, const GemmWQ6 * const * w, const GemmActQ6 * act, int ny, int ix, Store && store) {
    float out[R*C];
    const GemmActQ6 * a[C];
    int iy = 0;
    for (; iy + C <= ny; iy += C) {
        for (int c = 0; c < C; ++c) a[c] = act + (size_t)(iy + c)*ng;
        gemm_tile_q6<R, C>(ng, w, a, out);
        for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) store(ix + r, iy + c, out[r*C + c]);
    }
    if constexpr (C > 1) {
        if (iy < ny) {
            if constexpr (C > 2) { if (ny - iy >= 2) { gemm_rows_q6<R, 2>(ng, w, act + (size_t)iy*ng, ny - iy, ix, [&](int x, int y, float v) { store(x, iy + y, v); }); return; } }
            gemm_rows_q6<R, 1>(ng, w, act + (size_t)iy*ng, ny - iy, ix, [&](int x, int y, float v) { store(x, iy + y, v); });
        }
    }
}

// n must be a multiple of 256 (whole superblocks; Q6_K rows always are).
template <int R = 2, int C = 2, typename Store>
static void gemm_q6_K(int n, const void * vx, size_t bx, const char * const * y, int ny, int nrc_x, Store && store) {
    const int nsb = n / 256, ng = 2*nsb;
    auto & sc = gemm_scratch_q6();
    if (sc.act.size() < (size_t)ny*ng) sc.act.resize((size_t)ny*ng);
    if (sc.w.size() < (size_t)R*ng) sc.w.resize((size_t)R*ng);
    for (int iy = 0; iy < ny; ++iy) gemm_prep_act_q6(ng, y[iy], sc.act.data() + (size_t)iy*ng);
    const GemmWQ6 * w[R];
    for (int r = 0; r < R; ++r) w[r] = sc.w.data() + (size_t)r*ng;
    int ix = 0;
    for (; ix + R <= nrc_x; ix += R) {
        for (int r = 0; r < R; ++r) gemm_prep_q6_K(nsb, (const blk_q6_K *)((const char *)vx + (ix + r)*bx), sc.w.data() + (size_t)r*ng);
        gemm_rows_q6<R, C>(ng, w, sc.act.data(), ny, ix, store);
    }
    for (; ix < nrc_x; ++ix) {
        gemm_prep_q6_K(nsb, (const blk_q6_K *)((const char *)vx + ix*bx), sc.w.data());
        gemm_rows_q6<1, C>(ng, w, sc.act.data(), ny, ix, store);
    }
}

}
