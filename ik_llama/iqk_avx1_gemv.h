//
// Native 128-bit matrix x small-batch kernels for Sandy Bridge (AVX1, no AVX2/FMA/F16C).
//
// Weights: Q8_0 (fp16 scale + 32 int8) and MXFP4 (e8m0 scale + 32 fp4 nibbles).
// Activations: Q8_2_X4 rows as produced by ik's quantize_row_q8_2_x4 (groups of four
// blocks with 4 bf16 scales, 4 int16 block sums, 128 int8 quants; a remainder of
// n/32 % 4 blocks is stored as plain block_q8_2).
//
// The SIMDe-emulated AVX2 kernels split each 256-bit integer op into two 128-bit
// halves and move data between ymm halves, and they convert fp16 scales with a
// scalar fallback, so on this CPU they ran at about half of the memory bandwidth.
// These keep everything in xmm registers, reduce four block sums with one transpose
// and convert scales with plain integer ops.
//
#pragma once

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

namespace avx1_gemv {

struct blk_q8_2_x4 { uint16_t d[8]; int8_t qs[128]; };
struct blk_q8_2    { uint16_t d; uint16_t s; int8_t qs[32]; };
struct blk_q8_0    { uint16_t d; int8_t qs[32]; };
struct blk_mxfp4   { uint8_t e; uint8_t qs[16]; };
static_assert(sizeof(blk_q8_2_x4) == 144, "q8_2_x4");
static_assert(sizeof(blk_q8_2) == 36, "q8_2");
static_assert(sizeof(blk_q8_0) == 34, "q8_0");
static_assert(sizeof(blk_mxfp4) == 17, "mxfp4");

// [sum(p0), sum(p1), sum(p2), sum(p3)]
static inline __m128i sum4_epi32(__m128i p0, __m128i p1, __m128i p2, __m128i p3) {
    const __m128i p01 = _mm_add_epi32(_mm_unpacklo_epi32(p0, p1), _mm_unpackhi_epi32(p0, p1));
    const __m128i p23 = _mm_add_epi32(_mm_unpacklo_epi32(p2, p3), _mm_unpackhi_epi32(p2, p3));
    return _mm_add_epi32(_mm_unpacklo_epi64(p01, p23), _mm_unpackhi_epi64(p01, p23));
}

static inline float hsum_ps(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_movehdup_ps(v));
    return _mm_cvtss_f32(v);
}

// four fp16 values in the low halves of the 32-bit lanes -> fp32, denormals included
static inline __m128 fp16x4_to_f32(__m128i h) {
    const __m128i sign = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16);
    h = _mm_and_si128(h, _mm_set1_epi32(0x7fff));
    const __m128 fn = _mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(h, 13), _mm_set1_epi32(112 << 23)));
    const __m128 fd = _mm_mul_ps(_mm_cvtepi32_ps(h), _mm_set1_ps(5.9604644775390625e-08f)); // 2^-24
    const __m128 f  = _mm_blendv_ps(fn, fd, _mm_castsi128_ps(_mm_cmplt_epi32(h, _mm_set1_epi32(0x0400))));
    return _mm_or_ps(f, _mm_castsi128_ps(sign));
}

static inline float fp16_to_f32(uint16_t h) {
    return _mm_cvtss_f32(fp16x4_to_f32(_mm_cvtsi32_si128(h)));
}

// four bf16 scales -> fp32
static inline __m128 bf16x4_to_f32(const uint16_t * d) {
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)d)), 16));
}

static inline float bf16_to_f32(uint16_t d) {
    uint32_t u = (uint32_t)d << 16; float f; memcpy(&f, &u, 4); return f;
}

// signed x signed int8 dot of 32 values, as 4 int32 partial sums
static inline __m128i dot32_ss(__m128i x0, __m128i x1, const int8_t * y) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i y0 = _mm_loadu_si128((const __m128i *)y);
    const __m128i y1 = _mm_loadu_si128((const __m128i *)(y + 16));
    const __m128i s0 = _mm_maddubs_epi16(_mm_abs_epi8(x0), _mm_sign_epi8(y0, x0));
    const __m128i s1 = _mm_maddubs_epi16(_mm_abs_epi8(x1), _mm_sign_epi8(y1, x1));
    return _mm_add_epi32(_mm_madd_epi16(s0, ones), _mm_madd_epi16(s1, ones));
}

// unsigned (<= 24) x signed int8 dot of 32 values, as 4 int32 partial sums
static inline __m128i dot32_us(__m128i u0, __m128i u1, const int8_t * y) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i y0 = _mm_loadu_si128((const __m128i *)y);
    const __m128i y1 = _mm_loadu_si128((const __m128i *)(y + 16));
    return _mm_madd_epi16(_mm_add_epi16(_mm_maddubs_epi16(u0, y0), _mm_maddubs_epi16(u1, y1)), ones);
}

template <int ny>
struct Rows {
    const char * y[ny];
};

//
// Q8_0 weights. Each output: sum_b dx_b * dy_b * sum_i x_bi * y_bi
//
template <int ny, typename Store>
static void gemv_q8_0(int n, const void * vx, size_t bx, const Rows<ny> & rows, int nrc_x, Store && store) {
    const int nb = n / 32;
    const int ng = nb / 4;
    for (int ix = 0; ix < nrc_x; ++ix) {
        const blk_q8_0 * x = (const blk_q8_0 *)((const char *)vx + ix*bx);
        __m128 acc[ny];
        for (int iy = 0; iy < ny; ++iy) acc[iy] = _mm_setzero_ps();
        for (int g = 0; g < ng; ++g) {
            const blk_q8_0 * xb = x + 4*g;
            __m128i h = _mm_cvtsi32_si128(xb[0].d);
            h = _mm_insert_epi16(h, xb[1].d, 2);
            h = _mm_insert_epi16(h, xb[2].d, 4);
            h = _mm_insert_epi16(h, xb[3].d, 6);
            const __m128 dx = fp16x4_to_f32(h);
            __m128i p[ny][4];
            for (int k = 0; k < 4; ++k) {
                const __m128i x0 = _mm_loadu_si128((const __m128i *)xb[k].qs);
                const __m128i x1 = _mm_loadu_si128((const __m128i *)(xb[k].qs + 16));
                for (int iy = 0; iy < ny; ++iy) {
                    const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)rows.y[iy] + g;
                    p[iy][k] = dot32_ss(x0, x1, yb->qs + 32*k);
                }
            }
            for (int iy = 0; iy < ny; ++iy) {
                const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)rows.y[iy] + g;
                const __m128 d = _mm_mul_ps(dx, bf16x4_to_f32(yb->d));
                const __m128i s = sum4_epi32(p[iy][0], p[iy][1], p[iy][2], p[iy][3]);
                acc[iy] = _mm_add_ps(acc[iy], _mm_mul_ps(d, _mm_cvtepi32_ps(s)));
            }
        }
        float tail[ny] = {};
        for (int i = 4*ng; i < nb; ++i) {
            const __m128i x0 = _mm_loadu_si128((const __m128i *)x[i].qs);
            const __m128i x1 = _mm_loadu_si128((const __m128i *)(x[i].qs + 16));
            const float dx = fp16_to_f32(x[i].d);
            for (int iy = 0; iy < ny; ++iy) {
                const blk_q8_2 * yb = (const blk_q8_2 *)rows.y[iy] + i;
                const __m128i s = dot32_ss(x0, x1, yb->qs);
                tail[iy] += dx*bf16_to_f32(yb->d)*(float)_mm_cvtsi128_si32(sum4_epi32(s, s, s, s));
            }
        }
        for (int iy = 0; iy < ny; ++iy) store(ix, iy, hsum_ps(acc[iy]) + tail[iy]);
    }
}

//
// MXFP4 weights. Nibbles index kvalues_mxfp4 (twice the E2M1 values), shifted by +12 to make
// them unsigned for pmaddubsw; the shift is undone with the int16 block sums of the activations.
// Scale 2^(e-127) is built directly from the exponent byte; the extra 1/2 is applied at the end.
//
template <int ny, typename Store>
static void gemv_mxfp4(int n, const void * vx, size_t bx, const Rows<ny> & rows, int nrc_x, Store && store) {
    const int nb = n / 32;
    const int ng = nb / 4;
    const __m128i values = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m128i twelve = _mm_set1_epi32(12);
    for (int ix = 0; ix < nrc_x; ++ix) {
        const blk_mxfp4 * x = (const blk_mxfp4 *)((const char *)vx + ix*bx);
        __m128 acc[ny];
        for (int iy = 0; iy < ny; ++iy) acc[iy] = _mm_setzero_ps();
        for (int g = 0; g < ng; ++g) {
            const blk_mxfp4 * xb = x + 4*g;
            const uint32_t e4 = (uint32_t)xb[0].e | ((uint32_t)xb[1].e << 8) | ((uint32_t)xb[2].e << 16) | ((uint32_t)xb[3].e << 24);
            const __m128 dx = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(e4)), 23));
            __m128i p[ny][4];
            for (int k = 0; k < 4; ++k) {
                const __m128i q  = _mm_loadu_si128((const __m128i *)xb[k].qs);
                const __m128i u0 = _mm_shuffle_epi8(values, _mm_and_si128(q, m4));
                const __m128i u1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
                for (int iy = 0; iy < ny; ++iy) {
                    const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)rows.y[iy] + g;
                    p[iy][k] = dot32_us(u0, u1, yb->qs + 32*k);
                }
            }
            for (int iy = 0; iy < ny; ++iy) {
                const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)rows.y[iy] + g;
                const __m128i ys = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(yb->d + 4)));
                const __m128i s  = _mm_sub_epi32(sum4_epi32(p[iy][0], p[iy][1], p[iy][2], p[iy][3]), _mm_mullo_epi32(ys, twelve));
                const __m128  d  = _mm_mul_ps(dx, bf16x4_to_f32(yb->d));
                acc[iy] = _mm_add_ps(acc[iy], _mm_mul_ps(d, _mm_cvtepi32_ps(s)));
            }
        }
        float tail[ny] = {};
        for (int i = 4*ng; i < nb; ++i) {
            const __m128i q  = _mm_loadu_si128((const __m128i *)x[i].qs);
            const __m128i u0 = _mm_shuffle_epi8(values, _mm_and_si128(q, m4));
            const __m128i u1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
            const uint32_t eb = (uint32_t)x[i].e << 23; float dx; memcpy(&dx, &eb, 4);
            for (int iy = 0; iy < ny; ++iy) {
                const blk_q8_2 * yb = (const blk_q8_2 *)rows.y[iy] + i;
                const __m128i s = dot32_us(u0, u1, yb->qs);
                const int sum = _mm_cvtsi128_si32(sum4_epi32(s, s, s, s)) - 12*(int16_t)yb->s;
                tail[iy] += dx*bf16_to_f32(yb->d)*(float)sum;
            }
        }
        for (int iy = 0; iy < ny; ++iy) store(ix, iy, 0.5f*(hsum_ps(acc[iy]) + tail[iy]));
    }
}

} // namespace avx1_gemv
