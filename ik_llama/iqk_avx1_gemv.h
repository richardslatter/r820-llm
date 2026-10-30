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

//
// IQ4_NL weights: fp16 scale + 32 nibbles indexing a signed non-linear table. The table
// spans -127..113, too wide for the unsigned-offset trick, so the signed dot is used.
//
struct blk_iq4_nl { uint16_t d; uint8_t qs[16]; };
static_assert(sizeof(blk_iq4_nl) == 18, "iq4_nl");

template <int ny, typename Store>
static void gemv_iq4_nl(int n, const void * vx, size_t bx, const Rows<ny> & rows, int nrc_x, Store && store) {
    const int nb = n / 32;
    const int ng = nb / 4;
    const __m128i values = _mm_setr_epi8(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int ix = 0; ix < nrc_x; ++ix) {
        const blk_iq4_nl * x = (const blk_iq4_nl *)((const char *)vx + ix*bx);
        __m128 acc[ny];
        for (int iy = 0; iy < ny; ++iy) acc[iy] = _mm_setzero_ps();
        for (int g = 0; g < ng; ++g) {
            const blk_iq4_nl * xb = x + 4*g;
            __m128i h = _mm_cvtsi32_si128(xb[0].d);
            h = _mm_insert_epi16(h, xb[1].d, 2);
            h = _mm_insert_epi16(h, xb[2].d, 4);
            h = _mm_insert_epi16(h, xb[3].d, 6);
            const __m128 dx = fp16x4_to_f32(h);
            __m128i p[ny][4];
            for (int k = 0; k < 4; ++k) {
                const __m128i q  = _mm_loadu_si128((const __m128i *)xb[k].qs);
                const __m128i x0 = _mm_shuffle_epi8(values, _mm_and_si128(q, m4));
                const __m128i x1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
                for (int iy = 0; iy < ny; ++iy) {
                    const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)rows.y[iy] + g;
                    p[iy][k] = dot32_ss(x0, x1, yb->qs + 32*k);
                }
            }
            for (int iy = 0; iy < ny; ++iy) {
                const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)rows.y[iy] + g;
                const __m128 d = _mm_mul_ps(dx, bf16x4_to_f32(yb->d));
                acc[iy] = _mm_add_ps(acc[iy], _mm_mul_ps(d, _mm_cvtepi32_ps(sum4_epi32(p[iy][0], p[iy][1], p[iy][2], p[iy][3]))));
            }
        }
        float tail[ny] = {};
        for (int i = 4*ng; i < nb; ++i) {
            const __m128i q  = _mm_loadu_si128((const __m128i *)x[i].qs);
            const __m128i x0 = _mm_shuffle_epi8(values, _mm_and_si128(q, m4));
            const __m128i x1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
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
// Q6_K weights: 256-weight superblocks, fp16 scale, 16 int8 sub-block scales (one per 16
// weights), 6-bit quants split into low nibbles (ql) and 2-bit high parts (qh). Each 128-weight
// half of a superblock lines up with one Q8_2_X4 group. pmaddubsw multiplies the unsigned quants
// (0..63) with the activations and pmaddwd applies the sub-block scale while it adds the pairs;
// the -32 offset comes off afterwards as 32 * sum_j scale_j * (sum of the 16 activations of j).
// Those activation sums and the fp32 activation scales are worked out once per call, so each
// further activation row costs two multiplies per 16 weights and no sign or abs work.
// The integer sums are the same as with signed quants, so the results are too.
//
struct blk_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(blk_q6_K) == 210, "q6_K");

// per Q8_2_X4 group of one activation row: sums of its 8 runs of 16 quants and its 4 scales
struct q6_K_ygroup { __m128i ysum; __m128 d; };

template <int ny, typename Store>
static void gemv_q6_K(int n, const void * vx, size_t bx, const Rows<ny> & rows, int nrc_x, Store && store) {
    const int nsb = n / 256;
    const int ng  = 2*nsb;
    const __m128i m4  = _mm_set1_epi8(0x0f);
    const __m128i m30 = _mm_set1_epi8(0x30);
    q6_K_ygroup * yg = (q6_K_ygroup *)__builtin_alloca(sizeof(q6_K_ygroup)*ny*ng + 16);
    yg = (q6_K_ygroup *)(((uintptr_t)yg + 15) & ~(uintptr_t)15);
    {
        const __m128i bias = _mm_set1_epi8((char)0x80), zero = _mm_setzero_si128();
        for (int iy = 0; iy < ny; ++iy) {
            for (int g = 0; g < ng; ++g) {
                const blk_q8_2_x4 * yb = (const blk_q8_2_x4 *)rows.y[iy] + g;
                // psadbw against 0 sums 8 bytes; the bias makes them unsigned, 16*128 comes off after
                __m128i t[4];
                for (int k = 0; k < 4; ++k) {
                    const __m128i s0 = _mm_sad_epu8(_mm_xor_si128(_mm_loadu_si128((const __m128i *)(yb->qs + 32*k)), bias), zero);
                    const __m128i s1 = _mm_sad_epu8(_mm_xor_si128(_mm_loadu_si128((const __m128i *)(yb->qs + 32*k + 16)), bias), zero);
                    // [sum of run 2k, sum of run 2k+1] in the low 32 bits of each 64-bit lane
                    t[k] = _mm_add_epi32(_mm_unpacklo_epi64(s0, s1), _mm_unpackhi_epi64(s0, s1));
                }
                const __m128i t03 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t[0]), _mm_castsi128_ps(t[1]), 0x88));
                const __m128i t47 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(t[2]), _mm_castsi128_ps(t[3]), 0x88));
                q6_K_ygroup & o = yg[iy*ng + g];
                o.ysum = _mm_sub_epi16(_mm_packs_epi32(t03, t47), _mm_set1_epi16(16*128));
                o.d    = bf16x4_to_f32(yb->d);
            }
        }
    }
    for (int ix = 0; ix < nrc_x; ++ix) {
        const blk_q6_K * x = (const blk_q6_K *)((const char *)vx + ix*bx);
        __m128 acc[ny];
        for (int iy = 0; iy < ny; ++iy) acc[iy] = _mm_setzero_ps();
        for (int ib = 0; ib < nsb; ++ib) {
            const blk_q6_K * xb = x + ib;
            const __m128 dx = _mm_set1_ps(fp16_to_f32(xb->d));
            // 16 sub-block scales as int16
            const __m128i sc8  = _mm_loadu_si128((const __m128i *)xb->scales);
            const __m128i sc_lo = _mm_cvtepi8_epi16(sc8), sc_hi = _mm_cvtepi8_epi16(_mm_srli_si128(sc8, 8));
            for (int half = 0; half < 2; ++half) {
                const uint8_t * ql = xb->ql + 64*half;
                const uint8_t * qh = xb->qh + 32*half;
                const __m128i scs = half ? sc_hi : sc_lo;  // scales of this half, 8 x int16
                const __m128i scs32 = _mm_slli_epi16(scs, 5); // 32 * scale fits in int16
                const __m128i l0 = _mm_loadu_si128((const __m128i *)(ql +  0));
                const __m128i l1 = _mm_loadu_si128((const __m128i *)(ql + 16));
                const __m128i l2 = _mm_loadu_si128((const __m128i *)(ql + 32));
                const __m128i l3 = _mm_loadu_si128((const __m128i *)(ql + 48));
                const __m128i h0 = _mm_loadu_si128((const __m128i *)(qh +  0));
                const __m128i h1 = _mm_loadu_si128((const __m128i *)(qh + 16));
                // the eight 16-weight sub-blocks of this half, as unsigned 0..63
                __m128i xq[8];
                xq[0] = _mm_or_si128(_mm_and_si128(l0, m4), _mm_and_si128(_mm_slli_epi16(h0, 4), m30));
                xq[1] = _mm_or_si128(_mm_and_si128(l1, m4), _mm_and_si128(_mm_slli_epi16(h1, 4), m30));
                xq[2] = _mm_or_si128(_mm_and_si128(l2, m4), _mm_and_si128(_mm_slli_epi16(h0, 2), m30));
                xq[3] = _mm_or_si128(_mm_and_si128(l3, m4), _mm_and_si128(_mm_slli_epi16(h1, 2), m30));
                xq[4] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l0, 4), m4), _mm_and_si128(h0, m30));
                xq[5] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l1, 4), m4), _mm_and_si128(h1, m30));
                xq[6] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l2, 4), m4), _mm_and_si128(_mm_srli_epi16(h0, 2), m30));
                xq[7] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(l3, 4), m4), _mm_and_si128(_mm_srli_epi16(h1, 2), m30));
                // scale of 16-weight sub-block j (0..7 in this half), broadcast to 8 x int16
                __m128i s16[8];
                for (int j = 0; j < 8; ++j) {
                    s16[j] = _mm_shuffle_epi8(scs, _mm_set1_epi16((short)(0x0100 + 0x0202*j)));
                }
                const int g = 2*ib + half;
                // integer sums of all rows first, then the float work, so the multiplies of the
                // next row are not queued behind the conversion and scaling of the previous one
                __m128i sum[ny];
                for (int iy = 0; iy < ny; ++iy) {
                    const int8_t * yq = ((const blk_q8_2_x4 *)rows.y[iy] + g)->qs;
                    __m128i p[4];
                    for (int k = 0; k < 4; ++k) {
                        const __m128i y0 = _mm_loadu_si128((const __m128i *)(yq + 32*k));
                        const __m128i y1 = _mm_loadu_si128((const __m128i *)(yq + 32*k + 16));
                        p[k] = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(xq[2*k + 0], y0), s16[2*k + 0]),
                                             _mm_madd_epi16(_mm_maddubs_epi16(xq[2*k + 1], y1), s16[2*k + 1]));
                    }
                    // per 32-block: sum of scale * unsigned quant * y, less 32 * sum of scale * run sum
                    sum[iy] = _mm_sub_epi32(sum4_epi32(p[0], p[1], p[2], p[3]), _mm_madd_epi16(scs32, yg[iy*ng + g].ysum));
                }
                for (int iy = 0; iy < ny; ++iy) {
                    acc[iy] = _mm_add_ps(acc[iy], _mm_mul_ps(_mm_mul_ps(dx, yg[iy*ng + g].d), _mm_cvtepi32_ps(sum[iy])));
                }
            }
        }
        for (int iy = 0; iy < ny; ++iy) store(ix, iy, hsum_ps(acc[iy]));
    }
}

} // namespace avx1_gemv
