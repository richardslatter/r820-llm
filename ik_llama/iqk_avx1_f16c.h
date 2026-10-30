//
// fp16 <-> fp32 for CPUs without F16C (Sandy Bridge), included by iqk_avx1_shim.h.
//
// SIMDe's F16C emulation converts one value at a time, and the fp16 KV cache goes through
// _mm256_cvtph_ps on every attention step: with it flash attention was ~12 ms per token at
// short context. These are exact vector versions: fp16 -> fp32 for every input (denormals,
// inf, NaN) and fp32 -> fp16 with round to nearest even (the rounding argument is ignored;
// iqk only uses _MM_FROUND_TO_NEAREST_INT).
//
#pragma once

#include <immintrin.h>

// halves in the low 16 bits of each 32-bit lane -> floats
static inline __m128 iqk_avx1_half4_to_float(__m128i h) {
    const __m128i sign = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16);
    const __m128i em   = _mm_and_si128(h, _mm_set1_epi32(0x7fff));
    __m128i bits = _mm_add_epi32(_mm_slli_epi32(em, 13), _mm_set1_epi32(112 << 23));
    // exponent 31 (inf/NaN): rebias once more so the float exponent is 255
    bits = _mm_add_epi32(bits, _mm_and_si128(_mm_cmpgt_epi32(em, _mm_set1_epi32(0x7bff)), _mm_set1_epi32(112 << 23)));
    // exponent 0 (zero/denormal): value = mantissa * 2^-24
    const __m128 fd = _mm_mul_ps(_mm_cvtepi32_ps(em), _mm_set1_ps(5.9604644775390625e-08f));
    const __m128 f  = _mm_blendv_ps(_mm_castsi128_ps(bits), fd, _mm_castsi128_ps(_mm_cmplt_epi32(em, _mm_set1_epi32(0x0400))));
    return _mm_or_ps(f, _mm_castsi128_ps(sign));
}

// floats -> halves in the low 16 bits of each 32-bit lane
static inline __m128i iqk_avx1_float4_to_half(__m128 f) {
    const __m128i x    = _mm_castps_si128(f);
    const __m128i sign = _mm_srli_epi32(_mm_and_si128(x, _mm_set1_epi32((int)0x80000000)), 16);
    const __m128i a    = _mm_and_si128(x, _mm_set1_epi32(0x7fffffff));
    // normal result: rebias the exponent and round the 13 dropped bits to nearest even
    const __m128i odd = _mm_and_si128(_mm_srli_epi32(a, 13), _mm_set1_epi32(1));
    const __m128i n   = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(a, _mm_set1_epi32(((15 - 127) << 23) + 0xfff)), odd), 13);
    // denormal result (|f| < 2^-14): adding 0.5 lets the FPU do the rounding
    const __m128  dm = _mm_castsi128_ps(_mm_set1_epi32(126 << 23));
    const __m128i d  = _mm_sub_epi32(_mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(a), dm)), _mm_castps_si128(dm));
    __m128i r = _mm_blendv_epi8(n, d, _mm_cmplt_epi32(a, _mm_set1_epi32(113 << 23)));
    // |f| >= 65536 (and inf): inf; NaN: quiet NaN
    r = _mm_blendv_epi8(r, _mm_set1_epi32(0x7c00), _mm_cmpgt_epi32(a, _mm_set1_epi32(0x477fffff)));
    r = _mm_blendv_epi8(r, _mm_set1_epi32(0x7e00), _mm_cmpgt_epi32(a, _mm_set1_epi32(0x7f800000)));
    return _mm_or_si128(r, sign);
}

static inline __m128 iqk_avx1_mm_cvtph_ps(__m128i h) {
    return iqk_avx1_half4_to_float(_mm_cvtepu16_epi32(h));
}

static inline __m256 iqk_avx1_mm256_cvtph_ps(__m128i h) {
    const __m128 lo = iqk_avx1_half4_to_float(_mm_cvtepu16_epi32(h));
    const __m128 hi = iqk_avx1_half4_to_float(_mm_cvtepu16_epi32(_mm_srli_si128(h, 8)));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

static inline __m128i iqk_avx1_mm_cvtps_ph(__m128 f) {
    return _mm_packus_epi32(iqk_avx1_float4_to_half(f), _mm_setzero_si128());
}

static inline __m128i iqk_avx1_mm256_cvtps_ph(__m256 f) {
    return _mm_packus_epi32(iqk_avx1_float4_to_half(_mm256_castps256_ps128(f)),
                            iqk_avx1_float4_to_half(_mm256_extractf128_ps(f, 1)));
}
