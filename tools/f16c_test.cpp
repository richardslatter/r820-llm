// Exhaustive check of ik_llama/iqk_avx1_f16c.h against scalar references.
//   g++ -O2 -march=native -I../ik_llama f16c_test.cpp -o f16c_test && ./f16c_test
#include "iqk_avx1_f16c.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static float half_ref(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0)       v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else              v = std::ldexp((float)(1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}

// float -> half, round to nearest even (scalar form of the same algorithm, checked by hand)
static uint16_t to_half_ref(uint32_t u) {
    const uint32_t sign = (u >> 16) & 0x8000;
    uint32_t a = u & 0x7fffffff;
    uint16_t o;
    if (a >= 0x47800000) o = a > 0x7f800000 ? 0x7e00 : 0x7c00;
    else if (a < (113u << 23)) {
        float f; memcpy(&f, &a, 4); f += 0.5f; uint32_t b; memcpy(&b, &f, 4); o = (uint16_t)(b - 0x3f000000);
    } else {
        a += ((uint32_t)(15 - 127) << 23) + 0xfff + ((a >> 13) & 1);
        o = (uint16_t)(a >> 13);
    }
    return o | sign;
}

int main() {
    long bad = 0;
    for (uint32_t h = 0; h < 65536; h += 8) {
        uint16_t in[8]; for (int j = 0; j < 8; ++j) in[j] = (uint16_t)(h + j);
        float out[8];
        _mm256_storeu_ps(out, iqk_avx1_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)in)));
        for (int j = 0; j < 8; ++j) {
            const float r = half_ref(in[j]);
            uint32_t a, b; memcpy(&a, &out[j], 4); memcpy(&b, &r, 4);
            if (std::isnan(r) ? !std::isnan(out[j]) : a != b) {
                if (bad++ < 5) printf("half %04x: got %08x want %08x\n", in[j], a, b);
            }
        }
    }
    printf("half -> float: %ld mismatches of 65536\n", bad);

    // every float bit pattern; also checks against the true nearest-even half for finite inputs
    bad = 0;
    long bad_exact = 0;
    for (uint64_t u = 0; u < (1ull << 32); u += 8) {
        uint32_t in[8]; for (int j = 0; j < 8; ++j) in[j] = (uint32_t)(u + j);
        __m256 f; memcpy(&f, in, 32);
        uint16_t out[8];
        _mm_storeu_si128((__m128i *)out, iqk_avx1_mm256_cvtps_ph(f));
        for (int j = 0; j < 8; ++j) {
            const uint16_t r = to_half_ref(in[j]);
            float x; memcpy(&x, &in[j], 4);
            if (std::isnan(x) ? (out[j] & 0x7fff) <= 0x7c00 : out[j] != r) {
                if (bad++ < 5) printf("float %08x: got %04x want %04x\n", in[j], out[j], r);
            }
            // spot-check the reference itself: the result must be a nearest half
            if ((in[j] & 0xffff) == 0x1235 && std::isfinite(x) && std::fabs(x) < 65504.f) {
                const float h = half_ref(r), lo = half_ref(r - 1 + 0), hi = half_ref(r + 1);
                const bool ok = std::fabs(h - x) <= std::fabs(lo - x) && std::fabs(h - x) <= std::fabs(hi - x);
                if (!ok && (r & 0x7fff) != 0 && bad_exact++ < 5) printf("ref not nearest for %08x\n", in[j]);
            }
        }
    }
    printf("float -> half: %ld mismatches of 2^32, reference not-nearest: %ld\n", bad, bad_exact);
    return 0;
}
