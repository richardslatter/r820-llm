#!/usr/bin/env python3
"""Vector fp16 loads/stores in ggml.c for AVX CPUs without F16C (run from the ik_llama.cpp root).

Without F16C, ggml.c's GGML_F32Cx8_LOAD/STORE convert eight values one at a time through the
fp16 lookup table. DeepSeek-V4's attention runs on ggml's generic flash-attention path (ik
marks it so for numerical reasons) with an fp16 KV cache, so every K dot product and V update
went through that loop: ~12 ms of a 170 ms decode step. The replacements in
iqk/iqk_avx1_f16c.h are exact (checked on all 65536 halves and all 2^32 floats), so results
are bit-identical. Needs iqk_avx1_f16c.h copied into ggml/src/iqk/.
"""
import os
import shutil
import sys

here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "iqk_avx1_f16c.h"), "ggml/src/iqk/iqk_avx1_f16c.h")

p = "ggml/src/ggml.c"
s = open(p).read()
old = """#else
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    float arr[8];

    _mm256_storeu_ps(arr, y);

    for (int i = 0; i < 8; i++)
        x[i] = GGML_FP32_TO_FP16(arr[i]);
}
"""
new = """#else
// no F16C (Sandy Bridge): exact vector conversions instead of eight table lookups
#include "iqk/iqk_avx1_f16c.h"
static inline __m256 __avx_f32cx8_load(const ggml_fp16_t *x) {
    return iqk_avx1_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)x));
}
static inline void __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    _mm_storeu_si128((__m128i *)x, iqk_avx1_mm256_cvtps_ph(y));
}
"""
if s.count(old) != 1:
    sys.exit(f"{p}: anchor not found")
open(p, "w").write(s.replace(old, new))
print("vector fp16 load/store for AVX without F16C applied")
