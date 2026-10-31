//
// Force-included into the iqk sources when GGML_IQK_AVX1_SIMDE is on.
// Lets the AVX2/FMA kernels build on AVX-only CPUs (Sandy Bridge): the 256-bit
// integer and FMA intrinsics are provided by SIMDe on top of AVX/SSE4.
//

#pragma once

#include <immintrin.h>

#define SIMDE_X86_AVX2_NO_NATIVE
#define SIMDE_X86_FMA_NO_NATIVE
#define SIMDE_X86_F16C_NO_NATIVE
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <simde/x86/f16c.h>

// Defined only after the system intrinsic headers are in, so they keep their
// own target pragmas; from here on the iqk code takes its AVX2 paths.
#define __AVX2__ 1
#define __FMA__ 1

// SIMDe converts fp16 one value at a time; use exact vector versions instead.
#include "iqk_avx1_f16c.h"
#undef _mm_cvtph_ps
#undef _mm256_cvtph_ps
#undef _mm_cvtps_ph
#undef _mm256_cvtps_ph
#define _mm_cvtph_ps(a)       iqk_avx1_mm_cvtph_ps(a)
#define _mm256_cvtph_ps(a)    iqk_avx1_mm256_cvtph_ps(a)
#define _mm_cvtps_ph(a, r)    iqk_avx1_mm_cvtps_ph(a)
#define _mm256_cvtps_ph(a, r) iqk_avx1_mm256_cvtps_ph(a)
