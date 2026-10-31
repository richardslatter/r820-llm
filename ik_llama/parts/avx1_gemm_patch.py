#!/usr/bin/env python3
# Batched MXFP4 kernel (iqk_avx1_gemm.h) for prompt-sized expert batches on AVX1.
# GGML_AVX1_GEMM=<min columns> (default 3; 0 = off). Also keeps MXFP4 off the SIMDe Q8_0_R8 path.
import re, sys
root = sys.argv[1] if len(sys.argv) > 1 else '.'

def patch(path, old, new, count=1):
    p = f'{root}/{path}'
    s = open(p).read()
    if new in s:
        return
    n = s.count(old)
    assert n == count, f'{path}: expected {count} match(es), found {n}: {old[:60]!r}'
    open(p, 'w').write(s.replace(old, new))

patch('ggml/src/iqk/iqk_gemm_legacy_quants.cpp',
'''bool avx1_gemv_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_AVX1_GEMV"); return !e || atoi(e) != 0; }();
    return on;
}
#define AVX1_GEMV_SET(f, funcs) \\
    funcs[0] = f<1>; funcs[1] = f<2>; funcs[2] = f<3>; funcs[3] = f<4>; \\
    funcs[4] = f<5>; funcs[5] = f<6>; funcs[6] = f<7>; funcs[7] = f<8>;
}
#endif
''',
'''bool avx1_gemv_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_AVX1_GEMV"); return !e || atoi(e) != 0; }();
    return on;
}
#define AVX1_GEMV_SET(f, funcs) \\
    funcs[0] = f<1>; funcs[1] = f<2>; funcs[2] = f<3>; funcs[3] = f<4>; \\
    funcs[4] = f<5>; funcs[5] = f<6>; funcs[6] = f<7>; funcs[7] = f<8>;
}
#include "iqk_avx1_gemm.h"
namespace {
template <int ny> void mul_mat_mxfp4_q8_2_avx1_gemm(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if (n % 128) {
        if constexpr (ny <= 8) {
            mul_mat_mxfp4_q8_2_avx1<ny>(n, vx, bx, info, nrc_x);
        } else {
            auto i2 = info;
            mul_mat_mxfp4_q8_2_avx1<8>(n, vx, bx, i2, nrc_x);
            i2.cur_y += 8;
            mul_mat_mxfp4_q8_2_avx1<ny - 8>(n, vx, bx, i2, nrc_x);
        }
        return;
    }
    const char * y[ny];
    for (int iy = 0; iy < ny; ++iy) y[iy] = info.src1_row(iy);
    avx1_gemv::gemm_mxfp4<2, 2>(n, vx, bx, y, ny, nrc_x, [&info](int ix, int iy, float v) { info.store(ix, iy, v); });
}
template <int ny> void avx1_gemm_set(std::array<mul_mat_t, IQK_MAX_NY>& funcs, int min_ny) {
    if constexpr (ny > 1) avx1_gemm_set<ny - 1>(funcs, min_ny);
    if (ny >= min_ny) funcs[ny - 1] = mul_mat_mxfp4_q8_2_avx1_gemm<ny>;
}
}
#endif
''')

patch('ggml/src/iqk/iqk_gemm_legacy_quants.cpp',
'''            if (avx1_gemv_enabled()) { AVX1_GEMV_SET(mul_mat_mxfp4_q8_2_avx1, kernels) } else
''',
'''            if (avx1_gemv_enabled()) {
                AVX1_GEMV_SET(mul_mat_mxfp4_q8_2_avx1, kernels)
                if (const int m = iqk_avx1_gemm_min_ny(); m > 0) {
                    avx1_gemm_set<IQK_MAX_NY>(kernels, m);
                    func16 = mul_mat_mxfp4_q8_2_avx1_gemm<16>;
                }
            } else
''')

# shared switch, visible to iqk_mul_mat.cpp's is_dequant_better
patch('ggml/src/iqk/iqk_common.h',
'''struct DataInfo {''',
'''#ifdef GGML_IQK_AVX1_SIMDE
// GGML_AVX1_GEMM: smallest column count that uses the batched MXFP4 kernel (0 = off)
static inline int iqk_avx1_gemm_min_ny() {
    static const int v = [] { const char * e = getenv("GGML_AVX1_GEMM"); return e ? atoi(e) : 3; }();
    return v;
}
#endif

struct DataInfo {''')

patch('ggml/src/iqk/iqk_mul_mat.cpp',
'''            case GGML_TYPE_MXFP4  : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
''',
'''#ifdef GGML_IQK_AVX1_SIMDE
            case GGML_TYPE_MXFP4  : return nrc_y >= 32 && iqk_avx1_gemm_min_ny() == 0 ? GGML_TYPE_Q8_0_R8 : type;
#else
            case GGML_TYPE_MXFP4  : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
#endif
''', count=2)
print('ok')
