#!/usr/bin/env python3
# Batched Q6_K kernel (iqk_avx1_gemm.h) for multi-token dense matmuls on AVX1 (speculative
# verify and prompt batches). Uses the same GGML_AVX1_GEMM threshold as MXFP4 and keeps Q6_K
# off the SIMDe Q8_0_R8 dequant path while it is on. Apply after avx1_gemm_patch.py.
import sys
root = sys.argv[1] if len(sys.argv) > 1 else '.'

def patch(path, old, new, count=1):
    p = f'{root}/{path}'
    s = open(p).read()
    if new in s:
        return
    n = s.count(old)
    assert n == count, f'{path}: expected {count} match(es), found {n}: {old[:60]!r}'
    open(p, 'w').write(s.replace(old, new))

patch('ggml/src/iqk/iqk_gemm_kquants.cpp',
'''#define AVX1_GEMV_SET(f, funcs) \\
    funcs[0] = f<1>; funcs[1] = f<2>; funcs[2] = f<3>; funcs[3] = f<4>; \\
    funcs[4] = f<5>; funcs[5] = f<6>; funcs[6] = f<7>; funcs[7] = f<8>;
}
#endif
''',
'''#define AVX1_GEMV_SET(f, funcs) \\
    funcs[0] = f<1>; funcs[1] = f<2>; funcs[2] = f<3>; funcs[3] = f<4>; \\
    funcs[4] = f<5>; funcs[5] = f<6>; funcs[6] = f<7>; funcs[7] = f<8>;
}
#include "iqk_avx1_gemm.h"
namespace {
template <int ny> void mul_mat_q6_K_q8_2_avx1_gemm(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const char * y[ny];
    for (int iy = 0; iy < ny; ++iy) y[iy] = info.src1_row(iy);
    avx1_gemv::gemm_q6_K<2, 2>(n, vx, bx, y, ny, nrc_x, [&info](int ix, int iy, float v) { info.store(ix, iy, v); });
}
template <int ny> void avx1_gemm_set(std::array<mul_mat_t, IQK_MAX_NY>& funcs, int min_ny) {
    if constexpr (ny > 1) avx1_gemm_set<ny - 1>(funcs, min_ny);
    if (ny >= min_ny) funcs[ny - 1] = mul_mat_q6_K_q8_2_avx1_gemm<ny>;
}
}
#endif
''')

patch('ggml/src/iqk/iqk_gemm_kquants.cpp',
'''            if (avx1_gemv_enabled()) { AVX1_GEMV_SET(mul_mat_q6_K_q8_2_avx1, kernels) } else
''',
'''            if (avx1_gemv_enabled()) {
                AVX1_GEMV_SET(mul_mat_q6_K_q8_2_avx1, kernels)
                if (const int m = iqk_avx1_gemm_min_ny(); m > 0) {
                    avx1_gemm_set<IQK_MAX_NY>(kernels, m);
                    func16 = mul_mat_q6_K_q8_2_avx1_gemm<16>;
                }
            } else
''')

patch('ggml/src/iqk/iqk_mul_mat.cpp',
'''            case GGML_TYPE_Q6_K   : return nrc_y >= 64 ? GGML_TYPE_Q8_0_R8 : type;
''',
'''#ifdef GGML_IQK_AVX1_SIMDE
            case GGML_TYPE_Q6_K   : return nrc_y >= 64 && iqk_avx1_gemm_min_ny() == 0 ? GGML_TYPE_Q8_0_R8 : type;
#else
            case GGML_TYPE_Q6_K   : return nrc_y >= 64 ? GGML_TYPE_Q8_0_R8 : type;
#endif
''', count=2)
print('ok')
