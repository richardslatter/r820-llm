#!/usr/bin/env python3
"""Hook the native AVX1 Q8_0/MXFP4 kernels into ik's x86 legacy-quant dispatch (run from the ik_llama.cpp root).

Copies iqk_avx1_gemv.h next to the iqk sources. With GGML_IQK_AVX1_SIMDE on, Q8_0 and MXFP4
weights against Q8_2_X4 activations use these kernels for 1..8 activation rows (decode and
small batches); larger batches still repack to the _R8 layouts as before.
GGML_AVX1_GEMV=0 at run time goes back to the SIMDe kernels.
"""
import os
import shutil
import sys

here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "iqk_avx1_gemv.h"), "ggml/src/iqk/iqk_avx1_gemv.h")

p = "ggml/src/iqk/iqk_gemm_legacy_quants.cpp"
s = open(p).read()
if "avx1_gemv::" in s:
    sys.exit(f"{p}: already patched")

wrappers = '''#ifdef GGML_IQK_AVX1_SIMDE
#include "iqk_avx1_gemv.h"
namespace {
template <int ny> void mul_mat_q8_0_q8_2_avx1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    avx1_gemv::Rows<ny> rows;
    for (int iy = 0; iy < ny; ++iy) rows.y[iy] = info.src1_row(iy);
    avx1_gemv::gemv_q8_0<ny>(n, vx, bx, rows, nrc_x, [&info](int ix, int iy, float v) { info.store(ix, iy, v); });
}
template <int ny> void mul_mat_mxfp4_q8_2_avx1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    avx1_gemv::Rows<ny> rows;
    for (int iy = 0; iy < ny; ++iy) rows.y[iy] = info.src1_row(iy);
    avx1_gemv::gemv_mxfp4<ny>(n, vx, bx, rows, nrc_x, [&info](int ix, int iy, float v) { info.store(ix, iy, v); });
}
bool avx1_gemv_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_AVX1_GEMV"); return !e || atoi(e) != 0; }();
    return on;
}
#define AVX1_GEMV_SET(f, funcs) \\
    funcs[0] = f<1>; funcs[1] = f<2>; funcs[2] = f<3>; funcs[3] = f<4>; \\
    funcs[4] = f<5>; funcs[5] = f<6>; funcs[6] = f<7>; funcs[7] = f<8>;
}
#endif

'''
anchor = "bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {\n"
i = s.find(anchor)
if i < 0:
    sys.exit(f"{p}: set_kernels anchor not found")
s = s[:i] + wrappers + s[i:]

for old, f in [("            set_functions<Q8_0_Unpacker>(kernels);\n", "mul_mat_q8_0_q8_2_avx1"),
               ("            set_functions<MXFP4_Unpacker>(kernels);\n", "mul_mat_mxfp4_q8_2_avx1")]:
    j = s.find(old)
    if j < 0:
        sys.exit(f"{p}: anchor {old.strip()} not found")
    new = (f"#ifdef GGML_IQK_AVX1_SIMDE\n            if (avx1_gemv_enabled()) {{ AVX1_GEMV_SET({f}, kernels) }} else\n#endif\n"
           f"            {{ {old.strip()} }}\n")
    s = s[:j] + new + s[j + len(old):]

open(p, "w").write(s)
print("avx1 gemv kernels hooked for Q8_0 and MXFP4")
