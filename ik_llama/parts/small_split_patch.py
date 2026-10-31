#!/usr/bin/env python3
"""Even row split for small matmuls at small batch (run from the ik_llama.cpp root).

When a weight matrix has fewer than 32 rows per thread, iqk_mul_mat hands out 16-row tiles
round-robin. With 64 threads a 24-row matrix (DeepSeek-V4's 20480x24 hyper-connection
projections) ran on 2 threads, a 384-row router on 24, a 512-row projection on 32. For up to
IQK_MAX_NY activation rows and plain (non-interleaved) weight types this gives every thread
rows [Nx*ith/nth, Nx*(ith+1)/nth) instead, which is also the split GGML_NUMA_PLACE uses, so
the rows a thread reads are on its own node.
"""
import sys

p = "ggml/src/iqk/iqk_mul_mat.cpp"
s = open(p).read()
old = """        if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
            return false;
        }
        const int min_step = Ny <= 16 ? 16 : 32;
"""
new = """        if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
            return false;
        }
        if (Ny <= IQK_MAX_NY && MulMat::num_rows(ggml_type(typeA)) == 1) {
            const int first_x = (int)(Nx*ith/nth), last_x = (int)(Nx*(ith + 1)/nth);
            if (last_x > first_x) {
                DataInfo info{C + first_x, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, nullptr, 0};
                mm.mul_mat_NxM(ne00, (const char *)A + first_x*row_size_qx, row_size_qx, info, last_x - first_x, Ny);
            }
            return true;
        }
        const int min_step = Ny <= 16 ? 16 : 32;
"""
if s.count(old) != 1:
    sys.exit(f"{p}: anchor not found")
open(p, "w").write(s.replace(old, new))
print("even small-matmul split applied")
