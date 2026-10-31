#!/usr/bin/env python3
"""NUMA-local split for grouped matmuls on a reshaped 2D weight (run from the ik_llama.cpp root).

DeepSeek-V4's attention output projection multiplies 8 groups: wo_a [4096, 8192] reshaped to
[4096, 1024, 8] against the 8 head groups. iqk_mul_mat_4d gave each group to 8 threads spread
over all nodes, while GGML_NUMA_PLACE puts the 2D tensor's rows on nodes in 4 blocks, so 3/4
of the reads crossed QPI and wo_a ran at ~45 GB/s. When the groups are contiguous slices of
one matrix (nb02 == Nx*row size, no broadcast) and the batch is small, split all Nx*ne02 rows
evenly over the threads instead; every thread then reads rows on its own node.
"""
import sys

p = "ggml/src/iqk/iqk_mul_mat.cpp"
s = open(p).read()
old = """    if (ne13 == 1 && ne12 > 1 && ne12 == ne02 && Ny == 1 && nb02 < strideA) {
"""
new = """    if (ne13 == 1 && ne03 == 1 && ne02 > 1 && ne12 == ne02 && Ny <= IQK_MAX_NY && nb02 == Nx*strideA &&
        MulMat::num_rows(ggml_type(typeA)) == 1) {
        MulMat mm;
        if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
            return false;
        }
        const long R = Nx*ne02;
        const long r_end = R*(ith + 1)/nth;
        for (long r = R*ith/nth; r < r_end; ) {
            const long i02 = r/Nx, ix = r - i02*Nx;
            const long n = std::min(r_end - r, Nx - ix);
            DataInfo info{C + ix + i02*nb2, (const char *)B + i02*nb12, (size_t)stride_C, (size_t)strideB, 0, 1, nullptr, 0};
            mm.mul_mat_NxM(ne00, (const char *)A + r*strideA, strideA, info, (int)n, (int)Ny);
            r += n;
        }
        return true;
    }

    if (ne13 == 1 && ne12 > 1 && ne12 == ne02 && Ny == 1 && nb02 < strideA) {
"""
if s.count(old) != 1:
    sys.exit(f"{p}: anchor not found")
open(p, "w").write(s.replace(old, new))
print("NUMA-local grouped matmul split applied")
