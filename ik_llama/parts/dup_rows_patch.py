#!/usr/bin/env python3
"""Same-type copies with fewer rows in dim 1 than threads: split all rows across the threads.

ggml_compute_forward_dup_bytes split a row-by-row copy by dim-1 rows only. DeepSeek V4 copies
its permuted attention output ([4096, groups 8, tokens]) before attn_wo_a when there is more than
one token, so a 4-token verify batch copied 512 KB on 4 threads, ~55 us per layer. The copy now
splits all ne01*ne02*ne03 rows when ne01 < nth; every row is copied as before.

Usage: dup_rows_patch.py <ik_llama.cpp checkout>
"""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = root + "/ggml/src/ggml.c"
s = open(p).read()
old = """        // copy by rows
        const size_t rs = ggml_row_size(src0->type, ne00);
        for (int64_t i03 = 0; i03 < ne03; i03++) {"""
new = """        // copy by rows
        const size_t rs = ggml_row_size(src0->type, ne00);
        if (ne01 < nth) {
            // few rows in dim 1 (a few tokens): split all rows, not just dim 1
            const int64_t nrows = ne01*ne02*ne03;
            const int64_t r0 = nrows*ith/nth, r1 = nrows*(ith + 1)/nth;
            for (int64_t ir = r0; ir < r1; ++ir) {
                const int64_t i03 = ir/(ne01*ne02), i02 = (ir/ne01)%ne02, i01 = ir%ne01;
                memcpy(
                    ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                    ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                    rs);
            }
            return;
        }
        for (int64_t i03 = 0; i03 < ne03; i03++) {"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
