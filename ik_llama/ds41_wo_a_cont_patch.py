#!/usr/bin/env python3
"""Make the grouped output projection's input contiguous before its batched matmul (ik root).

DeepSeek-V4.1's attention output goes through 8 grouped LoRA projections: the [o_group_dim,
n_groups, n_tokens] attention result is permuted to [o_group_dim, n_tokens, n_groups] and fed to a
batched MUL_MAT against the Q8_0 wo_a weights. On CUDA (Pascal, sm_61) that MUL_MAT is wrong when
its src1 is the permuted, non-contiguous view and n_tokens > 1: per-tensor checksums against the
CPU build first diverge at attn_wo_a (7%), and GPU perplexity is ~720 instead of ~3.5. A contiguous
copy of the input (4096 x n_tokens x 8 floats per layer) avoids it. With one token the permute
does not change the layout, so decode is left as it was.
"""
import sys

p = "src/graphs/build_deepseek4.cpp"
s = open(p).read()
old = ("    attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);\n"
       "\n"
       "    ggml_tensor * oa = ggml_mul_mat(ctx0,\n")
new = ("    attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);\n"
       "    if (n_tokens > 1) {\n"
       "        // the batched matmul below is wrong on CUDA (sm_61) for this permuted view\n"
       "        attn = ggml_cont(ctx0, attn);\n"
       "    }\n"
       "\n"
       "    ggml_tensor * oa = ggml_mul_mat(ctx0,\n")
n = s.count(old)
if n == 0:
    sys.exit(f"{p}: anchor not found")
open(p, "w").write(s.replace(old, new))
print(f"wo_a contiguous-input patch applied at {n} site(s)")
