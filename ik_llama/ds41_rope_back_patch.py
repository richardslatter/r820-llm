#!/usr/bin/env python3
"""Out-of-place rope-back for DeepSeek-V4.1's attention output (run from the ik_llama.cpp root).

The graph de-rotates the attention output with an in-place rope turned into GGML_OP_ROPE_BACK with
op_params[15] = 1 (the rotated dims are the last n_dims of each head). On CUDA (Pascal, sm_61)
that in-place path leaves the result wrong for what follows. Per-tensor checksums against the CPU
build first diverge at attn_wo_a (7%), ffn_norm comes out 5.2 instead of 17.4, and GPU perplexity
is ~720 instead of ~3.5. With the out-of-place op the CUDA general kernel, which takes is_flipped
as a parameter, is used and the GPU matches the CPU to ~1%. The cost is one extra
[n_embd_head, n_head, n_tokens] buffer per layer.

The DSpark draft (build_dflash_dsv4) de-rotates the same way: with its layer 0 on the GPU, its
attn_wo_a came out 34% off and code acceptance fell from ~91% to ~57%.
"""
import sys

p = "src/graphs/build_deepseek4.cpp"
s = open(p).read()
sites = [
    # target model
    ("    attn = ggml_rope_ext_inplace(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,\n",
     "    attn = ggml_rope_ext(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,\n"),
    # DSpark draft
    ("        attn = ggml_rope_ext_inplace(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,\n",
     "        attn = ggml_rope_ext(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,\n"),
]
n = 0
for old, new in sites:
    n += s.count(old)
    s = s.replace(old, new)
if n == 0:
    sys.exit(f"{p}: anchor not found")
open(p, "w").write(s)
print(f"out-of-place rope-back applied at {n} site(s)")
