#!/usr/bin/env python3
"""Out-of-place rope-back for DeepSeek-V4.1's attention output (run from the ik_llama.cpp root).

The graph de-rotates the attention output with an in-place rope turned into GGML_OP_ROPE_BACK with
op_params[15] = 1 (the rotated dims are the last n_dims of each head). On CUDA (Pascal, sm_61)
that in-place path leaves the result wrong for what follows. Per-tensor checksums against the CPU
build first diverge at attn_wo_a (7%), ffn_norm comes out 5.2 instead of 17.4, and GPU perplexity
is ~720 instead of ~3.5. With the out-of-place op the CUDA general kernel, which takes is_flipped
as a parameter, is used and the GPU matches the CPU to ~1%. The cost is one extra
[n_embd_head, n_head, n_tokens] buffer per layer.
"""
import sys

p = "src/graphs/build_deepseek4.cpp"
s = open(p).read()
old = "    attn = ggml_rope_ext_inplace(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,\n"
new = "    attn = ggml_rope_ext(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,\n"
n = s.count(old)
if n == 0:
    sys.exit(f"{p}: anchor not found")
open(p, "w").write(s.replace(old, new))
print(f"out-of-place rope-back applied at {n} site(s)")
