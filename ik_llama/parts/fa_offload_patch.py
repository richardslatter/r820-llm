#!/usr/bin/env python3
"""Let large-batch flash attention run on the GPU with -ngl 0 (run from the ik root).

With every layer on the CPU, an op goes to the GPU only through the scheduler offload rule: one
of its sources is a CPU weight and the GPU backend both supports the op and wants it (batch >= 32).
DeepSeek-V4.1 attention has such a source, its per-head sinks, so prompt batches would offload,
but the Pascal hs512 kernel still reported sinks as unsupported. It applies them since
fattn_sinks_patch.py. With sinks allowed, a prompt batch runs attention on the GPU next to the Q
and output projections that are already there; decode and the draft verify (< 32 tokens) stay on
the CPU. K and V come from the CPU cache as split inputs.
"""
import sys


def patch(path, old, new, count=1):
    s = open(path).read()
    if new in s and old not in s:
        return
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: expected {count} match(es), found {n} for:\n{old[:200]}")
    open(path, "w").write(s.replace(old, new))


F = "ggml/src/ggml-cuda/fattn-tile-new.cu"

patch(F,
      "    if (K->ne[0] != 512 || V->ne[0] != 512 || !mask || sinks || max_bias != 0.0f || logit_softcap != 0.0f) {\n"
      "        return false;\n"
      "    }\n",
      "    if (K->ne[0] != 512 || V->ne[0] != 512 || !mask || max_bias != 0.0f || logit_softcap != 0.0f) {\n"
      "        return false;\n"
      "    }\n"
      "    // Sinks are applied by the kernel: one f32 per Q head.\n"
      "    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] != Q->ne[2])) {\n"
      "        return false;\n"
      "    }\n")

# The offload rule reads the batch from ne[1], but the flash attention result is [D, heads, tokens]:
# ne[1] is the head count (64), which would send every decode step to the GPU. Use the Q tokens.
F = "ggml/src/ggml-cuda.cu"
patch(F,
      "    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;\n",
      "    if (op->op == GGML_OP_FLASH_ATTN_EXT) {\n"
      "        return op->src[0]->ne[1] >= min_batch_size;\n"
      "    }\n"
      "\n"
      "    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;\n")
