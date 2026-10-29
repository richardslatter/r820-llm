#!/usr/bin/env python3
"""Attention sinks in ik's Pascal head-size-512 flash attention kernel (run from the ik root).

On Pascal (no tensor cores) every K = V = 512 attention goes to ggml_cuda_flash_attn_ext_tile_new
(written for Gemma-4), unless the DSA kernel takes it. DeepSeek-V4.1 layers without the sparse
indexer, and indexer layers at short context, all land there. The kernel never read the per-head
sink logits (src[4]), so every such layer computed softmax without them: perplexity 831 instead of
1.44 on the first wikitext chunk.

A sink is one extra logit per head that joins the softmax denominator but has no value row. As in
the CPU kernel: M' = max(M, s), S' = S*exp(M - M') + exp(s - M'), VKQ' = VKQ*exp(M - M'). It is
added once per (column, head), by the first KV block; with several KV-parallel blocks the combine
kernel folds it in through dst_meta like any other partial result.
"""
import sys


def patch(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: expected {count} match(es), found {n} for:\n{old[:200]}")
    open(path, "w").write(s.replace(old, new))


F = "ggml/src/ggml-cuda/fattn-tile-new.cu"

patch(F,
      "        const char * __restrict__ mask,\n"
      "        const int  * __restrict__ KV_max,\n"
      "        float      * __restrict__ dst,\n"
      "        float2     * __restrict__ dst_meta,\n"
      "        const float scale,\n",
      "        const char * __restrict__ mask,\n"
      "        const float * __restrict__ sinks,\n"
      "        const int  * __restrict__ KV_max,\n"
      "        float      * __restrict__ dst,\n"
      "        float2     * __restrict__ dst_meta,\n"
      "        const float scale,\n")

patch(F,
      "#pragma unroll\n"
      "    for (int jc0 = 0; jc0 < cpw; ++jc0) {\n"
      "        KQ_sum[jc0] = warp_reduce_sum(KQ_sum[jc0]);\n"
      "    }\n",
      "#pragma unroll\n"
      "    for (int jc0 = 0; jc0 < cpw; ++jc0) {\n"
      "        KQ_sum[jc0] = warp_reduce_sum(KQ_sum[jc0]);\n"
      "    }\n"
      "\n"
      "    // Attention sinks: one more logit per head in the softmax denominator, with no value row.\n"
      "    // Added once, by the first KV block (the combine kernel merges it through dst_meta).\n"
      "    if (sinks && blockIdx.y == 0) {\n"
      "#pragma unroll\n"
      "        for (int jc0 = 0; jc0 < cpw; ++jc0) {\n"
      "            const int jc = jc0 + (threadIdx.y/np)*cpw;\n"
      "            const float sink = sinks[head0 + jc % ncols2];\n"
      "            const float KQ_max_new   = fmaxf(KQ_max[jc0], sink);\n"
      "            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new);\n"
      "            KQ_max[jc0] = KQ_max_new;\n"
      "            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + expf(sink - KQ_max_new);\n"
      "#pragma unroll\n"
      "            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {\n"
      "                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;\n"
      "                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;\n"
      "            }\n"
      "        }\n"
      "    }\n")

patch(F,
      "        (const char *) mask->data,\n"
      "        KV_max.get(),\n",
      "        (const char *) mask->data,\n"
      "        KQV->src[4] ? (const float *) KQV->src[4]->data : nullptr,\n"
      "        KV_max.get(),\n")

print("tile_new attention-sink patch applied")
