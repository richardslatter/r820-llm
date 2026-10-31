#!/usr/bin/env python3
"""HC_POST with a few tokens: split by token and 64-wide chunk, not by token alone.

DeepSeek V4's hyper-connection output mix (80 per graph) split its work by token when there was
more than one, so a draft's ~4-token verify batch ran it on 4 of 64 threads: 15.4 ms per graph
against 1.2 ms of work. Every (token, 64-element chunk) pair is now a task, as the one-token path
already did, and each element is computed exactly as before. The tasks are ordered chunk-major,
so a thread handles the same columns of every token: the ones its share of the preceding matmul
wrote. GGML_HC_POST_SPLIT=1 orders them token-major, =0 goes back to the split by token.

Usage: hc_post_patch.py <ik_llama.cpp checkout>
"""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = root + "/ggml/src/ggml.c"
s = open(p).read()

start = s.index("    if (T == 1) {\n        const float * x_r    = (const float *)((const char *)x->data);")
end = s.index("static void ggml_compute_forward_hc_post(\n")
new = """    // tasks are (token, 64-element chunk) pairs, so a few tokens (a draft's verify batch) still
    // use every thread; each thread takes a contiguous range of them. GGML_HC_POST_SPLIT: 2
    // (default) orders them chunk-major, so a thread handles the same columns of every token, the
    // ones its share of the preceding matmul wrote; 1 token-major; 0 one task per token
    static int split_chunks = -1;
    if (split_chunks < 0) {
        const char * e = getenv("GGML_HC_POST_SPLIT");
        split_chunks = e == NULL ? 2 : atoi(e);
    }
    const int64_t nc = split_chunks || T == 1 ? (ne0 + 63)/64 : 1;
    const int64_t ntask = T*nc;
    const int64_t first_task = ntask*ith/nth, last_task = ntask*(ith + 1)/nth;

    for (int64_t ic = first_task; ic < last_task; ++ic) {
        const int64_t t = split_chunks == 2 ? ic % T : ic / nc;
        const int64_t c = split_chunks == 2 ? ic / T : ic % nc;
        const int first = nc > 1 ? 64*c : 0, last = nc > 1 ? MIN(first + 64, ne0) : ne0;
        const float * x_r    = (const float *)((const char *)x->data + t*x->nb[1]);
        const float * post_r = (const float *)((const char *)post->data + t*post->nb[1]);
        const float * comb_r = (const float *)((const char *)comb->data + t*comb->nb[2]);

        for (int i0 = first; i0 < last; ++i0) {
            for (int j = 0; j < S; ++j) {
                const float * res_r  = (const float *)((const char *)res->data + t*res->nb[2] + j*res->nb[1]);
                r[j] = res_r[i0];
            }
            for (int i = 0; i < S; ++i) {
                float sum = x_r[i0] * post_r[i];
                for (int j = 0; j < S; ++j) {
                    sum += comb_r[j*S + i] * r[j];
                }
                float * dst_r = (float *)((char *)dst->data + t*dst->nb[2] + i*dst->nb[1]);
                dst_r[i0] = sum;
            }
        }
    }
}

"""
s = s[:start] + new + s[end:]
open(p, "w").write(s)
print("patched", p)
