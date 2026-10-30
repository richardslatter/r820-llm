#!/usr/bin/env python3
"""Decode flash attention for one shared fp16 K/V head (run from the ik_llama.cpp root).

DeepSeek-V4 attends with 64 query heads over a single 512-wide latent K/V head, and ik sends
its attention to ggml's generic CPU flash-attention loop. That loop gives each thread one
head and walks every visible KV row for it: each row is converted from fp16 64 times per
layer, and the V accumulator is kept in fp16 (two conversions per element per row). At ~400
tokens of context that was ~0.9 ms per layer.

For f32 Q, f16 K/V with one K/V head and up to 8 query rows, threads are split into head
groups x chunks of the visible KV rows (consecutive threads share a chunk, so with
GGML_NUMA_BLOCKED each node reads only its own chunks). Each thread converts its rows once
(once for K and V: DeepSeek-V4 passes the same tensor for both), scores them against all
heads of its group four rows at a time, keeps an f32 online-softmax state per head, and
writes it out; after a barrier the chunk states are merged per head (and the attention
sink folded in). The kernel itself is in ds4_fa_mqa_kernel.c. GGML_FA_MQA=0 goes back to
the generic loop.

K/V is built every token as a concat of the raw and compressed caches; when the attention node
is its only user the concat is skipped and the kernel reads the visible rows from the caches
(GGML_FA_LAZY=0 turns that off). A computed source (the HCA layers' gathered compressed rows) is
read in place too when no node run before the attention node writes over it.
"""
import os
import sys

p = "ggml/src/ggml.c"
s = open(p).read()
if "ggml_fa_mqa_decode" in s:
    sys.exit(f"{p}: already patched")


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"{p}: anchor not found: {old[:70]!r}")
    s = s.replace(old, new)


here = os.path.dirname(os.path.abspath(__file__))
kernel = "\n" + open(os.path.join(here, "ds4_fa_mqa_kernel.c")).read() + """
static void ggml_compute_forward_flash_attn_ext_f16(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
"""

sub('''
static void ggml_compute_forward_flash_attn_ext_f16(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
''', kernel)

sub('''    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t Dk = nek0;
    const int64_t Dv = nev0;
    const int64_t N  = neq1;
''', '''    if (ggml_fa_mqa_decode(params, dst)) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t Dk = nek0;
    const int64_t Dv = nev0;
    const int64_t N  = neq1;
''')

sub('''                    cur = 3*sizeof(float)*D*n_tasks; // 3x head size/thread
#if GGML_USE_IQK_MULMAT
                    size_t size = iqk_fa_work_buffer_size(node, n_tasks);
                    cur = MAX(cur, size);
#endif
''', '''                    cur = 3*sizeof(float)*D*n_tasks; // 3x head size/thread
#if GGML_USE_IQK_MULMAT
                    size_t size = iqk_fa_work_buffer_size(node, n_tasks);
                    cur = MAX(cur, size);
#endif
                    cur = MAX(cur, sizeof(float)*ggml_fa_mqa_work_floats(node, n_tasks));
''')

# lazy K/V concat: marked in ggml_graph_plan, skipped here, read in place by the kernel
sub('''static void ggml_compute_forward_concat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
''', '''#define GGML_FA_MQA_LAZY 0x4c617a79

static void ggml_compute_forward_concat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    if (ggml_get_op_params_i32(dst, 1) == GGML_FA_MQA_LAZY) {
        return; // read in place by ggml_fa_mqa_decode
    }
''')

sub('''    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
''', '''    int max_tasks = 1;

    ggml_fa_mqa_mark_lazy((struct ggml_cgraph *) cgraph);

    // thread scheduling for the different operations + work buffer size estimation
''')

open(p, "w").write(s)
print("MQA decode flash attention applied")
