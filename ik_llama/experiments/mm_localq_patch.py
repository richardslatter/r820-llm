#!/usr/bin/env python3
"""Per-thread activation quantization for single-row matmuls (run from the ik_llama.cpp root).

At decode a quantized matmul gets one f32 activation row. ggml converts it to the weight's
vec_dot_type on thread 0 alone, then all threads meet at a barrier and read that one copy. When
the row was written by every thread (the split norms), thread 0 first pulls it from all four
sockets, and then 63 threads pull its copy from thread 0's cache: a fixed ~60 us on matmuls whose
weights stream in 20-80 us. With one f32 src1 row every thread now converts the row itself
into its own slot of the work buffer and goes straight to the dot products, with no barrier.
GGML_MM_LOCALQ=0 goes back to the shared copy.
"""
import sys

p = "ggml/src/ggml.c"
s = open(p).read()
if "ggml_mul_mat_local_q" in s:
    sys.exit(f"{p}: already patched")


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"{p}: anchor not found: {old[:70]!r}")
    s = s.replace(old, new)


helper = '''// one f32 src1 row: every thread converts it into its own slot of the work buffer, placed after
// the shared copy (returns the bytes the slots need, 0 = shared copy only)
static size_t ggml_mul_mat_local_q(const struct ggml_tensor * dst, int nth) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_MM_LOCALQ");
        enabled = !(e && atoi(e) == 0);
    }
    const struct ggml_tensor * src1 = dst->src[1];
    const enum ggml_type vec_dot_type = type_traits[dst->src[0]->type].vec_dot_type;
    if (!enabled || nth < 2 || src1->type != GGML_TYPE_F32 || vec_dot_type == GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 || ggml_nrows(src1) != 1) {
        return 0;
    }
    return 64 + nth*GGML_PAD(ggml_row_size(vec_dot_type, src1->ne[0]), 64);
}

static int ggml_compute_forward_mul_mat(
'''

sub('''
static int ggml_compute_forward_mul_mat(
''', "\n" + helper)

sub('''#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

#if IK_PRINT_TIMING
        int64_t t1 = ggml_time_us();
#endif
''', '''#endif

    const size_t nbq1 = ggml_row_size(vec_dot_type, ne10)*ggml_nrows(src1);
    char * wq_local = NULL;
    {
        const size_t ws = ggml_mul_mat_local_q(dst, nth);
        if (ws && params->wsize >= nbq1 + ws) {
            wq_local = (char *)(((uintptr_t) params->wdata + nbq1 + 63) & ~(uintptr_t) 63) + ith*GGML_PAD(nbq1, 64);
            from_float((const float *) src1->data, wq_local, ne10);
        }
    }

    if (src1->type != vec_dot_type && !wq_local) {
        char * wdata = params->wdata;

#if IK_PRINT_TIMING
        int64_t t1 = ggml_time_us();
#endif
''')

sub('''    const void * wdata    = (src1->type == vec_dot_type) ? src1->data : params->wdata;

    if (src1->type != vec_dot_type && dst->type == GGML_TYPE_F32) {''', '''    const void * wdata    = (src1->type == vec_dot_type) ? src1->data : wq_local ? wq_local : params->wdata;

    if (src1->type != vec_dot_type && dst->type == GGML_TYPE_F32) {''')

sub('''        // no IQK kernel: fall through to the generic vec_dot path
    }
''', '''        // no IQK kernel: fall through to the generic vec_dot path
    }

    if (wq_local) {
        // the generic path reads the shared copy
        if (ith == 0) {
            from_float((const float *) src1->data, params->wdata, ne10);
        }
        ggml_barrier(params->shared);
    }
''')

# work buffer: the shared copy, then the per-thread slots
sub('''                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type != GGML_TYPE_F32) {
                            cur += n_tasks*node->src[1]->ne[0]*sizeof(float); // src1->type -> f32 -> vec_dot_type
                        }
                    }
''', '''                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type != GGML_TYPE_F32) {
                            cur += n_tasks*node->src[1]->ne[0]*sizeof(float); // src1->type -> f32 -> vec_dot_type
                        }
                        cur += ggml_mul_mat_local_q(node, n_tasks);
                    }
''')

open(p, "w").write(s)
print("per-thread single-row quantization applied")
