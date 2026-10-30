#!/usr/bin/env python3
"""f32 matmuls with fewer rows than threads: split K the way the split rms_norm does.

DeepSeek V4's hyper-connection mixes (hc_attn_fn, hc_ffn_fn: f32, 24 rows of 20480, 80 per
token) multiply the output of an rms_norm that, with few tokens, is split along K: thread ith
writes slice ith of every row. Split by rows, 24 threads each then read the whole activation,
just written on all four sockets, one remote cache line at a time: ~175 us per call for 4 tokens
against ~35 us with the activation in cache.

Here thread ith takes the same K-slice of all rows, so it reads only what it wrote itself, and
writes its partial sums to the work buffer; after a barrier the partial sums are added in thread
order. GGML_MUL_MAT_KSLICE=0 goes back to the row split.

Usage: kslice_patch.py <ik_llama.cpp checkout>
"""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = root + "/ggml/src/ggml.c"
s = open(p).read()


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"{p}: anchor found {s.count(old)} times: {old[:70]!r}")
    s = s.replace(old, new)


helper = r'''// f32 x f32 matmuls with fewer rows than threads and a few long src1 rows (DeepSeek V4's
// hyper-connection mixes: 24 rows of 20480): every thread takes the same K-slice of all rows that
// the split rms_norm (ggml_rms_norm_f32_split) gives it, so it reads src1 values it wrote itself
// rather than whole rows written on other sockets. Partial sums go to the work buffer and are
// added after a barrier. GGML_MUL_MAT_KSLICE=0 turns it off.
static bool ggml_mul_mat_kslice_on(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MUL_MAT_KSLICE");
        enabled = env == NULL || atoi(env) != 0;
    }
    return enabled;
}

static bool ggml_mul_mat_kslice(const struct ggml_tensor * dst, int nth) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    return ggml_mul_mat_kslice_on() && nth > 1 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[1] < nth && src1->ne[1] <= 8 && src0->ne[0] >= 4096 && src0->ne[0] % 16 == 0 &&
        ggml_nrows(src0) == src0->ne[1] && ggml_nrows(src1) == src1->ne[1] &&
        src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float);
}

static size_t ggml_mul_mat_kslice_wsize(const struct ggml_tensor * dst, int nth) {
    return ggml_mul_mat_kslice(dst, nth) ? 64 + nth*dst->src[0]->ne[1]*dst->src[1]->ne[1]*sizeof(float) : 0;
}

'''
sub("static int ggml_compute_forward_mul_mat(\n", helper + "static int ggml_compute_forward_mul_mat(\n")

sub("""#if GGML_USE_IQK_MULMAT
    if (dst->type == GGML_TYPE_F32) {
        if (iqk_mul_mat_4d(ne01, ne11, ne00,""",
"""#if GGML_USE_IQK_MULMAT
    if (ggml_mul_mat_kslice(dst, nth)) {
        // the K-slice of ggml_rms_norm_f32_split; partial sums [thread][i11][i01]
        const int64_t k0 = (ne00/16)*ith/nth*16;
        const int64_t k1 = ith == nth - 1 ? ne00 : (ne00/16)*(ith + 1)/nth*16;
        const int64_t no = ne01*ne11;
        float * part = (float *)(((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
        if (k1 > k0) {
            if (!iqk_mul_mat(ne01, ne11, k1 - k0, GGML_TYPE_F32, (const char *)src0->data + k0*sizeof(float), nb01,
                        GGML_TYPE_F32, (const char *)src1->data + k0*sizeof(float), nb11, part + ith*no, ne01, 0, 1)) {
                GGML_ABORT("fatal error");
            }
        } else {
            memset(part + ith*no, 0, no*sizeof(float));
        }
        ggml_barrier(params->shared);
        for (int64_t o = no*ith/nth; o < no*(ith + 1)/nth; ++o) {
            float sum = 0;
            for (int j = 0; j < nth; ++j) sum += part[j*no + o];
            *(float *)((char *)dst->data + (o % ne01)*nb0 + (o / ne01)*nb1) = sum;
        }
        return node_n;
    }
    if (dst->type == GGML_TYPE_F32) {
        if (iqk_mul_mat_4d(ne01, ne11, ne00,""")

old_plan = """            case GGML_OP_MUL_MAT:
                {
                    const enum ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type != GGML_TYPE_F32) {
                            cur += n_tasks*node->src[1]->ne[0]*sizeof(float); // src1->type -> f32 -> vec_dot_type
                        }
                    }
"""
sub(old_plan, old_plan + """                    cur = MAX(cur, ggml_mul_mat_kslice_wsize(node, n_tasks));
""")
sub("""    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const int64_t nrows = t->ne[1];
    size_t placed = 0;
""", """    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const int64_t nrows = t->ne[1];
    size_t placed = 0;
    if (t->type == GGML_TYPE_F32 && nrows < 64 && t->ne[0] >= 4096 && t->ne[0] % 16 == 0 && t->ne[2] == 1 && t->ne[3] == 1 &&
        t->nb[1] == t->ne[0]*sizeof(float) && ggml_mul_mat_kslice_on()) {
        // few long f32 rows go through ggml_mul_mat_kslice: node k's threads read columns
        // [k*ne0/n, (k+1)*ne0/n) of every row, so those columns go to node k
        for (int64_t i1 = 0; i1 < nrows; ++i1) {
            const uintptr_t base = (uintptr_t)t->data + i1*t->nb[1];
            for (int k = 0; k < n_nodes; ++k) {
                const uintptr_t lo = (base + (k*t->ne[0]/n_nodes)*sizeof(float) + page - 1) & ~(page - 1);
                const uintptr_t hi = (base + ((k + 1)*t->ne[0]/n_nodes)*sizeof(float) + page - 1) & ~(page - 1);
                if (hi <= lo) continue;
                unsigned long mask[2] = { 1ul << k, 0 };
                madvise((void *)lo, hi - lo, 22 /*MADV_POPULATE_READ*/);
                if (syscall(SYS_mbind, (void *)lo, hi - lo, 2, mask, 8*sizeof(mask[0]) + 1, 2) == 0) {
                    placed += hi - lo;
                }
            }
        }
        return placed;
    }
""")
open(p, "w").write(s)
print("patched", p)
