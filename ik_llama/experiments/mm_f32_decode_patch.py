#!/usr/bin/env python3
"""Decode f32 matmuls with a node-local copy of the activation (run from the ik_llama.cpp root).

DeepSeek-V4's hyper-connection mixes (hc_attn_fn / hc_ffn_fn, f32 20480 x 24, 80 per token) and
the router (ffn_gate_inp, f32 5120 x 384) multiply f32 weights by one f32 activation row. Every
thread reads the whole row, mostly from another socket: for the hc mixes that is 24 x 80 KB of
cross-socket reads for 2 MB of weights, and the time follows the total (~25 GB/s, the same for
24 rows of 20480 or 48 of 10240). Splitting K alone does not help, since the activation bytes
read stay the same; it measured slower.

With GGML_NUMA_BLOCKED, the threads of each node now first copy their share of the row into a
per-node slot of the work buffer (the row then crosses the interconnect once per node), and
after a barrier compute from that copy. With fewer rows than threads each row is also cut into
nth/rows K-chunks whose partial sums are added after a second barrier. Consecutive tasks go to
consecutive threads, so with GGML_NUMA_PLACE a row is read on the node that holds it.
GGML_MM_F32LOCAL=0 goes back to the normal path.

Negative result, not applied: no gain in the model (see docs/plan.md).
"""
import sys

p = "ggml/src/ggml.c"
s = open(p).read()
if "ggml_mul_mat_f32_local" in s:
    sys.exit(f"{p}: already patched")


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"{p}: anchor not found: {old[:70]!r}")
    s = s.replace(old, new)


helper = '''// one f32 activation row against 2D f32 weights, threads blocked by NUMA node: K-chunks per
// row (1 = plain row split), 0 = not taken
static int ggml_mul_mat_f32_local(const struct ggml_tensor * dst, int nth) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_MM_F32LOCAL");
        enabled = !(e && atoi(e) == 0);
    }
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    if (!enabled || !g_numa_blocked || g_state.numa.n_nodes < 2 || nth < (int) g_state.numa.n_nodes ||
        src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        ggml_nrows(src1) != 1 || ggml_nrows(src0) != src0->ne[1] || src0->ne[0] < 1024 || !ggml_is_contiguous(src1)) {
        return 0;
    }
    const int nk = src0->ne[1] >= nth ? 1 : nth/(int) src0->ne[1];
    return src0->ne[0] >= 64*nk ? nk : 1;
}

// work buffer: partial sums (nk > 1), then one activation copy per node
static size_t ggml_mul_mat_f32_local_wsize(const struct ggml_tensor * dst, int nk) {
    const int64_t ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1];
    return GGML_PAD(ne01*nk*sizeof(float), 64) + g_state.numa.n_nodes*GGML_PAD(ne00*sizeof(float), 64) + 64;
}

static int ggml_compute_forward_mul_mat(
'''

sub('''
static int ggml_compute_forward_mul_mat(
''', "\n" + helper)

sub('''    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

#if GGML_USE_IQK_MULMAT
    if (dst->type == GGML_TYPE_F32) {
        if (iqk_mul_mat_4d(ne01, ne11, ne00,''', '''    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    const int nk_local = ggml_mul_mat_f32_local(dst, nth);
    if (nk_local && params->wsize >= ggml_mul_mat_f32_local_wsize(dst, nk_local)) {
        const int nk = nk_local;
        const int n_nodes = (int) g_state.numa.n_nodes;
        const int node  = ith*n_nodes/nth;
        const int first = (node*nth + n_nodes - 1)/n_nodes, last = ((node + 1)*nth + n_nodes - 1)/n_nodes;
        float * part = (float *)(((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
        float * xn   = (float *)((char *) part + GGML_PAD(ne01*nk*sizeof(float), 64) + node*GGML_PAD(ne00*sizeof(float), 64));
        {
            // this node's copy of the activation, each of its threads copying a slice
            const int64_t j = ith - first, m = last - first;
            const int64_t k0 = ne00*j/m, k1 = ne00*(j + 1)/m;
            memcpy(xn + k0, (const float *) src1->data + k0, (k1 - k0)*sizeof(float));
        }
        ggml_barrier(params->shared);
        // task t = row*nk + chunk; thread ith takes tasks [ceil(ith*nt/nth), ceil((ith+1)*nt/nth))
        const int64_t nt = ne01*nk;
        for (int64_t t = (ith*nt + nth - 1)/nth; t < ((ith + 1)*nt + nth - 1)/nth; ++t) {
            const int64_t r = t/nk, c = t - r*nk;
            const int64_t k0 = (ne00/16)*c/nk*16;
            const int64_t k1 = c == nk - 1 ? ne00 : (ne00/16)*(c + 1)/nk*16;
            ggml_vec_dot_f32((int)(k1 - k0), nk > 1 ? part + t : (float *) dst->data + r, 0,
                    (const float *)((const char *) src0->data + r*nb01) + k0, 0, xn + k0, 0, 1);
        }
        if (nk > 1) {
            ggml_barrier(params->shared);
            for (int64_t r = ith; r < ne01; r += nth) {
                float sum = 0.0f;
                for (int c = 0; c < nk; ++c) sum += part[r*nk + c];
                ((float *) dst->data)[r] = sum;
            }
        }
        return node_n;
    }

#if GGML_USE_IQK_MULMAT
    if (dst->type == GGML_TYPE_F32) {
        if (iqk_mul_mat_4d(ne01, ne11, ne00,''')

sub('''                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type != GGML_TYPE_F32) {
                            cur += n_tasks*node->src[1]->ne[0]*sizeof(float); // src1->type -> f32 -> vec_dot_type
                        }
                    }
                } break;''', '''                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type != GGML_TYPE_F32) {
                            cur += n_tasks*node->src[1]->ne[0]*sizeof(float); // src1->type -> f32 -> vec_dot_type
                        }
                    }
                    if (const int nk = ggml_mul_mat_f32_local(node, n_tasks)) {
                        cur = MAX(cur, ggml_mul_mat_f32_local_wsize(node, nk));
                    }
                } break;''')

open(p, "w").write(s)
print("node-local f32 decode matmul applied")
