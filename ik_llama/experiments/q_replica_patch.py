#!/usr/bin/env python3
# Small-batch matmuls on several NUMA nodes (GGML_NUMA_BLOCKED): each node quantizes src1 into its own
# copy of the work buffer with its own threads and waits for those threads only, so the matmul reads
# the quantized activations from its own node instead of from the few threads that wrote them.
# Not kept: the verify graph went from 294.5 to 302.3 ms. Splitting the quantization over all 64
# threads (one copy) was slower too (303.5 ms), and an AVX1 quantizer in place of the SIMDe one
# changed nothing (295.3 ms): the cost is moving the activations between sockets, not converting them.
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml.c"
s = open(p).read()

def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"expected {count} of:\n{old}\nfound {n}")
    s = s.replace(old, new)

# node-local barrier state
rep("""    _Alignas(64) atomic_int hb_global;
};""", """    _Alignas(64) atomic_int hb_global;
    // node-local barrier (ggml_node_barrier): only the threads of one node meet
    struct { _Alignas(64) atomic_int v; } hb_lcount[8];
    struct { _Alignas(64) atomic_int v; } hb_lrelease[8];
};""")
rep("""    for (int k = 0; k < n_nodes; ++k) {
        atomic_store(&shared->hb_count[k].v, 0);
        atomic_store(&shared->hb_release[k].v, 0);
    }""", """    for (int k = 0; k < n_nodes; ++k) {
        atomic_store(&shared->hb_count[k].v, 0);
        atomic_store(&shared->hb_release[k].v, 0);
        atomic_store(&shared->hb_lcount[k].v, 0);
        atomic_store(&shared->hb_lrelease[k].v, 0);
    }""")
rep("""#ifdef GGML_USE_OPENMP
static void ggml_barrier(struct ggml_compute_state_shared * shared) {""", """// the threads of the calling thread's node only (GGML_NUMA_BLOCKED)
static void ggml_node_barrier(struct ggml_compute_state_shared * shared) {
    const int node = g_hb_node;
    const int epoch = atomic_load_explicit(&shared->hb_lrelease[node].v, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&shared->hb_lcount[node].v, 1, memory_order_acq_rel) == shared->hb_node_n[node] - 1) {
        atomic_store_explicit(&shared->hb_lcount[node].v, 0, memory_order_relaxed);
        atomic_store_explicit(&shared->hb_lrelease[node].v, epoch + 1, memory_order_release);
        return;
    }
    while (atomic_load_explicit(&shared->hb_lrelease[node].v, memory_order_acquire) == epoch) {
#if defined(__SSE3__)
        _mm_pause();
#endif
    }
}

#ifdef GGML_USE_OPENMP
static void ggml_barrier(struct ggml_compute_state_shared * shared) {""")

helper = """// Quantize the rows of src1 (f32) for a matmul with threads ith of nth. For the x4 types the rows
// are split into units of 128 values (one x4 block), so a few long rows still use every thread.
static void ggml_quantize_src1_mt(const struct ggml_tensor * src1, enum ggml_type vec_dot_type, ggml_from_float_t from_float,
        char * wdata, size_t nbw1, size_t nbw2, size_t nbw3, int ith, int nth) {
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const bool x4 = (vec_dot_type == GGML_TYPE_Q8_0_X4 || vec_dot_type == GGML_TYPE_Q8_1_X4 || vec_dot_type == GGML_TYPE_Q8_2_X4) &&
                    ne10 % 128 == 0;
    const int64_t unit = x4 ? 128 : ne10;
    const size_t  unit_bytes = x4 ? ggml_row_size(vec_dot_type, 128) : nbw1;
    const int64_t upr = ne10/unit;
    const int64_t nu  = upr*ne11*ne12*ne13;
    const int64_t u_end = nu*(ith + 1)/nth;
    for (int64_t u = nu*ith/nth; u < u_end; ) {
        const int64_t ir = u/upr, iu = u - ir*upr;
        const int64_t n  = MIN(u_end - u, upr - iu);
        const int64_t i13 = ir/(ne11*ne12);
        const int64_t i12 = (ir - i13*ne11*ne12)/ne11;
        const int64_t i11 = ir - i13*ne11*ne12 - i12*ne11;
        from_float((const float *)((const char *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1]) + iu*unit,
                   wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + iu*unit_bytes, n*unit);
        u += n;
    }
}

// GGML_Q_REPLICA (default on): for a small batch, each NUMA node quantizes src1 into its own copy
// of the work buffer, so the matmul threads read it from their own node instead of from the few
// threads (on other nodes) that wrote it. Returns the copy of this thread, or -1.
#define GGML_Q_REPLICA_MAX_ROWS 32
static int ggml_q_replica_node(const struct ggml_compute_params * params, const struct ggml_tensor * src1) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GGML_Q_REPLICA");
        on = e ? atoi(e) : 1;
    }
    if (!on || params->shared->hb_n_nodes < 2 || src1->type != GGML_TYPE_F32 || ggml_nrows(src1) > GGML_Q_REPLICA_MAX_ROWS) {
        return -1;
    }
    return g_hb_node;
}

// quantize src1 into this node's copy (threads of the node only) and wait for the node
static void ggml_quantize_src1_node(const struct ggml_compute_params * params, int node, const struct ggml_tensor * src1,
        enum ggml_type vec_dot_type, ggml_from_float_t from_float, char * wdata, size_t nbw1, size_t nbw2, size_t nbw3) {
    struct ggml_compute_state_shared * sh = params->shared;
    const int i0 = (node*params->nth + sh->hb_n_nodes - 1)/sh->hb_n_nodes;
    ggml_quantize_src1_mt(src1, vec_dot_type, from_float, wdata, nbw1, nbw2, nbw3, params->ith - i0, sh->hb_node_n[node]);
    ggml_node_barrier(sh);
}

"""
rep("static int ggml_compute_forward_mul_mat(\n", helper + "static int ggml_compute_forward_mul_mat(\n")

# mul_mat: the replica path
rep("""    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

#if IK_PRINT_TIMING
        int64_t t1 = ggml_time_us();
#endif
""", """    const int q_node = src1->type != vec_dot_type ? ggml_q_replica_node(params, src1) : -1;
    const size_t q_off = q_node >= 0 ? q_node*GGML_PAD(ggml_row_size(vec_dot_type, ne10)*ne11*ne12*ne13, 64) : 0;
    if (q_node >= 0) {
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        GGML_ASSERT(params->wsize >= q_off + nbw1*ne11*ne12*ne13);
        ggml_quantize_src1_node(params, q_node, src1, vec_dot_type, from_float, (char *) params->wdata + q_off, nbw1, nbw1*ne11, nbw1*ne11*ne12);
    }
    else if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

#if IK_PRINT_TIMING
        int64_t t1 = ggml_time_us();
#endif
""")
old_w = """    const void * wdata    = (src1->type == vec_dot_type) ? src1->data : params->wdata;
"""
a = s.index(old_w, s.index("static int ggml_compute_forward_mul_mat(\n"))
s = s[:a] + """    const void * wdata    = (src1->type == vec_dot_type) ? src1->data : (const char *) params->wdata + q_off;
""" + s[a + len(old_w):]

# plan: room for one copy per node
rep("""                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type != GGML_TYPE_F32) {""", """                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type == GGML_TYPE_F32 && ggml_nrows(node->src[1]) <= GGML_Q_REPLICA_MAX_ROWS && g_state.numa.n_nodes > 1) {
                            cur = g_state.numa.n_nodes*GGML_PAD(cur, 64);  // ggml_q_replica_node
                        }
                        if (node->src[1]->type != GGML_TYPE_F32) {""")
rep("""                    if (src2->type != vec_dot_type) {
                        cur += ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                    }
                } break;
            case GGML_OP_OUT_PROD:""", """                    if (src2->type != vec_dot_type) {
                        cur += ggml_row_size(vec_dot_type, node->src[1]->ne[0]) * ggml_nrows(node->src[1]);
                        if (node->src[1]->type == GGML_TYPE_F32 && ggml_nrows(node->src[1]) <= GGML_Q_REPLICA_MAX_ROWS && g_state.numa.n_nodes > 1) {
                            cur = g_state.numa.n_nodes*GGML_PAD(cur, 64);  // ggml_q_replica_node
                        }
                    }
                } break;
            case GGML_OP_OUT_PROD:""")

# fused up_gate (dense shared expert)
rep("""    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                           (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                           ne10);
            }
        }
    }

    ggml_barrier(params->shared);

    float limit""", """    const int q_node = ggml_q_replica_node(params, src1);
    if (q_node >= 0) {
        wdata += q_node*GGML_PAD(ne13*nbw3, 64);
        GGML_ASSERT(params->wsize >= (size_t)(wdata - (char *)params->wdata) + ne13*nbw3);
        ggml_quantize_src1_node(params, q_node, src1, vec_dot_type, from_float, wdata, nbw1, nbw2, nbw3);
    } else {
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }

        ggml_barrier(params->shared);
    }

    float limit""")

open(p, "w").write(s)
print("patched")
