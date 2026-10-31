// rows of K/V converted and scored per block
#define GGML_FA_MQA_BR 32

// work buffer floats ggml_fa_mqa_decode needs (0 if the node is not its shape)
static size_t ggml_fa_mqa_work_floats(const struct ggml_tensor * dst, int nth) {
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    if (q->ne[1] > 8 || k->ne[2] != 1 || v->ne[2] != 1 || q->ne[3] != 1) return 0;
    const size_t H = q->ne[2], Dk = k->ne[0], Dv = v->ne[0], nkv = k->ne[1];
    return 16 + q->ne[1]*8*H*(Dv + 16) + (size_t)nth*GGML_PAD(GGML_FA_MQA_BR*(Dk + Dv) + 2*H*GGML_FA_MQA_BR + 2*nkv + 64, 16);
}

// whether ggml_fa_mqa_decode takes this node (it also needs the work buffer of ggml_fa_mqa_work_floats)
static bool ggml_fa_mqa_ok(const struct ggml_tensor * dst) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_FA_MQA");
        enabled = !e || atoi(e) != 0;
    }
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3], * sinks = dst->src[4];
    if (!enabled || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) return false;
    if (mask && mask->type != GGML_TYPE_F16) return false;
    if (sinks && sinks->type != GGML_TYPE_F32) return false;
    if (k->ne[3] != 1 || v->ne[3] != 1 || k->ne[1] != v->ne[1] || k->ne[0] % 8 || v->ne[0] % 16) return false;
    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    return max_bias == 0.0f && ggml_fa_mqa_work_floats(dst, 1) != 0;
}

// DeepSeek-V4 builds K (= V) as permute(concat(raw_k, extra_k, dim 2)) every token: ~1 MB copied
// per layer at 512 tokens of context, of which attention then reads only the visible rows. When
// such a concat feeds nothing but one node this kernel takes, ggml_graph_plan marks it lazy (op
// param 1): it is not computed, and the kernel reads the rows from its two sources. A source is a
// view of a persistent tensor (the KV caches), or of a computed one (the HCA layers' gathered
// compressed rows, one get_rows shared by a group of layers) that no node run between it and the
// attention node writes over.
#ifndef GGML_FA_MQA_LAZY
#define GGML_FA_MQA_LAZY 0x4c617a79
#endif
#define GGML_FA_MQA_LAZY_CAND 0x4c617a43

static struct ggml_tensor * ggml_fa_mqa_concat_of(const struct ggml_tensor * t) {
    if (t->op != GGML_OP_PERMUTE || t->view_offs != 0) return NULL;
    struct ggml_tensor * c = t->src[0];
    if (!c || c->op != GGML_OP_CONCAT || ggml_get_op_params_i32(c, 0) != 2) return NULL;
    if (t->ne[1] != c->ne[2] || t->nb[1] != c->nb[2] || c->ne[1] != 1 || c->ne[3] != 1) return NULL;
    for (int i = 0; i < 2; ++i) {
        const struct ggml_tensor * s = c->src[i];
        if (s->type != c->type || s->ne[0] != c->ne[0] || s->ne[1] != 1 || s->ne[3] != 1 ||
            s->nb[0] != ggml_type_size(s->type)) return NULL;
    }
    return c;
}

// true if computed source s of the concat read by gf->nodes[ifa] still holds its data there: its
// root is computed earlier and no node run in between writes into the root's bytes
static bool ggml_fa_mqa_intact(const struct ggml_cgraph * gf, int ifa, const struct ggml_tensor * s) {
    const struct ggml_tensor * r = s->view_src ? s->view_src : s;
    if (!r->data) return false;
    const char * lo = (const char *) r->data, * hi = lo + ggml_nbytes(r);
    for (int n = ifa - 1; n >= 0; --n) {
        const struct ggml_tensor * x = gf->nodes[n];
        if (x == r) return true;
        switch (x->op) {
            case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
                continue;
            case GGML_OP_CONCAT:
                if (x->op_params[1] == GGML_FA_MQA_LAZY || x->op_params[1] == GGML_FA_MQA_LAZY_CAND) continue;
                break;
            default:
                break;
        }
        if (x->data && (const char *) x->data < hi && lo < (const char *) x->data + ggml_nbytes(x)) return false;
    }
    return false;
}

static void ggml_fa_mqa_mark_lazy(struct ggml_cgraph * gf) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_FA_LAZY");
        enabled = !e || atoi(e) != 0;
    }
    bool any = false;
    for (int i = 0; i < gf->n_nodes; ++i) {
        struct ggml_tensor * t = gf->nodes[i];
        if (t->op == GGML_OP_CONCAT) {
            if (t->op_params[1] == GGML_FA_MQA_LAZY || t->op_params[1] == GGML_FA_MQA_LAZY_CAND) t->op_params[1] = 0;
        } else if (enabled && t->op == GGML_OP_FLASH_ATTN_EXT && ggml_fa_mqa_ok(t)) {
            struct ggml_tensor * c = ggml_fa_mqa_concat_of(t->src[1]);
            if (c && ggml_fa_mqa_concat_of(t->src[2]) == c && !(c->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                c->op_params[1] = GGML_FA_MQA_LAZY_CAND; c->op_params[2] = 0; c->op_params[3] = 0;
                any = true;
            }
        }
    }
    if (!any) return;
    // count the users of each candidate (op param 2) and of its permutes (op param 3)
    for (int i = 0; i < gf->n_nodes; ++i) {
        const struct ggml_tensor * t = gf->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            struct ggml_tensor * s = t->src[j];
            if (!s) continue;
            if (s->op == GGML_OP_CONCAT && s->op_params[1] == GGML_FA_MQA_LAZY_CAND) {
                s->op_params[2]++;
            } else if (s->src[0] && s->src[0]->op == GGML_OP_CONCAT && s->src[0]->op_params[1] == GGML_FA_MQA_LAZY_CAND) {
                s->src[0]->op_params[3]++;
            }
        }
    }
    // lazy only if its permutes are its only users, the attention node their only one, and its
    // computed sources are intact when the attention node runs
    for (int i = 0; i < gf->n_nodes; ++i) {
        const struct ggml_tensor * t = gf->nodes[i];
        if (t->op != GGML_OP_FLASH_ATTN_EXT || t->src[1]->op != GGML_OP_PERMUTE) continue;
        struct ggml_tensor * c = t->src[1]->src[0];
        if (!c || c->op != GGML_OP_CONCAT || c->op_params[1] != GGML_FA_MQA_LAZY_CAND) continue;
        const int users = t->src[1] == t->src[2] ? 1 : 2;
        bool ok = c->op_params[2] == users && c->op_params[3] == 2;
        for (int j = 0; j < 2 && ok; ++j) {
            const struct ggml_tensor * s = c->src[j];
            const struct ggml_tensor * r = s->view_src ? s->view_src : s;
            ok = r->op == GGML_OP_NONE || ggml_fa_mqa_intact(gf, i, s);
        }
        c->op_params[1] = ok ? GGML_FA_MQA_LAZY : 0;
    }
}

// the concat behind K or V if it was marked lazy
static const struct ggml_tensor * ggml_fa_mqa_lazy(const struct ggml_tensor * t) {
    const struct ggml_tensor * c = t->op == GGML_OP_PERMUTE ? t->src[0] : NULL;
    return c && c->op == GGML_OP_CONCAT && ggml_get_op_params_i32(c, 1) == GGML_FA_MQA_LAZY ? c : NULL;
}

// row i of K or V (t): from the sources of its lazy concat lz, if it has one
static inline const ggml_fp16_t * ggml_fa_mqa_row(const struct ggml_tensor * t, const struct ggml_tensor * lz, int64_t i) {
    if (lz) {
        const struct ggml_tensor * a = lz->src[0], * b = lz->src[1];
        return (const ggml_fp16_t *)(i < a->ne[2] ? (const char *) a->data + i*a->nb[2]
                                                  : (const char *) b->data + (i - a->ne[2])*b->nb[2]);
    }
    return (const ggml_fp16_t *)((const char *) t->data + i*t->nb[1]);
}

#if defined(__AVX__) && !defined(__AVX512F__)
// the four horizontal sums of a0..a3
static inline __m128 ggml_fa_hsum8x4(__m256 a0, __m256 a1, __m256 a2, __m256 a3) {
    const __m256 t01 = _mm256_hadd_ps(a0, a1), t23 = _mm256_hadd_ps(a2, a3);
    const __m256 t = _mm256_hadd_ps(t01, t23);
    return _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
}

static void ggml_fa_f16_to_f32_row(const ggml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i += 8) {
        _mm256_storeu_ps(y + i, GGML_F32Cx8_LOAD((ggml_fp16_t *)(x + i)));
    }
}

// scores of rows K[0..3] against heads q0, q1
static inline void ggml_fa_qk_2x4(const float * q0, const float * q1, const float * K, int64_t Dk, float * s0, float * s1) {
    __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
    __m256 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    for (int64_t d = 0; d < Dk; d += 8) {
        const __m256 x0 = _mm256_loadu_ps(q0 + d), x1 = _mm256_loadu_ps(q1 + d);
        __m256 k = _mm256_loadu_ps(K + 0*Dk + d);
        a00 = _mm256_add_ps(a00, _mm256_mul_ps(x0, k)); a10 = _mm256_add_ps(a10, _mm256_mul_ps(x1, k));
        k = _mm256_loadu_ps(K + 1*Dk + d);
        a01 = _mm256_add_ps(a01, _mm256_mul_ps(x0, k)); a11 = _mm256_add_ps(a11, _mm256_mul_ps(x1, k));
        k = _mm256_loadu_ps(K + 2*Dk + d);
        a02 = _mm256_add_ps(a02, _mm256_mul_ps(x0, k)); a12 = _mm256_add_ps(a12, _mm256_mul_ps(x1, k));
        k = _mm256_loadu_ps(K + 3*Dk + d);
        a03 = _mm256_add_ps(a03, _mm256_mul_ps(x0, k)); a13 = _mm256_add_ps(a13, _mm256_mul_ps(x1, k));
    }
    _mm_storeu_ps(s0, ggml_fa_hsum8x4(a00, a01, a02, a03));
    _mm_storeu_ps(s1, ggml_fa_hsum8x4(a10, a11, a12, a13));
}

static inline void ggml_fa_qk_1x4(const float * q0, const float * K, int64_t Dk, float * s0) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int64_t d = 0; d < Dk; d += 8) {
        const __m256 x0 = _mm256_loadu_ps(q0 + d);
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(x0, _mm256_loadu_ps(K + 0*Dk + d)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(x0, _mm256_loadu_ps(K + 1*Dk + d)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(x0, _mm256_loadu_ps(K + 2*Dk + d)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(x0, _mm256_loadu_ps(K + 3*Dk + d)));
    }
    _mm_storeu_ps(s0, ggml_fa_hsum8x4(a0, a1, a2, a3));
}

// o_h[0..Dv) = o_h*corr_h + sum_j p_h[j]*V[j] for 4 heads (o_h = o + h*ostride, p_h = p + h*GGML_FA_MQA_BR)
static inline void ggml_fa_pv_4(float * o, int64_t ostride, const float * p, const float * corr,
                                const float * V, int64_t Dv, int nb) {
    const __m256 c0 = _mm256_set1_ps(corr[0]), c1 = _mm256_set1_ps(corr[1]);
    const __m256 c2 = _mm256_set1_ps(corr[2]), c3 = _mm256_set1_ps(corr[3]);
    float * o0 = o, * o1 = o + ostride, * o2 = o + 2*ostride, * o3 = o + 3*ostride;
    const float * p0 = p, * p1 = p + GGML_FA_MQA_BR, * p2 = p + 2*GGML_FA_MQA_BR, * p3 = p + 3*GGML_FA_MQA_BR;
    for (int64_t d = 0; d < Dv; d += 16) {
        __m256 a00 = _mm256_mul_ps(c0, _mm256_loadu_ps(o0 + d)), a01 = _mm256_mul_ps(c0, _mm256_loadu_ps(o0 + d + 8));
        __m256 a10 = _mm256_mul_ps(c1, _mm256_loadu_ps(o1 + d)), a11 = _mm256_mul_ps(c1, _mm256_loadu_ps(o1 + d + 8));
        __m256 a20 = _mm256_mul_ps(c2, _mm256_loadu_ps(o2 + d)), a21 = _mm256_mul_ps(c2, _mm256_loadu_ps(o2 + d + 8));
        __m256 a30 = _mm256_mul_ps(c3, _mm256_loadu_ps(o3 + d)), a31 = _mm256_mul_ps(c3, _mm256_loadu_ps(o3 + d + 8));
        const float * vr = V + d;
        for (int j = 0; j < nb; ++j, vr += Dv) {
            const __m256 v0 = _mm256_loadu_ps(vr), v1 = _mm256_loadu_ps(vr + 8);
            __m256 pj = _mm256_broadcast_ss(p0 + j);
            a00 = _mm256_add_ps(a00, _mm256_mul_ps(pj, v0)); a01 = _mm256_add_ps(a01, _mm256_mul_ps(pj, v1));
            pj = _mm256_broadcast_ss(p1 + j);
            a10 = _mm256_add_ps(a10, _mm256_mul_ps(pj, v0)); a11 = _mm256_add_ps(a11, _mm256_mul_ps(pj, v1));
            pj = _mm256_broadcast_ss(p2 + j);
            a20 = _mm256_add_ps(a20, _mm256_mul_ps(pj, v0)); a21 = _mm256_add_ps(a21, _mm256_mul_ps(pj, v1));
            pj = _mm256_broadcast_ss(p3 + j);
            a30 = _mm256_add_ps(a30, _mm256_mul_ps(pj, v0)); a31 = _mm256_add_ps(a31, _mm256_mul_ps(pj, v1));
        }
        _mm256_storeu_ps(o0 + d, a00); _mm256_storeu_ps(o0 + d + 8, a01);
        _mm256_storeu_ps(o1 + d, a10); _mm256_storeu_ps(o1 + d + 8, a11);
        _mm256_storeu_ps(o2 + d, a20); _mm256_storeu_ps(o2 + d + 8, a21);
        _mm256_storeu_ps(o3 + d, a30); _mm256_storeu_ps(o3 + d + 8, a31);
    }
}

static inline void ggml_fa_pv_1(float * o, const float * p, float corr, const float * V, int64_t Dv, int nb) {
    const __m256 c = _mm256_set1_ps(corr);
    for (int64_t d = 0; d < Dv; d += 16) {
        __m256 a0 = _mm256_mul_ps(c, _mm256_loadu_ps(o + d)), a1 = _mm256_mul_ps(c, _mm256_loadu_ps(o + d + 8));
        const float * vr = V + d;
        for (int j = 0; j < nb; ++j, vr += Dv) {
            const __m256 pj = _mm256_broadcast_ss(p + j);
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(pj, _mm256_loadu_ps(vr)));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(pj, _mm256_loadu_ps(vr + 8)));
        }
        _mm256_storeu_ps(o + d, a0); _mm256_storeu_ps(o + d + 8, a1);
    }
}

// DeepSeek-V4 decode attention: one shared fp16 K/V head, f32 Q, up to 8 query rows.
// Returns false (before touching anything) when the node is not that shape.
static bool ggml_fa_mqa_decode(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!ggml_fa_mqa_ok(dst)) return false;
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3], * sinks = dst->src[4];
    // K/V of a lazy concat exist only as its sources: no other path can read them
    const struct ggml_tensor * kl = ggml_fa_mqa_lazy(k), * vl = ggml_fa_mqa_lazy(v);
    float scale, softcap;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&softcap,  (float *) dst->op_params + 2, sizeof(float));
    const int ith = params->ith, nth = params->nth;
    if (params->wsize < ggml_fa_mqa_work_floats(dst, nth)*sizeof(float)) {
        GGML_ASSERT(!kl && !vl);
        return false;
    }
    if (softcap != 0.0f) scale /= softcap;

    const int64_t Dk = k->ne[0], Dv = v->ne[0], H = q->ne[2], NQ = q->ne[1], nkv = k->ne[1];
    const int BR = GGML_FA_MQA_BR;
    // without a mask, src[5] lists the K rows of each query (negative entries are padding)
    const struct ggml_tensor * idx = !mask ? dst->src[5] : NULL;

    // 64-byte aligned: the state of a head is Dv accumulators, max, sum, padding to SD floats
    const int64_t SD = Dv + 16;
    float * part    = (float *)(((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
    float * scratch = part + NQ*8*H*SD + (size_t)ith*GGML_PAD(BR*(Dk + Dv) + 2*H*BR + 2*nkv + 64, 16);
    float * Kt   = scratch;
    float * Vt   = Kt + BR*Dk;
    float * P    = Vt + BR*Dv;
    float * corr = P + H*BR;
    float * mvs  = corr + H*BR;
    int32_t * rows = (int32_t *)(mvs + nkv);

    // the same chunking on every thread: by the number of visible rows of query 0
    int64_t n0 = 0;
    if (mask) {
        const uint16_t * m0 = (const uint16_t *) mask->data;
        for (int64_t i = 0; i < nkv; ++i) n0 += m0[i] != 0xfc00;
    } else if (idx) {
        const int32_t * i0 = (const int32_t *) idx->data;
        for (int64_t i = 0; i < idx->ne[0]; ++i) n0 += i0[i] >= 0;
    } else {
        n0 = nkv;
    }
    // chunks of >= 64 rows, at most 8 (more chunks: fewer conversions per row, more merging)
    int C = 1;
    while (2*C <= 8 && 2*C <= nth && nth % (2*C) == 0 && n0 >= 64*C) C *= 2;
    const int G = nth / C;
    // consecutive threads share a chunk: with GGML_NUMA_BLOCKED a node then reads only the rows
    // of its own chunks (once from memory, then from its L3 for the other head groups)
    const int c = ith / G, g = ith % G;
    // DeepSeek-V4 passes the same tensor as K and V: convert each row once
    const bool kv_same = k->data == v->data && k->nb[1] == v->nb[1] && Dk == Dv;
    const int64_t h0 = H*g/G, h1 = H*(g + 1)/G;

    for (int64_t iq1 = 0; iq1 < NQ; ++iq1) {
        // visible rows of this query and their mask values
        int64_t n = 0;
        if (mask) {
            const uint16_t * mp = (const uint16_t *)((const char *) mask->data + iq1*mask->nb[1]);
            for (int64_t i = 0; i < nkv; ++i) {
                if (mp[i] == 0xfc00) continue;
                rows[n] = (int32_t) i; mvs[n] = GGML_FP16_TO_FP32(mp[i]); ++n;
            }
        } else if (idx) {
            const int32_t * ip = (const int32_t *)((const char *) idx->data + iq1*idx->nb[1]);
            for (int64_t i = 0; i < idx->ne[0]; ++i) {
                if (ip[i] < 0) continue;
                rows[n] = ip[i]; mvs[n] = 0.0f; ++n;
            }
        } else {
            for (int64_t i = 0; i < nkv; ++i) { rows[n] = (int32_t) i; mvs[n] = 0.0f; ++n; }
        }
        const int64_t r0 = n*c/C, r1 = n*(c + 1)/C;

        // state of head h: Dv accumulators, running max, running sum
        float * st0 = part + (iq1*C + c)*H*SD;
        for (int64_t h = h0; h < h1; ++h) {
            float * st = st0 + h*SD;
            memset(st, 0, Dv*sizeof(float));
            st[Dv] = -INFINITY; st[Dv + 1] = 0.0f;
        }
        const float * q0 = (const float *)((const char *) q->data + iq1*q->nb[1]);

        for (int64_t rb = r0; rb < r1; rb += BR) {
            const int nb = (int) MIN(BR, r1 - rb), nb4 = (nb + 3) & ~3;
            // convert the block (padded to a multiple of 4 with copies of its last row)
            for (int j = 0; j < nb4; ++j) {
                const int64_t ic = rows[rb + MIN(j, nb - 1)];
                ggml_fa_f16_to_f32_row(ggml_fa_mqa_row(k, kl, ic), Kt + j*Dk, Dk);
                if (!kv_same) {
                    ggml_fa_f16_to_f32_row(ggml_fa_mqa_row(v, vl, ic), Vt + j*Dv, Dv);
                }
            }
            const float * Vr = kv_same ? Kt : Vt;

            // raw scores, P[(h - h0)*BR + j]
            int64_t h = h0;
            for (; h + 2 <= h1; h += 2) {
                const float * qa = (const float *)((const char *) q0 + h*q->nb[2]);
                const float * qb = (const float *)((const char *) q0 + (h + 1)*q->nb[2]);
                for (int j = 0; j < nb4; j += 4) {
                    ggml_fa_qk_2x4(qa, qb, Kt + j*Dk, Dk, P + (h - h0)*BR + j, P + (h + 1 - h0)*BR + j);
                }
            }
            for (; h < h1; ++h) {
                const float * qa = (const float *)((const char *) q0 + h*q->nb[2]);
                for (int j = 0; j < nb4; j += 4) ggml_fa_qk_1x4(qa, Kt + j*Dk, Dk, P + (h - h0)*BR + j);
            }

            // online softmax: new max, the factor for the old accumulators, probabilities
            for (h = h0; h < h1; ++h) {
                float * p  = P + (h - h0)*BR;
                float * st = st0 + h*SD;
                float mx = -INFINITY;
                for (int j = 0; j < nb; ++j) {
                    const float s = softcap == 0.0f ? p[j]*scale + mvs[rb + j] : softcap*tanhf(p[j]*scale) + mvs[rb + j];
                    p[j] = s; mx = MAX(mx, s);
                }
                const float M = MAX(st[Dv], mx);
                const float cf = st[Dv] == -INFINITY ? 0.0f : expf(st[Dv] - M);
                float S = st[Dv + 1]*cf;
                for (int j = 0; j < nb; ++j) { p[j] = expf(p[j] - M); S += p[j]; }
                corr[h - h0] = cf; st[Dv] = M; st[Dv + 1] = S;
            }

            // accumulators: o_h = o_h*corr_h + sum_j p_h[j]*V[j]
            for (h = h0; h + 4 <= h1; h += 4) {
                ggml_fa_pv_4(st0 + h*SD, SD, P + (h - h0)*BR, corr + (h - h0), Vr, Dv, nb);
            }
            for (; h < h1; ++h) {
                ggml_fa_pv_1(st0 + h*SD, P + (h - h0)*BR, corr[h - h0], Vr, Dv, nb);
            }
        }
    }

    ggml_barrier(params->shared);

    // merge the chunk states of each (query, head), fold in the sink, normalise
    const int64_t np = NQ*H;
    for (int64_t pi = np*ith/nth; pi < np*(ith + 1)/nth; ++pi) {
        const int64_t iq1 = pi / H, h = pi % H;
        float M = -INFINITY;
        for (int cc = 0; cc < C; ++cc) M = MAX(M, part[((iq1*C + cc)*H + h)*SD + Dv]);
        float * out = (float *)((char *) dst->data + (iq1*dst->ne[1] + h)*dst->nb[1]);
        float S = 0.0f;
        memset(out, 0, Dv*sizeof(float));
        if (M != -INFINITY) {
            for (int cc = 0; cc < C; ++cc) {
                const float * st = part + ((iq1*C + cc)*H + h)*SD;
                if (st[Dv] == -INFINITY) continue;
                const float w = expf(st[Dv] - M);
                S += w*st[Dv + 1];
                const __m256 wv = _mm256_set1_ps(w);
                for (int64_t d = 0; d < Dv; d += 8) {
                    _mm256_storeu_ps(out + d, _mm256_add_ps(_mm256_loadu_ps(out + d), _mm256_mul_ps(wv, _mm256_loadu_ps(st + d))));
                }
            }
        }
        if (sinks) {
            const float sk = ((const float *) sinks->data)[h];
            if (sk > M) {
                const float ms = expf(M - sk);
                for (int64_t d = 0; d < Dv; ++d) out[d] *= ms;
                S = S*ms + 1.0f;
            } else {
                S += expf(sk - M);
            }
        }
        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        for (int64_t d = 0; d < Dv; ++d) out[d] *= S_inv;
    }
    return true;
}
#else
static bool ggml_fa_mqa_decode(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    (void) params; (void) dst;
    return false;
}
#endif
