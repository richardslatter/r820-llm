#!/usr/bin/env python3
# ggml_fa_mqa_decode: with a mask, score each converted block of K/V rows against every query
# (the verify batch of speculative decoding) instead of converting the rows again per query.
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml.c"
s = open(p).read()

def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"expected {count} of:\n{old}\nfound {n}")
    s = s.replace(old, new)

# per-thread scratch: mask values of every query on the union of visible rows
rep("""    return 16 + q->ne[1]*8*H*(Dv + 16) + (size_t)nth*GGML_PAD(GGML_FA_MQA_BR*(Dk + Dv) + 2*H*GGML_FA_MQA_BR + 2*nkv + 64, 16);""",
    """    return 16 + q->ne[1]*8*H*(Dv + 16) + (size_t)nth*GGML_PAD(GGML_FA_MQA_BR*(Dk + Dv) + 2*H*GGML_FA_MQA_BR + (q->ne[1] + 1)*nkv + 64, 16);""")
rep("""    float * scratch = part + NQ*8*H*SD + (size_t)ith*GGML_PAD(BR*(Dk + Dv) + 2*H*BR + 2*nkv + 64, 16);""",
    """    float * scratch = part + NQ*8*H*SD + (size_t)ith*GGML_PAD(BR*(Dk + Dv) + 2*H*BR + (NQ + 1)*nkv + 64, 16);""")
rep("""    int32_t * rows = (int32_t *)(mvs + nkv);""",
    """    int32_t * rows = (int32_t *)(mvs + NQ*nkv);""")

# chunk size floor, tunable
rep("""    // chunks of >= 64 rows, at most 8 (more chunks: fewer conversions per row, more merging)
    int C = 1;
    while (2*C <= 8 && 2*C <= nth && nth % (2*C) == 0 && n0 >= 64*C) C *= 2;""",
    """    // chunks of >= min_rows rows, at most 8 (more chunks: fewer conversions per row, more merging)
    static int min_rows = -1;
    if (min_rows < 0) {
        const char * e = getenv("GGML_FA_MQA_MINROWS");
        min_rows = e ? MAX(1, atoi(e)) : 64;
    }
    int C = 1;
    while (2*C <= 8 && 2*C <= nth && nth % (2*C) == 0 && n0 >= min_rows*C) C *= 2;""")

# drop the shape debug print
a = s.index("    {\n        static int dbg = -1; static int ndbg = 0;")
b = s.index("    // DeepSeek-V4 passes", a)
s = s[:a] + s[b:]

old_loop_start = """    for (int64_t iq1 = 0; iq1 < NQ; ++iq1) {
        // visible rows of this query and their mask values"""
old_loop_end = """                ggml_fa_pv_1(st0 + h*SD, P + (h - h0)*BR, corr[h - h0], Vr, Dv, nb);
            }
        }
    }

    ggml_barrier(params->shared);"""
a = s.index(old_loop_start)
b = s.index(old_loop_end, a) + len(old_loop_end)

new_loop = """    // With a mask, the queries go through the rows together: the rows any of them sees, with -inf
    // for a query that does not see a row. Each block of rows is then converted once for all of
    // them, not once per query (the verify batch of speculative decoding sees nearly the same rows).
    static int multi = -1;
    if (multi < 0) {
        const char * e = getenv("GGML_FA_MQA_MULTI");
        multi = !e || atoi(e) != 0;
    }
    const int64_t nqp = mask && multi ? NQ : 1;

    for (int64_t iq = 0; iq < NQ; iq += nqp) {
        // rows of this pass, and the mask value of each of its queries on them (mvs + a*nkv)
        int64_t n = 0;
        if (mask) {
            const uint16_t * mq[8];
            for (int64_t a = 0; a < nqp; ++a) mq[a] = (const uint16_t *)((const char *) mask->data + (iq + a)*mask->nb[1]);
            for (int64_t i = 0; i < nkv; ++i) {
                int vis = 0;
                for (int64_t a = 0; a < nqp; ++a) vis |= mq[a][i] != 0xfc00;
                if (!vis) continue;
                rows[n] = (int32_t) i;
                for (int64_t a = 0; a < nqp; ++a) mvs[a*nkv + n] = mq[a][i] == 0xfc00 ? -INFINITY : GGML_FP16_TO_FP32(mq[a][i]);
                ++n;
            }
        } else if (idx) {
            const int32_t * ip = (const int32_t *)((const char *) idx->data + iq*idx->nb[1]);
            for (int64_t i = 0; i < idx->ne[0]; ++i) {
                if (ip[i] < 0) continue;
                rows[n] = ip[i]; mvs[n] = 0.0f; ++n;
            }
        } else {
            for (int64_t i = 0; i < nkv; ++i) { rows[n] = (int32_t) i; mvs[n] = 0.0f; ++n; }
        }
        const int64_t r0 = n*c/C, r1 = n*(c + 1)/C;

        // state of head h of query iq + a: Dv accumulators, running max, running sum
        for (int64_t a = 0; a < nqp; ++a) {
            float * st0 = part + ((iq + a)*C + c)*H*SD;
            for (int64_t h = h0; h < h1; ++h) {
                float * st = st0 + h*SD;
                memset(st, 0, Dv*sizeof(float));
                st[Dv] = -INFINITY; st[Dv + 1] = 0.0f;
            }
        }

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

            for (int64_t a = 0; a < nqp; ++a) {
                const float * q0  = (const float *)((const char *) q->data + (iq + a)*q->nb[1]);
                const float * mv  = mvs + a*nkv + rb;
                float       * st0 = part + ((iq + a)*C + c)*H*SD;

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
                        const float s = softcap == 0.0f ? p[j]*scale + mv[j] : softcap*tanhf(p[j]*scale) + mv[j];
                        p[j] = s; mx = MAX(mx, s);
                    }
                    const float M = MAX(st[Dv], mx);
                    if (M == -INFINITY) {
                        // nothing visible yet: leave the state as it is
                        for (int j = 0; j < nb; ++j) p[j] = 0.0f;
                        corr[h - h0] = 1.0f;
                        continue;
                    }
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
    }

    ggml_barrier(params->shared);"""
s = s[:a] + new_loop + s[b:]
open(p, "w").write(s)
print("patched")
