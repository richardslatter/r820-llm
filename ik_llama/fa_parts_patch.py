#!/usr/bin/env python3
"""Decode attention (ggml_fa_mqa_decode): split the heads across the NUMA nodes first.

The kernel split the visible K/V rows into at most 8 chunks and the 64 heads into nth/8 groups, so
with 64 threads every row was converted from fp16 by 8 threads, and every head's 8 chunk states,
written on all four sockets, were merged by one thread reading them remotely. Here the heads are
first split into GGML_FA_MQA_PARTS (default 4) partitions of consecutive threads, one per node with
GGML_NUMA_BLOCKED; a partition splits its rows into C chunks (about sqrt(n/2), up to its thread
count) and its heads into the rest. A node converts each row once or twice and merges only states
its own threads wrote. GGML_FA_MQA_PARTS=1 is the old split.

Also: the PV micro-kernels are kept out of line (inlined into the bigger loop GCC scheduled them
~1.6x slower), and the mask is scanned 16 entries at a time.

fa_bench (64 threads, 640 visible rows of 4096): see docs/plan.md. Same results to rounding: the
order of the per-chunk sums changes.

Usage: fa_parts_patch.py <ik_llama.cpp checkout>
"""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = root + "/ggml/src/ggml.c"
s = open(p).read()


def sub(old, new, count=1):
    global s
    if s.count(old) != count:
        sys.exit(f"{p}: anchor found {s.count(old)} times: {old[:70]!r}")
    s = s.replace(old, new)


# --- work buffer: states of NQ x C x H heads, C up to the threads of a partition
sub("""// work buffer floats ggml_fa_mqa_decode needs (0 if the node is not its shape)
static size_t ggml_fa_mqa_work_floats(const struct ggml_tensor * dst, int nth) {
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    if (q->ne[1] > 8 || k->ne[2] != 1 || v->ne[2] != 1 || q->ne[3] != 1) return 0;
    const size_t H = q->ne[2], Dk = k->ne[0], Dv = v->ne[0], nkv = k->ne[1];
    return 16 + q->ne[1]*8*H*(Dv + 16) + (size_t)nth*GGML_PAD(GGML_FA_MQA_BR*(Dk + Dv) + 2*H*GGML_FA_MQA_BR + 2*nkv + 64, 16);
}
""", """// head partitions: GGML_FA_MQA_PARTS (default 4) groups of consecutive threads, each with its
// share of the heads and all the rows (with GGML_NUMA_BLOCKED, one per node); 1 if they don't divide
static int ggml_fa_mqa_parts(int nth, int64_t H) {
    static int parts = -1;
    if (parts < 0) {
        const char * e = getenv("GGML_FA_MQA_PARTS");
        parts = e ? atoi(e) : 4;
    }
    return parts > 1 && nth % parts == 0 && nth >= 2*parts && H % parts == 0 ? parts : 1;
}

// most row chunks of a partition
static int ggml_fa_mqa_max_chunks(int nth, int64_t H) {
    const int NN = ggml_fa_mqa_parts(nth, H);
    return NN > 1 ? nth/NN : 8;
}

// work buffer floats ggml_fa_mqa_decode needs (0 if the node is not its shape)
static size_t ggml_fa_mqa_work_floats(const struct ggml_tensor * dst, int nth) {
    const struct ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    if (q->ne[1] > 8 || k->ne[2] != 1 || v->ne[2] != 1 || q->ne[3] != 1) return 0;
    const size_t H = q->ne[2], Dk = k->ne[0], Dv = v->ne[0], nkv = k->ne[1];
    return 16 + q->ne[1]*ggml_fa_mqa_max_chunks(nth, H)*H*(Dv + 16) + (size_t)nth*GGML_PAD(GGML_FA_MQA_BR*(Dk + Dv) + 2*H*GGML_FA_MQA_BR + 2*nkv + 64, 16);
}
""")

# --- out-of-line PV kernels
sub("""static inline void ggml_fa_pv_4(float * o, int64_t ostride,""",
    """// out of line: inlined into ggml_fa_mqa_decode GCC schedules it ~1.6x slower
static __attribute__((noinline)) void ggml_fa_pv_4(float * o, int64_t ostride,""")
sub("""static inline void ggml_fa_pv_1(float * o,""", """static __attribute__((noinline)) void ggml_fa_pv_1(float * o,""")

# --- mask scan, 16 entries at a time
sub("""// DeepSeek-V4 decode attention: one shared fp16 K/V head, f32 Q, up to 8 query rows.""",
    """// visible entries of an f16 mask row (not -inf), 16 at a time; with rows, also their indices and values
static int64_t ggml_fa_mqa_mask_rows(const uint16_t * m, int64_t n, int32_t * rows, float * mvs) {
    const __m128i ninf = _mm_set1_epi16((short) 0xfc00);
    int64_t cnt = 0, i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128i a = _mm_loadu_si128((const __m128i *)(m + i)), b = _mm_loadu_si128((const __m128i *)(m + i + 8));
        const int hidden = _mm_movemask_epi8(_mm_packs_epi16(_mm_cmpeq_epi16(a, ninf), _mm_cmpeq_epi16(b, ninf)));
        if (hidden == 0xffff) continue;
        if (!rows) {
            cnt += 16 - __builtin_popcount(hidden);
            continue;
        }
        for (int j = 0; j < 16; ++j) {
            if (hidden >> j & 1) continue;
            rows[cnt] = (int32_t)(i + j); mvs[cnt] = GGML_FP16_TO_FP32(m[i + j]); ++cnt;
        }
    }
    for (; i < n; ++i) {
        if (m[i] == 0xfc00) continue;
        if (rows) { rows[cnt] = (int32_t) i; mvs[cnt] = GGML_FP16_TO_FP32(m[i]); }
        ++cnt;
    }
    return cnt;
}

// DeepSeek-V4 decode attention: one shared fp16 K/V head, f32 Q, up to 8 query rows.""")

sub("""    if (mask) {
        const uint16_t * m0 = (const uint16_t *) mask->data;
        for (int64_t i = 0; i < nkv; ++i) n0 += m0[i] != 0xfc00;
    } else if (idx) {""", """    if (mask) {
        n0 = ggml_fa_mqa_mask_rows((const uint16_t *) mask->data, nkv, NULL, NULL);
    } else if (idx) {""")

sub("""    // chunks of >= 64 rows, at most 8 (more chunks: fewer conversions per row, more merging)
    int C = 1;
    while (2*C <= 8 && 2*C <= nth && nth % (2*C) == 0 && n0 >= 64*C) C *= 2;
    const int G = nth / C;
    // consecutive threads share a chunk: with GGML_NUMA_BLOCKED a node then reads only the rows
    // of its own chunks (once from memory, then from its L3 for the other head groups)
    const int c = ith / G, g = ith % G;
    // DeepSeek-V4 passes the same tensor as K and V: convert each row once
    const bool kv_same = k->data == v->data && k->nb[1] == v->nb[1] && Dk == Dv;
    const int64_t h0 = H*g/G, h1 = H*(g + 1)/G;
""", """    // NN partitions of T consecutive threads, each with HP heads and all the rows: with
    // GGML_NUMA_BLOCKED a partition is a node, and merges only states its own threads wrote.
    // A partition splits its rows into C chunks and its heads into G = T/C groups. More chunks:
    // fewer conversions of each row, more states to merge; about sqrt(n0/2) balances the two.
    // With one partition: chunks of >= 64 rows, at most 8.
    const int NN = ggml_fa_mqa_parts(nth, H), T = nth / NN, pp = ith / T, t = ith % T;
    const int64_t HP = H / NN, hp0 = pp*HP;
    int C = 1;
    if (NN > 1) {
        while (2*C <= T && T % (2*C) == 0 && 8*C*C <= n0) C *= 2;
    } else {
        while (2*C <= 8 && 2*C <= nth && nth % (2*C) == 0 && n0 >= 64*C) C *= 2;
    }
    const int G = T / C;
    const int c = t / G, g = t % G;
    // DeepSeek-V4 passes the same tensor as K and V: convert each row once
    const bool kv_same = k->data == v->data && k->nb[1] == v->nb[1] && Dk == Dv;
    const int64_t h0 = hp0 + HP*g/G, h1 = hp0 + HP*(g + 1)/G;
""")

sub("""        if (mask) {
            const uint16_t * mp = (const uint16_t *)((const char *) mask->data + iq1*mask->nb[1]);
            for (int64_t i = 0; i < nkv; ++i) {
                if (mp[i] == 0xfc00) continue;
                rows[n] = (int32_t) i; mvs[n] = GGML_FP16_TO_FP32(mp[i]); ++n;
            }
        } else if (idx) {""", """        if (mask) {
            n = ggml_fa_mqa_mask_rows((const uint16_t *)((const char *) mask->data + iq1*mask->nb[1]), nkv, rows, mvs);
        } else if (idx) {""")

# state of (query, partition, chunk, head): part + (((iq1*NN + pp)*C + c)*HP + h - hp0)*SD
sub("""        // state of head h: Dv accumulators, running max, running sum
        float * st0 = part + (iq1*C + c)*H*SD;
""", """        // state of head h: Dv accumulators, running max, running sum
        float * st0 = part + ((iq1*NN + pp)*C + c)*HP*SD - hp0*SD;
""")

sub("""    // merge the chunk states of each (query, head), fold in the sink, normalise
    const int64_t np = NQ*H;
    for (int64_t pi = np*ith/nth; pi < np*(ith + 1)/nth; ++pi) {
        const int64_t iq1 = pi / H, h = pi % H;
        float M = -INFINITY;
        for (int cc = 0; cc < C; ++cc) M = MAX(M, part[((iq1*C + cc)*H + h)*SD + Dv]);""",
    """    // merge the chunk states of each (query, head) of this partition, fold in the sink, normalise
    const int64_t np = NQ*HP;
    for (int64_t pi = np*t/T; pi < np*(t + 1)/T; ++pi) {
        const int64_t iq1 = pi / HP, h = hp0 + pi % HP;
        const float * sth = part + ((iq1*NN + pp)*C*HP + h - hp0)*SD;
        float M = -INFINITY;
        for (int cc = 0; cc < C; ++cc) M = MAX(M, sth[cc*HP*SD + Dv]);""")
sub("""            for (int cc = 0; cc < C; ++cc) {
                const float * st = part + ((iq1*C + cc)*H + h)*SD;""", """            for (int cc = 0; cc < C; ++cc) {
                const float * st = sth + cc*HP*SD;""")

sub("""    float * scratch = part + NQ*8*H*SD + (size_t)ith*""",
    """    float * scratch = part + NQ*ggml_fa_mqa_max_chunks(nth, H)*H*SD + (size_t)ith*""")

open(p, "w").write(s)
print("patched", p)
