#!/usr/bin/env python3
"""Decode attention (ggml_fa_mqa_decode): PV kernels out of line, mask scanned 16 entries at a time.

- Inlined into ggml_fa_mqa_decode, GCC schedules the PV micro-kernels ~1.6x slower (single-thread
  timing of the same loop); out of line they keep their own register allocation.
- The f16 mask row (one entry per KV row, -inf for hidden ones) was read one entry at a time,
  twice per call; at 4K context that is 8K scalar compares per thread before any work.

Results are bit-identical.

Usage: fa_decode_tweaks_patch.py <ik_llama.cpp checkout>
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


sub("""static inline void ggml_fa_pv_4(float * o, int64_t ostride,""",
    """// out of line: inlined into ggml_fa_mqa_decode GCC schedules it ~1.6x slower
static __attribute__((noinline)) void ggml_fa_pv_4(float * o, int64_t ostride,""")
sub("""static inline void ggml_fa_pv_1(float * o,""", """static __attribute__((noinline)) void ggml_fa_pv_1(float * o,""")

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

sub("""        if (mask) {
            const uint16_t * mp = (const uint16_t *)((const char *) mask->data + iq1*mask->nb[1]);
            for (int64_t i = 0; i < nkv; ++i) {
                if (mp[i] == 0xfc00) continue;
                rows[n] = (int32_t) i; mvs[n] = GGML_FP16_TO_FP32(mp[i]); ++n;
            }
        } else if (idx) {""", """        if (mask) {
            n = ggml_fa_mqa_mask_rows((const uint16_t *)((const char *) mask->data + iq1*mask->nb[1]), nkv, rows, mvs);
        } else if (idx) {""")

open(p, "w").write(s)
print("patched", p)
