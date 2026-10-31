#!/usr/bin/env python3
"""Add GGML_TYPE_MXFP8 (type id 43) as a gather-only type to a mainline llama.cpp or an
ik_llama.cpp checkout (run from its root).

Unsloth's DeepSeek-V4.1-Flash GGUFs store the Engram table (per_layer_token_embd.weight) as
MXFP8: blocks of 32 E4M3 values sharing one E8M0 scale, 33 bytes per block. The table is only
ever row-gathered, so the type gets a to_float and a GET_ROWS case, nothing else.
"""
import os
import sys


def patch(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        sys.exit(f"{path}: expected {count} match(es), found {n} for:\n{old[:200]}")
    open(path, "w").write(s.replace(old, new))


def insert_in_function(path, func_sig, anchor, text):
    """Insert text before the first `anchor` after `func_sig` (for case lists repeated per op)."""
    s = open(path).read()
    i = s.index(func_sig)
    j = s.index(anchor, i)
    open(path, "w").write(s[:j] + text + s[j:])


IK = os.path.isdir("ggml/src/iqk")

# --- type id ---------------------------------------------------------------------------------
if IK:
    patch("ggml/include/ggml.h",
          "        GGML_TYPE_Q2_0      = 42,  // same type id as mainline GGML_TYPE_Q2_0\n",
          "        GGML_TYPE_Q2_0      = 42,  // same type id as mainline GGML_TYPE_Q2_0\n"
          "        GGML_TYPE_MXFP8     = 43,  // E4M3 + E8M0 scale, gather-only (Unsloth DeepSeek-V4.1 Engram table)\n")
else:
    patch("ggml/include/ggml.h",
          "        GGML_TYPE_COUNT   = 43,\n",
          "        GGML_TYPE_MXFP8   = 43, // E4M3 + E8M0 scale, gather-only (Unsloth DeepSeek-V4.1 Engram table)\n"
          "        GGML_TYPE_COUNT   = 44,\n")

# --- block layout ------------------------------------------------------------------------------
patch("ggml/src/ggml-common.h",
      'static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size/padding");\n',
      'static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2, "wrong mxfp4 block size/padding");\n'
      "\n"
      "// MXFP8 (OCP microscaling): 32 E4M3 values sharing one E8M0 scale. Only read by GET_ROWS.\n"
      "#define QK_MXFP8 32\n"
      "typedef struct {\n"
      "    uint8_t e;            // E8M0 scale\n"
      "    uint8_t qs[QK_MXFP8]; // E4M3 codes\n"
      "} block_mxfp8;\n"
      'static_assert(sizeof(block_mxfp8) == sizeof(uint8_t) + QK_MXFP8, "wrong mxfp8 block size/padding");\n')

# --- dequantization ----------------------------------------------------------------------------
q8_decl = ("GGML_API " if not IK else "") + \
    "void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);\n"
patch("ggml/src/ggml-quants.h", q8_decl,
      q8_decl + q8_decl.replace("q8_0", "mxfp8"))

with open("ggml/src/ggml-quants.c", "a") as f:
    f.write(r'''
// MXFP8: E4M3 (the "fn" variant: no infinities, S.1111.111 is NaN) times a shared E8M0 scale.
// Every code times every scale is exact in fp32.
static inline float mxfp8_e4m3_to_fp32(uint8_t x) {
    const uint32_t e = (x >> 3) & 0xF;
    const uint32_t m = x & 0x7;
    float v;
    if (e == 0) {
        v = (float) m * (1.0f/512.0f); // subnormal: m/8 * 2^-6
    } else if (e == 0xF && m == 0x7) {
        v = NAN;
    } else {
        union { uint32_t u; float f; } b = { ((e + 120) << 23) | (m << 20) }; // (1 + m/8) * 2^(e-7)
        v = b.f;
    }
    return x & 0x80 ? -v : v;
}

void dequantize_row_mxfp8(const block_mxfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_MXFP8 == 0);
    const int64_t nb = k / QK_MXFP8;
    for (int64_t i = 0; i < nb; i++) {
        const float d = x[i].e == 0xFF ? NAN : GGML_E8M0_TO_FP32(x[i].e);
        for (int j = 0; j < QK_MXFP8; ++j) {
            y[i*QK_MXFP8 + j] = mxfp8_e4m3_to_fp32(x[i].qs[j]) * d;
        }
    }
}
''')

if IK:
    patch("ggml/src/ggml-quants.c",
          "        case GGML_TYPE_MXFP4: break;\n",
          "        case GGML_TYPE_MXFP4: break;\n        case GGML_TYPE_MXFP8: break;\n")
else:
    patch("ggml/src/ggml-quants.c",
          "        case GGML_TYPE_I64:\n            // nothing to validate\n",
          "        case GGML_TYPE_MXFP8:\n        case GGML_TYPE_I64:\n            // nothing to validate\n")

# --- type traits -------------------------------------------------------------------------------
traits = '''    [GGML_TYPE_MXFP8] = {
        .type_name                = "mxfp8",
        .blck_size                = QK_MXFP8,
        .type_size                = sizeof(block_mxfp8),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp8,
'''
if IK:
    traits += "        .nrows                    = 1,\n        .row_meta_size            = 0,\n"
traits += "    },\n"
patch("ggml/src/ggml.c", "    [GGML_TYPE_MXFP4] = {\n", traits + "    [GGML_TYPE_MXFP4] = {\n")

# --- GET_ROWS ----------------------------------------------------------------------------------
if IK:
    insert_in_function("ggml/src/ggml.c", "static void ggml_compute_forward_get_rows(\n",
                       "        case GGML_TYPE_MXFP4:\n", "        case GGML_TYPE_MXFP8:\n")
else:
    insert_in_function("ggml/src/ggml-cpu/ops.cpp", "void ggml_compute_forward_get_rows(\n",
                       "        case GGML_TYPE_MXFP4:\n", "        case GGML_TYPE_MXFP8:\n")

print("mxfp8 patch applied (%s)" % ("ik_llama.cpp" if IK else "llama.cpp"))
