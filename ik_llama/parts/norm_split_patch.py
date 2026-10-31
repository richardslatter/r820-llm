#!/usr/bin/env python3
"""Column-split RMS norms at decode (run from the ik_llama.cpp root, after decode_ops_patch.py).

RMS_NORM and FUSED_RMS_NORM give each thread whole rows. At batch 1 DeepSeek-V4 normalises one
row of 20480 floats (the four hyper-connection streams) twice per layer, so one thread does it
alone (~45 us) while 63 wait. With fewer rows than threads and rows of at least 4096 floats,
every thread now takes a 64-byte-aligned column slice of every row: it writes its partial sums
of squares to the work buffer, waits at a barrier, adds up the partials and scales its slice.
GGML_NORM_SPLIT=0 goes back to whole rows.
"""
import sys

p = "ggml/src/ggml.c"
s = open(p).read()
if "ggml_rms_norm_split" in s:
    sys.exit(f"{p}: already patched")


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"{p}: anchor not found: {old[:70]!r}")
    s = s.replace(old, new)


split_fn = '''// decode: fewer rows than threads and long rows -> every thread takes a column slice of every
// row, and the partial sums of squares go through the work buffer (0 = not split)
static size_t ggml_rms_norm_split(const struct ggml_tensor * dst, int nth) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GGML_NORM_SPLIT");
        enabled = !(e && atoi(e) == 0);
    }
    const struct ggml_tensor * src0 = dst->src[0];
    const int64_t nr = ggml_nrows(src0);
    if (!enabled || nth < 2 || nr >= nth || nr > 16 || src0->ne[0] < 4096 || src0->type != GGML_TYPE_F32) {
        return 0;
    }
    if (dst->op == GGML_OP_FUSED_RMS_NORM && dst->op_params[1] > 1) {
        return 0; // grouped
    }
    return 64 + nr*nth*sizeof(ggml_float);
}

// the split form of both norms (c: the fused weight, or NULL)
static void ggml_rms_norm_f32_split(const struct ggml_compute_params * params, struct ggml_tensor * dst,
        const float * c, float eps) {
    const struct ggml_tensor * src0 = dst->src[0];
    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t nr = ne01*ne02*ne03;
    const int64_t k0 = (ne00/16)*ith/nth*16;
    const int64_t k1 = ith == nth - 1 ? ne00 : (ne00/16)*(ith + 1)/nth*16;
    // partial sums [row][thread]: each reader then pulls nth/8 lines (one slot per thread made it
    // read nth lines just written on other sockets, ~4x slower)
    ggml_float * part = (ggml_float *)(((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);

    for (int64_t ir = 0; ir < nr; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = ir - i03*ne02*ne01 - i02*ne01;
        const float * x = (const float *) ((const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        part[ir*nth + ith] = ggml_sumsq_f32(k1 - k0, x + k0);
    }

    ggml_barrier(params->shared);

    for (int64_t ir = 0; ir < nr; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = ir - i03*ne02*ne01 - i02*ne01;
        const float * x = (const float *) ((const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
              float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
        ggml_float sum = 0.0;
        for (int j = 0; j < nth; ++j) sum += part[ir*nth + j];
        const float scale = 1.0f/sqrtf((float)(sum/ne00) + eps);
        if (c) {
            for (int64_t j = k0; j < k1; ++j) y[j] = scale * c[j] * x[j];
        } else {
            for (int64_t j = k0; j < k1; ++j) y[j] = scale * x[j];
        }
    }
}

static void ggml_compute_forward_rms_norm_f32(
'''

sub('''
static void ggml_compute_forward_rms_norm_f32(
''', "\n" + split_fn)

# plain RMS_NORM
sub('''    GGML_ASSERT(eps > 0.0f);

    // TODO: optimize
    const int64_t nr = ne01*ne02*ne03;
''', '''    GGML_ASSERT(eps > 0.0f);

    if (dst->op == GGML_OP_RMS_NORM || dst->op == GGML_OP_FUSED_RMS_NORM) {
        const size_t ws = ggml_rms_norm_split(dst, nth);
        if (ws && params->wsize >= ws) {
            ggml_rms_norm_f32_split(params, dst, NULL, eps);
            return;
        }
    }

    // TODO: optimize
    const int64_t nr = ne01*ne02*ne03;
''')

# FUSED_RMS_NORM with a weight
sub('''    int nrows = ne03*ne02*ne01;
    int nrows_per_thread = (nrows + nth - 1)/nth;
    int first = ith*nrows_per_thread;
    int last  = MIN(nrows, first + nrows_per_thread);

    const float * c = (float *) src1->data;
''', '''    {
        const size_t ws = ggml_rms_norm_split(dst, nth);
        if (ws && params->wsize >= ws) {
            ggml_rms_norm_f32_split(params, dst, (const float *) src1->data, eps);
            return;
        }
    }

    int nrows = ne03*ne02*ne01;
    int nrows_per_thread = (nrows + nth - 1)/nth;
    int first = ith*nrows_per_thread;
    int last  = MIN(nrows, first + nrows_per_thread);

    const float * c = (float *) src1->data;
''')

# work buffer
sub('''            case GGML_OP_COUNT:
                {
                    GGML_ABORT("fatal error");
                }
            default:
                break;
        }

        work_size = MAX(work_size, cur);
''', '''            case GGML_OP_RMS_NORM:
            case GGML_OP_FUSED_RMS_NORM:
                {
                    cur = ggml_rms_norm_split(node, n_tasks);
                } break;
            case GGML_OP_COUNT:
                {
                    GGML_ABORT("fatal error");
                }
            default:
                break;
        }

        work_size = MAX(work_size, cur);
''')

open(p, "w").write(s)
print("norm split applied")
