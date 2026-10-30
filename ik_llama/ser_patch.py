#!/usr/bin/env python3
"""Smart expert reduction (-ser min,thresh) back on (run from the ik_llama.cpp root).

ik's -ser drops a selected expert when its weight is below thresh x the largest selected weight,
keeping at least min experts; dropped slots get id -1 and the expert matmuls skip them. The
call was commented out in llm_build_moe_ffn, so the flag did nothing. Back on, with:
  - a top-k selection instead of argsort_thresh's bubble sort (~70k compares over 384 experts
    per layer, one thread per token);
  - the threshold compared on the unbiased weights (probs), not the bias-shifted selection
    scores, when the model has an expert bias (DeepSeek-V3/V4);
  - iqk_mul_multi_add skipping zero-weight slots: the expert outputs of dropped slots are never
    written, and 0 x (stale NaN) would poison the sum.
"""
import sys


def patch(p, subs):
    s = open(p).read()
    for old, new in subs:
        if s.count(old) != 1:
            sys.exit(f"{p}: anchor not found: {old[:70]!r}")
        s = s.replace(old, new)
    open(p, "w").write(s)


patch("ggml/src/ggml.c", [
('''        result = ggml_argsort_thresh(ctx, a, min_entries, thresh);
    }
''', '''        result = ggml_argsort_thresh(ctx, a, min_entries, thresh);
        ggml_set_op_params_i32(result, 2, k); // only the first k entries are read
    }
'''),
('''    //if (ith == 0) printf("%s: min_entries = %d, thresh = %g\\n", __func__, min_entries, (double)thresh);

    for (int64_t i = ith; i < nr; i += nth) {
        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }
''', '''    //if (ith == 0) printf("%s: min_entries = %d, thresh = %g\\n", __func__, min_entries, (double)thresh);

    const int k = ggml_get_op_params_i32(dst, 2);

    for (int64_t i = ith; i < nr; i += nth) {
        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        if (k > 0 && k < ne0) {
            // the k largest by insertion, then reordered and thresholded by the weights in src[1]
            // (the unbiased expert weights) when given
            const float * ref = dst->src[1] ? (const float *)((const char *) dst->src[1]->data + i*dst->src[1]->nb[1]) : src_data;
            int n = 0;
            for (int32_t e = 0; e < (int32_t) ne0; ++e) {
                const float v = src_data[e];
                if (n == k && v <= src_data[dst_data[k - 1]]) continue;
                int j = n < k ? n++ : k - 1;
                for (; j > 0 && src_data[dst_data[j - 1]] < v; --j) dst_data[j] = dst_data[j - 1];
                dst_data[j] = e;
            }
            for (int j = 1; j < k; ++j) {
                const int32_t e = dst_data[j];
                int jj = j;
                for (; jj > 0 && ref[dst_data[jj - 1]] < ref[e]; --jj) dst_data[jj] = dst_data[jj - 1];
                dst_data[jj] = e;
            }
            const float max_value = ref[dst_data[0]];
            for (int j = min_entries; j < k; ++j) {
                if (ref[dst_data[j]] < max_value*thresh) dst_data[j] = -1;
            }
            for (int64_t j = k; j < ne0; ++j) dst_data[j] = -1;
            continue;
        }

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }
'''),
])

patch("src/llama-build-context.cpp", [
('''            //selected_experts = ggml_top_k_thresh(ctx, selection_probs, n_expert_used,
            //        lctx.cparams.min_experts, lctx.cparams.thresh_experts); // [n_expert_used, n_tokens]
            selected_experts = ggml_top_k(ctx, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
''', '''            if (lctx.cparams.min_experts > 0 && lctx.cparams.thresh_experts > 0) {
                selected_experts = ggml_top_k_thresh(ctx, selection_probs, n_expert_used,
                        lctx.cparams.min_experts, lctx.cparams.thresh_experts); // [n_expert_used, n_tokens]
                if (selection_probs != probs) {
                    selected_experts->view_src->src[1] = probs; // threshold on the unbiased weights
                }
            } else {
                selected_experts = ggml_top_k(ctx, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
            }
'''),
])

patch("ggml/src/iqk/iqk_cpu_ops.cpp", [
('''        auto x1 = (const float *)c1;
        for (int k = 0; k < ne00; ++k) y[k] = x0[k] * x1[0];
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0 + k0;
            x1 = (const float *)c1;
            for (int k = 0; k < ne00; ++k) y[k] += x0[k] * x1[0];
        }
''', '''        auto x1 = (const float *)c1;
        // a zero weight is an expert dropped by -ser: its output was never written
        if (x1[0] != 0.0f) {
            for (int k = 0; k < ne00; ++k) y[k] = x0[k] * x1[0];
        } else {
            for (int k = 0; k < ne00; ++k) y[k] = 0.0f;
        }
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0 + k0;
            x1 = (const float *)c1;
            if (x1[0] == 0.0f) continue;
            for (int k = 0; k < ne00; ++k) y[k] += x0[k] * x1[0];
        }
'''),
])

print("smart expert reduction applied")
