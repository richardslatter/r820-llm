#!/usr/bin/env python3
"""Let a CPU split start before the device results it needs later are ready.

ggml's scheduler copies every input of a split before running any of it, so a CPU split
that adds a GPU result at the end (the shared expert, put on the P2200 with
-ot shexp=CUDA0) first waits for the GPU. This patch runs the nodes before the first use
of a device result, then copies it and runs the rest: the CPU's routed experts and the
GPU's shared expert run side by side. GGML_SCHED_DEFER=0 turns it off.

Usage: sched_defer_patch.py <ik_llama.cpp checkout>
"""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = root + "/ggml/src/ggml-backend.cpp"
s = open(p).read()

helper_anchor = "static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {"
helper = r'''// A CPU split whose inputs include results of an asynchronous device backend normally waits for
// that device before its first node. When those results are only needed further in (e.g. the
// shared expert computed on the GPU and added after the routed experts on the CPU), the nodes
// before their first use run first, so the CPU and the device work side by side.
// Returns the index of that first use, or 0 when the split runs as usual. GGML_SCHED_DEFER=0 turns it off.
static int ggml_backend_sched_deferred_split_point(ggml_backend_sched_t sched, ggml_backend_sched_split * split,
        ggml_backend_sched_split & ready, ggml_backend_sched_split & late) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_SCHED_DEFER");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || sched->callback_eval || sched->callback_copy || split->n_inputs == 0) return 0;
    if (!ggml_backend_is_cpu(sched->backends[split->backend_id])) return 0;
    ready = *split;
    late  = *split;
    ready.n_inputs = late.n_inputs = 0;
    for (int j = 0; j < split->n_inputs; ++j) {
        ggml_tensor * input = split->inputs[j];
        ggml_backend_t ib = ggml_backend_sched_get_tensor_backend(sched, input);
        if (!(input->flags & GGML_TENSOR_FLAG_INPUT) && ib && !ggml_backend_is_cpu(ib)) {
            late.inputs[late.n_inputs++] = input;
        } else {
            ready.inputs[ready.n_inputs++] = input;
        }
    }
    if (late.n_inputs == 0) return 0;
    auto uses_late = [&](const ggml_tensor * t) {
        for (; t; t = t->view_src) {
            for (int j = 0; j < late.n_inputs; ++j) {
                if (t == tensor_copy(late.inputs[j], split->backend_id, sched->cur_copy)) return true;
            }
        }
        return false;
    };
    for (int k = 0; k < split->graph.n_nodes; ++k) {
        const ggml_tensor * node = split->graph.nodes[k];
        if (uses_late(node)) return k;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (node->src[i] && uses_late(node->src[i])) return k;
        }
    }
    return 0;
}

'''
assert s.count(helper_anchor) == 1
s = s.replace(helper_anchor, helper + helper_anchor)

old = """        // copy the input tensors to the split backend
        ggml_backend_sched_copy_inputs(sched, split, sched->needs_sync, ids, unique_ids, last_ids_tensor);
"""
new = """        // run the part of a CPU split that does not need device results while the device computes them
        ggml_backend_sched_split ready_part, late_part;
        if (int k = ggml_backend_sched_deferred_split_point(sched, split, ready_part, late_part); k > 0) {
            ggml_backend_sched_copy_inputs(sched, &ready_part, sched->needs_sync, ids, unique_ids, last_ids_tensor);
            sched->needs_sync[split_backend_id] = true;
            ggml_cgraph head = ggml_graph_view(&split->graph, 0, k);
            auto ec = ggml_backend_graph_compute_async(split_backend, &head);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            ggml_backend_sched_copy_inputs(sched, &late_part, sched->needs_sync, ids, unique_ids, last_ids_tensor);
            ggml_cgraph tail = ggml_graph_view(&split->graph, k, split->graph.n_nodes);
            ec = ggml_backend_graph_compute_async(split_backend, &tail);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy]);
            }
            continue;
        }

        // copy the input tensors to the split backend
        ggml_backend_sched_copy_inputs(sched, split, sched->needs_sync, ids, unique_ids, last_ids_tensor);
"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
