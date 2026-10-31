#!/usr/bin/env python3
# ggml_backend_sched: an op whose weight is a view (e.g. a reshaped weight) stays on the weight's
# backend. The view gets its buffer only when the graph is allocated, so the pass-1 check missed it
# and the op could land on the GPU, uploading the CPU weight on every step.
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml-backend.cpp"
s = open(p).read()

old = """        if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);"""
new = """        // a view of a weight (e.g. a reshaped weight) gets its buffer only when the graph is allocated
        const struct ggml_tensor * ws = src->buffer == NULL && src->view_src != NULL ? src->view_src : src;
        if (ws->buffer != NULL && ws->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, ws, tensor);"""
if s.count(old) != 1:
    sys.exit(f"expected 1 of:\n{old}\nfound {s.count(old)}")
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
