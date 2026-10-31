#!/usr/bin/env python3
"""Scheduler: wait for the backend that holds an input-flagged tensor before copying it out.

Tensors flagged as graph inputs are copied into a split right away, after waiting only for the
split's own backend: user inputs live in host memory, which nothing else writes. The DFlash/DSpark
draft's K/V cache is flagged as an input too, but lives in its layer's buffer, so with the draft
on the GPU it is written by SET_ROWS on the CUDA compute stream and read by the CPU attention
(sinks at head size 512 have no Pascal kernel). The copy runs on cudaStreamPerThread, which does
not wait for the non-blocking compute stream, so the attention could read rows the GPU had not
written yet: draft acceptance fell from ~90% to ~35% with the draft on the GPU.

Usage: sched_input_sync_patch.py <ik_llama.cpp checkout>
"""
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = root + "/ggml/src/ggml-backend.cpp"
s = open(p).read()


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"{p}: anchor found {s.count(old)} times: {old[:70]!r}")
    s = s.replace(old, new)


sub("""                synced_on_input = true;
            }
            ggml_backend_tensor_copy(input, input_cpy);""", """                synced_on_input = true;
            }
            // an input kept on another backend (the DFlash K/V cache) may still be being written there
            if (input_backend != split_backend && input_backend != last_input_backend) {
                ggml_backend_synchronize(input_backend);
                last_input_backend = input_backend;
            }
            ggml_backend_tensor_copy(input, input_cpy);""")

open(p, "w").write(s)
print("patched", p)
