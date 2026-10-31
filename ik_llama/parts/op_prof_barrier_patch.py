#!/usr/bin/env python3
# GGML_OP_PROF: hold the other threads after each node until thread 0 has read their times.
# Without it a thread can run the next node and overwrite its slot first, so small nodes looked
# as if thread 0 did all the work (the "slowest" / "average" columns were wrong, not the wall time).
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "ggml/src/ggml.c"
s = open(p).read()

old = """        if (state->shared->ec != GGML_STATUS_SUCCESS) {
            break;
        }
    }
#if IK_PRINT_TIMING
    int64_t t_end"""
new = """        // keep the other threads from overwriting their time with the next node's before it is read
        if (prof) ggml_barrier(state->shared);

        if (state->shared->ec != GGML_STATUS_SUCCESS) {
            break;
        }
    }
#if IK_PRINT_TIMING
    int64_t t_end"""
if s.count(old) != 1:
    sys.exit(f"expected 1 of:\n{old}\nfound {s.count(old)}")
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
