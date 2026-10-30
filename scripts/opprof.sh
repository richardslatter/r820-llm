#!/bin/bash
# opprof.sh <tag> [mode]: op profile of tg32 after a 512-token prompt (GGML_OP_PROF=2, or 3 for a per-node trace)
M=${MDIR:-/mnt/models/dq6k}/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
B=${BIN:-$HOME/src/ik_llama.cpp/build-dev/bin}
GGML_OP_PROF=${2:-2} GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 $B/llama-bench -m $M -t 64 -p 0 -n 32 -gp 512,32 -r 1 --numa distribute > ~/bench/opprof-$1.out 2> ~/bench/opprof-$1.err
touch ~/bench/opprof-$1.done
