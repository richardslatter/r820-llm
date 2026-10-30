#!/bin/bash
# tgd.sh <label> [extra llama-bench args]: decode after a 512-token prompt (tg32 at depth ~512), NUMA-pinned and placed
L=$1; shift
M=${MDIR:-/mnt/models/dq8}/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
B=${BIN:-$HOME/src/ik_llama.cpp/build-cpu/bin}
GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 $B/llama-bench -m $M -t 64 -p 0 -n 32 -gp 512,32 -r 1 --numa distribute "$@" > ~/bench/tgd-$L.out 2> ~/bench/tgd-$L.err
echo "$L: $(grep -E "tg32|pp512" ~/bench/tgd-$L.out | awk -F"|" "{printf \"%s %s ; \", \$(NF-2), \$(NF-1)}")"
