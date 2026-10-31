#!/bin/bash
# DSpark decode: CPU build vs CUDA build (-ngl 0), n_max 3 and 4, with the batched MXFP4 kernel on (default)
cd ~/src/ik_llama.cpp
M=/mnt/models/dq6k/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
D="-md /mnt/models/dspark/dspark-ik.gguf"
for B in ${BS:-build-cpu build-cuda}; do
  G=""; [ $B = build-cuda ] && G="-ngl 0 -ngld 0"
  for N in ${NS:-3 4}; do
    echo "== $B n_max=$N ${GE:+GGML_AVX1_GEMM=$GE}"
    env ${GE:+GGML_AVX1_GEMM=$GE} GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 ./$B/bin/llama-spec-bench -m $M $D $G -t 64 --numa distribute -c 4096 --temp 0 --seed 123 -n 192 --spec-type dspark:n_max=$N --task ${TASKS:-code,story} 2>&1 | grep -E "^\| [a-z]|rror"
  done
done
echo DONE
