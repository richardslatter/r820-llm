#!/bin/bash
# pp512 A/B: old MXFP4 path vs batched kernel, CPU build, with op profile
cd ~/src/ik_llama.cpp
M=/mnt/models/dq6k/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
B=${B:-build-cpu}
for G in ${GS:-0 3}; do
  echo "== $B GGML_AVX1_GEMM=$G"
  env GGML_AVX1_GEMM=$G GGML_OP_PROF=1 LLAMA_OT_CPU_MMAP=1 GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 ./$B/bin/llama-bench -m $M -t 64 ${NGL:-} --numa distribute -fa 1 -p ${P:-512} -n 0 -r 1 2>&1 | grep -E "^\| deep|op profile|^(MOE|MUL_MAT|FLASH)" | head -12
done
echo DONE
