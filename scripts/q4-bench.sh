#!/bin/bash
# Imatrix IQ4_NL dense (q6_K head / 4-bit head): KLD vs the BF16-dense original, then decode speed
cd ~/bench
T=~/models/test/wikitext-2-raw/wiki.test.raw
B=$HOME/src/ik_llama.cpp/build-cpu/bin
for v in diq4i diq4io; do
  GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 $B/llama-perplexity -m /mnt/models/$v/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
    -f $T -c 256 -b 256 -ub 8 --chunks 16 -t 64 --numa distribute --kl-divergence-base kld-base.bin --kl-divergence > kld-$v.out 2> kld-$v.err
  echo "== kld $v"; grep -E "Mean +KLD|Same top p" kld-$v.out
done
cd ~/src/ik_llama.cpp
for v in diq4 diq4i diq4io; do
  echo "== speed $v"
  env GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 ./build-dev/bin/llama-bench -m /mnt/models/$v/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
    -t 64 -p 0 -n 32 -gp 512,32 -r 2 --numa distribute 2>&1 | grep -E "^\| deep|rror"
done
echo DONE
