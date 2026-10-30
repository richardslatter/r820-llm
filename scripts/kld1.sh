#!/bin/bash
# kld1.sh <variant> <tag>: KL divergence of one variant against kld-base.bin with the build in $BIN (default build-dev)
cd ~/bench
T=~/models/test/wikitext-2-raw/wiki.test.raw
B=${BIN:-$HOME/src/ik_llama.cpp/build-dev/bin}
GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 $B/llama-perplexity -m /mnt/models/$1/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
    -f $T -c 256 -b 256 -ub 8 --chunks 16 -t 64 --numa distribute --kl-divergence-base kld-base.bin --kl-divergence > kld-$2.out 2> kld-$2.err
touch kld-$2.done
