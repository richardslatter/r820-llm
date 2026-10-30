#!/bin/bash
# kldub1.sh <variant> <tag>: as kld1.sh with one-token batches (the decode graph); all 16 chunks of kld-base.bin (~40 min)
cd ~/bench
T=~/models/test/wikitext-2-raw/wiki.test.raw
B=${BIN:-$HOME/src/ik_llama.cpp/build-dev/bin}
GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 $B/llama-perplexity -m /mnt/models/$1/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
    -f $T -c 256 -b 256 -ub 1 -t 64 --numa distribute --kl-divergence-base kld-base.bin --kl-divergence > kld-$2.out 2> kld-$2.err
touch kld-$2.done
