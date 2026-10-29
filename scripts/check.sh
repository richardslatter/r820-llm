#!/bin/bash
# ik (SIMDe AVX1 build) vs mainline: perplexity on wikitext-2, same quantized files
cd ~/models/test
ML=~/src/llama.cpp/build-cpu/bin; IK=~/src/ik_llama.cpp/build-cpu/bin
for q in Q8_0 Q4_K_M; do [ -f g-$q.gguf ] || $ML/llama-quantize granite-1b-a400m-f16.gguf g-$q.gguf $q >/dev/null 2>&1; done
T=wikitext-2-raw/wiki.test.raw
for q in Q8_0 Q4_K_M; do
  f=g-$q.gguf; [ $q = F16 ] && f=granite-1b-a400m-f16.gguf
  echo "== $q mainline"; $ML/llama-perplexity -m $f -f $T -c 512 --chunks 40 -t 16 2>&1 | grep -E "Final estimate|error|abort" 
  echo "== $q ik";       $IK/llama-perplexity -m $f -f $T -c 512 --chunks 40 -t 16 2>&1 | grep -E "Final estimate|error|abort|nan"
  echo "== $q ik -fmoe"; $IK/llama-perplexity -m $f -f $T -c 512 --chunks 40 -t 16 -fmoe 2>&1 | grep -E "Final estimate|error|abort|nan"
done
echo "== tg speed Q8_0"
$ML/llama-bench -m g-Q8_0.gguf -t 16 -p 0 -n 64 2>&1 | tail -n 3
$IK/llama-bench -m g-Q8_0.gguf -t 16 -p 0 -n 64 -fmoe 1 2>&1 | tail -n 3
echo ALLDONE
