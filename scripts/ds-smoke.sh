#!/bin/bash
# Greedy 32-token completion of a fixed prompt on DeepSeek-V4.1-Flash; prints the text and timings.
#   ds-smoke.sh ml|ik <threads> [extra args...]   (env vars such as GGML_NUMA_BLOCKED pass through)
E=$1; T=$2; shift 2
M=${MDIR:-/mnt/models/UD-Q8_K_XL}/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
P="The three largest moons of Jupiter are"
case $E in
  ml) BIN="$HOME/src/llama.cpp/build-cpu/bin/llama-completion -no-cnv" ;;
  ik) BIN="$HOME/src/ik_llama.cpp/build-cpu/bin/llama-cli" ;;
esac
OUT=~/bench/smoke-$E-t$T${TAG:+-$TAG}
/usr/bin/time -v timeout 1800 $BIN -m $M -p "$P" -n 32 --temp 0 -t $T --numa distribute -c 2048 "$@" > $OUT.out 2> $OUT.err
echo "rc=$?"; cat $OUT.out; echo
grep -E "load time|prompt eval time|eval time|NUMA-placed|Maximum resident|Elapsed \(wall|error|failed|abort" $OUT.err | grep -v "^ *$" | tail -n 12
