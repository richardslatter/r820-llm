#!/bin/bash
# greedy.sh <tag>: 64 greedy tokens after a ~700-token wikitext prompt (decode with the HCA top-k gather active)
M=${MDIR:-/mnt/models/dq6k}/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
B=${BIN:-$HOME/src/ik_llama.cpp/build-dev/bin}
[ -f /tmp/longprompt.txt ] || head -c 3200 ~/models/test/wikitext-2-raw/wiki.test.raw | tail -c 3000 > /tmp/longprompt.txt
GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 $B/llama-cli -m $M -t 64 --numa distribute -fa on -c 2048 -f /tmp/longprompt.txt -n 64 --temp 0 --ignore-eos > ~/bench/greedy-$1.out 2> ~/bench/greedy-$1.err
touch ~/bench/greedy-$1.done
