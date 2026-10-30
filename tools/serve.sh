#!/bin/bash
# serve.sh: LAN chat server on the current pick (dense weights in Q6_K), web UI on :8080
# DRAFT=/mnt/models/dspark/dspark-ik.gguf adds the DSpark draft model (NMAX tokens per draft, default 3).
M=${MDIR:-/mnt/models/dq6k}/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
B=${BIN:-$HOME/src/ik_llama.cpp/build-cpu/bin}
exec env GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 $B/llama-server -m $M -t 64 --numa distribute -fa on \
  ${DRAFT:+-md $DRAFT --spec-type dspark:n_max=${NMAX:-3}} \
  -c ${CTX:-16384} --jinja --chat-template-file ${TMPL:-$HOME/bench/v4_chat.jinja} --reasoning-format deepseek --reasoning ${REASONING:-off} --host 0.0.0.0 --port ${PORT:-8080} -a ${ALIAS:-DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL} "$@"
