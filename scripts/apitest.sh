#!/bin/bash
# apitest.sh [port]: end-to-end speed through the LAN server's chat API (code and prose prompt, 256 tokens)
P=${1:-8080}
for prompt in "Write a Python function that parses an ISO 8601 date string into a datetime without using any libraries, with docstring and tests." \
              "Write a short story about a lighthouse keeper who finds a strange bottle on the shore."; do
  curl -s http://localhost:$P/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":256,\"temperature\":${TEMP:-0}}" |
    python3 -c "import json,sys; t=json.load(sys.stdin)['timings']; print(f\"prompt {t['prompt_n']} tok at {t['prompt_per_second']:.1f} t/s, gen {t['predicted_n']} tok at {t['predicted_per_second']:.2f} t/s\")"
done
