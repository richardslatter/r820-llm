#!/usr/bin/env python3
# Stream one chat completion from the LAN server and log each chunk with its arrival time.
#   record.py <url> <out.json> <prompt>
import json, sys, time, urllib.request

url, out, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = {
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2000,
    "temperature": 0,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": True},
}
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
events, timings = [], None
t0 = time.monotonic()
with urllib.request.urlopen(req, timeout=900) as r:
    for raw in r:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        j = json.loads(data)
        t = time.monotonic() - t0
        if j.get("timings"):
            timings = j["timings"]
        for ch in j.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("reasoning_content"):
                events.append({"t": t, "k": "r", "s": d["reasoning_content"]})
            if d.get("content"):
                events.append({"t": t, "k": "c", "s": d["content"]})
end = time.monotonic() - t0
json.dump({"prompt": prompt, "events": events, "end": end, "timings": timings}, open(out, "w"), ensure_ascii=False)
n_r = sum(1 for e in events if e["k"] == "r")
n_c = len(events) - n_r
print(f"{len(events)} chunks ({n_r} reasoning, {n_c} answer) in {end:.1f} s; first at {events[0]['t']:.2f} s")
if timings:
    print(f"server: prompt {timings['prompt_n']} tok at {timings['prompt_per_second']:.1f} t/s, "
          f"gen {timings['predicted_n']} tok at {timings['predicted_per_second']:.2f} t/s")
