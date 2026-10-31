# r820-llm

DeepSeek V4.1 Flash (552B MoE) on a 2012 Dell PowerEdge R820, made as fast as the hardware allows with no new parts.

- 4 × Xeon E5-4640 (Sandy Bridge: AVX only, no AVX2/FMA/F16C), 4 NUMA nodes, 64 threads
- 768 GB DDR3-1333, ~110 GB/s when each socket reads its own memory
- Quadro P2200 (Pascal, 5 GB)
- Engine: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) with `ik_llama/r820.patch`

The full log with every measurement is in [docs/plan.md](docs/plan.md). BIOS, iDRAC and power fixes are in [docs/hardware-notes.md](docs/hardware-notes.md).

## Result

Unsloth's UD-Q8_K_XL with the BF16 dense weights requantised to Q6_K (KLD 0.018 against BF16). Experts stay MXFP4.

| | Decode tok/s |
|---|---|
| Stock llama.cpp (Q8_0 dense) | 1.58 |
| Stock ik_llama.cpp (Q8_0 dense) | 2.60 |
| **This repo, no draft** | **6.7** (5.8 at 4K–16K context) |
| **This repo + DSpark draft, code** | **9.6–9.7** (`n_max=3`), 10.9 (`n_max=4`) |
| This repo + DSpark draft, free prose | 6.0–6.7 |

Prompt processing runs at 15.2 tok/s (pp512) on the CUDA build with `-ngl 0`, against 10.8 on the CPU-only build.

## What was done

Each row is measured decode, in order.

| Change | Effect |
|---|---|
| **Power:** pulled the dead PSU, disabled Hot Spare and the iDRAC power cap | A hidden 343 W Node Manager clamp had held the cores at 0.4–1.1 GHz. They now hold 2.5 GHz, and run-to-run swings of 4× are gone. |
| **Model in a tmpfs RAM disk** on 4 KiB pages | Pages can be moved per NUMA node (2 MB pages are larger than a 1.6 MB row quarter) at the same bandwidth. |
| **Dense BF16 → Q8_0, later Q6_K** (`tools/requant_shard.c`, rewrites one 17 GB shard) | 12.7 (Q8_0) → 10.9 GB read per token. Q6_K has a native kernel, so the smaller read is a real gain. |
| **ik on AVX1** (SIMDe shim), MXFP8 type, Unsloth GGUF layout | Makes ik build and load the model at all: 2.60 tok/s, against mainline's 1.58 |
| **NUMA-local engine:** threads pinned per node, static per-node expert split, each weight's row quarters `mbind`-ed to their node | 2.60 → 3.27 |
| **Native 128-bit matmul kernels** (Q8_0, Q6_K, IQ4_NL, MXFP4, 1–8 rows) in place of SIMDe's emulated AVX2 | 3.27 → 3.90; 108–110 GB/s across 4 nodes in isolation |
| **Two-level NUMA barrier**, even split for small matmuls | 4.02 → 5.08 |
| **NUMA-local grouped output projection** (`wo_a`), Engram projections placed | → 5.66 |
| **Exact vector fp16 without F16C**, decode flash attention for the shared K/V head | Attention 15% faster; perplexity 3.5485 → 3.4919 (fp32 accumulation) |
| **Q6_K dense** with its native kernel | 5.94 → 6.52 |
| **Lazy K/V concat**, column-split norms, small ops spread over all threads | → 6.68 |
| **DSpark draft model** (V4.1's MTP): 3 missing GGUF keys added (`tools/gguf_set_kv.py`) | Code 9.0 with `n_max=3` |
| **Cheaper draft:** shared output head instead of a single-node copy, SSE2 softmax | Draft round 66–70 → ~33 ms |
| **Faster verify batch:** Q6_K batch kernel, HC_POST and copy splits, f32 K-slice | `pp4` 11.15 → 12.9; code 9.6–9.7, 10.9 at `n_max=4` |
| **CUDA correctness on Pascal:** out-of-place rope-back, attention sinks in the hs512 kernel, contiguous `wo_a` input, scheduler input sync | GPU output matches the CPU (perplexity 3.5047 vs 3.5081); the DSpark draft also runs correctly on the GPU |
| **CUDA build with `-ngl 0`** | Prompts +40–47% (large batches go to the GPU), decode −2–5% |

## What didn't work

- **Dense layers on the GPU.** It helped while the CPU kernels were slow (3.14 → 3.54). Once they doubled, it costs 16–19%. Each CPU↔GPU hand-off inside ggml costs 100–165 µs, and PCIe adds no bandwidth worth having.
- **Shared expert alone on the GPU:** −13%, even when the CPU split was allowed to start before the GPU copy (`experiments/sched_defer_patch.py`). The ceiling with zero overhead was 3–8%.
- **DSpark draft on the GPU.** It first gave wrong output; two bugs were fixed (a scheduler race and an in-place rope). It is correct now but ~10% slower than on the CPU, for the same hand-off cost.
- **Closing the ~12 ms gap of the small matmuls.** They run ~3× slower in the model than alone. Split-K, per-thread activation quantisation and node-local activation copies (`experiments/mm_*`) gained nothing; the cause is unmeasured (no `perf` on this box).
- **K-slice for the router:** doubled its time; strided reads cost more than they save.
- **Long-context attention.** Head partitions, out-of-line PV kernels, a vector mask scan and a faster fp16 conversion (`experiments/fa_*`, `f16_magic_patch.py`) were all within noise. Attention is only ~7.5% of decode at 4K.
- **4-bit dense (IQ4_NL, with or without imatrix):** +8–11% speed, but KLD 0.08 against 0.018. Below the Q6 quality floor. A 4-bit output head saved nothing measurable.
- **Smart expert reduction (`-ser`):** +5–9% but KLD 0.066–0.151. It is re-enabled, but opt-in only.
- **Draft-free speculation (suffix / n-gram):** +5–11% only where the output repeats the prompt, nothing on prose. DSpark replaced it.
- **Other quant types and layouts** (Q4_K, IQ4_K, IQ5_K, `_r4`/`_r8`): they only run through SIMDe on AVX1, well below the native kernels' speed (1.1–1.9 against 2.4–5.0 GB/s per core).
- **Fewer threads:** 32 and 48 threads are slower than 64.
- **Graph reuse:** impossible for DeepSeek-V4, since shapes change every token; worth ~7 ms anyway.

The one remaining idea is a physical change for the owner: 2 DIMMs per channel at 1600 MT/s, estimated at +15–20% decode.

## Setup

Base commit: ik_llama.cpp `bdebfb6a0e8f92fc390e530a820bc245210d86db`. Needs `libsimde-dev` (0.8.2).

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp && cd ik_llama.cpp
git checkout bdebfb6a0e8f92fc390e530a820bc245210d86db
git apply ../r820-llm/ik_llama/r820.patch
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_IQK_AVX1_SIMDE=ON -DLLAMA_CURL=OFF
cmake --build build-cpu -j 32
```

The CUDA build (sm_61, gcc-13 as nvcc's host compiler) is in `scripts/build-cuda.sh`.

Dense weights to Q6_K: only the last shard holds BF16 tensors, so the other eight are hard-linked.

```bash
gcc -O2 -march=native -fopenmp -I$ML/ggml/include tools/requant_shard.c -o requant_shard -L$ML/build-cpu/bin -lggml-base -Wl,-rpath,$ML/build-cpu/bin
mkdir -p /mnt/models/dq6k && for i in 1 2 3 4 5 6 7 8; do ln /mnt/models/UD-Q8_K_XL/*-0000$i-of-00009.gguf /mnt/models/dq6k/; done
./requant_shard /mnt/models/UD-Q8_K_XL/*-00009-of-00009.gguf /mnt/models/dq6k/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00009.gguf q6_K token_embd.weight=keep
```

DSpark draft and the LAN chat server (web UI and OpenAI API on :8080):

```bash
python3 tools/gguf_set_kv.py dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf dspark-ik.gguf \
    dflash.attention.indexer.head_count=u32:32 dflash.attention.indexer.key_length=u32:128 dflash.attention.indexer.top_k=u32:512
DRAFT=/mnt/models/dspark/dspark-ik.gguf tools/serve.sh
```

`serve.sh` sets the two runtime switches the patch adds:
- `GGML_NUMA_BLOCKED=1` (with `--numa distribute`): thread *i* runs on node ⌊*i* × 4 ÷ 64⌋, and node *k* only computes row quarter *k* of each weight.
- `GGML_NUMA_PLACE=1`: after loading, row quarter *k* of every weight is moved to node *k*.

## Layout

| Path | What |
|---|---|
| `ik_llama/r820.patch` | Every change to ik_llama.cpp: apply this |
| `ik_llama/parts/` | Most of the same changes as separate, commented scripts, by topic (NUMA, kernels, attention, CUDA fixes, MXFP8, GGUF layout) |
| `ik_llama/experiments/` | Tried and not kept (see above) |
| `mainline/mxfp8.patch` | MXFP8 for mainline llama.cpp, the correctness reference |
| `tools/` | Requantiser, GGUF tools, server script and chat template, kernel and bandwidth benchmarks, `mmcheck` (matmuls against a double-precision reference) |
| `scripts/` | Build, perplexity, KLD, decode-speed and profile scripts used on the server; RAM disk restore |
| `system/`, `idrac/` | OS tuning files as installed; iDRAC BIOS-job abort request |
| `docs/` | The plan and log, and the hardware notes |
