# r820-llm

Running DeepSeek V4.1 Flash (552B MoE + Engram tables) on an old Dell PowerEdge R820 as fast as it will go, with no new hardware.

- 4 × Xeon E5-4640 v1 (Sandy Bridge-EP: **AVX only**, no AVX2/FMA/F16C), 4 NUMA nodes
- 768 GB DDR3 (48 × 16 GB, 1333 MT/s), measured ~110 GB/s when every socket reads its own memory
- Quadro P2200 (Pascal, 5 GB), PCIe 3.0 x16 on node 1
- Ubuntu 26.04.1, kernel 7.0, NVIDIA 580 + CUDA 12.4
- Engine: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), with mainline llama.cpp kept as the correctness reference

This is a working log, not a polished project. The plan and all measurements are in [docs/plan.md](docs/plan.md); hardware and iDRAC notes are in [docs/hardware-notes.md](docs/hardware-notes.md).

## Status

| Milestone | State |
|---|---|
| Boot fix, OS install, BIOS and power tuning | done |
| Memory bandwidth benchmarks | done: 110 GB/s local, ~30 interleaved, 3–6 cross-socket |
| ik_llama.cpp on AVX1 (SIMDe shim, type-table fix) | done, kernels verified against a double-precision reference |
| NUMA-local placement in ik (blocked pinning, static MoE split, mbind) | done, verified on small models |
| NVIDIA driver + CUDA builds | done |
| Model (UD-Q8_K_XL, 474 GiB) copied into a RAM disk | done |
| MXFP8 type (the Engram table's format) for both engines | done |
| ik reads the Unsloth/mainline DeepSeek-V4.1 GGUF layout | done |
| BF16 dense weights → Q8_0 (rewrite of one 17 GB shard) | done |
| DeepSeek runs on both engines | done: ik 2.31 tok/s, mainline 1.22 (stock placement) |
| ik vs mainline perplexity check | done: 3.5081 vs 3.5073 |
| NUMA-local placement (4 KiB pages, per-CPU pinning) | done: 99.7–100% of every row quarter on its node |
| iDRAC 433 W user cap | fixed: 936 W |
| iDRAC Node Manager policy 13 (343 W clamp) | fixed: pulled dead PSU1, Hot Spare off, cap off; cores hold 2.5 GHz under load |
| GPU offload (dense on P2200, experts mmapped on CPU) | done: correct (perplexity 3.5047) after the rope-back fix |
| Q8_0 vs BF16 dense quality | done: perplexity 3.5081 vs 3.4907 (+0.5%) |
| tok/s: stock vs NUMA-local, thread counts (CPU) | done: ik NUMA-local **3.27 tok/s** (stock ik 2.60, mainline 1.58) |
| GPU offload speed | done: **3.54 tok/s** with 20 layers' dense weights on the P2200 (CPU only 3.14) |
| Phase 3a: decode profile | done: SIMDe GEMV at ~60 GB/s, libgomp barrier spin 27%, small matmuls on 2–32 of 64 threads |
| Phase 3b: native 128-bit Q8_0 / MXFP4 GEMV kernels | done: 108–110 GB/s across 4 nodes in isolation (was ~60); decode 3.27 → 3.90 tok/s |
| Phase 3c: two-level NUMA barrier, even split for small matmuls | done: 4.02 → **5.08 tok/s** |
| Phase 3d: NUMA-local grouped wo_a, Engram projections placed | done: **5.66 tok/s** (171 ms/token) |
| Phase 3e: exact vector fp16 for AVX1, decode flash attention for the shared K/V head | done: attention 15% faster, perplexity 3.5485 → 3.4919 (fp32 accumulation) |
| Phase 3e: dense format (Q8_0 / Q6_K / IQ4_NL kernels at 106–110 GB/s) | done: **Q6_K** picked, KLD 0.018 vs BF16, decode 5.94 → 6.52 tok/s |
| Phase 3e: lazy K/V concat, column-split norms and small ops | done: **6.68 tok/s** (6.17 after a 512-token prompt) |
| Phase 3e: small matmuls in the model (~12 ms/token above their standalone time) | open: split-K, per-thread activation quantisation and node-local activation copies all tried, no gain |
| Smart expert reduction (`-ser`) re-enabled | done, opt-in only: +5–9% decode but KLD 0.066–0.151 |
| Draft-free speculative decoding | done: `--spec-type suffix:n_max=2,n_min=1` gives +5–11% where output repeats the prompt, 0 on free prose; no MTP tensors in the GGUF |
| DSpark draft model (V4.1's MTP) | done: loads after adding 3 GGUF keys; `dspark:n_max=3` gives **9.0 tok/s** code, 8.0 extraction, 6.3 prose (dq6k) |
| Long context, threads, prompt speed | done: −14% decode at 16K context; 64 threads best; prompts 10.8 tok/s, compute-bound at 60–70% of AVX1 peak |
| Imatrix 4-bit dense, 4-bit output head | done: KLD 0.082 → 0.078 (within error), no speed gain; dq6k stays the pick |
| GPU retest with current kernels | done: layer offload now slows decode; `-ngl 0` on build-cuda gives prompts +40–47% for −2–5% decode; 672 W peak |

First runs (60 threads, stock placement, Q8_0 dense):
- ik_llama.cpp: **2.31 tok/s** decode, 5.3 tok/s prompt.
- mainline: 1.22 tok/s decode.
- Each token reads ~12.7 GB, so the NUMA-local work should give the next big step.

Estimate: ~6.5 tok/s NUMA-local on the CPU; ~12 with the dense part on the GPU at ~4.5 bits; ~16–19 best case with speculative decoding. The plan has the per-tensor breakdown.

## Layout

| Path | What |
|---|---|
| `docs/plan.md` | The plan, measured results, decisions |
| `docs/hardware-notes.md` | Boot fix, iDRAC/BIOS recipes, power cap, network |
| `ik_llama/r820.patch` | All local changes to ik_llama.cpp (AVX1 SIMDe build + NUMA-local engine), including the new shim header |
| `ik_llama/numa_patch.py` | Script that applies the NUMA part of the patch to a clean checkout |
| `ik_llama/mxfp8_patch.py` | Adds the MXFP8 type (gather-only) to ik or mainline; the Engram table needs it |
| `ik_llama/ds41_rope_back_patch.py` | Out-of-place attention rope-back: fixes the CUDA (sm_61) path for DeepSeek-V4.1 |
| `ik_llama/fattn_sinks_patch.py` | Attention sinks in ik's Pascal head-size-512 flash-attention kernel |
| `ik_llama/ds41_wo_a_cont_patch.py` | Contiguous input for the grouped output projection's batched matmul |
| `scripts/restore-ramdisk.sh` | Refill the RAM disk from the NAS after a power cycle |
| `system/net/` | `net-reconnect.service`: retry DHCP once a minute after a cable-less boot |
| `ik_llama/ds41_gguf_patch.py` | Lets ik load DeepSeek-V4.1 GGUFs in the Unsloth/mainline layout |
| `mainline/mxfp8.patch` | The MXFP8 change as applied to mainline llama.cpp |
| `tools/requant_shard.c` | Requantizes the BF16 tensors of one GGUF shard, copying the rest (optionally imatrix-weighted) |
| `tools/gguf_kv.py` | Dependency-free GGUF metadata dump |
| `tools/gguf_set_kv.py` | Copies a GGUF, adding or replacing metadata keys (used to make the DSpark draft load in ik) |
| `ik_llama/numa_pin_patch.py` | Follow-up to numa_patch.py: one CPU per thread, populate before mbind |
| `tools/placecheck.py` | Reports which node each weight's row quarters actually sit on (move_pages) |
| `tools/splitpages.py` | Splits tmpfs huge pages to 4 KiB in place (pid/address debugfs form) |
| `tools/rd2.c`, `tools/rdfile.c`, `tools/clearpol.c` | Bandwidth tests (4 KiB vs 2 MB pages, tmpfs files), policy reset |
| `ik_llama/iqk_avx1_gemv.h`, `ik_llama/avx1_gemv_patch.py` | Native 128-bit Q8_0 and MXFP4 matrix × small-batch kernels for Sandy Bridge, hooked into iqk for 1–8 rows (`GGML_AVX1_GEMV=0` turns them off) |
| `ik_llama/numa_barrier_patch.py` | Two-level NUMA barrier for `GGML_NUMA_BLOCKED` (`GGML_HBARRIER=0` off) and the `GGML_OP_PROF=1` per-op / per-weight profiler |
| `ik_llama/small_split_patch.py` | Even row split for matmuls with < 32 rows per thread at small batch |
| `ik_llama/grouped_split_patch.py` | NUMA-local split for grouped matmuls on a reshaped 2D weight (wo_a) |
| `tools/gemv_bench.cpp` | Correctness and per-node bandwidth test for the AVX1 kernels |
| `ik_llama/iqk_avx1_shim.h` | Force-included into iqk for the SIMDe AVX1 build |
| `ik_llama/iqk_avx1_f16c.h`, `ik_llama/ggml_f16_avx1_patch.py` | Exact vector fp16 ↔ fp32 without F16C, for iqk and ggml.c |
| `ik_llama/ds4_fa_mqa_kernel.c`, `ik_llama/ds4_fa_mqa_patch.py` | Decode flash attention for one shared fp16 K/V head, and the lazy K/V concat (`GGML_FA_MQA=0`, `GGML_FA_LAZY=0` off) |
| `ik_llama/decode_ops_patch.py` | Vector RMS-norm sums, column split for `MUL_MULTI_ADD` at batch 1, `GGML_OP_PROF=3` per-node trace |
| `ik_llama/norm_split_patch.py` | Column-split RMS norms for long single rows (`GGML_NORM_SPLIT=0` off) |
| `tools/fa_bench.c`, `tools/hc_bench.c` | Decode attention and hyper-connection norm + matmul in isolation |
| `tools/mm_bench.c` | Decode matmuls (one activation row) of any type in isolation, NUMA-placed, on 4 KB pages |
| `ik_llama/ser_patch.py` | Re-enables `-ser`: top-k selection, threshold on the unbiased weights, zero-weight slots skipped |
| `ik_llama/mm_localq_patch.py`, `ik_llama/mm_f32_decode_patch.py` | Negative results, kept for reference: per-thread activation quantisation, node-local f32 activation copy |
| `tools/f16c_test.cpp` | Exhaustive check of the fp16 conversions |
| `tools/serve.sh`, `tools/v4_chat.jinja` | LAN chat server (llama-server, web UI on :8080) and a plain chat template it can parse |
| `tools/mmcheck.c` | ggml `mul_mat` / `mul_mat_id` checker against a double-precision reference |
| `tools/gguf_tensors.py` | Dependency-free GGUF tensor lister (name, type, shape, bytes) |
| `tools/rd.c` | Read-bandwidth benchmark (256-bit loads, OpenMP) |
| `scripts/` | Build, perplexity, KL-divergence (`kld1.sh`, `kldub1.sh`), decode-speed (`tgd.sh`), op-profile (`opprof.sh`) and greedy-output (`greedy.sh`) scripts used on the server |
| `system/` | OS tuning files as installed on the server |
| `idrac/` | WS-Man request for aborting a pending BIOS configuration |

## Applying the ik_llama.cpp patch

Base commit: ik_llama.cpp `bdebfb6a0e8f92fc390e530a820bc245210d86db`. Needs `libsimde-dev` (0.8.2).

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp && cd ik_llama.cpp
git checkout bdebfb6a0e8f92fc390e530a820bc245210d86db
git apply ../r820-llm/ik_llama/r820.patch
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_IQK_AVX1_SIMDE=ON -DLLAMA_CURL=OFF
cmake --build build-cpu -j 32
```

CUDA build: see `scripts/build-cuda.sh` (sm_61, gcc-13 as the nvcc host compiler).

Mainline reference: llama.cpp `f7c01c5226cac1de1052c17920ff38cdefea0ae6` (b11979) plus `mainline/mxfp8.patch` (`git apply`), same build flags without `GGML_IQK_AVX1_SIMDE`.

### Runtime switches added by the patch

- `GGML_NUMA_BLOCKED=1` with `--numa distribute`: thread *i* goes to node ⌊*i* × nodes ÷ threads⌋, and MoE expert chunks are split statically by node, so node *k* only ever computes rows quarter *k* of each weight matrix.
- `GGML_NUMA_PLACE=1`: after loading, row quarter *k* of every weight matrix (each expert slice separately) is moved to node *k* with `mbind(MPOL_BIND, MPOL_MF_MOVE)`. Token and Engram lookup tables are left alone.

## Dense weights to Q8_0

All BF16 tensors are in the last shard, so only that one is rewritten. The others are hard-linked:

```bash
gcc -O2 -march=native -fopenmp -I$ML/ggml/include tools/requant_shard.c -o requant_shard -L$ML/build-cpu/bin -lggml-base -Wl,-rpath,$ML/build-cpu/bin
mkdir -p /mnt/models/dq8 && for i in 1 2 3 4 5 6 7 8; do ln /mnt/models/UD-Q8_K_XL/*-0000$i-of-00009.gguf /mnt/models/dq8/; done
./requant_shard /mnt/models/UD-Q8_K_XL/*-00009-of-00009.gguf /mnt/models/dq8/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00009.gguf q8_0 token_embd.weight=keep
```

The variant in use now, `dq6k`, is made the same way with `q6_K` as the type (`iq4_nl output.weight=q6_K` for `diq4`). The plan compares them.

`--imatrix file.gguf` (before the other arguments) weights the quantisation by the mean squared activations of a GGUF imatrix, such as Unsloth's `imatrix_unsloth.gguf`.

## DSpark draft model

```bash
python3 tools/gguf_set_kv.py dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf dspark-ik.gguf \
    dflash.attention.indexer.head_count=u32:32 dflash.attention.indexer.key_length=u32:128 dflash.attention.indexer.top_k=u32:512
DRAFT=/mnt/models/dspark/dspark-ik.gguf tools/serve.sh
```

## Tools

```bash
gcc -O2 -march=native -fopenmp tools/rd.c -o rd && numactl --cpunodebind=0 --membind=0 ./rd 8
python3 tools/gguf_tensors.py model-*.gguf > tensors.tsv
gcc -O2 -march=native -I$IK/ggml/include tools/mmcheck.c -o mmcheck -L$IK/build-cpu/ggml/src -lggml -lm -Wl,-rpath,$IK/build-cpu/ggml/src
./mmcheck 64                    # all types, mul_mat and mul_mat_id
MMCHECK_NUMA=1 GGML_NUMA_BLOCKED=1 ./mmcheck 64   # static NUMA expert split
```
