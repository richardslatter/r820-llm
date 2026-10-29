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
| **iDRAC Node Manager policy 13 holds the system at 343 W** (no inlet sensor) | blocking: needs the control panel or PSU1 checked |
| GPU offload (dense on P2200, experts mmapped on CPU) | works with `LLAMA_OT_CPU_MMAP=1`; perplexity check pending |
| Q8_0 vs BF16 dense quality | done: perplexity 3.5081 vs 3.4907 (+0.5%) |
| tok/s: stock vs NUMA-local, thread counts, GPU offload | after the cap is off |

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
| `ik_llama/ds41_gguf_patch.py` | Lets ik load DeepSeek-V4.1 GGUFs in the Unsloth/mainline layout |
| `mainline/mxfp8.patch` | The MXFP8 change as applied to mainline llama.cpp |
| `tools/requant_shard.c` | Requantizes the BF16 tensors of one GGUF shard, copying the rest |
| `ik_llama/numa_pin_patch.py` | Follow-up to numa_patch.py: one CPU per thread, populate before mbind |
| `tools/placecheck.py` | Reports which node each weight's row quarters actually sit on (move_pages) |
| `tools/splitpages.py` | Splits tmpfs huge pages to 4 KiB in place (pid/address debugfs form) |
| `tools/rd2.c`, `tools/rdfile.c`, `tools/clearpol.c` | Bandwidth tests (4 KiB vs 2 MB pages, tmpfs files), policy reset |
| `tools/mmcheck.c` | ggml `mul_mat` / `mul_mat_id` checker against a double-precision reference |
| `tools/gguf_tensors.py` | Dependency-free GGUF tensor lister (name, type, shape, bytes) |
| `tools/rd.c` | Read-bandwidth benchmark (256-bit loads, OpenMP) |
| `scripts/` | Build and perplexity-check scripts used on the server |
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

## Tools

```bash
gcc -O2 -march=native -fopenmp tools/rd.c -o rd && numactl --cpunodebind=0 --membind=0 ./rd 8
python3 tools/gguf_tensors.py model-*.gguf > tensors.tsv
gcc -O2 -march=native -I$IK/ggml/include tools/mmcheck.c -o mmcheck -L$IK/build-cpu/ggml/src -lggml -lm -Wl,-rpath,$IK/build-cpu/ggml/src
./mmcheck 64                    # all types, mul_mat and mul_mat_id
MMCHECK_NUMA=1 GGML_NUMA_BLOCKED=1 ./mmcheck 64   # static NUMA expert split
```
