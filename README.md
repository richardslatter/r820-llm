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
| Model (UD-Q8_K_XL, 474 GiB) copied into a RAM disk | in progress |
| DeepSeek smoke test, ik vs mainline | next |
| tok/s: stock vs NUMA-local, 32 vs 64 threads, GPU offload | next |

Current estimate (from the tensor list of 7 of 9 shards; to be replaced by measurements):
- The download (Unsloth UD-Q8_K_XL) stores the routed experts as **MXFP4**, ~4.5 GB per token over 40 MoE layers.
- It stores attention, shared experts and the output head as **BF16**, ~15.5 GB per token.
- So a token reads ~20 GB as downloaded: **~4 tok/s** NUMA-local on the CPU.
- Requantizing only the BF16 part to Q8_0 (near-lossless) gives ~12.7 GB/token and **~6.5 tok/s**.
- At ~4.5 bits the dense part (~4.4 GB) might fit on the 5 GB GPU, leaving the CPU only the experts: **~12 tok/s**.
- With speculative decoding on top, **~16–19 tok/s** best case.

## Layout

| Path | What |
|---|---|
| `docs/plan.md` | The plan, measured results, decisions |
| `docs/hardware-notes.md` | Boot fix, iDRAC/BIOS recipes, power cap, network |
| `ik_llama/r820.patch` | All local changes to ik_llama.cpp (AVX1 SIMDe build + NUMA-local engine), including the new shim header |
| `ik_llama/numa_patch.py` | Script that applies the NUMA part of the patch to a clean checkout |
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

Mainline reference: llama.cpp `f7c01c5226cac1de1052c17920ff38cdefea0ae6` (b11979), stock.

### Runtime switches added by the patch

- `GGML_NUMA_BLOCKED=1` with `--numa distribute`: thread *i* goes to node ⌊*i* × nodes ÷ threads⌋, and MoE expert chunks are split statically by node, so node *k* only ever computes rows quarter *k* of each weight matrix.
- `GGML_NUMA_PLACE=1`: after loading, row quarter *k* of every weight matrix (each expert slice separately) is moved to node *k* with `mbind(MPOL_BIND, MPOL_MF_MOVE)`. Token and Engram lookup tables are left alone.

## Tools

```bash
gcc -O2 -march=native -fopenmp tools/rd.c -o rd && numactl --cpunodebind=0 --membind=0 ./rd 8
python3 tools/gguf_tensors.py model-*.gguf > tensors.tsv
gcc -O2 -march=native -I$IK/ggml/include tools/mmcheck.c -o mmcheck -L$IK/build-cpu/ggml/src -lggml -lm -Wl,-rpath,$IK/build-cpu/ggml/src
./mmcheck 64                    # all types, mul_mat and mul_mat_id
MMCHECK_NUMA=1 GGML_NUMA_BLOCKED=1 ./mmcheck 64   # static NUMA expert split
```
