# DeepSeek V4.1 Flash on the R820: inference plan

Goal: run DeepSeek V4.1 Flash locally and push single-stream tokens/sec as high as this hardware allows, **without buying anything**. Phase 1 tunes Q8, Phase 2 moves to Q4.

## Hardware (as measured)

| Part | Detail |
|---|---|
| CPUs | 4 × Xeon E5-4640 v1 (Sandy Bridge-EP), 32 cores / 64 threads, 2.4–2.8 GHz. **AVX only: no AVX2, F16C or FMA** |
| RAM | 48 × 16 GB dual-rank DDR3 RDIMM, rated 1600 but **running at 1333** (3 per channel). 768 GB total, **170.6 GB/s on paper** |
| NUMA | 4 nodes, ~190 GB each. Socket 0↔2 and 1↔3 are **two hops apart** (distance 30) |
| GPU | NVIDIA Quadro P2200 (Pascal GP106), 5 GB GDDR5X, ~200 GB/s |
| Disk | 131 GB on RAID-1 (PERC H710), too small for the model |
| Network | 4 × 1 GbE (BCM5720), 2 × 10 GbE SFP+ (Intel 82599) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0, host `r820` (`r820.local`, DHCP, currently 192.168.1.136) |

Free RAM option: pull 16 sticks to run 2 per channel (512 GB). If the BIOS then runs them at 1600, that's **204.8 GB/s (+20%)**. Too small for Q8, fine for Q4.

## Measured (Phase 0, 2026-10-29)

Read-only bandwidth, AVX 256-bit loads, 48 sticks at 1333:

| Test | As delivered (DAPC profile, 433 W cap) | Now (Performance profile, cap removed) |
|---|---|---|
| All 32 cores, each reading its own node | 85 | **110** |
| All 64 threads, each reading its own node | 95.7 | 104 |
| One socket, local memory | 28–29 | 17 on a cold first run, then 27–28 |
| One socket, memory 1 hop away | 4.7 | 5.3–6.3 |
| One socket, memory 2 hops away | 4.0 | 3.2–4.0 |
| All 32 cores, memory interleaved over 4 nodes | 23.2 | 28–31 |

**What changed:**
- **The server was power-capped at 433 W** (DCMI power limit, almost certainly set by a previous owner). Under load it throttled the cores to ~1,800 MHz. Removed with `ipmitool dcmi power deactivate`. It now draws ~630 W at full memory load with cores at ~2,600 MHz, within one 1100 W PSU.
- **BIOS System Profile** changed from PerfPerWattOptimizedDapc to **PerfOptimized**: C-states and C1E off, processor power MaxPerf. Everything else was already right: memory 1333 (the ceiling for 3 dual-rank sticks per channel at 1.5 V), Optimizer mode, QPI at maximum, node interleave off, prefetchers on. BIOS 2.7.0 is the last release. CollectSystemInventoryOnRestart is now off, which made the first BIOS job stall.
  - If an iDRAC BIOS job ever sticks: `racadm racreset hard`, then `racadm jobqueue delete -i JID_CLEARALL_FORCE`, then set the attribute and create the job again.

**Consequences:**
- Cross-socket reads stay slow, at ~3–6 GB/s; that's the platform, not a setting. Every byte of weights must be read by a thread on the socket that holds it.
- **Interleave is only for stock baselines** (~30 GB/s). The tuned engine needs NUMA-local placement.
- **NUMA-local placement moves from Phase 5 into Phase 1.** Split every weight matrix by rows into 4 quarters, one per node: each quarter is `mbind`-placed on its node and computed only by that node's 16 threads, so only the small activations cross sockets. This covers both dense and per-expert matrices, so the load stays balanced whatever experts get picked.
- **Usable ceiling: ~110 GB/s.** Decode speed ≈ 110 ÷ bytes per token (see "Model facts").

**Done on the server:**
- **OS:** `/etc/sysctl.d/90-llm.conf` sets numa_balancing=0, swappiness=1 and max_map_count. `llm-tune.service` sets the performance governor and THP=always. The GRUB command line adds the C-state limits.
- **Builds:** mainline llama.cpp b11979 at `~/src/llama.cpp/build-cpu`. ik_llama.cpp at `~/src/ik_llama.cpp/build-cpu` on local branch `r820-avx1`. Both already support DeepSeek V4.1, with DSpark in ik.
- **ik on AVX1:**
  - ik's own no-AVX2 fallback is broken: stubs with stale signatures, ops that abort, and `GGML_IQK_MUL_MAT=OFF` no longer compiles.
  - Fixed by building the IQK AVX2 kernels through **SIMDe**, a header library providing AVX2/FMA intrinsics on AVX/SSE. Enable with option `GGML_IQK_AVX1_SIMDE=ON`; the shim is `ggml/src/iqk/iqk_avx1_shim.h`. Verified: no AVX2 or FMA instructions in `libggml.so`.
  - Also added SSE versions of two sigmoid helpers in `ggml.c`.
  - `ggml.c`'s type table picks each type's activation format by `__AVX2__`. It now follows `GGML_IQK_X86_AVX2`, which is true for real AVX2 or the SIMDe build, so it matches the kernels. Without this the fused MoE kernel aborted.
  - Patch: `~/src/ik-r820-avx1.patch`, plus the untracked `ggml/src/iqk/iqk_avx1_shim.h`, also copied to `~/src/`.
  - This replaces most of the Phase 3 port. Phase 3 becomes profiling the SIMDe hot loops and hand-tuning them.
- **ik correctness on this CPU** (test harness `~/bench/mmcheck.c`, ggml ops against a double-precision reference):
  - `mul_mat` and `mul_mat_id` (the MoE expert path) are correct for f16, bf16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, **MXFP4**, IQ4_XS and IQ4_NL, at 1 to 512 tokens. Relative error is ~3–7×10⁻⁴, which is just the 8-bit activations.
  - Qwen3-0.6B Q8_0 perplexity: ik 21.667, mainline 21.704.
  - Granite-MoE gives garbage on ik (fine on mainline). This is Granite-specific in ik, since its kernels check out; not relevant to DeepSeek.
- **GPU:**
  - **Driver:** NVIDIA 580.178.04 (`nvidia-headless-580-server`, closed module). Quadro P2200 on PCIe 3.0 x16, attached to **NUMA node 1**.
  - **Two fixes were needed:** `/etc/modprobe.d/blacklist-nouveau.conf`, and an override `/etc/udev/rules.d/71-nvidia.rules` without the `modprobe -r` remove rules, which ping-ponged the driver when it was loaded live.
  - **CUDA 12.4** from Ubuntu's `nvidia-cuda-toolkit`; nvcc uses gcc-13 as its host compiler.
- **Network:** avahi, so the server is `r820.local`. netplan uses `dhcp-identifier: mac` from the next boot.
- **Builds:** `build-cpu` and `build-cuda` (sm_61) for both ik and mainline, under `~/src/*/`.
- **Model in RAM:** `/mnt/models` is a 560 GB tmpfs, `mpol=interleave` and `huge=always`, not in fstab. It's filled by a one-off stream from the NAS: `nc -l 9000 | tar x` on the server, `tar c | nc -N` on the NAS. The NAS link is 1 GbE, ~117 MB/s, ~72 min for the Q8.

**NUMA-local engine work in ik (done, opt-in via environment variables; patch in `~/src/ik-r820-avx1-numa.patch`):**
- `GGML_NUMA_BLOCKED=1` with `--numa distribute`:
  - **Blocked pinning:** thread *i* is pinned to node ⌊*i* × nodes ÷ threads⌋ instead of *i* mod nodes. IQK's static contiguous row split then gives node *k* rows quarter *k* of every dense matrix.
  - **Static MoE split:** expert chunks are handed out by node (node *k*'s threads take the chunks of quarter *k* of each active expert) instead of through the shared atomic counter. Applies to both `mul_mat_id` and the fused up/gate path.
- `GGML_NUMA_PLACE=1`: after loading, `ggml_numa_place_tensor()` `mbind(MPOL_BIND, MPOL_MF_MOVE)`s row quarter *k* of every weight matrix (each expert slice separately) to node *k*. This migrates the existing tmpfs or page-cache pages, so it needs no extra RAM. Token and Engram embeddings are skipped, since they're only row-looked-up.
- **Verified:** static expert split results are identical to the stock path (MXFP4, Q8_0, Q4_K at 1–33 tokens, 64 threads); Qwen3 perplexity is unchanged (21.6671).
- **Caveat:** a merged up+gate expert tensor would need each half placed separately. DeepSeek's GGUF keeps them separate.

## Model facts

- 552B parameters in the main model, plus 196.6B in the **Engram** lookup tables (read one row per token, negligible bandwidth).
- 384 routed experts per layer, 6 active per token. Separate **DSpark** drafter for speculative decoding (~8 GB at Q8).
- Q8 download is ~508 GB (474 GiB, 9 shards, SHA256 verified) at `nas.local:/volume1/@home/richard/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-Q8_K_XL`.
- **The "Q8_K_XL" file stores the routed experts as MXFP4** (ggml type 39, 4.25 bits per weight). Each expert tensor is 5120 × 2304 × 384 experts, 2.41 GB per tensor and ~18.8 MB per expert's three matrices. Only the non-expert weights are Q8.
  - So **requantizing the experts to Q4_K would not shrink them** (4.5 bpw against 4.25). The download is effectively already the "Q4 experts" model, and Phase 2 needs rethinking: see below.
- **Estimated bytes per token:** 6 experts × 18.8 MB ≈ 113 MB per MoE layer, plus attention and shared weights at Q8.
  - The 203 GB shard is likely the Engram tables, of which only one row per table is read per token. If so, the experts span roughly 42 layers, giving **~5–8 GB per token**, not the ~17 GB first assumed.
  - To confirm from the full tensor list (`~/bench/gguf_tensors.py`) once all shards are copied.

Decode speed ≈ usable bandwidth ÷ bytes per token. Q8 is ~1.06 bytes/weight (~17 GB/token), Q4 experts ~0.56 (~9.5 GB/token). **The quantisation level is a bigger lever than any RAM or CPU change.**

## Compatibility issues

| Area | Issue | Fix |
|---|---|---|
| NVIDIA driver | Pascal only works with the **proprietary 580 legacy branch**. The open kernel modules are Turing+ only | **Resolved:** Ubuntu's `nvidia-headless-580-server` (580.178.04) builds on kernel 7.0 as is |
| A driver written from scratch | **Not feasible.** Pascal clock and power control needs NVIDIA-signed firmware; nouveau runs the card stuck at low boot clocks | Use the 580 proprietary driver |
| CUDA | CUDA 13 dropped Pascal | CUDA 12.x, `CMAKE_CUDA_ARCHITECTURES=61` |
| Compiler | CUDA 12.x won't accept Ubuntu 26.04's newest GCC | Install gcc-13 or gcc-14 for nvcc |
| llama.cpp | V4.1 needs **b11870 or newer**; old-format GGUFs don't load | Build latest; convert old downloads with the provided script |
| ik_llama.cpp fast paths | `iqk_mul_mat` **needs AVX2**; AVX-only builds must set `GGML_USE_IQK_MULMAT=OFF` and fall back to slow generic code | Rewrite the IQK routines for AVX (Phase 3) |
| ik_llama.cpp AVX-only bugs | NaN logits reported on the non-IQK path; IQ*_KT quantisation silently broken without AVX2 | Apply the fork fix; quantise IQ formats on the Mac (ARM NEON path) |
| ik_llama.cpp V4.1 support | In PRs #2455 / #2506 / #2545; merge state unverified | Check main, merge PRs if needed |
| CPU build | Default builds target AVX2 | Build on the server with `-march=native` |
| Everything else | PERC H710, both NICs and NUMA tools all fine on kernel 7.0 | Nothing needed |

## Base engine

**ik_llama.cpp**, with mainline llama.cpp (b11870+) kept as the correctness reference. ik brings fused MoE (`-fmoe`), MLA attention (`-mla 3`), load-time repacking (`-rtr`) and IQ4_K/IQ4_KS formats. Its speed advantage only arrives once its kernels are rewritten for AVX, which is the custom-kernel work anyway.

## Phases

### Phase 0: preparation (during the download)
- **OS:** NUMA auto-balancing off (`kernel.numa_balancing=0`), performance governor, huge pages.
- **BIOS:** System Profile = Performance, C-states off, memory mode Optimizer, Node Interleaving **off**.
- **Memory benchmark** per socket and across all sockets: the real bandwidth ceiling.
- **Build ik_llama.cpp** with `GGML_USE_IQK_MULMAT=OFF`, apply the AVX-only NaN fix, confirm or merge V4.1 support.
- **Build mainline llama.cpp b11870+** as the reference.
- **RAM disk:** ~~interleaved tmpfs~~. Measured interleave is 23 GB/s, so it's ruled out. Instead, load with `--no-mmap` and place each quarter of every tensor on its own node (see "Measured"). The NAS copy stays the source of truth.
- **GPU driver:** 580 legacy + CUDA 12.x, patched for kernel 7.0 if needed.

### Phase 1: Q8 tuning (all 48 sticks, 1333)
1. Run both builds on an identical prompt and **compare output probabilities** to prove ik's AVX path is correct.
2. Calculate exact bytes per token from the GGUF.
3. Tune NUMA placement, thread count (32 vs 64), pinning per socket, and ik options (`-fmoe -mla 3 -rtr`).
4. GPU: dense layers and KV cache on the P2200 (`-ngl 99 -ot "exps=CPU"`), then GPU prompt processing for long prompts.
5. **Expected: ~3.5–5 tok/s baseline, ~5–7 tuned.**

### Phase 2: Q4
1. Requantise the experts to **Q4_K_M** (works on both builds today), keep dense layers at Q8: `llama-quantize --allow-requantize`. Write the result to the NAS, since Q8 and Q4 won't both fit in RAM.
2. Repeat the Phase 1 tuning.
3. **RAM test:** remove 16 sticks (1 per channel) and check the configured speed. Keep that layout if it runs at 1600.
4. Quality check against Q8: perplexity plus your own prompts.
5. **Expected: ~7–10 tok/s, +20% if the RAM runs at 1600.**

### Phase 3: rewrite ik's fast routines for AVX
1. Port `iqk_mul_mat` to AVX: 128-bit integer maths (SSSE3/AVX: `pmaddubsw`, `pmaddwd`), 256-bit float, no FMA. Start with Q8_0 and Q4_K, then IQ4_K/IQ4_KS.
2. Fix IQ quantisation on this CPU, or quantise on the Mac.
3. Compare output probabilities against mainline at every step.
4. **Expected: approaching the bandwidth limit, ~10–13 tok/s at Q4.**

### Phase 4: speculative decoding
- DSpark drafter on CPU cores (or the GPU if a version under 5 GB exists), batch verification in the engine, tune draft length.
- MoE limits the gain: each drafted token needs its own 6 experts. **Expected ×1.3–1.5, to be measured.**

### Phase 5: remaining engine work inside ik_llama.cpp
1. **Expert-per-socket placement:** experts split across the 4 nodes, dense weights replicated on each, only activations cross sockets.
2. Huge pages, software prefetching, streaming loads.
3. GPU prompt processing with CUDA 12 kernels for sm_61.
4. Layer-by-layer comparison against the reference implementation.
5. **Expected at Q4: ~13.5 tok/s at 1333, ~17 at 1600, 20–25 with GPU and speculative decoding.**

## Targets (no purchases)

| | Q8 | Q4 experts |
|---|---|---|
| Stock llama.cpp | 3.5–5 | 6–9 |
| Tuned, with GPU | 5–7 | 8–12 |
| + speculative decoding | 6.5–10 | 10–16 |
| Custom engine, everything combined | **10–12** | **20–25** |

All figures depend on the measured bytes per token from Phase 1.

## Sources

- [Unsloth: DeepSeek-V4.1-Flash run locally guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash)
- [ISTA-DASLab: DeepSeek-V4.1-Flash GGUF (llama.cpp b11870+)](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
- [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- [ik_llama.cpp build docs](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md)
- [Fork PR: AVX1 non-IQK mul_mat fix](https://github.com/FabioLeitao/ik_llama.cpp/pull/4)
- [Fork issue: non-IQK CPU path NaN logits](https://github.com/FabioLeitao/ik_llama.cpp/issues/1)
- [ik_llama.cpp #2605: IQ*_KT broken without AVX2](https://github.com/ikawrakow/ik_llama.cpp/issues/2605)
- [ik_llama.cpp #2455: add DeepSeek-V4.1](https://github.com/ikawrakow/ik_llama.cpp/pull/2455)
- [ik_llama.cpp #2506: converter support](https://github.com/ikawrakow/ik_llama.cpp/pull/2506)
- [ik_llama.cpp #2545: vision support](https://github.com/ikawrakow/ik_llama.cpp/pull/2545)
- [llama.cpp #28696: convert DeepSeek V4.1](https://github.com/ggml-org/llama.cpp/pull/28696)
- [unslothai/llama.cpp #293: deepseek41 engram host placement](https://github.com/unslothai/llama.cpp/pull/293)
