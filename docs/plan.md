# DeepSeek V4.1 Flash on the R820: inference plan

Goal: run DeepSeek V4.1 Flash locally and push single-stream tokens/sec as high as this hardware allows, **without buying anything**. Phase 1 tunes the download as is; Phase 2 requantizes its BF16 dense weights.

## Hardware (as measured)

| Part | Detail |
|---|---|
| CPUs | 4 × Xeon E5-4640 v1 (Sandy Bridge-EP), 32 cores / 64 threads, 2.4–2.8 GHz. **AVX only: no AVX2, F16C or FMA** |
| RAM | 48 × 16 GB dual-rank DDR3 RDIMM, rated 1600 but **running at 1333** (3 per channel). 768 GB total, **170.6 GB/s on paper** |
| NUMA | 4 nodes, ~190 GB each. Socket 0↔2 and 1↔3 are **two hops apart** (distance 30) |
| GPU | NVIDIA Quadro P2200 (Pascal GP106), 5 GB GDDR5X, ~200 GB/s |
| Disk | 131 GB on RAID-1 (PERC H710), too small for the model |
| Network | 4 × 1 GbE (BCM5720), 2 × 10 GbE SFP+ (Intel 82599) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0, host `r820` at 192.168.1.137 |

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

## First DeepSeek runs (2026-10-29)

Greedy 16-token completion of a fixed prompt, 60 threads, `--numa distribute`, model pages interleaved over the 4 nodes (stock placement, no NUMA patch active):

| Engine | Dense weights | Decode tok/s | Prompt tok/s (7 tokens) |
|---|---|---|---|
| mainline llama.cpp | BF16 (as downloaded) | 0.43 | 4.2 |
| mainline llama.cpp | Q8_0 (`requant_shard`) | 1.22 | 2.1 |
| **ik_llama.cpp** | Q8_0 | **2.31** | 5.3 |

- **Never run 64 threads with anything else busy.** With 64 threads on the 64 logical CPUs, one background job (the SHA256 check) made every run ~10× slower: 0.10–0.12 tok/s. perf showed 95% of the time spinning in OpenMP barriers. The engine syncs threads hundreds of times per token, so one descheduled thread stalls all of them. Use ≤60 threads, or keep the machine idle.
- BF16 dense weights cost mainline 2.8× (scalar BF16 code on AVX1), more than their 1.6× extra bytes. Q8_0 dense is the working format from here on.
- ik is 1.9× faster than mainline on the same file before any of our NUMA work.
- **Outputs differ from the first generated token** (ik ": Ganymede, Europa, Callisto…", mainline " Io, Europa, and Ganymede…"). Both are coherent, but it needs a perplexity comparison before trusting ik's numbers.
- Getting ik to load the file took two patches:
  - `mxfp8_patch.py`: MXFP8 type, also applied to mainline.
  - `ds41_gguf_patch.py`: ik expects a different Engram GGUF layout. Same rolling-xor hash and maths, different key names; one shared table instead of per-layer tables; key and query gate weights folded into one tensor. The shim maps the Unsloth/mainline layout onto ik's.

## Correctness, NUMA placement and the throttling problem (2026-10-29 evening)

- **ik is correct on DeepSeek:** wikitext-2 perplexity (6 × 256 tokens, Q8_0 dense) is ik 3.5081 and mainline 3.5073, within ±0.32.
- **Placement works:**
  - Two fixes in `numa_pin_patch.py`:
    - fault the range in (`MADV_POPULATE_READ`) before `mbind(MPOL_MF_MOVE)`, since the kernel only migrates pages a process has mapped;
    - pin each blocked thread to one CPU (`GGML_NUMA_SKIP=n` leaves n CPUs per node to the OS).
  - The tmpfs is remounted `huge=never`, and the expert and dense shards are split to 4 KiB pages (`splitpages.py`). An expert's row quarter is 1.6 MB, smaller than a 2 MB huge page.
  - 4 KiB pages read as fast as 2 MB ones (109 vs 110 GB/s).
  - The debugfs file-path split form returns EINVAL on these files; the pid/address form works.
  - Measured with `placecheck.py` (move_pages): every expert and dense row quarter is **99.7–100% on its node**. The page move takes 209 s once and persists in tmpfs.
- **Run-to-run speed swings (0.6–2.7 tok/s, same command) were the iDRAC power cap.** Under load the cores run at **430–1100 MHz effective** (turbostat APERF/MPERF):
  - P-state pinned at the minimum (ratio 12), power-limit status active;
  - RAPL throttled 50–80% of the time at only ~30 W per package, though MSR 0x610 says PL1 = 95 W, so the limit comes in over PECI;
  - system at ~320 W.
  - `ipmitool delloem powermonitor getpowerbudget` still shows **Power cap 433 W**, below the 545 W "minimum power". The earlier `ipmitool dcmi power deactivate` only cleared the DCMI side; DCMI now says no limit is set.
  - Short runs straight after idle escape the clamp for a few seconds (~530 W, 2.4 GHz, ~2.2–2.7 tok/s), which explains the bimodal results.
  - In-band changes are refused ("Power cap set feature is not enabled").
  - **Fixed via racadm (2026-10-30):** `System.Power.Cap.Enable` was already Disabled, yet `#Cap.ActivePolicy` (name "iDRAC") still enforced 433 W. The value can only be changed while the cap is enabled, so the fix was `racadm set System.Power.Cap.Enable Enabled` then `racadm set System.Power.Cap.Watts 936`. The active policy now reads 936 W (100% = `#Cap.MaxThreshold`), so it never binds. Under-load clocks are still to be verified from the OS.
  - **That was not the whole story.** With the iDRAC cap at 936 W, sustained load still ran at 0.9–1.2 GHz, ~310 W system, 0.60 tok/s. Intel Node Manager (`ipmitool nm`, NM 2.5, in-band) shows the real limiter:
    - platform **policy 13**: enabled, no trigger, **343 W**, 5 s correction, no T-states, non-volatile. It holds the system at ~335 W average (CPU domain ~143 W, memory ~108 W).
    - policy 1 is the iDRAC user cap (now 936 W); policy 4 is 100 W on a missing power reading.
    - Policy 13 refuses host-side changes (`nm control disable per_policy 13` and re-adding it as disabled both return "Policy ID Invalid"), so the iDRAC owns it.
    - NM has **no inlet temperature** ("Invalid Mode"), and the iDRAC shows Inlet Temp "No Reading". The inlet sensor is on the front control panel, unplugged since the VLT0204 fix.
    - **Most likely cause:** the iDRAC holds a safe power level because it can't see inlet temperature. Less likely: PSU1 (the Hot Spare primary) has no AC.
  - Also seen in `racadm get System.Power`: `HotSpare.Enable=Enabled` with `HotSpare.PrimaryPSU=PSU1` (the PSU with no AC), and `RedundancyPolicy=Not Redundant`. Left as is.
  - Not the cause: DIMM thermal throttling (~0 throttle cycles in the iMC counters), CPU temperature (≈60 °C), kernel threads, scheduler stacking.
- **The other misleading result:** a leftover remote `sha256sum` kept running after the local task was stopped, which made all 64-thread runs ~10× slower until it was killed.
- **Tool notes:**
  - `llama-bench` / `llama-sweep-bench` runs last long enough to always hit the clamp, which is why they read 0.6–0.9 tok/s.
  - All numbers so far are with the cap active; best unclamped readings: ik 2.2 tok/s (interleaved) and 2.71 (pinned + placed, llama-bench straight after the placement run).

## Q8_0 quality and GPU offload (2026-10-30)

- **Q8_0 dense weights are near-lossless.** ik perplexity (same 6 × 256 wikitext chunks): BF16 dense (as downloaded) 3.4907, Q8_0 dense 3.5081 (+0.5%, inside ±0.31).
- **Capping the CPU clock in-band does not escape the iDRAC cap.** With `scaling_max_freq` at 1.6/1.8/2.0 GHz, the busy clock still sank to 1.0–1.4 GHz and decode stayed at 0.6–0.8 tok/s. DRAM power is ~120–137 W of the budget by itself, and the memory looks throttled too.
- **GPU offload works with the experts kept on the file mapping:**
  - In ik's CUDA build `-ot exps=CPU` is silently turned into the pinned `CUDA_Host` type (`llama-load-tensors.cpp`), which copies all 289 GB of experts into anonymous memory → OOM kill at 285 GB.
  - `-ngl 0` with `=CUDA0` overrides doesn't help: ik ignores GPU overrides when no layers are offloaded.
  - Fix: new opt-in `LLAMA_OT_CPU_MMAP=1` keeps CPU overrides as the plain CPU type, so they stay mmapped (in `ik_llama/r820.patch`).
  - Working command: `LLAMA_OT_CPU_MMAP=1 llama-cli -ngl 12 -ot exps=CPU …`. That puts 12 layers' dense weights on the P2200 (2.1 GB `CUDA0` buffer, ~0.95 GB compute buffer); experts and Engram stay on the CPU, zero copies.
  - First reading, under the cap: 1.18 tok/s decode and 3.5 tok/s prompt, against ~0.7 CPU-only under the same throttle.
  - The GPU output is coherent but differs from the CPU text (different arithmetic for attention), so it needs the perplexity check.
  - **The perplexity check fails:** chunk 1 is **830.95** on the GPU configuration (`-ngl 12 -ot exps=CPU`, 256-token batches) against 1.4371 on the CPU. Batched evaluation on the offloaded layers is broken, which points at one of the DeepSeek-V4.1 ops on CUDA sm_61; single-token decode looked plausible. GPU numbers are not usable until this is found.
    - Suspect: DeepSeek-V4.1 attention has 512-wide heads (key/value length 512). Batched CUDA flash attention uses tile/MMA kernels, and Pascal has no tensor cores, so it takes a fallback that may not handle D=512; single-token decode uses the vector kernel. Test: perplexity with `-fa 0`, and with attention kept on the CPU.

## Model facts (measured from the GGUF, 2026-10-29)

Unsloth UD-Q8_K_XL: 474 GiB in 9 shards (SHA256 verified on the NAS), at `nas.local:/volume1/@home/richard/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-Q8_K_XL`. 40 layers, all MoE: 384 routed experts (6 active) plus 1 shared expert per layer. 4 hyper-connection streams (`embedding_length_out` 20480).

| Part | Format | Total | Read per token | Shards |
|---|---|---|---|---|
| Routed experts (gate/up/down, 5120×2304×384 each) | **MXFP4** (type 39, 4.25 bpw) | 288.8 GB | **4.51 GB** (6/384 of each layer) | 00002–00007 |
| Attention (q_a, q_b, kv, output_a/b, indexer, compressor) | **BF16** | 10.3 GB | 10.3 GB | 00009 |
| Shared experts | **BF16** | 2.8 GB | 2.8 GB | 00009 |
| Output head | BF16 | 1.3 GB | 1.3 GB | 00009 |
| Engram projections (`engram_kv`, 2 layers) | BF16 | 0.6 GB | 0.6 GB | 00009 |
| Router, hyper-connection weights, norms | F32 | 0.5 GB | 0.5 GB | 00009 |
| Engram table (`per_layer_token_embd`, 256 × 768M rows) | **MXFP8** (type 43, 8.25 bpw) | 202.8 GB | ~0 (row lookups) | 00008 |
| Token embedding | BF16 | 1.3 GB | ~0 (row lookup) | 00009 |

- **~20 GB read per token as downloaded**, of which 15.5 GB is BF16 dense weights. The earlier "~5–8 GB" estimate assumed those were Q8; they're BF16.
- **All the BF16 weights are in the last shard (00009, 16.8 GB).** Requantizing them only means rewriting that one file: Q8_0 makes it ~8.2 GB per token (~12.7 GB per token in all), Q4_K-class ~4.4 GB (~8.9 in all). The expert and Engram shards stay as they are.
- **The experts are already 4-bit.** Requantizing them to Q4_K would not shrink them.
- **MXFP8** (OCP microscaling: 32 E4M3 values with one E8M0 scale) is not in mainline llama.cpp or ik. It comes from the unmerged llama.cpp PR #30817, which Unsloth's guide builds from. Ported as a gather-only type with `ik_llama/mxfp8_patch.py` (to_float + GET_ROWS), and applied to both trees. The decoder is verified exact for all 256 codes; sampled Engram rows decode cleanly (RMS 0.88, max 5, no NaN).
- **DSpark:** the metadata describes it (block size 5, 128 experts, 3 active), but the drafter weights are not in this download.

Decode speed ≈ usable bandwidth ÷ bytes per token. **The BF16 dense weights are now the biggest lever**: 15.5 of the 20 GB per token.

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
| Unsloth GGUF | Engram table is **MXFP8 (type 43)**, from unmerged llama.cpp PR #30817; mainline and ik reject the file | **Resolved:** own gather-only MXFP8 port (`mxfp8_patch.py`) for both trees |
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
5. **Expected as downloaded (~20 GB/token): ~1.5 tok/s stock interleaved, ~4 NUMA-local.**

### Phase 2: requantize the BF16 dense weights (was "Q4")
1. Rewrite only shard 00009: BF16 → **Q8_0** first (near-lossless), then try Q6_K / Q5_K / Q4_K for attention and shared experts. Experts (MXFP4) and the Engram table (MXFP8) stay as they are. Only ~17 GB needs rewriting, so it fits in RAM next to the model.
2. Repeat the Phase 1 tuning.
3. GPU: with the dense part at ~4.5 bits (~4.4 GB) it may fit on the 5 GB P2200, leaving only the experts for the CPU.
4. Quality check against the BF16 original: perplexity plus your own prompts.
5. RAM test (2 DIMMs per channel at 1600) is off the table: 512 GB is too small for the 474 GiB model.
6. **Expected: ~6.5 tok/s with Q8_0 dense on the CPU; ~12 with ~4.5-bit dense on the GPU and experts NUMA-local on the CPU.**

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

## Targets (no purchases), revised for the measured model

| | As downloaded (BF16 dense, ~20 GB/token) | Dense → Q8_0 (~12.7 GB/token) | Dense ~4.5-bit on the GPU, experts on the CPU |
|---|---|---|---|
| Stock llama.cpp (interleaved) | ~1.5 | ~2.5 | – |
| NUMA-local engine, CPU only | ~4 | ~6.5 | – |
| + GPU | – | – | ~12 |
| + speculative decoding (DSpark, ×1.3–1.5) | – | ~8–10 | **~16–19** |

To be replaced by measurements.

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
