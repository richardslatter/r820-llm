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
- **Network:** avahi, so the server is `r820.local`. netplan uses `dhcp-identifier: mac`, which took effect at the 2026-10-30 boot: the server is now **192.168.1.137**.
  - `net-reconnect.service` (`system/net/`): booted without a cable (it's often on the iDRAC port)? It re-runs DHCP once a minute on any wired NIC with carrier until there's a default route, then exits.
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
  - **Update (2026-10-30):** after pulling the dead PSU1 and an AC cycle, policy 13 reads **740 W** (was 343 W). Read over IPMI-over-LAN by the owner.
    - The trigger was most likely the PSU state (PSU1 installed with no AC and set as the Hot Spare primary). The inlet sensor is still missing.
    - `nm control disable global` is blocked by Dell: "Command subfunction disabled or unavailable (d6)".
    - Hot Spare was then disabled (`racadm set System.Power.HotSpare.Enable Disabled`).
    - **Fixed, verified 2026-10-30 00:18.** Sequence: AC cycle, dead PSU1 pulled, Hot Spare disabled, iDRAC user cap disabled, then a reboot.
      - Node Manager now shows policies 1, 4 and 13 all "not enabled", and nothing is limiting in any domain.
      - Under all-socket memory load: busy 97.6%, **2500 MHz**, CPU packages 292 W, DRAM 223 W (clamped: 400–1100 MHz, ~120 W, ~120 W).
    - The iDRAC user cap was then disabled entirely (`Cap.Enable=Disabled`). `#Cap.ActivePolicy` still shows the last value (now 936 W, the maximum), so it's harmless even if it is still enforced.
    - Measured full CPU load is ~630 W, so 740 W should no longer bind. It may clip with the GPU busy (+~75 W). Still to be verified under load from the OS.
    - Front panel test: with the control-panel cable connected, the SEL logs "system board 5V PG voltage is outside of range" and "fail-safe voltage is outside of range", and Inlet Temp stays Unknown. So the panel loads the 5 V rail, and its sensor isn't seen either.
  - Also seen in `racadm get System.Power`: `HotSpare.Enable=Enabled` with `HotSpare.PrimaryPSU=PSU1` (the PSU with no AC), and `RedundancyPolicy=Not Redundant`. Left as is.
  - Not the cause: DIMM thermal throttling (~0 throttle cycles in the iMC counters), CPU temperature (≈60 °C), kernel threads, scheduler stacking.
- **The other misleading result:** a leftover remote `sha256sum` kept running after the local task was stopped, which made all 64-thread runs ~10× slower until it was killed.
- **Tool notes:**
  - `llama-bench` / `llama-sweep-bench` runs last long enough to always hit the clamp, which is why they read 0.6–0.9 tok/s.
  - All numbers so far are with the cap active; best unclamped readings: ik 2.2 tok/s (interleaved) and 2.71 (pinned + placed, llama-bench straight after the placement run).

## First unclamped benchmarks (2026-10-30, power clamp fixed)

llama-bench decode, Q8_0 dense weights (~12.7 GB/token), tmpfs on 4 KiB pages (`huge=never`), cores at 2.5 GHz:

| Config | Decode tok/s |
|---|---|
| mainline, stock, 64 threads | 1.58 |
| ik, stock, 64 threads (pages interleaved) | 2.60 |
| ik, `GGML_NUMA_BLOCKED=1` (per-CPU pinning), interleaved | 2.66 |
| **ik, pinned + `GGML_NUMA_PLACE=1`** (99.8–100% of every row quarter on its node) | **3.27** (tg32), 3.10 (tg64) |
| ik, pinned + placed, 48 threads | 2.69 |
| ik, pinned + placed, 32 threads (one per physical core) | 2.94 |

- Prompt processing, pinned + placed, 64 threads: 8.2 tok/s (pp64).
- NUMA-local placement: +26% over stock ik. ik: 2× mainline.
- **3.27 tok/s × 12.7 GB ≈ 42 GB/s, against ~110 GB/s local bandwidth.** Decode is no longer memory-bound; the kernels (SIMDe-emulated AVX2 on AVX1) and per-op synchronisation are the bottleneck now. That is the Phase 3 custom-kernel work, which needs profiling first.

## GPU offload speed (2026-10-30, after the GPU fixes)

64 threads, `GGML_NUMA_BLOCKED=1`, NUMA-placed pages, `LLAMA_OT_CPU_MMAP=1 -ot exps=CPU` (all experts stay mmapped on the CPU):

| Config | Prompt pp64 tok/s | Decode tg32 tok/s |
|---|---|---|
| CPU only (build-cpu) | 8.02 | 3.14 |
| `-ngl 12` (2.1 GB dense on the P2200) | 7.51 | 3.33 |
| **`-ngl 20`** | 7.79 | **3.54** |

- The GPU gives +13% decode at 20 layers, and prompt speed is unchanged.
- Decode is bound by CPU kernel compute and per-op synchronisation, not by bytes read: ~42 GB/s out of ~110. Moving dense weights off the CPU therefore only helps a little, and each offloaded layer adds CPU↔GPU syncs.
- **Phase 0 is done.** Everything is measured and correct on both paths. The next gains are Phase 3: profile and hand-tune the AVX1 (SIMDe) MXFP4/Q8_0 kernels and cut per-op barriers.

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
    - **Actual cause, found by tensor-by-tensor checksum comparison** (`llama-eval-callback`, CPU build vs CUDA build with one layer offloaded): CUDA's **in-place** flipped `ROPE_BACK`, which DeepSeek-V4.1 uses to de-rotate the attention output.
      - Before the fix, the first divergence was `attn_wo_a` (7%), then `ffn_norm` 5.2 instead of 17.4.
      - Making that rope out of place (`ds41_rope_back_patch.py`) brings the GPU within ~1% of the CPU.
      - **GPU perplexity is now 3.5047** (CPU 3.5081, mainline 3.5073).
      - The sink fix and a contiguous-input fix (`fattn_sinks_patch.py`, `ds41_wo_a_cont_patch.py`) are kept. They changed nothing measurable here: identical perplexity with and without, and identical with `-nkvo`.
    - Earlier hypothesis (kept for the record), **from reading ik's CUDA source at the base commit:** on Pascal, DeepSeek-V4.1 attention (K = V = 512, sparse indexer in `src[5]`) runs down `ggml_cuda_flash_attn_ext` like this:
      1. the sparse MMA kernel needs tensor cores, so it fails;
      2. `ggml_cuda_dsa_attn_ext` (handles sinks) declines short contexts: KV length ≤ top-k 512, or < 2048 for batches ≤ 16;
      3. `is_pascal_hs512` → `ggml_cuda_flash_attn_ext_tile_new`, the Gemma-4 kernel, which **has no attention-sink support** (never reads `src[4]`).
      So every short-context GPU attention drops the per-head `attn_sinks`, decode included; the coherent decode text was luck. Sparse indices are also only turned into a mask when no mask exists.
    - Workaround to test: `-nkvo`, keeping the KV cache and the attention op on the CPU and only the projections on the GPU.
    - Proper fix (custom kernel, `ik_llama/fattn_sinks_patch.py`, written but not yet compiled or tested): sinks in `tile_new`'s online softmax. Seed each head's running max with its sink logit and the running sum with 1 (exp(sink − max) after rescaling).

## Phase 3e: attention, small ops and the dense format (2026-10-30)

All decode figures: llama-bench, 64 threads, `GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1 --numa distribute`, tg32 and tg32 after a 512-token prompt.

**fp16 without F16C** (`ggml_f16_avx1_patch.py`, `iqk_avx1_f16c.h`):
- SIMDe and ggml's own fallback converted fp16 one value at a time. Every K dot product and V update of the fp16 KV cache went through it, ~12 ms of a 170 ms decode step.
- The vector versions are exact: checked on all 65536 halves and all 2^32 floats (`tools/f16c_test.cpp`).

**Decode flash attention for one shared K/V head** (`ds4_fa_mqa_patch.py`, `ds4_fa_mqa_kernel.c`):
- 64 query heads over one 512-wide fp16 K/V head. Threads take head groups × chunks of the visible rows; each row is converted once and scored against all heads of the group; f32 online softmax; chunk states merged after a barrier.
- 15% faster than the generic loop at 512 tokens of context, and better: the generic loop accumulated V in fp16. Perplexity 3.4919 against 3.5485 (BF16 dense reference 3.4907), greedy text identical.
- **Lazy K/V concat.** The graph builds K = V as concat(raw cache rows, compressed rows) every token, ~1 MB copied per layer at 512 tokens. When the attention node is the concat's only reader, the concat is skipped and the kernel reads the rows from its sources:
  - CSA layers: both sources are cache views. ~1.8 ms per token saved.
  - HCA layers: the compressed rows are one `get_rows` shared by a group of four layers. They are read in place when no node between the `get_rows` and the attention node writes into its bytes (checked at plan time). 20 × 82 µs → ~0 per token.
  - A first attempt redirected the `get_rows` into the concat's buffer; that cannot work because one `get_rows` feeds four concats.

**Small ops on one thread** (`decode_ops_patch.py`, `norm_split_patch.py`):
- RMS norms of single 20480-float rows (the four hyper-connection streams, twice per layer): vectorised sum of squares, then split by columns across all threads with a barrier between the partial sums and the scaling. 44.5 → 11.7 µs per norm.
- The partial sums are laid out [row][thread], eight threads to a cache line. One padded line per thread was **4× slower** (42 µs): every reader then pulls 64 lines just written on other sockets, at ~0.5 µs each.
- `MUL_MULTI_ADD` split by token, so at batch 1 one thread did all of it (121 nodes per token, ~28 µs each); now split by columns.
- **Negative result, split-K for the hc_pre matmul** (24 × 20480 f32, 80 per token, ~50 µs each): per-node partial sums with the last thread to arrive reducing. Slower in the model (~74 µs against 49–64 µs) and in isolation (`tools/hc_bench.c`). Cross-socket line traffic and contended atomics cost more than the split saves. An empty graph launch alone costs ~54 µs warm and ~185 µs with cold caches.

**Dense format.** The dense weights (attention, shared experts, hc, Engram projections) were requantised from BF16 with `requant_shard`; experts stay MXFP4. KL divergence against BF16 dense, 16 × 256 wikitext tokens, ub 8:

| Dense weights | Decode tok/s | after 512 tokens | Mean KLD | Same top token |
|---|---|---|---|---|
| Q8_0 (`dq8`) | 5.94 | 5.38 | 0.0139 | 97.0% |
| **Q6_K (`dq6k`)** | 6.52 | 5.84 | 0.0178 | 96.8% |
| IQ4_NL (`diq4`) | 7.30 | 6.50 | 0.0818 | 92.8% |

- **Pick: Q6_K.** It is 10% faster than Q8_0 at a small quality cost. IQ4_NL is another 12% faster, but its KLD is 6× higher. `tools/serve.sh` defaults to `dq6k`.
- With the attention, norm and small-op changes, dq6k decodes at **6.68 tok/s** (6.17 after a 512-token prompt). With `GGML_FA_LAZY=0`: 6.62 / 6.06.
- The lazy concat is exact: KLD per chunk identical with it on and off (one-token batches), and 64 greedy tokens after a 694-token prompt (HCA gather active) byte-identical.

Still open in Phase 3e:
- ~4 ms per token in the hc_pre matmuls (fixed cost per small node).
- ~2 ms in ROPE.
- The q6_K small matmuls (`qr` 83 µs, `kv_b` 56 µs).

## Where the decode time goes, and kernel speed on AVX1 (2026-10-30)

**Per-token breakdown** at 512 tokens of context, dq6k, 161.5 ms per token:
- The node loop takes 153.9 ms.
- Graph build 2.3 ms, scheduler allocation 3.0 ms, the rest of the host side ~2 ms.
- Graph reuse never happens for DeepSeek-V4: the model-state hash includes the compression-plan sizes and n_kv, which change every token. Reuse would need dynamic shapes, so it isn't worth chasing for ~7 ms.

**Matmuls in the model** (op profile, per token):

| Weights | ms | GB/s |
|---|---|---|
| Experts up+gate (MXFP4) | 31.4 | 96 |
| Experts down (MXFP4) | 15.8 | 95 |
| `attn_output_b`, `attn_q_b` (Q6_K) | 14.3, 13.5 | 96, 102 |
| `attn_output_a` (Q6_K) | 13.8 | 80 |
| `output` (Q6_K) | 5.0 | 108 |
| `ffn_down_shexp` (Q6_K) | 4.9 | 78 |
| `ffn_gate_inp` (f32) | 4.4 | 71 |
| `attn_q_a` (Q6_K) | 3.9 | 55 |
| `attn_kv` (Q6_K, 5120 × 512) | 3.1 | 28 |
| `hc_attn_fn` + `hc_ffn_fn` (f32, 20480 × 24) | 4.8 | 32 |

The big matmuls are near the ~110 GB/s the memory gives. The small ones lose ~12 ms between them.

**Kernel speed by type** (`tools/mm_bench.c`). One thread, weights in L2 (4096 × 32), so this is the arithmetic alone. Native means a hand-written AVX1 kernel (`iqk_avx1_gemv.h`); everything else goes through SIMDe's AVX2 emulation.

| Type | GB/s per core | Kernel |
|---|---|---|
| Q8_0 | 5.0 | native |
| Q6_K | 2.9 | native |
| MXFP4 | 2.7 | native |
| IQ4_NL | 2.4 | native |
| Q5_K, Q4_K, Q4_0 | 1.8–1.9 | SIMDe |
| IQ4_XS, IQ4_KS, IQ4_K | 1.3 | SIMDe |
| IQ5_KS, IQ5_K, Q6_0 | 1.1 | SIMDe |
| IQ6_K | 0.7 | SIMDe |

- Filling 110 GB/s from 32 cores takes ~3.4 GB/s per core (a little less with both hyper-threads).
- So Q6_K and MXFP4 are already at the compute limit, and every SIMDe type would be compute-bound well below the memory speed.
- A lower-bit dense format only pays off with a native kernel at least as fast as IQ4_NL's. That is a precondition for the mixed dense quant idea below.
- The row-interleaved `_r4`/`_r8` layouts are slower here too: no native kernels.
- At 64 threads with the weights streaming from DRAM, Q6_K and Q8_0 move about the same bytes per second (74 vs 71.5 GB/s). Q6_K wins on time because it is smaller.

**Correction: mm_bench and huge pages.** Transparent huge pages are on (`always`) for anonymous memory, and mm_bench's weights were anonymous. `ggml_numa_place_tensor` binds each quarter of a tensor to its node, but a quarter that only covers part of a 2 MB page cannot be moved. `mbind` still returns success, so the tool reported 100% placed while most of a small tensor stayed on the node that wrote it. mm_bench now maps its buffer with `MADV_NOHUGEPAGE`, as the model's tmpfs pages are 4 KB anyway. Before and after, 64 threads:

| Matmul | Huge pages | 4 KB pages |
|---|---|---|
| f32 20480 × 24 (`hc_*_fn`) | 80 µs, 25 GB/s | 25 µs, 77 GB/s |
| f32 5120 × 384 (`ffn_gate_inp`) | 328 µs, 24 GB/s | 83 µs, 95 GB/s |
| Q8_0 5120 × 384 | 52 µs, 40 GB/s | 21 µs, 102 GB/s |
| Q6_K 5120 × 384 | 30 µs, 54 GB/s | 17 µs, 94 GB/s |

This replaces an earlier note here that small Q6_K matmuls were compute-bound (53 µs for 5120 × 512); that was the same artefact. With 4 KB pages, 5120 × 512 takes 22 µs (97 GB/s) streaming, and Q8_0 27 µs (104 GB/s). In a graph of only 4 matmuls, the graph launch makes it ~47 µs.

So small matmuls reach the memory speed when run alone. In the model they don't: `attn_kv` (5120 × 512) takes 77 µs per call and `hc_*_fn` ~60 µs, against 22 and 25 µs here, about 12 ms per token in all. The weights are placed, so the suspect was the input, which threads on all four sockets have just written.

**Negative result, node-local activation copy** (`mm_f32_decode_patch.py`). For f32 matmuls with one input row, the threads of each node first copied the row into a per-node buffer and then computed from that copy. Where there were fewer rows than threads, each row was also split along K. In mm_bench, the hc shape got slower (25 → 40 µs) and `gate_inp` was unchanged. In the model (two alternating runs each, tg32 / tg32@pp512):
- Off: 6.74 / 6.18 and 6.59 / 6.10 tok/s.
- On: 6.39 / 5.89 and 6.60 / 6.13 tok/s.

So there was no gain, and it is not applied. Together with the per-thread quantisation result below, this suggests the in-model gap for small matmuls is not activation traffic. Load imbalance carried in from the previous op is the next suspect, but it is unmeasured: no `perf` on this box.

An earlier split-K-only version, without the node-local copy, was also slower.

**Negative result, per-thread activation quantisation** (`mm_localq_patch.py`). Each thread converted the single activation row itself instead of thread 0 converting it and everyone waiting at a barrier. 6.07 against 6.05 tok/s: no gain. It is not applied.

**Lesson:** killing a benchmark binary over ssh does not stop the shell loop that started it. A stray 64-thread job made a later decode run 40× slower (0.16 tok/s). Kill the loop's shell too, and check `top` before each benchmark.

## Smart expert reduction (`-ser`), measured (2026-10-30)

In this tree `-ser` did nothing: the call was commented out. `ik_llama/ser_patch.py` turns it back on, with three changes:
- a top-k selection in place of the bubble sort;
- the threshold applied to the unbiased weights;
- zero-weight slots skipped when the expert outputs are summed.

It is faster, but the quality cost is too high, so it stays opt-in. dq6k itself has KLD 0.018 against Q8.

| `-ser` | Speed | KLD |
|---|---|---|
| 5,1 | +5% | 0.066 |
| 4,1 | +9% | 0.151 |

## Speculative decoding without a draft model (2026-10-30)

The GGUF has no MTP/nextn tensors, so only draft-free methods are possible. The question is the cost of verifying a batch: each extra token in the batch brings its own experts.

Verify cost (`llama-bench -p N`, dq6k, 64 threads):

| Tokens in batch | 1 | 2 | 3 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| ms per batch | 144 | 225 | 289 | 345 | 462 | 561 | 1088 |
| × one token | 1.0 | 1.56 | 2.0 | 2.4 | 3.2 | 3.9 | 7.5 |

So a draft must be accepted most of the time to pay off, and long drafts never do.

`llama-spec-bench`, built-in tasks, 192 tokens, greedy, tok/s (one run each, noise about ±0.2):

| Setting | code | extract | story |
|---|---|---|---|
| No drafting | 6.44 | 6.08 | 6.42 |
| `suffix:n_max=2,n_min=1` | 6.75 | 6.77 | 6.47 |
| `suffix:n_max=3,n_min=1` | 6.64 | 6.72 | 6.50 |
| `suffix:n_max=4,n_min=2` | 6.87 | 6.59 | 6.30 |
| `suffix:n_max=8,n_min=2` | 6.05 | 4.76 | 6.46 |
| `ngram-mod:n_max=3,n_min=1` | 6.36 | 5.95 | 6.24 |
| `ngram-mod:n_max=8,n_min=2` | 6.59 | 5.46 | 6.56 |

- **Suffix drafting with short drafts is a small, free win.** It gives +5–11% where the output repeats text from the prompt (code, extraction) and nothing on free prose, where it never drafts.
- `n_max=2` was the most even: 96% first-token acceptance on extract.
- Long drafts lose even at 93% first-token acceptance, as the verify table predicts.
- `ngram-mod` drafts rarely and accepts less.
- The gain should be larger in multi-turn chats and code edits that repeat earlier text.

Recommended server flag: `--spec-type suffix:n_max=2,n_min=1`.

Note: `--jinja` with the GGUF's own template fails in ik's jinja engine ("Callee is not a function"). The server works around this with `--chat-template-file`.

## Ideas for cutting the per-token weight read (not yet tried, 2026-10-30)

At dq6k each token reads ~10.9 GB: experts 4.5, dense 6.1, f32 router/hc 0.3. That is ~98 ms at 110 GB/s. None of the following has been run yet. They are listed best first.

1. **2 DIMMs per channel at 1600 MT/s** (+20% bandwidth, ~98 → ~82 ms).
   - The DIMMs are rated 1600; 3 per channel caps them at 1333.
   - This was ruled out in Phase 2 because 512 GiB was too small for the Q8 download. dq6k needs only ~465 GiB (experts 269 + Engram table 189 + dense 6), which leaves ~40 GiB if the tmpfs holds only dq6k.
   - Optional: quantise the Engram table (MXFP8, lookup only) to 4-bit for ~100 GB more headroom, at no speed cost.
   - You'd need to pull 16 sticks, one per channel. Then check that the BIOS reports 1600 at 2 per channel, and refill from the NAS (~72 min).
   - Biggest unknown: whether the R820 actually runs 1600 at 2 per channel.
2. **ik's smart expert reduction (`-ser`).**
   - 5 of 6 experts saves ~0.75 GB, ~7 ms per token.
   - A threshold variant saves less but is safer.
   - Needs only a KLD check.
3. **Imatrix-guided mixed dense quant** (IQ5_KS/IQ4_KS, ~5 bpw).
   - `diq4` was made without an imatrix, which is likely why its KLD is 0.082 against 0.018.
   - Target: dq6k-level KLD with dense 6.1 → ~4.7 GB, ~13 ms per token.
   - Cost: an imatrix run, and possibly AVX1 kernels for those types.
4. **Shared expert on the P2200** running alongside the CPU routed experts. Hides ~1.15 GB, ~10 ms per token; needs the engine to run GPU and CPU splits in parallel.
5. **Speculative decoding with no download.**
   - Prompt-lookup (n-gram) drafting.
   - Check whether the GGUF has MTP/nextn tensors.
   - The DSpark drafter isn't in the download, so it needs your permission first.
   - Dense reads amortise across drafted tokens; expert reads mostly don't.
6. **Smaller items:**
   - Output head at 4-bit: ~2–3 ms.
   - Experts below native FP4: ~1.2 GB, but likely a real quality loss; low priority.

Ruled out:
- 2 MB vs 4 KiB pages: measured equal.
- BIOS settings: prefetchers, interleave, profile and turbo are already right.
- Non-temporal loads: no effect on write-back memory.
- Compression: AVX1 is compute-bound.
- Caching hot experts: ~75 MB per layer against 16 MB of L3.

Estimate for 1 + 2 + 3 together: ~8.6 GB at ~132 GB/s, ~72 ms read + ~23 ms other, so ~10.5 tok/s CPU-only against ~7.1. Speculative decoding on top could give ~13–15.

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
5. RAM test (2 DIMMs per channel at 1600) is off the table: 512 GB is too small for the 474 GiB model. *Reopened:* dq6k needs only ~465 GiB (see "Ideas for cutting the per-token weight read").
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
