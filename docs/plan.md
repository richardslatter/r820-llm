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
    - Proper fix (custom kernel, `ik_llama/parts/fattn_sinks_patch.py`, now applied, see above): sinks in `tile_new`'s online softmax. Seed each head's running max with its sink logit and the running sum with 1 (exp(sink − max) after rescaling).

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

In this tree `-ser` did nothing: the call was commented out. `ik_llama/parts/ser_patch.py` turns it back on, with three changes:
- a top-k selection in place of the bubble sort;
- the threshold applied to the unbiased weights;
- zero-weight slots skipped when the expert outputs are summed.

It is faster, but the quality cost is too high, so it stays opt-in. dq6k itself has KLD 0.018 against Q8.

| `-ser` | Speed | KLD |
|---|---|---|
| 5,1 | +5% | 0.066 |
| 4,1 | +9% | 0.151 |

## Speculative decoding without a draft model (2026-10-30)

The main GGUF has no MTP/nextn tensors. V4.1's predictor is a separate DSpark draft (next section); this section covers the draft-free methods. The question is the cost of verifying a batch: each extra token in the batch brings its own experts.

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

## DSpark draft model (2026-10-30)

V4.1's "MTP" is a separate 3-block DSpark draft. It reads the target's hidden states from layers 37–39 and borrows the target's embedding and output head. It drafts a block of up to 5 tokens in one pass.

Unsloth ships it as `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (8.0 GB, arch `dflash`, experts MXFP4). ik's tensor names match, but ik reads the DSA indexer keys for every DeepSeek-V4 stage and the file has none, so loading fails with "key not found in model: dflash.attention.indexer.head_count".

The draft's stages are uncompressed, so the keys are never used. Adding them with the target's values makes it load:

```
tools/gguf_set_kv.py dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf dspark-ik.gguf \
    dflash.attention.indexer.head_count=u32:32 dflash.attention.indexer.key_length=u32:128 \
    dflash.attention.indexer.top_k=u32:512
```

Usage: `-md dspark-ik.gguf --spec-type dspark:n_max=N`. The draft's experts add ~8 GB of RAM.

`llama-spec-bench`, dq6k, 64 threads, 192 tokens, greedy, tok/s (accepted fraction of drafted tokens in brackets):

| Setting | code | extract | story |
|---|---|---|---|
| No drafting | 6.31 | 6.20 | 6.54 |
| `dspark:n_max=1` | 7.55 (98%) | 6.83 (93%) | 6.21 (60%) |
| `dspark:n_max=2` | 7.99 (95%) | 7.58 (93%) | 6.02 (56%) |
| `dspark:n_max=3` | 9.04 (94%) | 8.02 (89%) | 6.25 (59%) |
| `dspark:n_max=4` | 10.03 (93%) | 8.39 (90%) | 5.69 (51%) |

- **Code and extraction: +27–59%.** Acceptance stays around 90% at every draft position, so longer drafts keep paying even though verifying 5 tokens costs 2.8× one token.
- **Free prose: no gain.** Acceptance is ~60% for the first token and falls with position, and n_max=4 is 13% slower than no drafting.
- A round costs the verify batch plus ~60–90 ms of drafting. The draft is 3 MoE blocks plus the shared 129k-row output head.
- Combining with suffix drafting is refused ("two-stage speculative mode currently requires a self-spec stage first").
- The code task stops at end-of-text after 90 tokens in some runs and runs to 192 in others. Batched verification changes the rounding, which is enough to flip a greedy choice late in the answer.
- **`p_min` does not rescue prose.** The draft is confident even where it is wrong, so the cutoff rarely shortens a draft:

  | Setting | code | extract | story |
  |---|---|---|---|
  | `n_max=4,p_min=0.5` | 10.08 | 8.18 | 5.60 |
  | `n_max=4,p_min=0.7` | 9.99 | 8.34 | 5.68 |
  | `n_max=5,p_min=0.7` | 9.86 | 8.09 | 5.68 |
  | `n_max=3,p_min=0.5` | 9.17 | 8.05 | 6.18 |

Recommended: `-md dspark-ik.gguf --spec-type dspark:n_max=3`. It gives +27–43% on code and extraction and costs ~4% on free prose. Use `n_max=4` for code-only work.

## Context length, threads and prompt processing (2026-10-30)

dq6k, build-dev, 64 threads unless noted:

| Test | tok/s |
|---|---|
| Decode, 32 / 48 / 64 threads | 6.27 / 6.26 / 6.54 |
| Decode after 0 / 4K / 8K / 16K prompt tokens | 6.69 / 5.95 / 5.86 / 5.78 |
| Prompt, pp512 | 10.8 |

- **Long context costs little:** −11% at 4K, −14% at 16K. The 128-token sliding window and the compressed/sparse layers keep attention small.
- **64 threads stays best.** Hyper-threads still help by 4%.
- **Prompt processing is compute-bound, not memory-bound.** One pp512 graph takes 24.0 s:

  | Part | Time | Rate |
  |---|---|---|
  | Dense Q6_K matmuls | 8.7 s | ~0.87 TFLOP/s |
  | Routed experts, MXFP4 (up/gate 7.2 s, down 3.3 s) | 10.5 s | ~0.83 TFLOP/s |
  | Flash attention | 2.2 s | |
  | Shared expert | 1.5 s | |

  Peak AVX1 float throughput for 32 cores at 2.4–2.8 GHz is ~1.2–1.4 TFLOP/s, so the batched kernels (SIMDe path) already run at 60–70% of peak. A faster CPU kernel can add at most ~1.4×. A 4K-token prompt takes ~6 minutes. The GPU is the only large lever for prompts (next section).

  *Correction (2026-10-31):* the profiler averaged a short warm-up with the pp512 graph, so these times are half the real ones and the kernels ran at ~30–35% of peak. Native batched kernels took pp512 to 23.9 tok/s on the CPU alone (see "Batched AVX1 kernels for prompts and the verify batch").

## GPU retest with the current kernels (2026-10-30)

build-cuda (all current patches, sm_61), dq6k, 64 threads, `LLAMA_OT_CPU_MMAP=1 -ot exps=CPU`:

| `-ngl` | Prompt pp512 tok/s | Decode tg32 tok/s |
|---|---|---|
| CPU build | 10.8 | 6.69 |
| 0 | 15.2 | 6.38 |
| 20 | 16.1 | 5.59 |
| 28 | 16.5 | 5.44 |
| 34 | context creation fails (5 GB) | |

- **Offloading layers now slows decode.** Decode used to be +13% at `-ngl 20`, but the CPU kernels have since doubled in speed. The CPU↔GPU syncs per offloaded layer now cost more than the dense weights the GPU takes over.
- **The GPU still helps prompts, with no layers on it.** At `-ngl 0` ik copies the weights of large-batch ops to the GPU (`offload-batch-size`, 32 rows), so dense matmuls and attention run there.
  - The routed experts stay on the CPU: they only offload when batch × 6 ≥ 32 × 384.
  - That is correct here, since a layer's experts are ~7 GB over PCIe 3.
  - pp2048: CPU 10.1, GPU with `-ub` 512 / 1024 / 2048 (`offload-batch-size=512`): 14.8 / 14.9 / 14.7. The batch size makes no difference.
- **Server choice:** build-cuda with `-ngl 0` gives prompts +40–47% for −2–5% decode. Plain CPU is better for short-prompt, decode-heavy use.
- **Power:** peak 672 W at the wall with the CPUs and GPU loaded (GPU ≤ 38 W), under the 740 W policy. Cores held 2.5 GHz.

## Server setups compared (2026-10-30)

The LAN server has been running `diq4` (IQ4_NL dense, no imatrix) on build-cpu. Measured against the current build and drafting:

| Build, model | tg32 | tg32 after 512 prompt tokens |
|---|---|---|
| build-cpu, diq4 | 7.25 | 6.49 |
| build-dev, diq4 | 7.34 | 6.71 |

`llama-spec-bench`, diq4, build-dev, tok/s (acceptance in brackets):

| Setting | code | extract | story |
|---|---|---|---|
| No drafting | 7.26 | 6.76 | 7.13 |
| `dspark:n_max=3` | 7.84 (82%) | 8.01 (88%) | 6.53 (62%) |

- build-cpu is only 1–3% behind build-dev on decode. Rebuilding the server is not worth a restart on its own.
- The draft accepts less against diq4 (82% on code against 94% at dq6k). The 4-bit target disagrees with the draft more often, so dspark gains less there.
- **Best overall: dq6k + `dspark:n_max=3`.** Code 9.0, extraction 8.0, prose 6.3 tok/s, at KLD 0.018 instead of 0.082. diq4 alone is only faster on free prose (7.1).
- `tools/serve.sh` takes `DRAFT=/mnt/models/dspark/dspark-ik.gguf` (and `NMAX`) to add the draft.

## Imatrix 4-bit dense and a 4-bit output head (2026-10-30)

`requant_shard --imatrix` with Unsloth's `imatrix_unsloth.gguf` (558 chunks of 4096 tokens). It covers 349 of the 350 dense BF16 tensors plus the output head; token_embd has no entry. attn_output_a's imatrix holds 8 slices, one per group of 1024 rows.

KLD against the BF16-dense model (16 × 256 tokens of wikitext-2), decode on build-dev (2 runs):

| Variant | Dense | Head | KLD | Same top token | tg32 | tg32 after 512 |
|---|---|---|---|---|---|---|
| dq6k | Q6_K | Q6_K | 0.018 | 96.8% | 6.69 | 6.17 |
| diq4 | IQ4_NL | Q6_K | 0.082 ± 0.006 | 92.8% | 7.40 | 6.74 |
| diq4i | IQ4_NL + imatrix | Q6_K | 0.078 ± 0.006 | 93.1% | 7.25 | 6.73 |
| diq4io | IQ4_NL + imatrix | IQ4_NL + imatrix | 0.080 ± 0.006 | 93.0% | 7.35 | 6.68 |

- **The imatrix barely helps:** −5%, inside the error bars. The loss comes from 4-bit storage of the dense attention weights, not from unweighted rounding. Idea 3's guess (imatrix as the reason for 0.082) was wrong.
- **The 4-bit head costs nothing measurable in quality, but gains nothing measurable in speed.** It saves 0.17 GB per token (~1.5 ms), below the run-to-run noise.
- Reaching dq6k quality at ~4.7 GB would need a 5-bit mix (IQ5_KS / Q5_K on the attention weights) and AVX1 kernels for those types. At most ~10 ms per token (~6%), so it stays low priority.
- dq6k stays the pick. Each variant was a 40 s rewrite of shard 9 next to hard links of shards 1–8.

## Shared expert on the GPU (2026-10-31)

The shared expert (0.15 GB of Q6_K per token) is the one dense FFN piece small enough to keep on the P2200 while every layer stays on the CPU: `-ngl 0 -ot shexp=CUDA0`, build-cuda, dq6k.

- ik dropped all `-ot` overrides at `-ngl 0`. The patch now keeps them when a GPU is present (`src/llama.cpp`), so single tensors can go to the GPU with no layers offloaded.
- ggml's scheduler copies a split's inputs before running any of it, so the CPU waits for the GPU's result before starting the routed experts. `ik_llama/experiments/sched_defer_patch.py` (not applied) runs a CPU split up to the first node that needs a device result, then copies it in: the routed experts and the GPU's shared expert run side by side.

| Setting | tg32 | tg32 after 512 |
|---|---|---|
| All on the CPU | 6.70 | 6.03 |
| Shared expert on the GPU | 5.83 | 5.42 |
| Shared expert on the GPU + deferred copy | 6.10 | 5.45 |

- **Slower in every form.** Each CPU↔GPU hand-off inside ggml costs ~100–165 µs (set/get on `cudaStreamPerThread` plus a sync), against 15–25 µs for the same copies in a standalone CUDA test. Two hand-offs per layer add ~18 ms per token. The CPU saves only ~4.5 ms.
- Even with zero overhead the ceiling is 3–8%, so this is closed.
- **Draft model on the GPU** (`-draft "-ngl 99 -ot exps=CPU"`: dense draft weights on the P2200, draft experts on the CPU) computed wrong: acceptance fell from 94% to 55% on code and from 59% to 34% on prose. Fixed; see "Draft on the GPU: two bugs" below.

## Draft cost on the CPU (2026-10-31)

Draft rounds took ~66–70 ms on build-cuda (16–19% of all time) while the draft's weight read is ~1.3 GB (~14 ms). `GGML_OP_PROF=4` (only graphs of ≤ 1000 nodes, i.e. the draft) and timers around the draft's steps showed where the time went.

1. **The draft's output head was a single-node copy.** The draft borrows the target's 129k-row output head. When the two buffer types differ (on build-cuda the draft's is pinned `CUDA_Host`, the target's plain CPU), ik copies the head into one fresh allocation. NUMA placement runs before that copy, so the copy sits on one node and reads at 14.5 GB/s instead of 64–70. That cost ~30 ms per round.
   - Fix: share the tensor whenever both buffers are in host memory (`src/llama-spec-features-dflash.cpp`). Draft compute went from 56.5 to 26.5 ms per round, and 0.5 GB of RAM is saved.
   - build-cpu was never affected: both buffer types are plain CPU there, so the head was already shared.
2. **The draft's confidence check ran scalar double-precision `exp` over the whole vocabulary.** With `p_min` > 0 (default 0.75) each drafted token's probability is a softmax over 129,280 logits. Only an AVX2+FMA path existed, so this CPU ran scalar `exp` at 2.4 ms per token, ~7 ms per round.
   - Fix: an SSE2 version of the same polynomial (`common/sampling.cpp`): 0.23 ms instead of 1.84 ms, and relative error ≤ 5e-5 against the double sum.
   - This applies to every build here, including the server's.

A draft round now costs ~33 ms on build-cuda:

| Step | ms |
|---|---|
| Graph build | 0.6 |
| K/V update from the target's features | 2.9–3.6 |
| Draft compute (3 blocks + output head) | 26.5 |
| Token pick + confidence | 2.0 |

Of the remaining compute, the output head is ~8 ms (0.43 GB of Q6_K at ~56 GB/s), the draft experts ~7 ms and the attention projections most of the rest. The draft is now ~8% of a round; the target's verify batch (~4 tokens at ~2.4× the cost of one) is the rest.

Neither fix changes what the target outputs, and draft acceptance is unchanged. `llama-spec-bench`, build-cuda, dq6k, tok/s:

| Setting | code | story |
|---|---|---|
| `dspark:n_max=3` before | 8.51 | 5.80 |
| + shared output head | 9.14 | 6.27 |
| + SSE2 confidence | 8.99–9.30 | 6.11–6.24 |

Draft time per run (code, 53 rounds) went 3.64 → 2.04 → 1.77 s. Run-to-run noise is ±5%, so the second fix (~2% of a round) is too small to show in tok/s on its own.

Draft length again, with the cheaper draft (tok/s):

| Setting | code | story |
|---|---|---|
| `n_max=2` | 8.26 | 6.17 |
| `n_max=3` | 9.30 | 6.11 |
| `n_max=3,p_min=0.5` | 9.07 | 6.37 |
| `n_max=3,p_min=0.9` | 9.20 | 6.11 |
| `n_max=4` | 10.39 (stopped at 90 tokens) | 5.84 |
| `n_max=4,p_min=0.9` | 9.89 | 6.02 |
| `n_max=5` | 10.28 | 5.81 |

The picture is the same as before: `n_max=3` is the balanced setting, and 4–5 only pays for code-like output. ik's `--spec-autotune` refuses DSpark.

## The verify batch on the CPU (2026-10-31)

With a draft, the target runs batches of ~4 tokens. `GGML_OP_PROF=1` on `llama-bench -p 4 -ub 4` (build-cuda, dq6k, `-ngl 0`) showed where a 4-token graph goes, against one token:

| Op | batch 1 (ms) | batch 4 (ms) |
|---|---|---|
| Dense matmuls (Q6_K) | 62 | 101 |
| Routed experts (MXFP4) | 16.5 | 46 |
| f32 matmuls (hyper-connection mixes, router) | 9 | 19.5 |
| Whole graph | 146 | 315 |

At one token the Q6_K matmuls run at ~98 GB/s, close to the memory limit. At four they are compute-bound: Sandy Bridge has one port for `pmaddubsw`, `pmaddwd` and `mulps`, and the old kernel spent ~23.5 of those per 128 weights and activation row (sign/abs work for signed quants, plus a float multiply per 16 weights). The experts' growth is mostly more distinct experts being read.

**New Q6_K kernel** (`iqk_avx1_gemv.h`):
- It multiplies the unsigned quants (0..63) directly and takes the −32 offset off afterwards, as 32 × Σ scale × (sum of the 16 activations).
- Those activation sums and the activation scales are worked out once per call, not per weight row.
- The integer sums are the same as before, so the output is bit-identical; `mmcheck` agrees with a double-precision reference.

| | before | after |
|---|---|---|
| `pp2` (tok/s) | 8.81 | 9.11 |
| `pp4` (tok/s) | 11.15 | 11.89 |
| Q6_K matmuls at batch 2 (ms/graph) | 78.9 | 69.5 |
| Q6_K matmuls at batch 4 (ms/graph) | 101.0 | 97.0 |
| Batch 1 | unchanged | unchanged |

What is left of the kernel is close to the port's floor (~19 multiplies per 128 weights and row), so further gains there would be small.

**The f32 matmuls do not get faster by cutting rows into K-pieces.** The hyper-connection mixes are 24 rows, so only 24 of 64 threads work. Cutting each row into K-pieces so that all 64 threads work made them slower: 2.8 → 3.3 ms per token at batch 1, and 6.8 → 10.4 ms at batch 4. The kernel itself is fine (19.5 GB/s on one thread from cache). As found earlier (`mm_split_k`, `mm_f32_decode`), the cost is every thread reading the activation, written by all four sockets. That patch was dropped; the K-slice below gets the split right.

### Small ops that ran on a few threads

A per-node trace (`GGML_OP_PROF=3`) of the batch-4 graph showed two ops that split their work by token, so 4 tokens kept 4 of 64 threads busy:

- **HC_POST** (the hyper-connection output mix, 80 per graph): 15.4 ms per graph for ~1.2 ms of work. It now splits by (token, 64-element chunk), as the one-token path already did, and every element is computed as before. 15.4 → 6.1 ms. `ik_llama/parts/hc_post_patch.py`, `GGML_HC_POST_SPLIT=0` off.
- **The copy before `attn_wo_a`** (`dup_bytes`, permuted attention output, 512 KB at 4 tokens) split by dim-1 rows only. It now splits all rows when there are fewer dim-1 rows than threads. CONT 8.5 → 3.4 ms. `ik_llama/parts/dup_rows_patch.py`.

`llama-bench`, `GGML_HC_POST_SPLIT` 0 vs 1 (two runs each, tok/s):

| | off | on |
|---|---|---|
| `pp2` | 8.75 / 8.33 | 9.27 / 9.06 |
| `pp4` | 10.87 / 11.67 | 12.38 / 12.24 |

### K-slice for the hyper-connection mixes

With a few tokens the RMS norm before each mix is split along K (`norm_split_patch.py`): thread *i* writes slice *i* of every row. So the mix now splits the same way. Thread *i* multiplies its own K-slice of all 24 weight rows by the activation slice it just wrote, puts partial sums in the work buffer, and after a barrier each thread adds up a share of the outputs in thread order. Nothing is read from another socket except the small partial-sum buffer.

At first this made one token slower, because 3/4 of each thread's weight slice sat on other nodes (rows were placed by row quarters). These weights (f32, < 64 rows, ≥ 4096 columns) are now placed by column quarters instead.

`ik_llama/parts/kslice_patch.py`, `GGML_MUL_MAT_KSLICE=0` off. `mmcheck` (`MMCHECK_F32=1`) shows errors of ~4e-9 against the double-precision reference, lower than the row split's.

| | off | on |
|---|---|---|
| f32 matmuls, batch 4 (ms/graph) | 21.7 | 10.6 |
| `pp2` (tok/s) | 9.37 / 9.29 | 9.58 / 9.69 |
| `pp4` (tok/s) | 12.53 / 12.46 | 12.87 / 12.92 |
| `tg16` (tok/s) | 6.70 / 6.49 | 6.74 / 6.74 |

The batch-4 graph went 315 → 294 ms over this section.

**The K-slice changes the summation order, so greedy output can differ.** KL divergence against BF16 (dq6k, `kld1.sh`, same build, 16 chunks):

| | off | on |
|---|---|---|
| Mean KLD | 0.0170 ± 0.0012 | 0.0188 ± 0.0014 |
| 99th percentile KLD | 0.268 | 0.297 |
| Same top token | 97.10 ± 0.37% | 96.90 ± 0.39% |
| PPL(Q)/PPL(BF16) | 1.0207 ± 0.0068 | 1.0075 ± 0.0062 |

These are level within the error, so the K-slice stays on.

Spec-bench (build-cuda, dq6k, DSpark, tok/s, two runs each):

| Stage | `n_max=3` code | `n_max=3` story | `n_max=4` code | `n_max=4` story |
|---|---|---|---|---|
| Before the Q6_K kernel | 8.99–9.30 | 6.11–6.24 | | |
| + Q6_K kernel | 9.29 / 9.52 | 6.16 / 6.58 | 10.22 / 10.64 | 5.78 / 5.99 |
| + HC_POST and copy splits | 9.62 / 9.67 | 6.73 / 6.75 | 10.43 / 10.45 | 6.06 / 6.00 |
| + K-slice | 9.69 / 9.59 | 6.28 / 6.24 | 10.96 / 10.92 | 6.00 |
| + chunk-major HC_POST (same output) | 9.65 / 9.72 | 6.32 / 6.36 | 10.20 / 10.99 (stops at 90 tokens) | 5.95 / 5.95 |

With the K-slice the story run takes a different greedy path (83 of 167 drafted tokens accepted instead of 96 of 162), so its tok/s is not comparable to the row above. It shows how much prose speed depends on the text.

`llama-bench` with all of the above (no draft): `pp1` 6.96, `pp2` 10.29, `pp3` 11.92, `pp4` 12.68, `pp5` 13.55, `tg16` 6.69 tok/s.

What is left at batch 4 (~290 ms per graph):
- Q6_K matmuls 97 ms, at the multiply port's floor.
- Routed experts 88 + 46 ms, set by how many distinct experts the 4 tokens pick.
- Shared expert 14 ms.
- The f32 router (`ffn_gate_inp`, 384 rows, ~6 ms).
- HC_POST 6 ms, RMS norms 5 ms.

**The K-slice does not help the router.** Taking the 384-row router through the same path (weights placed by column quarters too) doubled the f32 matmuls at batch 4 (10.5 → 19.5 ms per graph) and slowed decode (6.59 / 6.84 → 6.35 / 6.17 tok/s). Each thread then reads 384 strided 256-byte pieces, which costs more than the remote activation reads it saves. Not kept.

**HC_POST order.** Token-major tasks give thread *i* a few whole tokens, so it reads columns that 64 other threads wrote in the matmul before. Chunk-major tasks (now the default; `GGML_HC_POST_SPLIT=1` for token-major) give it the same 64 columns of every token, which is exactly what its share of that matmul wrote. Output is bit-identical. HC_POST itself did not get faster (6.1 vs 6.5 ms), but the op that next reads its output, MUL_MULTI_ADD, went 3.4 → 2.1 ms (per-thread average 2.85 → 1.05), and the batch-4 graph went 289.8 → 286.9 ms. `pp4` was 13.08 / 12.80 vs 13.07 / 13.11 tok/s, within the noise.

## Draft on the GPU: two bugs (2026-10-31)

With the DSpark draft's dense weights on the P2200 (`-draft "-ngl 99 -ot exps=CPU"`), code acceptance fell from ~91% to ~55%. Two separate bugs caused it. Both are fixed, and the GPU draft now accepts exactly the same tokens as the CPU draft.

**1. The scheduler copied the draft's K/V cache before the GPU had written it** (`sched_input_sync_patch.py`).
- The draft's K/V cache tensors are flagged as graph inputs but live in their layer's buffer, so with the draft on the GPU they are written by SET_ROWS on the CUDA compute stream.
- They are read by the CPU attention: sinks at head size 512 have no Pascal kernel.
- The scheduler copies input tensors after waiting only for the split's own backend. The copy runs on `cudaStreamPerThread`, which does not wait for the non-blocking compute stream, so the CPU could read rows not yet written.
- `CUDA_LAUNCH_BLOCKING=1` raised acceptance to 59%, which pointed at a race. The fix waits for the tensor's own backend before the copy, once per backend per split.
- With only this fix: `-ngl 1` 92.5%, `-ngl 2` 91.8%, `-ngl 3` 57.9%, `-ngl 99` 56.6%. So a second bug was in the draft's layer 0 (draft layer *n* is offloaded once `-ngl` passes it).

**2. In-place ROPE_BACK in the draft graph** (`ds41_rope_back_patch.py`, second site).
- `build_dflash_dsv4` de-rotates the attention output with an in-place rope turned into ROPE_BACK with flipped dims, the same pattern that broke the target model on CUDA earlier.
- A per-node trace (eval callback, CPU vs GPU layer 0) matched the K/V cache and every node up to the de-rotation; the next matmul, `attn_wo_a`, was 34% off.
- A standalone CPU-vs-CUDA op test ruled out the REPEAT and concat nodes nearby: both were exact.
- Out of place, the CUDA kernel takes the flip as a parameter and the result matches.

With both fixes, `-ngl 99` and `-ngl 3` give 91.22% (135/148) on code, the same as the CPU draft.

**Speed (dq6k, DSpark `n_max=3`, n=192, build-cuda `-ngl 0` for the target):**

| Draft placement | code tok/s | code accepted | story tok/s | story accepted |
|---|---|---|---|---|
| CPU (`-ngl 0`) | 9.68 / 9.53 | 135/148 | 6.29 / 6.01 | 83/167 |
| GPU (`-ngl 99 -ot exps=CPU`) | 8.48 / 8.68 | 135/148 | 5.50 / 5.47 | 83/168 |

Two runs each, interleaved. The GPU draft is ~10% slower on code and ~11% on prose. Each draft layer hands its experts' input and output across PCIe, and each hand-off inside ggml costs ~100–165 µs, the same overhead that sank the shared expert on the GPU. The draft stays on the CPU. The two fixes stay in the patch set because they make any split CPU/GPU DFlash setup correct.

## Decode attention at long context (2026-10-31)

At 4K context, decode drops from ~6.5 to ~5.8 tok/s (dq6k, build-cuda `-ngl 0`, no draft). `GGML_OP_PROF=2` puts the decode attention at 40 calls × ~330 µs = 13.2 ms per token, ~7.5% of decode and most of the gap. The kernel's own timers (`GGML_FA_MQA_PROF=1`, 1360 calls), average µs per call:

| setup | fp16→fp32 | QK | softmax | PV | barrier | merge |
|---|---|---|---|---|---|---|
| 35 | 88 | 72 | 12 | 47 | 28 | 35 |

Three changes were tried. None moved end-to-end decode beyond run-to-run noise, so none is in `r820.patch`:

1. **Head partitions** (`fa_parts_patch.py`, `GGML_FA_MQA_PARTS`): threads split over head groups as well as KV rows, so each converts fewer rows.
2. **Out-of-line PV kernels and a 16-wide mask scan** (`fa_decode_tweaks_patch.py`). In a single-thread loop, inlining made the PV kernels ~1.6× slower. Output is bit-identical.
3. **One-multiply fp16→fp32** (`f16_magic_patch.py`): ~25% faster conversion in isolation, and bit-identical over all 65536 inputs.

| Variant | tg32 | tg32 @ 4096 |
|---|---|---|
| Partitions 1 (default) | 6.38 / 6.71 | 5.71 / 5.96 |
| Partitions 4 | 6.69 / 6.75 | 5.63 / 5.62 |
| Partitions 16 | 6.83 / 6.75 | 5.82 / 5.85 |
| Base | 6.39 / 6.60 | 5.93 / 5.75 |
| + PV out of line, mask scan | 6.54 / 6.60 | 5.84 / 5.73 |
| + one-multiply fp16 | 6.53 / 6.40 | 5.85 / 5.80 |

(tok/s, two runs each in alternating order.) Even removing all of the attention would gain only ~8% at 4K. What is left there is mostly the fp16 conversion and the QK dot products, both set by the cache size. Long context is closed for now.

## Batched AVX1 kernels for prompts and the verify batch (2026-10-31)

**The earlier prompt profile was off by 2×.** `llama-bench -r 1` runs a short warm-up and then the pp512 graph, and the profiler averages over both, so the times under "Context length, threads and prompt processing" are half the real ones. A pp512 takes ~48 s on the CPU build: dense Q6_K 17 s, expert up/gate 14 s, expert down 6.7 s, attention 4.4 s. The kernels ran at ~0.45 TFLOP/s, not "60–70% of peak", so there was room left.

**Why the old kernels were slow on batches.** The gemv kernels handle up to 8 columns per call. For each 8 columns they unpack every weight block again, transpose four block sums per row and column, and spill their accumulators. At 512 columns the SIMDe Q8_0_R8 path took over, which emulates AVX2 instructions.

**New kernels** (`iqk_avx1_gemm.h`, `avx1_gemm_patch.py`, `q6_gemm_patch.py`):
- A tile of 2 weight rows and all activation columns is first unpacked into int8 scratch. The four blocks of a 128-value group are interleaved in 4-byte lanes, so `pmaddubsw` + `pmaddwd` leave block *b*'s dot product in lane *b*, with no transpose.
- The unpack, the scales and the block-sum corrections are done once and shared by every row and column of the tile.
- Q6_K keeps its quants unsigned (0..63). Four `pmaddubsw` results can overflow int16, but `paddw` wraps, and the −32 offset comes off with one `psubw` against precomputed activation-pair sums. The true value fits in int16 (8 × 32 × 127 = 32512), so the result is exact. The sub-block scales then go on with `pmaddwd`.
- `GGML_AVX1_GEMM=<n>` (default 3) is the smallest batch that uses them; 0 turns them off. While they are on, MXFP4 and Q6_K stay off the SIMDe dequant path.

Test harness (`tools/gemm_bench.cpp`, `tools/q6_bench.cpp`, 64 threads, GMAC/s). Output matches the gemv kernels bit for bit, and both match a double-precision reference.

| Batch | Q6_K old | Q6_K new | Q6_K new, weights streamed from RAM (GB/s) |
|---|---|---|---|
| 1 | 232 | 228 | 105 |
| 4 | 353 | 473 | 87 (old 67) |
| 8 | 352 | 551 | 56 (old 36) |
| 16 | 352 | 590–674 | |
| 32 | 353 | 582–714 | |

The MXFP4 kernel gains in the same way.

In the model (dq6k, 64 threads):

| | before | MXFP4 kernel | + Q6_K kernel |
|---|---|---|---|
| pp512, CPU build (tok/s) | 10.55 | 14.69 | **23.86** |
| pp512, build-cuda `-ngl 0` | 15.2 | 26.13 | (Q6_K runs on the GPU there) |
| Dense Q6_K per pp512 (s) | 17.3 | 17.1 | 5.5 |
| Expert up/gate + down per pp512 (s) | 21.0 | 7.8 | 7.8 |
| `pp4` graph, CPU build (ms) | | 285 | 256 |

`llama-spec-bench`, dq6k, DSpark, 192 tokens, tok/s:

| Build | `n_max` | code | story |
|---|---|---|---|
| build-cpu, MXFP4 kernel | 3 | 10.02 | 6.31 |
| build-cpu, MXFP4 kernel | 4 | 10.41 | 6.19 |
| build-cuda `-ngl 0`, MXFP4 kernel | 3 | 9.71 | 6.13 |
| build-cuda `-ngl 0`, MXFP4 kernel | 4 | 11.32 | 5.95 |
| **build-cpu, both kernels** | 3 | **10.69** | **6.75** |
| **build-cpu, both kernels** | 4 | **12.29** | 6.45 |

- Acceptance and rounds are unchanged, token for token, so the output is the same.
- The `n_max=4` code run stops at an end-of-text token after 90 tokens, so its rate is for a shorter run.
- The batch-4 experts are untouched: each expert sees one or two of the four tokens, so they still run as gemv.
- **Server choice changes:** the CPU build now has prompts at 23.9 tok/s and the fastest decode, so the server runs it, not build-cuda.
- The verify batch is now 256 ms against 146 ms for one token, which caps what drafting can gain on prose.

KL divergence against BF16 on the prompt path (`llama-perplexity -ub 256`, 16 × 256 tokens), which used to dequantise Q6_K to Q8_0_R8:

| | old path (`GGML_AVX1_GEMM=0`) | batched kernels |
|---|---|---|
| Mean KLD | 0.0191 ± 0.0013 | 0.0201 ± 0.0016 |
| 99th percentile KLD | 0.319 | 0.328 |
| Same top token | 96.80 ± 0.39% | 96.85 ± 0.39% |
| PPL(Q)/PPL(BF16) | 1.0111 ± 0.0066 | 1.0102 ± 0.0065 |

Level within the error. At batch 3–8 (verify, `-ub 8`) the kernels give the same output as the gemv kernels.

Through the LAN server (build-cpu, DSpark `n_max=3`, `scripts/apitest.sh`, 256 tokens, temperature 0): code 10.40 tok/s, prose 7.02, prompt of 32 tokens at 14.0 tok/s. With thinking on, a short coding request gave 418 thinking tokens at 8.0 tok/s and a 200-token code answer at 10.2 (`docs/media/r820-deepseek-demo.mp4`).

**Not done:** flash attention on the GPU for prompt batches. The tile kernel's support check still rejects sinks, though the kernel applies them since `fattn_sinks_patch.py`, so attention stays on the CPU in build-cuda (~5 s per pp512). Lifting that check would help build-cuda prompts only, and the CPU build already meets the 20 tok/s target. *Done in the next section.*

## Second targets: 30 tok/s prompts, 15 tok/s decode (2026-10-31)

### Prompts: 35 tok/s on build-cuda

- **Prompt attention on the GPU** (`fa_offload_patch.py`). With `-ngl 0` an op reaches the GPU only through the scheduler's offload rule (a CPU weight as a source, batch ≥ 32). Attention qualifies through its per-head sinks, but the Pascal hs512 tile kernel still reported sinks as unsupported. Allowing them moves prompt attention to the GPU, next to the projections already there. Decode and the verify batch (< 32 tokens) stay on the CPU. pp512 26.1 → 29.1–29.9.
- **Pinned dense weights** (`cuda_pin_patch.py`, `GGML_CUDA_PIN_MAX_GB=10`). After NUMA placement (pinned pages cannot move), `cudaHostRegister` the host buffers of at most 10 GB: the dense shard, which large batches upload. The uploads then run at PCIe speed instead of through pageable staging. The expert shards are used on the CPU only and stay unpinned. pp512 → **35.05**, 35.12 ± 0.40 on the final build.
- Quality of the GPU prompt path (build-cuda `-ngl 0`, `-ub 256`, so attention runs on the GPU, 16 × 256 tokens against BF16): mean KLD 0.0195 ± 0.0015, 99th percentile 0.314, same top token 96.41 ± 0.41%, PPL(Q)/PPL(BF16) 1.0125 ± 0.0058. Level with the CPU path (0.0191–0.0201) within the error.
- **The server goes back to build-cuda**: prompts 35.1 against 23.9, and decode at `n_max=4` is the same (code 12.28 against 12.29).

Through the LAN server (`scripts/apitest.sh`, 256 tokens, temperature 0):

| Server | code | prose |
|---|---|---|
| build-cuda, `n_max=4` (runs now) | 10.09 | 7.01 |
| build-cuda, `n_max=3` | 9.31 | 6.85 |
| build-cpu, `n_max=4` | 10.34 | 7.11 |

build-cpu decodes ~2.5% faster, but build-cuda reads prompts 47% faster, so the server runs build-cuda at `n_max=4`. The chat-API code prompt gets ~3 tokens per round (84–87% of drafts accepted, against 93% and 4.68 tokens on the benchmark's code task), which is why it shows ~10 tok/s, not 12. Demo, same prompt as before with thinking on: 832 tokens at 8.98 tok/s (thinking 8.4, code answer 11.3), against 622 at 8.68 (`docs/media/r820-deepseek-demo.mp4`).

### The verify batch: where the time goes

`GGML_OP_PROF=5` on build-cpu, DSpark `n_max=4`, code. The main graph holds layers 0–36; the DSpark eval callback splits layers 37–39 and the output head into separate graphs.

| Part of a 5-token verify | ms | Note |
|---|---|---|
| Routed experts (up/gate + down) | 144.5 | 136–140 GB/s: the bandwidth floor |
| Dense Q6_K (q_a, kv, q_b, wo_a, wo_b, shared expert) | 88.8 | 52–72 GB/s, bound by port 0 (`pmaddubsw`, `pmaddwd`, `mulps`) |
| Attention | 18.7 | |
| f32 matmuls (router, hyper-connection mixes) | 11.1 | |
| Norms, hyper-connections, rope, copies, adds | ~31 | |
| **Main graph** | **294.5** | |
| Layers 37–39 and the head, separate graphs | ~45 | |
| Draft | ~30 | |
| **Round** | **~382** | 4.68 tokens accepted per round → 12.3 tok/s |

15 tok/s at the same acceptance needs a round of ≤ 312 ms, 70 ms less. The experts cannot go faster on this memory, so the cut would have to come almost entirely from the dense kernel, the small ops and the draft.

### Kept

- **Verify attention shares the K/V conversion across queries** (`fa_multi_patch.py`). The decode attention kernel converted the fp16 K/V rows to f32 once per query. With a mask, the verify queries now walk the union of their visible rows together, with −inf for a row a query cannot see, so each block is converted once for all of them. Attention per verify 20.7 → 15.7–18.7 ms (noisy). `GGML_FA_MQA_MULTI=0` turns it off. `GGML_FA_MQA_MINROWS` sets the chunk floor (default 64; 32 and 16 were no better).
- **Scheduler: a view of a weight stays with the weight** (`sched_view_weight_patch.py`). A reshaped weight gets its buffer only when the graph is allocated, so the pass-1 check missed it. The op could then land on the GPU and upload the CPU weight on every step.
- **Profiler fix** (`op_prof_barrier_patch.py`). Threads could run the next node and overwrite their time before thread 0 read it. Small matmuls then looked as if thread 0 did 5–20× the work of the others. It was an artifact: with the fix all threads take the same time.
- `LLAMA_DECODE_TRACE=1` (`decode_trace_patch.py`): one line per `llama_decode` with tokens, layers and graph time.

### Tried, not kept

- **GPU sidecar for the verify's dense matmuls.** The P2200's Q6_K kernel at 5 columns runs ~250 GMAC/s, below the CPU's ~400–500, and the activations cross PCIe at ~1 GB/s. With `wo_a`/`wo_b` and the shared expert on the GPU, the draft scheduler also ran out of GPU memory.
- **Fewer threads on the routed experts, the rest on the shared expert.** Expert bandwidth drops with fewer threads (48 threads 114 GB/s, 32 threads 120 GB/s, 64 threads 140), so the overlap loses.
- **Faster activation quantisation for the small matmuls** (`experiments/q_replica_patch.py`). The quantisation phase took 8–147 µs per matmul (147 µs for `wo_a`'s 40 rows), on the 5 threads that own the 5 token rows. Main-graph time for each variant:

  | Variant | ms |
  |---|---|
  | Baseline | 294.5 |
  | AVX1 quantiser instead of SIMDe's emulated AVX2 | 295.3 |
  | Split over all 64 threads | 303.5 (`wo_a` faster, every other matmul slower) |
  | One copy per NUMA node, node-local barrier | 302.3 |

  The cost is moving activations between sockets, not converting them. Spreading the writers out makes every reader fetch from more sockets.

**Known issue:** greedy output is garbage at `-t 32` and `-t 48`, with or without the new attention path. It is older than this round. The server uses 64 threads, where the output is correct.

## Ideas for cutting the per-token weight read (2026-10-30)

At dq6k each token reads ~10.9 GB: experts 4.5, dense 6.1, f32 router/hc 0.3. That is ~98 ms at 110 GB/s. They are listed best first.

Status: 2 measured (`-ser`, opt-in only), 3 and 6 measured (4-bit sections below), 5 done (suffix and DSpark drafting). 1 needs the physical DIMM change; 4 needs engine work.

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
- **MXFP8** (OCP microscaling: 32 E4M3 values with one E8M0 scale) is not in mainline llama.cpp or ik. It comes from the unmerged llama.cpp PR #30817, which Unsloth's guide builds from. Ported as a gather-only type with `ik_llama/parts/mxfp8_patch.py` (to_float + GET_ROWS), and applied to both trees. The decoder is verified exact for all 256 codes; sampled Engram rows decode cleanly (RMS 0.88, max 5, no NaN).
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

## Targets (no purchases): estimates against measurements (2026-10-31)

Decode tok/s. The estimates are the ones this plan started with; the measurements are llama-bench / llama-spec-bench at 64 threads, `GGML_NUMA_BLOCKED=1 GGML_NUMA_PLACE=1`.

| Setup | Estimate | Measured |
|---|---|---|
| Stock llama.cpp, as downloaded (BF16 dense) | ~1.5 | 1.22 (mainline, 60 threads) |
| Stock, Q8_0 dense | ~2.5 | 1.58 mainline, 2.60 ik |
| NUMA-local engine, CPU only, Q8_0 dense | ~6.5 | 5.94 |
| NUMA-local engine, CPU only, Q6_K dense (the pick) | – | **6.7–7.0** (5.8 at 16K context) |
| + dense part on the GPU | ~12 | slower: −2–5% at `-ngl 0`, −16–19% with layers on the GPU; shared expert alone −13% |
| + speculative decoding (DSpark), code-like output | ~16–19 (with the GPU) | **10.7** (`n_max=3`), **12.3** (`n_max=4`) |
| + speculative decoding (DSpark), free prose | | 6.5–6.8 |
| Second targets (no purchases): prompts 30, decode 15 | | prompts **35.1** met; decode **12.3** on code, not met (see "Second targets") |
| Prompt processing, pp512 | – | 23.9 CPU only; **35.1** build-cuda `-ngl 0` with attention on the GPU and the dense shard pinned |

- The GPU plan assumed the P2200 would add bandwidth. The CPU kernels doubled in speed instead, and each CPU↔GPU hand-off in ggml costs 100–165 µs, so offloading anything per token now loses.
- Speculative decoding gives ×1.45–1.6 on code, about what Phase 4 expected. On prose the draft is accepted too rarely to gain.
- The DSpark draft on the GPU now computes correctly (two bugs fixed, see "Draft on the GPU: two bugs"), but it is ~10% slower than on the CPU, so the draft stays on the CPU.
- Long context: 5.7–5.9 tok/s at 4K. Attention is ~7.5% of decode there, and three kernel changes gained nothing measurable.
- Still to try, needs you to do it: 2 DIMMs per channel at 1600 MT/s (see "Ideas for cutting the per-token weight read"), about +15–20% decode.

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
