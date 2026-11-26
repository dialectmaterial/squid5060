# squid5060 — Theoretical Performance Ceilings (`Inkling-2-Mini` on 8 GB RTX 5060 Laptop)

## 1. Measured Hardware (RTX 5060 Laptop 8 GB + Intel Core Ultra 7 255H)

Measured directly with custom `sm_120a` CUDA microbenchmarks and live telemetry on AC power:

| Metric | Value | Notes |
|---|---:|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop (`sm_120a`, Blackwell) | **26 SMs**, **32.0 MiB L2**, `canMapHost=1`, `asyncEngines=2`, `85 W` AC cap |
| CPU | Intel Core Ultra 7 255H (16 hybrid cores) | **6P @ 5.1 GHz + 8E @ 4.4 GHz** + 2 LP-E (excluded), `AVX-VNNI`, `WAITPKG` |
| VRAM | 7,799.1 MiB CUDA-reported GDDR7 | **7.47 GiB** (`7,651 MiB`) peak alongside desktop compositor, **0 host KV spill** |
| **Measured GDDR7 Pure Read (`uint4`, 64 MiB)** | **284.5 GB/s** | On-chip GDDR7 streaming ceiling |
| **Measured PCIe Gen5 x8 H2D DMA (`cudaMemcpyAsync`)** | **24.8 GB/s** | `0.182 ms` per `4.27 MiB` expert |
| **Measured Zero-Copy Mapped Host Read (`uint4`)** | **24.1 GB/s** | `97.2%` of DMA speed (`23.7–23.9 GB/s` in MoE miss kernels) |
| **Measured Dual-Path Cached MoE VRAM Hit (L2+GDDR7)** | **740–1,270 GB/s** | `B=1..3` half-warp PDL+L2; `B>=4` Blackwell `mma.m16n8k16` TC (42 regs, 0 spills) |
| **Effective CPU AVX-VNNI Miss Bandwidth (12t `6P+6E`)** | **~52–60 GB/s** | Q8-activation NVFP4 `_mm256_dpbusd_avx_epi32` + atomic 32-row work-stealing |
| **Combined Parallel Miss Bandwidth (`DDR5 VNNI || PCIe`)** | **~75–82 GB/s** | **3.1×** GPU-only PCIe (`24.1 GB/s`) via cost-balanced `auto` (`2/3`) split |
| CUDA Toolkit / Driver | CUDA 13.4 / Driver `615.78.08` | Native `sm_120a` NVFP4 (`cvt.rn.f16x2.e2m1x2`), PDL, selective `L2::evict_first` |

## 2. Why `Inkling-2-Mini` Fits 8 GB So Well

- **Loop Geometry:** 24 physical layers expanded into **48 logical visits** (`0..5` prelude, `6..17` × **3 passes** = 36 looped visits, `18..23` coda).
- **75% of all MoE visits** (`36 / 48`) hit only **50% of physical experts** (`1,536 / 3,072` in layers `6..17`), reaching **`93.0%–95.7%`** VRAM hit rate in the looped core.
- **Unified NVFP4 Expert Slot Size:** `4,478,976 B` (`4.2715 MiB`) per physical expert (`w13_data` 2.53 MiB + `w13_scales` 0.32 MiB + `w2_data` 1.27 MiB + `w2_scales` 0.16 MiB).
- **8 GB Memory Budget (`ctx=32768`, `--kv q4`, `--mode nvfp4`, `--mtp 7`):**
  - Resident non-expert + DSpark weights (`dense + controllers + routers + lm_head + dspark`): **`0.79 GiB`**
  - Pinned zero-copy host `embed`: `0.86 GiB` host RAM (**`0.00 GiB` VRAM**)
  - Runtime VRAMArena (`40` local BF16 SWA KV + `8` global Q4 KV @ 32K, tail-aliased prefill): **`0.32 GiB`**
  - Unified VRAM MoE Expert Cache: **`6,279 MiB` = `1,470 / 3,072` physical experts (`47.9%`)** (`1,529` slots with `--mtp 0`)
  - **256K (`--ctx 262144`, `--kv q4`, `--mtp 7`) Verified Budget:** `1.48 GiB` runtime VRAM + `5,092 MiB` (`1,192` experts, `38.8%`) — fits 100% in 8 GB with **0 host KV spill** (`106.3 tok/s` smoke decode).
  - Total peak `nvidia-smi` (including desktop & 5-way CUDA Graph): **`7.47 GiB`** (`7,651 MiB`).
  - Pinned host-RAM full expert pool (`24 × 128 × 4.27 MiB`): **`12.81 GiB`** system RAM.

## 3. Zero-Host-Sync GPU Event Breakdown & Hybrid Roofline (`v3.7` / Phase 5)

Measured with zero-host-sync `cudaEventRecord` / `cudaEventElapsedTime` (`SQUID_PROF_DECODE=1`, preserving full PDL and CPU-GPU overlap) plus live CUDA-Graph step timings (`ctx=32768`, AC power):

| Batch Width | Non-MoE (`emb+ctrl+dense+attn+rtr+lm`) | GPU MoE (`mgu+mdn`) | CPU Tail (`cmerge`) | Non-Graph Event Total | **CUDA-Graph Step** | **Measured Cost Ratio** |
|---|---:|---:|---:|---:|---:|---:|
| **`B=1` (`--mtp 0`)** | `6.33 ms` | `5.96 ms` | `1.16 ms` | `13.45 ms` | **`10.01 ms`** | **`1.00×`** |
| **`B=2`** | `~6.45 ms` | `~8.45 ms` | `~1.80 ms` | `~16.70 ms` | **`14.55 ms`** | **`1.45×`** |
| **`B=3` (Sweet Spot)** | `6.67 ms` | `10.67 ms` | `2.50 ms` | `19.88 ms` | **`16.99 ms`** | **`1.70×`** |
| **`B=4` (TC MoE)** | `6.81 ms` | `12.94 ms` | `3.56 ms` | `23.35 ms` | **`19.55 ms`** | **`1.95×`** |
| **`B=8` (TC MoE, Hybrid)** | `7.80 ms` | `20.66 ms` | `4.17 ms` | `32.67 ms` | **`29.81 ms`** | **`2.98×`** |
| `B=8` (GPU-only, `--cpu-moe off`) | `7.36 ms` | `45.93 ms` | `0.00 ms` | `53.34 ms` | `~49.5 ms` | `4.95×` |

### Quantifying the Remaining Gap to the Physical Hardware Ceiling

1. **Non-Speculative `B=1` (`--mtp 0`) Ceiling vs. Achieved:**
   - At `90.95%` warm decode cache hit rate (`95.7%` loop, `73.1%` prelude, `79.9%` coda):
   - Irreducible GPU non-MoE inside CUDA Graph (with full PDL): **`~4.90 ms`**
   - MoE VRAM hits (`~349` experts × `4.48 MB`, ~`2.4×` L2 loop reuse → `~650 MB` GDDR7 @ `284.5 GB/s`): **`~2.28 ms`**
   - MoE RAM misses (`~35` experts = `157 MB` split `2/3` CPU DDR5 VNNI || `1/3` GPU PCIe): `max(105 MB / 55 GB/s = 1.91 ms, 2.28 ms + 52 MB / 23.8 GB/s = 4.46 ms)` → **`~4.46 ms`**
   - **Physical Hybrid `B=1` Floor:** `4.90 + 4.46 = 9.36 ms/tok` (**`106.8 tok/s`**).
   - **Achieved `B=1` (`--mtp 0`):** **`10.01 ms/tok` (`99.92 tok/s`)** warm CLI (`88.54 tok/s` 6-prompt `T=0.6` median) = **`93.5%` of the physical hybrid ceiling**!
   - **Remaining `0.65 ms` gap:** cross-socket `cmerge` system-memory fence tail across 48 visits + concurrent `copy_stream` async promotion contention.

2. **Adaptive Speculative (`--mtp 7`, 5-Way `{1, 2, 3, 4, 8}`) Ceiling vs. Achieved:**
   - With Phase 1 (AVX-VNNI 12t work-stealing + cost-balanced `auto` split + fused router dispatch), Phase 2 (Blackwell `mma.m16n8k16` TC MoE at `B>=4` with 42 regs / 0 spills + PDL + selective `L2::evict_first`), Phase 3 (instant bulk-DMA tail refill + dynamic tail sizing + warm-preserving prefill LRFU), and Phase 4 (5-way `{1,2,3,4,8}` switch calibrated to `{1.00, 1.45, 1.70, 2.05, 3.20}` + PDL/L2/early-exit Markov):
   - **Diverse 6-prompt chat/code (`dl ≈ 1.85–2.20`):** theoretical hybrid ceiling **`~108–115 tok/s`** → achieved **`101.1–103.1 tok/s` (`T=0.6`) / `102.0 tok/s` (`T=1.0`) / `105.7–107.1 tok/s` (`T=0.0`)** (**`91–94%` of ceiling**, **`2.77×` llama.cpp**).
   - **High-acceptance coding / verbatim long-context recall (`dl ≈ 3.0–3.5`):** theoretical hybrid ceiling **`~145–160 tok/s`** → achieved **`136.2 tok/s` SSE peak (`3.59×` llama.cpp)**, **`125.9 tok/s` 18.8K Needle (`3.92×` llama.cpp, `148.6` CLI)**, and **`117.7 tok/s` at 29.9K 3-depth (`10%/50%/90%`) Needle (`3/3` PASS)** (**`92–95%` of ceiling**).
   - **Irreducible physical limiters of the remaining ~5–8%:** laptop Dynamic Boost shared chassis power envelope (`73–78 W` GPU while 12 CPU cores run at 105 °C), dual-channel DDR5 bus arbitration between CPU VNNI reads and GPU PCIe zero-copy misses, and 48 per-visit GPU↔CPU release/acquire barriers.

## 4. Verification & Quality Summary (`v3.7` / Phase 5)

- **Exact FP64 KLD (`7,784` tokens, `tools/kld_corpus.txt`):** GPU TC vs legacy `mean_KLD = 0.008599` (`top1 = 97.10%`); Hybrid TC+VNNI vs legacy `mean_KLD = 0.009136` (`top1 = 96.97%`, `cand_PPL = 2.7299`, `ΔPPL = -0.0088` — closer to exact BF16 `2.7048` than legacy FP16-accum `2.7387`).
- **Multi-Depth Long-Context Retrieval:** `18.8K` Needle **`3/3` PASS** (`125.3–125.9 tok/s`), `29.9K` (`ctx=32768`) 3-depth (`10% / 50% / 90%`) Needle **`3/3` PASS** (`113.3–117.7 tok/s`), `256K` (`--ctx 262144`) VRAM allocation **PASS** (`0` host KV spill).
- **`ptxas -v` ISA Audit (551 kernels across 3 `.cu` files):** **0 register spills** on every active NVFP4 decode kernel (`moe_gate_up/down_nvfp4_tc_cached`: 42 regs / 0 spills) and all 26 DSpark/Markov kernels.

## 8. Phases 6–8: Quality Parity, Lazy 256K KV & Agentic UX (`v3.8`)

- **Frozen Spark-Bench Quality Gate (Phase 6):** Evaluated `squid5060` (NVFP4 + Q4 KV, 8 GB GPU) on `knapcio/inkling-2-mini-spark-bench` (`inkqc`, 295 items): **`286 / 295`** (`gsm8k 195/200`, `qeval 71/75` tied #1 overall, `lc 20/20` 100%, median decode **`127.2 tok/s`**, peak **`161.9 tok/s`**), `128K` needle **PASS** (`63.2 tok/s`), plus `chat_template_kwargs` (`reasoning_effort`, `enable_thinking`) compatibility.
- **Lazy Grow-on-Demand `32K → 64K → 128K → 256K` Global KV (Phase 7):** Booting `--ctx 262144` allocates only `32K` global KV in the base arena (`0.32 GiB` runtime VRAM, **`1,470 / 3,072`** hot experts = identical to `--ctx 32768`), caps DSpark's global KV at `32K` on 8 GB GPUs (reclaiming ~66 expert slots at `256K`), and tail-aliases larger global KV tiers onto `d_cache_pool` on demand with lossless host-bounce KV preservation mid-session (`32K → 64K` preserves `30,073` cached tokens and recalls an early needle at `101.9 tok/s`), auto-shrinking back to `32K` (`1,470` experts) on fresh short chats. **`240K` needle (`240,104` tok): PASS @ `45.5 tok/s`** (`1,215` active experts, 0 host KV spill).
- **Empirical Calibrations (Kept Opt-In):**
  - `Q_TOK=64` prefill attention (`SQUID_PREFILL_Q64=1`) regressed `-2%` at `37.4K` (`1,796` vs `1,833 tok/s`) due to 1 CTA/SM occupancy, so `Q_TOK=32` stays default.
  - Sun et al. (ICLR 2025, Algorithm 2) Block Verification (`SQUID_BLOCK_VERIFY=1`) cost `-8.2%` wall-clock (`88.8` vs `96.7 tok/s`) at `mtp=3` due to the extra 204K-vocab residual-mass pass (`lm_head_bv_rm_kernel`), so token-by-token verification + always-on `RSFIX` positive-mass fallback stays default (`SQUID_BLOCK_VERIFY=1` opt-in).
- **Agentic Speed & UX (`pi` Harness, Phase 8):**
  - **Sampled-Safe Prompt-Copy (`SQUID_PCOPY_MIN_CTX=8192`, 64K history, 16-tok exact match, exact `q=1.0` point-mass verification):** **`110.3 → 142.7 tok/s` (`+29.4%`)** on `11.4K` verbatim code-edit prompts at `T=1.0, top_p=0.95` with zero distribution bias.
  - **8-Slot Rolling Prefix Cache + Exact Raw Tool-Call History Cache:** Sub-4K `512`-tok grid boosts `1,825`-tok few-shot prefixes from `0 → 1,536` cached (`84.2%`); rolling turn checkpoints (`4..7`) hit `847/867` (`97.7%`) on Turn-2 branch rollbacks; `ToolRawHistoryCache` hits `727/745` (`97.6%`) even when clients re-serialize prior `tool_calls` JSON with different whitespace/key order.
  - **Live SSE `tool_calls` Streaming:** Incremental UTF-8-safe JSON scanner in `InklingStreamParser` emits `tool_calls` headers and `function.arguments` deltas token-by-token (**424 live chunks** over 3.08s, 0 duplicates, clean fallback when `"args"` precedes `"name"`).
