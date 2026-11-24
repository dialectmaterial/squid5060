# squid5060 — Theoretical Performance Ceilings (`Inkling-2-Mini` on 8 GB RTX 5060 Laptop)

## 1. Measured Hardware (NVIDIA GeForce RTX 5060 Laptop GPU)

Measured directly with custom `sm_120a` CUDA microbenchmarks on 2026-11-24:

| Metric | Value | Notes |
|---|---:|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop (`sm_120a`, Blackwell) | **26 SMs**, **32.0 MiB L2**, `canMapHost=1`, `asyncEngines=2` |
| Memory | 7,799.1 MiB CUDA-reported GDDR7 | ~7.47 GiB peak usable alongside desktop compositor |
| **Measured GDDR7 Pure Read (`uint4`, 64 MiB)** | **284.5 GB/s** | On-chip GDDR7 decode ceiling |
| **Measured PCIe Gen5 x8 H2D DMA (`cudaMemcpyAsync`)** | **24.8 GB/s** | `0.182 ms` per 4.27 MiB expert |
| **Measured Zero-Copy Mapped Host Read (`uint4`)** | **24.1 GB/s** | 97.2% of DMA speed |
| **Measured Dual-Path Cached MoE `w13` PCIe Miss** | **23.8–23.9 GB/s** | Coalesced SMEM scales + `uint4` (96.4% of DMA) |
| **Measured Dual-Path Cached MoE `w2` PCIe Miss** | **23.7 GB/s** | Up from `14.7 GB/s` (`+61%`) with `uint2` quarter-warp! |
| **Measured Dual-Path Cached MoE VRAM Hit (L2+GDDR7)** | **740–1,270 GB/s** | Zero-SMEM `uint2` fast path |
| CUDA Toolkit / Driver | CUDA 13.4 / Driver `615.78.08` | Native `sm_120a` NVFP4 (`cvt.rn.f16x2.e2m1x2`) + `m16n8k16` TC |

## 2. Why `Inkling-2-Mini` Fits 8 GB So Well

- **Loop Geometry:** 24 physical layers expanded into **48 logical visits** (`0..5` prelude, `6..17` × **3 passes** = 36 looped visits, `18..23` coda).
- **75% of all MoE visits** (`36 / 48`) hit only **50% of physical experts** (`1,536 / 3,072` in layers `6..17`).
- **Unified NVFP4 Expert Slot Size:** `4,478,976 B` (`4.2715 MiB`) per physical expert (`w13_data` 2.53 MiB + `w13_scales` 0.32 MiB + `w2_data` 1.27 MiB + `w2_scales` 0.16 MiB).
- **8 GB Memory Budget (`ctx=32768`, `--kv q4`, `--mode nvfp4`, `--mtp 0`):**
  - Resident non-expert weights (`dense + controllers + routers + lm_head`): **`0.60 GiB`**
  - Pinned zero-copy host `embed`: `0.86 GiB` host RAM (**`0.00 GiB` VRAM**)
  - Runtime VRAMArena (`40` local BF16 SWA KV + `8` global Q4 KV @ 32K + `B=512` prefill scratch): **`0.70 GiB`**
  - Unified VRAM MoE Expert Cache: **`6,078 MiB` = `1,423 / 3,072` physical experts (`46.3%`)**
  - Total peak `nvidia-smi` (including desktop): **`7.47 GiB`**!
  - Pinned host-RAM full expert pool (`24 × 128 × 4.27 MiB`): **`12.81 GiB`** system RAM.

## 3. Decode Roofline & Measured Breakdown (`B = 1`, `mtp = 0`)

At `~89.5–90.5%` decode cache hit rate (~`95%` in the 36 looped visits, ~`68–78%` in prelude/coda):
- **Non-MoE dense + attention + controllers + sconv (48 visits):** **`4.46 ms`** (measured via `SQUID_PROF_DECODE=1`)
- **LM head (`200,058 × 2,304` NVFP4, 208 blocks = `8×26` SMs):** **`0.73 ms`**
- **MoE VRAM hits (~345 experts × 4.48 MB, with L2 reuse across 3 loop passes):** **`~3.5 ms`**
- **MoE PCIe zero-copy misses (~39 experts × 4.48 MB = 175 MB @ 23.8 GB/s):** **`~7.35 ms`**
- **Total:** **`16.0–16.6 ms/tok` = `60.2–62.5 tok/s`** (measured live: **`60.31 tok/s` median / `63.08 tok/s` peak** on P1 coding; **`55.01 tok/s` median** across 6 diverse `bench.py` prompts!).

Every **+1% cache hit rate** saves `3.84 × 4.48 MB / 23.8 GB/s = 0.72 ms/tok` (**~2.6 tok/s**).

## 4. Speculative Decoding Trade-Off on PCIe-Bound 8 GB vs. Full-VRAM 32 GB

On RTX 5090 (all weights in 1,792 GB/s GDDR7), `mtp=3` (`B=4`) always wins because loading extra unique experts for draft tokens is cheap. On 8 GB PCIe Gen5 x8 (`24.8 GB/s`):
- Loading DSpark consumes **`~224 MiB` VRAM** (`-52` cached expert slots → ~`-1.5%` hit rate).
- Each rejected draft token still pulls its unique missed experts over PCIe during target verify.
- **Empirical crossover:**
  - **Mixed / prose workloads (`bench.py` 6-prompt suite, `dl ~ 1.8–2.2`):** **`--mtp 0`** wins decisively (**`55.01 tok/s`** vs `44.59 tok/s` at `mtp=2`).
  - **Code-heavy / high-acceptance workloads (`dl >= 2.35`, e.g. P1 coding or 24K needle recall):** **`--mtp 2`** (`B=3`, Pareto optimum over `mtp=3`) wins — reaching **`66.94 tok/s`** on P1 coding and **`55.95 tok/s`** on 23.9K long-context decode (`+14%` over `mtp=0`'s `49.03 tok/s`)!
