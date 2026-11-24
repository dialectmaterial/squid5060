# squid5060 — 8 GB Port Plan & Experiment Ledger

**Target:** [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on **8 GB NVIDIA RTX 5060 Laptop GPU** (`sm_120a`, 26 SMs, 32 MiB L2, 284.5 GB/s GDDR7, 24.8 GB/s PCIe Gen5 x8).
**Baseline to beat:** `llama.cpp` `ink-dedup` (`UD-Q3_K_XL`, `--moe-cache-mib 5600`): **37.2 tok/s** 6-prompt decode median, **174 tok/s** 3.3K prefill, **106 tok/s** 22K prefill (210 s), **32.1 tok/s** 30.5K decode, 3/3 needle retrieval.

## Completed Optimization Milestones

1. **M0 — Hardware Characterization & Toolchain**
   - Measured real RTX 5060 Laptop ceilings: GDDR7 read `284.5 GB/s`, PCIe Gen5 x8 H2D DMA `24.8 GB/s`, zero-copy GPU read `24.1 GB/s` (`uint4`).
   - Updated `build.zig` for `squid5060`, `/opt/cuda` default, and RPath.
2. **M1 — Unified 3,072-Expert VRAM Cache + Zero-Copy Pinned Host Fallback**
   - Packed all 24 layers' NVFP4 experts (converting L20–23 MXFP8→NVFP4 in 32-expert GPU chunks) into uniform `4.27 MiB` slots (`EXPERT_SLOT_BYTES = 4,478,976`).
   - Pinned `embed` (`0.86 GiB`) and `dspark.markov_w1` (`98 MiB`) in zero-copy mapped host RAM.
   - Added cached CUDA kernels for `B=1` decode, `B=2..8` grouped MTP verify, and `B=512` Tensor Core prefill — verified **0.0 bit-for-bit parity** against `squid5090` all-VRAM kernels.
   - Preserved persistent `cudaGraphExec_t` (zero per-layer host sync stalls).
   - Initial `mtp=0` speed: **41.34 → 43.75 tok/s**; `mtp=3`: **49.54–51.84 tok/s**.
3. **M2 — Aging-LFU (`LRFU`) + Zero-Race Async Overlapped `copy_stream` Promotion**
   - Added global Aging-LFU with hysteresis (`decay=0.998`, `hyst=1.50`) and normalized prefill chunk warming, cutting promotion thrashing by **11.6×** vs CLOCK.
   - Added dedicated `copy_stream` async promotion using 8 reserved staging slots + 1-token victim quarantine so H2D copies overlap with compute with zero race conditions.
4. **M3 — Dual-Path CTA-Uniform PCIe `uint4` + SMEM-Scale Kernels & 26-SM Grid Tuning (Biggest Win: +38%!)**
   - Decode micro-profiler (`SQUID_PROF_DECODE=1`) exposed `moe_down` taking `8.2 ms/tok` (only `14.7 GB/s` on PCIe misses due to 4-expert warp packing and strided 8-byte scale reads).
   - Rewrote `moe_down_nvfp4_b1_cached_kernel` to slotted 1-expert-per-CTA (`moe_slotted=1` in `mega_mlp_post_*`) and added **CTA-uniform dual-path dispatch** across all 4 cached MoE decode/verify kernels:
     - VRAM hit (`cslot >= 0`): zero-SMEM `uint2` (`740–1,270 GB/s`).
     - PCIe miss (`cslot < 0`): 1-burst coalesced SMEM scale preload + `uint4` (`128B`/`256B`) weight bursts (**`23.7 GB/s` on `w2` = +61%**, **`23.8–23.9 GB/s` on `w13` = 96% of PCIe DMA ceiling**!).
   - Aligned persistent grids to 26 SMs (`LM_HEAD_BLOCKS = 208`, `DSPARK_UNEMBED_BLOCKS = 208`, `MARKOV_BLOCKS = 104`), dropping `lm_head` from `0.95 ms → 0.73 ms`.
   - `moe_down` dropped **`8.43 ms → 2.85 ms` (3× faster!)**, jumping `mtp=0` from **43.75 → 60.31 tok/s median (63.08 peak)** and `mtp=2` to **62.95 tok/s median (66.94 peak)**!
5. **M4 — Full `bench.py` Apples-to-Apples Validation & 128K `longctx`**
   - Verified 131,072 (128K) `q4` KV context allocates and executes in 8 GB VRAM with 4.94 GiB (1,156 slots) still cached.
   - Ran the exact `bench.py` suite (`6×512` decode, `3.3K` & `12.9K` cold prefill, `23.9K` Needle retrieval `3/3`, peak VRAM):
     - **`--mtp 0` (new 8 GB default):** **`55.01 tok/s`** 6-prompt median (up to `61.31`), **`255.1 / 624.7 / 1,094.7 tok/s`** prefill, **`49.03 tok/s`** @ 23.9K, **`3/3` PASS**, **`7.47 GiB` peak VRAM**.
     - **`--mtp 2` (code/high-acc opt-in):** **`44.59 tok/s`** 6-prompt median (up to `56.15` / `66.94` on P1), **`997.9 tok/s`** 23.9K prefill, **`55.95 tok/s`** @ 23.9K, **`3/3` PASS**, **`7.47 GiB` peak VRAM**.

## Experiment Ledger (RTX 5060 Laptop 8 GB)

| Run | Config | Workload | Decode Median (Peak) | Prefill | Hit Rate / Notes | Peak VRAM | Verdict |
|---|---|---|---:|---:|---|---:|---|
| Baseline | `llama.cpp` `ink-dedup` (`UD-Q3_K_XL`) | `bench.py` 6-prompt + 22K | `37.2 tok/s` (`32.1` @ 30.5K) | `174` (3.3K) / `106` (22K) | 94.3% hit, 48 syncs/tok, 3/3 | `7.6–7.8 GiB` | Baseline |
| M1a | `squid5060` CLOCK sync (`mtp=0`) | P1 coding (512 tok) | `43.25 tok/s` | — | 90.7% hit, 18.3K prom | `~7.4 GiB` | Keep |
| M1b | `squid5060` LRFU async (`mtp=0`) | P1 coding (512 tok) | `43.75 tok/s` (`44.00`) | — | 89.1% hit (loop 94.7%) | `~7.4 GiB` | Keep |
| M1c | `squid5060` CLOCK async (`mtp=3`, full vocab) | P1 coding (512 tok) | `49.54 tok/s` (`51.84`) | — | `acc=61.3%`, `dl=2.82` | `~7.4 GiB` | Keep |
| M1-neg | Heuristic `draft_vocab_32k.bin` (`mtp=3`) | P1 coding (512 tok) | `45.62 tok/s` | — | `acc=50.1%` (missed high-ID code toks) | `~7.4 GiB` | Reverted |
| M3a | **Dual-path PCIe `uint4` + SMEM-sc + 26-SM** (`mtp=0`, LRFU `998/150`) | P1 coding (512 tok) | **`60.31 tok/s`** (`62.14` / `63.08` @ `999/180`) | `372 ms` | `moe_dn 8.43 → 2.85 ms`, 88.7% hit, 973 prom | `~7.5 GiB` | **Keep (+38%!)** |
| M3b | Same (`mtp=1`, LRFU `998/150`) | P1 coding (512 tok) | `61.97 tok/s` (`65.42`) | `380 ms` | `acc=75.9%`, `dl=1.76`, 525 prom | `~7.5 GiB` | Keep |
| M3c | Same (`mtp=2`, LRFU `998/150` / CLOCK) | P1 coding (512 tok) | **`62.10 / 62.95 tok/s`** (**`66.94`**) | `369 ms` | `acc=67.7–69.9%`, `dl=2.35–2.39`, 456 prom | `~7.5 GiB` | **Best Code/MTP** |
| M3d | Same (`mtp=3` / `mtp=4`, LRFU) | P1 coding (512 tok) | `60.08` / `54.47 tok/s` | `374 ms` | Extra draft PCIe misses offset `dl` | `~7.5 GiB` | `mtp=2` dominates |
| **M4a** | **`squid5060 serve --mtp 2 --ctx 32768 --kv q4`** | **`bench.py` full suite** | `44.59 tok/s` (**`55.95`** @ 23.9K) | `210` / `514` / **`998 tok/s`** | **`3/3` Needle PASS**, `ok: true` | **`7.47 GiB`** | **Best Long-Ctx Decode** |
| **M4b** | **`squid5060 serve --mtp 0 --ctx 32768 --kv q4`** | **`bench.py` full suite** | **`55.01 tok/s`** (`61.31` peak, `49.03` @ 23.9K) | **`255` / `625` / `1,095 tok/s`** | **`3/3` Needle PASS**, `197.8 ms` TTFT, `ok: true` | **`7.47 GiB`** | **Default Winner** |
| **M4c** | **`squid5060 longctx --ctx 131072 --kv q4`** | **128K allocation & scaling** | `37.5–39.3 tok/s` (synth) | — | 1,156 cached slots (`4.94 GiB`) + 128K Q4 KV | **`<7.6 GiB`** | **128K Fits!** |
