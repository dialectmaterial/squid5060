# `squid5090` Technical Reference & Learnings

Complete consolidated engineering reference for `squid5090`: roofline analysis, Blackwell `sm_120a` decode & prefill kernels, Hybrid Prompt-Lookup + DSpark speculative decoding, 1M-context memory architecture, TigerStyle Zig systems design, an explicit audit against [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill) (what we applied *and* what remains unexploited), and the complete Fisher-LDA runtime-gated abliteration math and `uv` generation workflow.

*(For a high-level plain-English 3Blue1Brown-style overview, read [`LITE.md`](LITE.md). For quickstart build & install, see [`README.md`](README.md).)*

---

## 1. Benchmark Summary & Full Progression (Single RTX 5090, 32 GiB GDDR7)

### End-to-End Hill-Climb (`76.6 → 589.5 / 771.6 tok/s`, Levers 0–13 at dedicated `P0`, Lever 14 unified matrix at `P1`)

| Milestone / Lever | Core Change | Code (`c=1`) | Chat (`c=1`) | `mtp=0` | Prefill (20K / 64K) | Max Ctx / VRAM |
|---|---|---:|---:|---:|---:|---|
| **v0 Eager Baseline** | Full 48-visit Inkling-2-Mini in `exact` (BF16 + NVFP4/MXFP8) | — | — | 76.6 tok/s | ~900 tok/s | ~200K (`18.8 GiB`) |
| **Levers 1–3: Graphs + NVFP4 PTX** | Persistent `cudaGraphExec_t`, Blackwell `cvt.rn.f16x2.e2m1x2`, load-time NVFP4 quant (`5.47 → 2.80 GB/tok`), fused `qkvrg` | — | — | 190.1 tok/s | ~900 tok/s | ~256K (`15.0 GiB`) |
| **Levers 4–7: DSpark + 99%-BW GEMV + 1M TC Split-K** | Pruned 32K draft vocab (`868 → 428 µs`), half-warp `lm_head` (`1,497 GB/s`), zero-atomic deterministic MoE, `--kv q4` + host-pinned embed, unified TC Split-K FlashDecoding, `B=512` TC prefill | 369.6 tok/s | 307.3 tok/s | 243.8 tok/s | 5,077 / 4,637 tok/s | **1M** (`22.7 GiB`) |
| **Lever 8: TigerStyle Zig Consolidation** | Rewrote host engine, BPE tokenizer (`32.8M tok/s`), 2-arena `VRAMArena`, zero-malloc HTTP/SSE server in Zig (`build.zig`) | 369.6 tok/s | 307.3 tok/s | 243.8 tok/s | 5,077 / 4,637 tok/s | **1M** (`22.7 GiB`) |
| **Lever 9: Blackwell PDL + TC `mma.m16n8k16` Decode** | `cudaLaunchAttributeProgrammaticStreamSerialization` + `pdl_wait_fenced`, register-permuted TC `moe_up_tc` / 144-CTA `moe_down_tc` / dense NVFP4 GEMVs | 470.2 tok/s | 307.3 tok/s | **259.0–262.8 tok/s** | 5,077 / 4,637 tok/s | **1M** (`22.7 GiB`) |
| **Lever 10: Adaptive `{2,4,8}` Switch Graph + Overlap** | Markov online softmax confidence, CUDA 13.1 `cudaGraphCondTypeSwitch`, overlapped next-round drafting | 585.9 tok/s | **332.5 tok/s** | 259.0–262.8 tok/s | 5,077 / 4,637 tok/s | **1M** (`22.7 GiB`) |
| **Lever 11: `B=2048` WGEMM + 2D-Tiled Short Conv** | Multi-stage `cp.async` `128×128×64` grouped MoE GEMM + 576-CTA 4-tap FIR sconv (`-42.9 ms/chunk`) | 585.9 tok/s | **332.5 tok/s** | 259.0–262.8 tok/s | **8,755 / 7,742 tok/s** *(9,181 chunk peak)* | **1M** (`23.22 GiB`) |
| **Lever 12: Runtime-Gated Fisher-LDA `--ablation`** | 1.8 MB gated projection fused into decode & prefill inside single `VRAMArena` | **541.1 tok/s** *(ablated)* | **332.5 tok/s** *(0 diff 53/53)* | 259.0 tok/s | 8,755 / 7,742 tok/s | **1M** (`23.22 GiB`) |
| **Lever 13: Hybrid O(1) Prompt-Lookup + DSpark** | Zero-malloc 2-entry hash index (`8,192` ring, `min_match_len=16`), bypasses DSpark into `verify_b8_graph_exec` on verbatim matches | **589.5 tok/s** code<br>**771.6 tok/s** quote/RAG *(P0)* | **332.5 tok/s** *(P0)* | 259.0 tok/s | **8,755 / 7,742 tok/s** | **1M** (`23.22 GiB`) |
| **Lever 14: Rejection Sampling + Long-Ctx KLD + 512K Recall + Hardening** | Exact Leviathan/Chen rejection sampling (`q_i(v)` drafted in CUDA Graph + deterministic fixed-point top-p + residual resampling), FP32-accum FP8 GEMV, 32 MiB HTTP/chunked tokenizer, streaming FP64 `kld` eval (`7,784` tok), real 128K–512K needle recall | **547.0 / 485.1** (`T=0/1`, P1)<br>**610.5 / 518.5** copy (`T=0/1`, P1) | **315.7 / 279.3** (`T=0/1`, P1)<br>**308.4 / 272.6** prose (`P1`) | 248.5 tok/s *(P1)* | **5,988 / 4,575 / 3,062 tok/s** *(128K/256K/512K)* | **512K real PASS**<br>**1M** (`23.23 GiB`) |
| **Lever 15: Blackwell TC FP8 MMA + PDL (`serve` default)** | Added fused `qkvrg_fused_fp8_mma`, `linear_fp8_mma`, and `lm_head_fp8_mma` (`mma.m16n8k16.f32.f16` + PDL), boosting `fp8/q4` `T=1.0` code **`+66.4%`** (`253.9 → 422.4 tok/s`) while keeping official L0–23 MoE experts & `0.0061–0.0096` KLD (`ΔPPL <= +0.0035`); promoted `--mode fp8` to default `serve` mode | **427.3 / 422.4** (`fp8/q4`, `T=0/1`, P1)<br>**486.8 / 433.0** copy (`T=0/1`) | **252.3 / 226.2** (`fp8/q4`, `T=0/1`, P1)<br>**239.1 / 207.6** prose (`P1`) | 195.2 tok/s *(fp8)* | 5,988 / 4,575 / 3,062 tok/s | **512K** (`22.59 GiB`, `fp8/q4`)<br>**256K** (`22.65 GiB`, `fp8/fp8`) |

### Built-In Exact FP64 KLD vs. Official `exact/bf16`

#### A. Long-Context Reproducible Corpus — [`tools/kld_corpus.txt`](tools/kld_corpus.txt) (`7,784` tokens, 7.6× past 1,024-tok local window)
Directly stresses the 8 quantized global KV layers across Python, C++/CUDA, Zig, math, and cross-section recall (`ref_PPL = 2.7048`):

| Mode | KV Format | Mean KLD (nats) | Median KLD | p95 KLD | Top-1 Match | Candidate PPL | ΔPPL |
|---|---|---:|---:|---:|---:|---:|---:|
| `exact` | `bf16` | **0.000000** | 0.000000 | 0.000000 | **100.00%** | 2.7048 | +0.0000 |
| `exact` | `fp8` | 0.003509 | 0.000323 | 0.015505 | 98.39% | 2.7007 | -0.0041 |
| `exact` | `q4` | 0.007752 | 0.000891 | 0.034746 | 97.15% | 2.7063 | +0.0015 |
| **`fp8`** | **`fp8`** | **0.006143** | 0.000627 | 0.025082 | **97.42%** | 2.7019 | -0.0030 |
| **`fp8`** | **`q4`** | **0.009574** | 0.001189 | 0.042074 | **96.93%** | 2.7083 | +0.0035 |
| `nvfp4` | `fp8` | 0.022998 | 0.002948 | 0.095089 | 94.99% | 2.7485 | +0.0437 |
| **`nvfp4`** | **`q4`** | **0.026291** | 0.003453 | 0.110257 | **94.81%** | 2.7607 | +0.0559 |

#### B. Short-Context Reference (`529` tokens, inside 1,024-tok local window, `ref_PPL = 2.5555`)

| Mode | KV Format | Mean KLD (nats) | Median KLD | p95 KLD | Top-1 Match | Candidate PPL | ΔPPL |
|---|---|---:|---:|---:|---:|---:|---:|
| `exact` | `bf16` | **0.000000** | 0.000000 | 0.000000 | **100.00%** | 2.5555 | +0.0000 |
| `exact` | `fp8` | 0.002682 | 0.000061 | 0.013487 | 99.05% | 2.5645 | +0.0090 |
| `exact` | `q4` | 0.005170 | 0.000237 | 0.024269 | 97.73% | 2.5811 | +0.0256 |
| `fp8` | `bf16` | 0.003544 | 0.000118 | 0.016707 | 99.24% | 2.5616 | +0.0061 |
| **`fp8`** | **`fp8`** | **0.003031** | 0.000108 | 0.016044 | **98.49%** | 2.5706 | +0.0151 |
| **`fp8`** | **`q4`** | **0.004969** | 0.000307 | 0.023981 | **97.16%** | 2.5820 | +0.0265 |
| `nvfp4` | `bf16` | 0.015811 | 0.001053 | 0.075243 | 94.90% | 2.5761 | +0.0207 |
| `nvfp4` | `fp8` | 0.016548 | 0.001192 | 0.082989 | 96.03% | 2.5900 | +0.0345 |
| **`nvfp4`** | **`q4`** | **0.018413** | 0.001068 | 0.089071 | **95.84%** | 2.5952 | +0.0397 |

- **Takeaway:** Even 7.6× beyond the 1,024-token local window (`7,784` tokens), `q4` global KV adds only `+0.0032–0.0078` KLD (`ΔPPL <= +0.0043` in `fp8/q4`). Most of `nvfp4`'s KLD (`0.0184` short / `0.0263` long) comes from quantizing dense attention + L20–23 experts + `lm_head` to 4-bit; `--mode fp8 --kv q4` cuts KLD **2.9–3.7×** while still fitting 512K–1M contexts without host spill.

### Long-Context Scaling & Prefix Cache Latency (`--mode nvfp4 --kv q4`)

| Workload / Context Depth | Prefill Latency / Rate | Decode Speed (`c=1`) | Total VRAM | Notes |
|---|---:|---:|---:|---|
| **Short Code (1,024 tok)** | `29.0 ms` | **589.5 tok/s** P0 / **547.0** P1 (`T=0`), **444.9** (`T=1.0`) | **15.59 GiB** | `acc=73.1%` (`T=0`), `63.9%` (`T=1.0` rejection sampling) |
| **Verbatim Code Copy / RAG** | `59.7–91.2 ms` | **771.6 tok/s** P0 / **610.5** P1 (`T=0`), **518.5** (`T=1.0`) | **15.59 GiB** | Prompt-Lookup + DSpark (`acc=79.7–91.5%`) |
| **Chat / Prose (512 tok)** | `28.0 ms` | **315.7 / 308.4** (`T=0`), **279.3 / 272.6** (`T=1.0`, P1) | **15.59 GiB** | Dedicated P0 chat: `332.5 tok/s` |
| **Live 16.1K HTTP Turn 1 (warm prefix)** | **`71.9 ms`** (`16,033` cached) | **446.7 tok/s** (`dl=3.10`) | **23.22 GiB** | `107.4 ms` end-to-end TTFT; `32 MiB` static HTTP buffer |
| **Cold 20K (`19,978` tok)** | **`2,281.9 ms`** (`8,755 tok/s`) | **453.6 tok/s** | **16.71 GiB** | `9,181 tok/s` `B=2048` chunk peak |
| **Real 64K Needle (PASS)** | **`8,464.8 ms`** (`7,742 tok/s`) | **317.6–331.0 tok/s** | **16.71 GiB** | Verbatim needle recall PASS |
| **Real 128K Needle (PASS)** | **`21,887.8 ms`** (`5,988 tok/s`) | **166.6 tok/s** | **17.40 GiB** | Verbatim recall PASS (`COBALT-FALCON-7492`) |
| **Real 256K Needle (PASS)** | **`57,288.3 ms`** (`4,575 tok/s`) | **123.9 tok/s** | **17.90 GiB** | Verbatim recall PASS (`OBSIDIAN-HERON-3815`) |
| **Real 512K Needle (PASS)** | **`171,127.7 ms`** (`3,062 tok/s`) | **74.4 tok/s** | **20.30 GiB** | Verbatim recall PASS (`VERMILION-LYNX-9064`) |
| **Warm 82.5K turn (past 65K cap)** | **`62.4 ms`** (`82,454` cached) | **237.5 tok/s** | **23.22 GiB** | **>1,400× faster** than cold prefill |
| **Synthetic `longctx` 256K / 512K / 1M (`1,048,000`)** | *zeroed KV* | **`B=1`: 114.6 / 74.6 / 45.7 tok/s** (`P1`)<br>*`B=4 @dl=2.5` synthetic: 214.1 / 154.3 / 102.4 (`P1`), 108.0 (`P0`)* | **17.9 / 20.3 / 23.23 GiB** | Real KV alloc & step latency (`96.1` in `ledger.jsonl` was pre-PDL Lever 7C) |

---

## 2. Measured Hardware Ceilings & Inkling-2-Mini Roofline (`sm_120a`)

### 2.1 Measured RTX 5090 Hardware Ceilings
Measured directly on-device (dedicated `P0` state; background GPU workloads silently clamp the card to `P1` at `13,801 MHz` mem and cost ~10% decode speed):

| Metric | Measured Value | Notes |
|---|---:|---|
| GPU | NVIDIA GeForce RTX 5090 (`sm_120a`, Blackwell) | 170 SMs, 96.0 MiB L2, 100 KiB dynamic SMEM/SM |
| Memory | 32,607 MiB GDDR7 (512-bit, `14,001 MHz` DDR) | Spec peak `1,792.1 GB/s` |
| **Pure Read BW (`1 GiB`, `__ldg`)** | **`1,512.9 GB/s`** | **Real decode bandwidth ceiling** (`lm_head` hits `1,497 GB/s` = 99%) |
| D2D Copy BW (`1 GiB`) | `1,235.1 GB/s` | Read + write |
| BF16 Tensor Core GEMM (`4096³` / `8192³`) | `176.2` / `174.3 TFLOPS` | Sustained compute reference |
| Toolchain | CUDA 13.1 (`V13.1.115`) / Driver `590.48.01` | Native `sm_120a` NVFP4/FP8/PDL/Graph Switch |

### 2.2 Inkling-2-Mini Geometry & Resident Weight Footprint
Checkpoint: `thinkingmachines/Inkling-2-Mini` (`model-00001-of-00001.safetensors` = `19.53 GB`, `mtp.safetensors` = `0.64 GB`).
- **48 logical visits from 24 physical layers:** Prelude `0..5` (6 visits) → Looped Core `6..17` × **3 passes** (36 visits; physical weights reused 3×, distinct KV/sconv state per visit) → Coda `18..23` (6 visits). Includes AttnRes block-boundary depth attention, cross-pass low-rank controllers, and static + dynamic 4-tap short convolutions.
- **Hybrid Attention (why 1M context fits in `23.22 GiB`):** 24 Q heads, 8 KV heads (`head_dim=128`, `d_rel=16`, `rel_extent=1024`). 40 visits are local sliding-window (`W=1024`, fixed `1280`-slot BF16 ring = ~210 MB total); only **8 global visits** (`layers 5, 11, 17, 23`) store full context (`32.8 KB/tok` BF16 → `16.4 KB/tok` FP8 → **`~8.5 KB/tok` Q4**). Combined with zero-copy host-pinned BF16 embeddings (`0.86 GiB` saved), `1,048,576` tokens fit in `23.22 GiB` with `8.8 GiB` headroom.
- **MoE & DSpark:** 128 routed experts (top-8, `intermediate=1152`, softcapped SwiGLU `gate_cap=4, up_cap=25`), `hidden=2304`, vocab `200,058` (`201,024` padded). Built-in 3-layer DSpark MTP drafter (`intermediate=9216`, `block_size=8` → 7 drafts) + rank-256 Markov RNN head tapping target visits `[5, 23, 41, 48]`.

| Weight Component (Text + MTP) | `exact` | `fp8` Dense | `nvfp4` All-Linear |
|---|---:|---:|---:|
| MoE Experts L0–19 (NVFP4) | 11.466 GB | 11.466 GB | 11.466 GB |
| MoE Experts L20–23 | 4.204 GB (MXFP8) | 4.204 GB (MXFP8) | 2.293 GB (NVFP4) |
| Attention Dense + Router (24 layers) | 1.308 GB (BF16) | 0.654 GB (FP8) | 0.368 GB (NVFP4) |
| Controllers + Norms + SConv + AttnRes | 0.058 GB | 0.058 GB | 0.058 GB |
| Embedding + LM Head (`unembed`) | 1.852 GB (BF16) | 1.389 GB | 1.187 GB (BF16 + NVFP4) |
| DSpark MTP (3 layers + Markov) | 0.636 GB (BF16) | ~0.240 GB (NVFP4 + 32K head) | ~0.240 GB (NVFP4 + 32K head) |
| **Total Resident Weights** | **18.28 GiB** | **16.85 GiB** | **14.57 GiB** (`13.70 GiB` w/ host embed) |

### 2.3 Decode Bandwidth Roofline (`B=1` & Speculative `B=2..8`)

| Per-Token Traffic (`B=1`, short ctx) | `exact` (MB) | `fp8` (MB) | `nvfp4` (MB) |
|---|---:|---:|---:|
| Experts L0–19 (44 visits × 8/128) + L20–23 (4 visits) | 1,839.3 | 1,839.3 | 1,719.8 |
| Attention Dense + Router (48 visits) + Controllers (24 visits) | 2,700.0 | 1,392.0 | 820.0 |
| LM Head (`unembed`) | 926.3 | 463.2 | 260.5 |
| **Total Bytes / Token (`B=1`)** | **5,465.6 MB** | **3,694.5 MB** | **2,800.3 MB** |
| **Raw BW Ceiling (`1,512.9 GB/s` / `1,792 GB/s` spec)** | **277 / 328 tok/s** | **410 / 485 tok/s** | **540 / 640 tok/s** |

- **Why `nvfp4` dense + LM head mattered so much:** In the official `exact` checkpoint, BF16 attention + router + controllers + LM head are **66.3%** of all decode traffic (`3,626.3 MB`), dwarfing the MoE experts (`1,839.3 MB`). Quantizing them to NVFP4 nearly doubled the single-token ceiling before speculation even began.
- **Why speculation (`B=2..8`) breaks past `~265 tok/s`:** Across 48 visits with activation/KV/sconv/norm overhead, `B=1` saturates at `259–268 tok/s`. In a `B = K + 1` verify pass, dense weights and LM head are read **once** (`B×` amortized) and MoE experts deduplicate across consecutive correlated tokens (empirical unique-expert multiplier $u(4)\approx 2.8$, $u(8)\approx 4.5\text{--}4.9$ vs i.i.d. $3.64 / 6.45$):

| Speculative Regime (`nvfp4`) | Verify `B` | Accepted `a` | Est. `u(B)` | Verify + Draft MB | MB / Accepted Tok | **Roofline (`1,513 GB/s`)** | **Achieved `squid5090`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Autoregressive (`mtp=0`) | 1 | 1.0 | 1.0 | 2,800 | 2,800 | 540 tok/s | **262.8 tok/s** |
| Fixed `mtp=3` (prose/code) | 4 | 2.6–3.0 | 2.9 | 6,218 | ~2,100–2,390 | ~633–720 tok/s | **307–470 tok/s** |
| Adaptive `{2,4,8}` (code) | 2..8 | 3.8–5.2 | ~3.8–4.9 | ~8,000–9,658 | ~1,857 | ~815 tok/s | **589.5 tok/s** |
| Hybrid Prompt-Lookup (quote/RAG) | 8 | 5.4–6.5 | ~4.5 | ~8,820 *(0 draft)* | ~1,380–1,630 | ~928–1,096 tok/s | **771.6 tok/s** |

---

## 3. Blackwell Decode & Prefill Kernel Learnings

### Decode (`cuda/kernels_common.cuh`, `kernels_gemv.cu`, `kernels_inkling.cu`, `kernels_dspark.cu`, `cuda_abi.cu`)
- **NVFP4 PTX + streaming loads:** `cvt.rn.f16x2.e2m1x2` + `ld.global.nc.L1::no_allocate.L2::256B` hits `1,497 GB/s` (`99%` of measured ceiling) on half-warp `lm_head`. **CUDA 13.1 pitfall:** never enable `L2::cache_hint` (`ptxas` 13.1 codegen bug).
- **Blackwell PDL:** `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization` + `pdl_wait_fenced(zero)` + `pdl_trigger()` overlaps producer epilogues with consumer weight-load prologues across ~600 kernel boundaries per step (`mtp=0`: `243.8 → 262.8 tok/s`).
- **Register-permuted Tensor Core `mma.m16n8k16` for `M=1..8`:** Replaces CUDA-core `half2` FMA in `moe_up_tc_kernel`, 144-CTA fused `moe_down_tc_kernel`, `qkvrg_fused_mma`, `linear_nvfp4_mma` (`K ∈ {2304, 3072, 9216}`), and `lm_head_mma` (`mtp=3`: `369.6 → 470.2 tok/s`).
  - *Numerical note:* `B=1` (FP16-pair CUDA core) vs `B>=2` (FP32-accum TC MMA) can 1-ULP tie-break differently on long greedy rollouts; `B=2`, `B=4`, `B=8`, and adaptive `{2,4,8}` share the exact same FP32-accum path and match bit-for-bit.
- **Adaptive `{2, 4, 8}` CUDA Graph Conditional Switch + Overlapped Drafting:**
  - Markov head computes online softmax `(m, sum)` under softcap `30.0` alongside argmax to emit per-draft confidences `p_0..p_6`.
  - Device kernel `z_adaptive_select_width_kernel` chooses `B=8` (if `p_0, p_1, p_2 > threshold`), `B=4` (if `p_0 > threshold`), or `B=2` otherwise, and dispatches inside a single CUDA 13.1 `cudaGraphCondTypeSwitch` with zero host sync.
  - Next round's `draft_graph_exec` is enqueued behind `done_event` so drafting overlaps host D2H/SSE emission.
  - **Zero-rollback guarantee:** 16-slot modular sconv rings `(p & 15)` (`8 + 3 = 11 < 16`) + position-indexed KV/DSpark caches mean rejected tokens strictly sit ahead of `*d_pos = P + n_acc` and are overwritten on the next step.
- **Hybrid O(1) Prompt-Lookup + DSpark Drafter (`src/engine.zig`):**
  - Zero-malloc `PromptLookupIndex` (`8,192`-token ring + `16,384`-slot 2-entry hash table storing `(pos0, pos1)`, ~320 KiB host RAM, 0 VRAM).
  - **Key threshold lesson:** `min_match_len = 6` caused 37 false triggers on short Python test boilerplate (`assert cache.get("...`), regressing regular code from `586 → 565 tok/s`. Raising `min_match_len = 16` eliminated false triggers completely, lifting regular code to **`589.5 tok/s`** and verbatim code/RAG quotes from **`706.9 → 771.6 tok/s`** (`+9.2%`, `91.5%` acceptance, 100% bit-exact!).
  - Uses a `draft_in_flight` state machine: on a hit, a 28-byte FIFO H2D overwrites `d_drafts` and launches `verify_b8_graph_exec` (which still runs `commit_dspark = true` so DSpark stays synchronized), then sets `draft_in_flight = false` so every subsequent round in the matching passage skips `draft_graph_exec` completely!

### Prefill (`cuda/kernels_prefill.cu`)
- **`B=2048` Multi-Stage `cp.async` `128×128×64` Grouped MoE GEMM (`p_wgemm_kernel`):** Raises average tokens/expert `32 → 128`, uses 3-stage swizzled FP16 activation `cp.async` + 3–7-stage weight `cp.async` (~98–99 KiB SMEM) + `ldmatrix.x4` + FP16 MMA, contiguous sorted `Up → Down` handoff, and pigeonhole tile bound (`B<=512, TR=32` or `B<=2048, TR=128` → `<= 256` tiles, zero host sync). Cut 48-visit MoE prefill for 2,048 tokens to `92.2 ms` (`2.4×` faster).
- **2D Time-Tiled Short Conv (`p_sconv_post_res_kernel`):** `SQUID_PROF_PREFILL=1` revealed the 4-tap FIR sconv was running serially over `B=2048` on only `9` CTAs (`70.3 ms` = 26% of prefill!). Tiling across `SCONV_T_BLK = 32` (`576` CTAs) + a separate 9-CTA `p_sconv_ring_commit_kernel` (preventing the race where `(pos0 + 2047) & 15 == (pos0 - 1) & 15` overwrites CTA 0's read!) saved **`42.9 ms/chunk`** and jumped chunk prefill from `7,689 → 9,181 tok/s`!

### TigerStyle Zig Architecture (`src/`)
Measured head-to-head against our earlier C++20 host engine on the exact same GPU and CUDA kernels:
- **2 `cudaMalloc` bump arenas** (`1` weights + `1` runtime, paired `assert_exact_fill()`) vs `1,359` allocations (**680× fewer** driver allocs).
- **0 post-init / per-SSE-token host `malloc`s** (`ServerBuffers`), **243+ always-on release assertions**, `0.07 tok/s` run-to-run jitter.
- **Native Zig BPE tokenizer (`src/tokenizer.zig`):** `135 ms` cold mmap load (`1.84×` faster), **`32.8M tok/s`** encode (`2.05×` faster), 100% exact parity with HF `tokenizers`.
- **4-checkpoint host-RAM prefix KV cache + 4K global-KV side-request shield** (`0` extra VRAM): reuses 16K–82.5K prefixes in `62–72 ms` without short title/classifier side requests evicting the session.
- **Warning-free `zig build`:** treats any `nvcc` warning stderr as a build failure, enforcing clean kernel pruning.

---

## 4. Audit vs. [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill)

We used `inference-optimisation-skill` (0xSero / Sybil Solutions) as the core optimization methodology (Section 2 hardware ceilings → `PLAN.md` log → `ledger.jsonl` → single-variable A/B → long-context needles → greedy parity gates).

### What We Learned & Applied
- **Ceilings first (`ceilings.md`):** Measured real copy BW (`1,512.9 GB/s`) before writing kernels.
- **Lever ordering (`SKILL.md` §3, `levers.md`):** Decode graphs → native 4-bit quant (`NVFP4`) → 32K draft vocab pruning → GPU-side argmax/verify → decode-style TC Split-K verify attention → per-workload adaptive draft depth → explicit prefix caching (`cached_tokens > 0`) → stage-level prefill profiling (`SQUID_PROF_PREFILL=1`) → grouped MoE weight reuse → static memory arenas with headroom.
- **Measurement discipline (`methodology.md`):** Real tokenizer counts (not SSE chunks), dedicated `P0` vs. shared `P1` GPU state, `a / step_time` separation, 1-ULP `M=1` vs `M=k` numeric distinction, built-in FP64 KLD (`7,784` tok), and real 64K–512K verbatim needle verification.

### What We Did *Not* Apply Yet (Remaining Headroom)
1. **Native Blackwell FP8 / INT8 Tensor Core Prefill Dense GEMMs (`levers.md`):** `launch_prefill_gemm_h` currently dequantizes to FP16 scratch and runs FP16 `cublasGemmEx` (`dense_tc = 29.5 ms` at `B=2048`). Switching to native Blackwell FP8 `cublasLtMatmul` (labelled lossy trade) could shave `~12–15 ms/chunk` toward **`10,000+ tok/s`** prefill.
2. **SASS / `cuobjdump --dump-sass` ISA Audit (`levers.md`, `methodology.md`):** We tuned PTX/CUDA, but haven't yet inspected `sm_120a` SASS on `moe_down_tc_kernel` and `p_wgemm_kernel` for hidden register spills or bank conflicts.
3. **Multi-User Continuous Batching (`cN`, `SKILL.md` §3.4):** `squid5090` is intentionally single-slot (`c=1`) for peak responsiveness and 1M context. Supporting `c2`–`c4` requires paged global KV and per-slot sconv/local-KV ring banks.
4. **Automated Interleaved `control → candidate → control` Bootstrap-CI Harness & Full 1M Multi-Needle Soak (`methodology.md`, `publishing.md`):** We ran 3–6-run medians, 64K/128K/256K/512K single-needle PASSes (50% depth), and 1M `longctx` scaling, but haven't yet run an automated bootstrap-95%-CI harness or a real 1,048,000-token multi-depth needle soak.
5. **Wider Prompt-Lookup Verify Graph (`B=16`) + Early-Exit Markov RNN:** Extending prompt-lookup hits from `B=8` (`771.6 tok/s`) to `B=16` could push long code-copy/RAG bursts past `1,000 tok/s`, and early-exiting Markov steps `3..6` on low draft-0 confidence could save `~100–150 µs` on chat steps.

*(Stack-specific items in the skill that do not apply: Triton/Inductor/DeepGEMM JIT caches — we are AOT native Zig/CUDA with `~3 s` startup; vLLM/SGLang/llama.cpp flags; Intel Xe/AMD ROCm/unified-memory cgroup workarounds; external FA2/oneDNN swaps incompatible with Inkling-2-Mini's L2-clipped Q/K/R + rel-bias + sinks + Q4 KV.)*

---

## 5. Runtime-Gated Fisher-LDA Abliteration (`--ablation` & `tools/ablate.py`)

Adapted from [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (Apache-2.0).

### Why Weight-Space Abliteration Fails on Inkling-2-Mini
1. **Looped weight sharing:** Physical layers `6..17` run 3× as 36 logical visits; editing physical weights corrupts coding/reasoning passes alongside refusal passes.
2. **4-bit quantization floor:** Re-quantizing rank-1 edits back into NVFP4 rounds them to zero or adds noise.
3. **Unconditional harmless drift:** Always subtracting $r$ shifts harmless token distributions even when no refusal is occurring.

### The Per-Visit Gated Residual-Write Formula
Instead of modifying the 19 GB weights, `squid5090` applies a 1.8 MB sidecar after `post_attention_convolution` and `post_mlp_convolution` at each of the **48 logical visits**:

$$y \leftarrow y - w_v \cdot \mathbb{I}\!\left(d_v \cdot y > \tau_v\right) \cdot \max\!\left(r_v \cdot y - c_v,\; 0\right) \cdot r_v$$

- **$r_v \in \mathbb{R}^{2304}$ (Projected Refusal Direction, unit norm):** Harmful minus harmless mean read activations (`336` prompts each from `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`), captured at both direct-answer and reasoning starts, orthogonalized against the harmless mean ($\text{diff} - (\text{diff}\cdot \hat{h})\hat{h}$), and blended by `dir_mix`.
- **$c_v = \mathbb{E}_{\text{harmless}}[r_v \cdot y]$:** Harmless mean offset so $\max(r_v \cdot y - c_v, 0)$ subtracts only excess refusal projection.
- **$d_v \in \mathbb{R}^{2304}, \tau_v$ (Shrinkage Fisher-LDA Detector):** Solved from the full `2304 × 2304` FP64 harmless write covariance $\Sigma_v$ (harmless chat + reasoning + general text) and 4 harmful-minus-harmless write displacements $\delta_{m,v}$:
  $$s_m = \left(\Sigma_v + 0.1\,\overline{\operatorname{diag}(\Sigma_v)}\,I\right)^{-1}\delta_{m,v}, \quad u_v = \sum_{m=1}^{4}\frac{s_m}{\sqrt{\delta_{m,v}^\top s_m}}, \quad d_v = \frac{u_v}{\sqrt{u_v^\top \Sigma_v u_v}}$$
  Because $\operatorname{Std}_{\text{harmless}}(d_v \cdot y) \equiv 1$, setting $\tau_v = \mathbb{E}_{\text{harmless}}[d_v \cdot y] + k_{\text{det}}$ ($k_{\text{det}} \in [1.5, 4.5]$) fires only when a write stands out by $>k_{\text{det}}$ harmless standard deviations — leaving normal coding/chat **bit-for-bit identical** (`53/53` verified on harmless chat).
- **$w_v$ (Heretic Linear-Decay Kernel) + Optuna TPE:** Two 4-param triangular visit kernels `(max_w, pos, min_frac, dist)` for attention & MLP + `dir_mix` + `k_det` (10 parameters total). Optuna minimizes refusals (evaluated fast via `squid5090 serve --ablation`) and capability damage (`max(KL_answers, KL_text)` in BF16 reference `tools/inkling_ref.py`).
- **CUDA Fusion:** Packed as `[48, 4]` `float4` `{w, c, floor, tau}` inside the single weight `VRAMArena` and fused into decode's 576-thread mega-kernels (`apply_ablation_576`) and prefill's `p_ablate_post_res_kernel` (`541.1 tok/s` ablated code decode).

### Generating Your Own Abliterated `.safetensors` with `uv`

Core `squid5090` requires zero Python. The offline generator in `tools/` runs cleanly via [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Sync isolated offline environment (CUDA 13.0 PyTorch + Inkling-2-Mini transformers + Optuna)
uv sync --project tools

# 2. Build squid5090 (defaults to -Doptimize=fast / ReleaseFast)
zig build

# 3. Run the end-to-end pipeline
export SQUID_MODEL_DIR=/opt/models/inkling-2-mini
WORK=work/ablate-run1 && mkdir -p "$WORK"

uv run --project tools tools/ablate.py directions --out "$WORK/directions.pt"
uv run --project tools tools/ablate.py directions --out "$WORK/directions_think.pt" --think
uv run --project tools tools/ablate.py stats      --work "$WORK"
uv run --project tools tools/ablate.py baseline   --out "$WORK/baseline_val.pt" --split val
uv run --project tools tools/ablate.py baseline   --out "$WORK/baseline.pt" --split report
uv run --project tools tools/ablate.py search     --work "$WORK" --batches 12 --n 8
uv run --project tools tools/ablate.py show       --work "$WORK" --top 20
uv run --project tools tools/ablate.py export     --work "$WORK" --trial <N> --out "$WORK/custom-abliterated.safetensors"
uv run --project tools tools/ablate.py verify \
  --ablation "$WORK/custom-abliterated.safetensors" \
  --baseline "$WORK/baseline.pt" \
  --server ./zig-out/bin/squid5090 \
  --out "$WORK/verify.json"

# 4. Serve & A/B test 32 mild over-refusal prompts (zero extra deps via uv PEP 723)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 --ablation "$WORK/custom-abliterated.safetensors"
uv run tools/refusal_eval.py --base-url http://127.0.0.1:8000/v1 --label ablated --out "$WORK/refusals-ablated.json"
uv run tools/refusal_eval.py --compare "$WORK/refusals-original.json" "$WORK/refusals-ablated.json"
```

---

## 6. Repository Map

```text
squid5090/
├── build.zig                 # Native Zig build (parallel content-hashed nvcc sm_120a + Zig .fast)
├── LICENSE / NOTICE          # Apache-2.0 license and complete open-source attributions
├── README.md                 # Benchmarks, KLD frontier, & Linux build/install quickstart
├── LITE.md                   # Plain-English 3Blue1Brown-style visual guide to optimization & abliteration
├── LEARNINGS.md              # Consolidated technical reference, hardware ceilings & skill audit
├── PLAN.md / ledger.jsonl    # Historical archive (v0–Lever 7C) & machine-readable experiment ledger
├── deploy/squid5090.service  # Persistent systemd service unit
├── cuda/                     # Native Blackwell sm_120a CUDA kernels & C-ABI bridge
│   ├── kernels_common.cuh    # Blackwell PTX (e2m1, e4m3, m16n8k16, ldmatrix, PDL), Q4/FP8 KV, TC Split-K
│   ├── kernels_gemv.cu       # PDL + TC mma.m16n8k16 MoE & dense NVFP4/FP8 GEMVs, rejection sampling, 99%-BW lm_head
│   ├── kernels_inkling.cu    # Mega-fused depth-attn/sconv/ablation, 1-launch controllers, router, attention
│   ├── kernels_dspark.cu     # DSpark 3-layer drafter, TC base logits, folded Markov head + online softmax
│   ├── kernels_prefill.cu    # B=2048 128x128x64 cp.async WGEMM, 2D-tiled sconv, FlashAttention, ablation
│   └── cuda_abi.cu           # Thin extern "C" bridge + CUDA 13.1 cudaGraphCondTypeSwitch manager
├── src/                      # 100% TigerStyle Zig engine & zero-malloc OpenAI server
│   ├── tiger.zig / constants.zig / cuda.zig / vram_arena.zig / safetensors.zig
│   ├── tokenizer.zig         # Flat open-addressed BPE tokenizer (32.8M tok/s, chunked >2 MiB) & Inkling template
│   ├── weights.zig           # Two-pass single-VRAMArena weight + optional ablation loader
│   ├── engine.zig            # Single runtime VRAMArena, Switch Graph, PromptLookupIndex, Prefix Cache, streaming logits
│   ├── server.zig            # Zero-malloc static-buffer POSIX HTTP/1.1 + SSE OpenAI API server (32 MiB)
│   └── main.zig              # Single-binary CLI (serve, generate, bench, longctx, logits, kld, tokenize)
└── tools/                    # Reproducible KLD corpus + optional offline uv Python abliteration & refusal-eval toolchain
    ├── kld_corpus.txt        # 7,784-token multi-domain KLD evaluation corpus (>1K global KV)
    ├── pyproject.toml        # uv project (CUDA 13.0 PyTorch + Inkling-2-Mini transformers + Optuna)
    ├── ablate.py / inkling_ref.py / refusal_eval.py
```
