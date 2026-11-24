# squid5060

**A custom, zero-dependency TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible API server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26B looped MoE) — engineered for 8 GB Blackwell GPUs (RTX 5060 Laptop / Desktop, `sm_120a`).**

Forked from [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090), `squid5060` brings native lossless NVFP4 Blackwell PTX inference, persistent CUDA Graphs, batched `B=512` Tensor Core prefill, built-in DSpark MTP speculative decoding, and up to **128K+ context** into **< 7.5 GiB VRAM** — beating tuned `llama.cpp` (`UD-Q3_K_XL` + MoE-cache loop dedup) by **+48% on decode** and **10.3× on 24K cold prefill**.

---

## Measured RTX 5060 Laptop (8 GB) Benchmarks vs. `llama.cpp`

Measured apples-to-apples on the same **NVIDIA GeForce RTX 5060 Laptop GPU** (`sm_120a`, 26 SMs, 32 MiB L2, 7.8 GiB usable VRAM, PCIe Gen5 x8) with the same benchmark harness (`c=1`, 6 diverse prompts × 512 decode, cold 3.3K & 12.9K random-filler prefill, 23.9K Needle-in-a-Haystack retrieval, continuous `nvidia-smi` peak VRAM monitoring):

| Metric (`c=1`, RTX 5060 Laptop 8 GB) | `llama.cpp` `ink-dedup` (`UD-Q3_K_XL`, 5.6 GiB MoE cache) | **`squid5060` (`nvfp4`, `q4` KV, `--mtp 0`, default)** | **`squid5060` (`nvfp4`, `q4` KV, `--mtp 2`)** | **`squid5060` Speedup vs. `llama.cpp`** |
|---|---:|---:|---:|---:|
| **6-prompt `c=1` decode median** | `37.2 tok/s` | **`55.01 tok/s`** *(up to `61.31`)* | `44.59 tok/s` *(up to `56.15`)* | **+47.9% faster** |
| **Coding prompt P1 (512 tok, 3-run median)** | ~38 tok/s | **`60.31 tok/s`** *(peak `63.08`)* | **`62.95 tok/s`** *(peak `66.94`, `dl=2.35`)* | **+66% – +76% faster** |
| **Short-prompt TTFT** | ~250–350 ms | **`197.8 ms`** | `224.0 ms` | **Sub-200 ms TTFT** |
| **Cold prefill (3.3K random filler)** | `174 tok/s` | **`255.1 tok/s`** *(12.9 s)* | `210.3 tok/s` | **+46.6% faster** |
| **Cold prefill (12.9K random filler)** | ~106–140 tok/s | **`624.7 tok/s`** *(20.6 s)* | `513.7 tok/s` | **~4.5–5.9× faster** |
| **Cold prefill (23.9K Needle haystack)** | `106 tok/s` *(210 s)* | **`1,094.7 tok/s`** *(21.8 s)* | `997.9 tok/s` *(23.9 s)* | **10.3× faster prefill** |
| **Long-ctx decode (23.9K context)** | `32.1 tok/s` | **`49.03 tok/s`** | **`55.95 tok/s`** *(`dl=2.48`)* | **+53% (`mtp=0`) / +74% (`mtp=2`)** |
| **Needle-in-a-Haystack accuracy** | `3/3` | **`3/3` PASS** | **`3/3` PASS** | **100% exact** |
| **Peak VRAM (`nvidia-smi`, incl. desktop)** | `7.6–7.8 GiB` | **`7.47 GiB`** *(1,423/3,072 experts cached)* | **`7.47 GiB`** *(1,371/3,072 experts cached)* | **Fits cleanly in 8 GB** |
| **Weight precision** | 3-bit GGUF re-quant | **Native lossless NVFP4** | **Native lossless NVFP4** | **No GGUF loss** |

*On 128K context (`--ctx 131072 --kv q4`), `squid5060` allocates all 131,072 tokens in 1.63 GiB runtime VRAM while keeping **4.94 GiB (1,156 physical experts)** resident in the VRAM MoE cache.*

---

## How `squid5060` Fits a 26B Looped MoE into 8 GB VRAM at 55–67 tok/s

Full `Inkling-2-Mini` NVFP4 weights are ~14.54 GiB — too large for an 8 GB GPU. `squid5060` automatically activates its **8 GB Unified Expert Cache** mode on any GPU with `< 15 GiB` VRAM (while preserving full-VRAM RTX 5090 mode when `>= 15 GiB` is detected):

1. **Compact Always-Resident Footprint (`0.60–0.78 GiB` weights + `0.70–0.74 GiB` runtime):**
   - Dense attention projections, controllers, routers, norms, and LM head stay resident in VRAM in native NVFP4 (`0.60 GiB` with `mtp=0`, `0.78 GiB` with DSpark MTP).
   - The BF16 embedding table (`0.86 GiB`) and DSpark `markov_w1` (`98 MiB`) are pinned in zero-copy mapped host RAM (`cudaHostAllocMapped`) — only 1 row/token is ever read.
   - Global KV caches default to `--kv q4` (group-64 symmetric INT4 + FP16 scales) — fitting 32K context in `0.70 GiB` runtime VRAM and 128K context in `1.63 GiB`.
2. **Unified 3,072-Expert VRAM Cache (`~5.85–6.12 GiB`, 1,371–1,432 physical slots):**
   - All 24 physical layers' 3,072 experts (including L20–23 converted from MXFP8 → NVFP4 on load in 32-expert GPU chunks) are packed into a single uniform `4,478,976 B` (`4.27 MiB`) slot layout (`w13_data`, `w13_scales`, `w2_data`, `w2_scales`, all 256-byte aligned so 1 DMA transfers an entire expert).
   - **Automatic 3× loop deduplication:** Because Inkling-2-Mini loops layers `6..17` three times per token (48 logical visits over 24 physical layers), loop experts naturally accumulate 3× touch frequency in our global Aging-LFU (`LRFU`, `decay=0.998`, `hysteresis=1.50`) cache — reaching **~95% hit rate in the 36 looped visits** with zero per-layer manual budgets.
   - **Zero-race async overlapped promotion:** A dedicated `copy_stream` overlaps H2D expert promotions with the next token's forward pass using 8 reserved staging slots and 1-token victim quarantine, with zero CUDA Graph invalidation.
3. **Dual-Path CTA-Uniform CUDA Kernels Saturating PCIe Gen5 x8 (96% of Hardware Ceiling):**
   - Inside every MoE kernel, `cslot = __ldg(slot_map + expert)` is uniform across the entire CTA (`1 expert per CTA`, eliminating cross-expert warp stalls and `__syncthreads` barriers).
   - **VRAM Hit (`cslot >= 0`):** Zero-SMEM `uint2` path (**>740 GB/s** on `w2`, **>1,270 GB/s** on `w13` across L2+GDDR7).
   - **PCIe Miss (`cslot < 0`):** 1-burst coalesced SMEM scale preload (`1,152 B` / `2,304 B`) + `uint4` (`128 B` / `256 B`) weight bursts hitting **23.7–23.9 GB/s** directly from pinned host RAM (**96% of the 24.8 GB/s PCIe Gen5 x8 DMA ceiling**; **+61% faster** than strided `uint2` on `moe_down`!).
   - Persistent `cudaGraphExec_t` stays completely intact across every decode token — **zero per-layer host sync stalls** (unlike `llama.cpp`'s 48 sync barriers/token).
4. **26-SM RTX 5060 Persistent Grid Tuning:**
   - `LM_HEAD_BLOCKS = 208` (`8×26` SMs), `DSPARK_UNEMBED_BLOCKS = 208`, and `MARKOV_BLOCKS = 104` (`4×26` SMs) eliminate wave tails (`lm_head` dropped from `0.95 ms` → `0.73 ms`).
5. **Batched `B=512` Blackwell Tensor Core Prefill + Multi-Checkpoint Prefix Cache:**
   - Reuses `squid5090`'s cuBLAS FP16 TC dense/controller/router GEMMs, `m16n8k16` FlashAttention, and grouped-GEMM NVFP4 TC MoE kernels with cache warming after each chunk — scaling from `255 tok/s` (3.3K) to **`1,095 tok/s`** (23.9K).
   - Keeps `squid5090`'s 4-checkpoint host-RAM prefix KV cache + 4K side-request shield (0 MiB extra VRAM) for fast multi-turn coding-agent reuse.

---

## Quickstart

### Dependencies
- **GPU:** NVIDIA Blackwell GPU (`sm_120a`, e.g. RTX 5060 8 GB / RTX 5070 / RTX 5090)
- **CUDA Toolkit:** CUDA 13.x (`nvcc` + `libcublas` + `libcudart`, default `/opt/cuda`)
- **Zig:** `0.17.1+`
- **Model:** [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (~20.2 GB safetensors)

### Build & Run
```bash
git clone https://github.com/dialectmaterial/squid5060.git
cd squid5060

# Build zig-out/bin/squid5060 (override CUDA path if installed at /usr/local/cuda)
zig build -Dcuda-path=/opt/cuda

# Download Inkling-2-Mini safetensors
hf download thinkingmachines/Inkling-2-Mini --local-dir /path/to/inkling-2-mini

# Start OpenAI-compatible HTTP/SSE server (auto-configures 8 GB cache, nvfp4, q4 KV, ctx=32768, mtp=0)
./zig-out/bin/squid5060 serve /path/to/inkling-2-mini --host 127.0.0.1 --port 8000

# Code-heavy workload with built-in DSpark MTP speculation (peaks ~66.9 tok/s)
./zig-out/bin/squid5060 serve /path/to/inkling-2-mini --host 127.0.0.1 --port 8000 --mtp 2

# One-shot CLI generation
./zig-out/bin/squid5060 generate /path/to/inkling-2-mini \
  --prompt "Write a Python binary search function with type hints, docstring, and unit tests." \
  --max-tokens 512 --ctx 32768 --mtp 0

# Benchmark (3-run or 6-run median)
./zig-out/bin/squid5060 bench /path/to/inkling-2-mini \
  --prompt "Write a Python binary search function with type hints, docstring, and unit tests." \
  --max-tokens 512 --runs 6 --ctx 32768 --mtp 0

# Long-context scaling sweep (128K in 8 GB VRAM)
./zig-out/bin/squid5060 longctx /path/to/inkling-2-mini --kv q4 --ctx 131072
```

### CLI Flags & Cache Tuning Environment Variables

| Option / Env Var | Default (8 GB GPU) | Description |
|---|---|---|
| `--mode` | `nvfp4` | Weight precision (`nvfp4` \| `fp8` \| `exact`) |
| `--kv` | `q4` | Global KV format (`q4` \| `fp8` \| `bf16`) |
| `--ctx` | `32768` (`serve`) | Max context length (tested up to `131072` in 8 GB VRAM) |
| `--mtp` | `0` (`serve` 8 GB), `3` (≥15 GB) | DSpark MTP draft count (`0` = best mixed-workload default; `2` = best for code/high-acceptance) |
| `--cache-mib` | `0` (auto) | Explicit VRAM MoE cache pool size in MiB (`0` = auto-fill remaining VRAM minus headroom) |
| `--cache-headroom-mib` | `180` | Free VRAM margin reserved above the MoE cache pool |
| `--expert-cache` / `--no-expert-cache` | auto (`< 15 GiB`) | Force unified VRAM expert cache on or off |
| `SQUID_CACHE_POLICY` | `lrfu` | Replacement policy (`lrfu` \| `clock`) |
| `SQUID_CACHE_ASYNC` | `1` | Overlapped `copy_stream` promotion with victim quarantine (`1` \| `0`) |
| `SQUID_CACHE_DECAY` | `998` (`0.998`) | Aging-LFU per-step decay (`800..1000` ÷ `1000`) |
| `SQUID_CACHE_HYST` | `150` (`1.50`) | Eviction hysteresis ratio (`100..500` ÷ `100`) |
| `SQUID_DRAFT_VOCAB` | `full` (unless `draft_vocab_32k.bin` present) | Draft head vocabulary (`full` \| `32k` \| `49k`) |
| `SQUID_API_KEY` | unset | Optional `Authorization: Bearer <key>` auth for `serve` |

---

## Repository Structure

```text
squid5060/
├── build.zig                 # Native Zig build system (parallel cached nvcc sm_120a + Zig ReleaseFast)
├── deploy/
│   ├── squid5060.service     # systemd service unit for 8 GB RTX 5060 OpenAI API server
│   └── squid5090.service     # Upstream 32 GB RTX 5090 systemd unit
├── README.md                 # Quickstart, RTX 5060 benchmarks & architecture guide
├── THEORY.md                 # Measured RTX 5060 Laptop hardware & PCIe/GDDR7 roofline analysis
├── PLAN.md                   # Optimization milestones & experiment ledger
├── cuda/                     # Pure Blackwell sm_120a CUDA kernels & C-ABI bridge
│   ├── config.hpp            # Shared geometry + unified 4.27 MiB NVFP4 expert slot offsets
│   ├── weights.hpp           # POD device-pointer descriptors (resident + cached MoE banks)
│   ├── kernels.hpp           # Kernel launcher declarations
│   ├── kernels_common.cuh    # Blackwell PTX (e2m1, e4m3, m16n8k16, ldmatrix), KV pack/unpack, TC Split-K
│   ├── kernels_gemv.cu       # Dual-path VRAM/PCIe MoE GEMV (23.9 GB/s PCIe), 26-SM lm_head
│   ├── kernels_inkling.cu    # Mega-fused depth-attn/sconv, controllers, router, TC attention
│   ├── kernels_dspark.cu     # DSpark 3-layer block drafter, TC global layer 1, 26-SM Markov RNN
│   ├── kernels_prefill.cu    # Batched B=512 TC FlashAttention + cached NVFP4 TC MoE prefill
│   └── cuda_abi.cu           # Thin extern "C" bridge exposing sm_120a kernels & CUDA runtime to Zig
└── src/                      # 100% TigerStyle Zig engine & OpenAI server
    ├── expert_cache.zig      # Unified 3,072-expert VRAM cache (LRFU/CLOCK, async copy_stream, telemetry)
    ├── weights.zig           # Two-pass resident VRAMArena + pinned zero-copy host expert banks
    ├── engine.zig            # Single runtime VRAMArena, 48-visit CUDA Graph orchestrator & prefix cache
    ├── server.zig            # Zero-malloc static-buffer POSIX HTTP/1.1 + SSE OpenAI API server
    ├── tokenizer.zig         # Flat open-addressed BPE tokenizer (32.8M tok/s) & Inkling chat template
    ├── safetensors.zig       # POSIX mmap safetensors loader
    ├── vram_arena.zig        # Single-cudaMalloc bump VRAMArena with paired exact-fill assertions
    ├── constants.zig         # Comptime-verified Inkling-2-Mini, DSpark & slot geometry invariants
    ├── cuda.zig              # Explicit extern "C" kernel & CUDA runtime bindings
    ├── tiger.zig             # Always-on release assertions, monotonic clock, ByteWriter, POSIX C-ABI
    └── main.zig              # Single-binary CLI entry point (serve, generate, bench, longctx, tokenize)
```

## Credits
Built on [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090). All upstream RTX 5090 (`>= 15 GiB`) full-VRAM paths remain intact and auto-selected when sufficient VRAM is present.
