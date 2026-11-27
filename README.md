# squid5060

[![Release](https://img.shields.io/github/v/release/dialectmaterial/squid5060?color=34d399&label=version)](https://github.com/dialectmaterial/squid5060/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/Blackwell-sm__120a%20%C2%B7%208%20GB%20VRAM-76b900)](THEORY.md)
[![Stack](https://img.shields.io/badge/Zig%20%2B%20CUDA-Single%20Binary-f7a41d)](build.zig)

### A 26.4B frontier MoE + vision model — running at 100+ tokens/second on an 8 GB laptop GPU.

[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) weighs **14.5 GB** even in 4-bit — almost twice the VRAM of an RTX 5060. Conventional engines either refuse to load it or stall waiting on PCIe transfers.

**`squid5060`** is a standalone, zero-Python inference engine and OpenAI-compatible server written in Zig and Blackwell CUDA (`sm_120a`). It fits the full model, speculative drafting, on-demand vision, and up to **256K context** inside **< 7.5 GiB VRAM** — turning an 8 GB laptop or desktop GPU into a genuinely fast local coding and vision workstation.

| **103 – 107 tok/s** | **256K context** | **286 / 295** | **0.22 GiB idle** |
| :---: | :---: | :---: | :---: |
| Single-stream decode (`136 tok/s` peak, `2.9–3.9×` vs `llama.cpp`) | Grows lazily `32K → 256K` in `< 7.5 GiB` (`240K` needle PASS) | Spark-Bench quality (`195/200` GSM8K, `20/20` long-ctx) | Auto-sleeps when idle, wakes in `~330 ms` |

![squid5060 RTX 5060 Laptop 8 GB Benchmarks vs llama.cpp](assets/benchmark-5060.png)

---

## Built for Real Daily Use

Most high-speed engines assume a dedicated datacenter card. `squid5060` is designed to live comfortably on the same machine you actually work and play on:

- **One binary, zero Python:** Builds with `zig build` into a single native executable serving OpenAI-compatible HTTP/SSE endpoints (`/v1/chat/completions`, `/v1/completions`, tool calling, reasoning, and vision).
- **Gives your GPU back automatically:** With `--idle-sleep`, `squid5060` parks your active chat to RAM and drops VRAM from **`7.47 GiB → 0.22 GiB`** when idle — freeing over 7 GiB for games or creative apps, then waking seamlessly in **~330 ms** on the next request.
- **256K context without the speed tax:** Starting with `--ctx 262144` only allocates a 32K KV cache at boot so short chats keep all **1,470 hot experts** in VRAM. Longer sessions grow lazily (`32K → 64K → 128K → 256K`) mid-conversation without losing cached tokens, then shrink back automatically on fresh chats.
- **Snappy agentic coding & multi-chat:** Streams `tool_calls` arguments live token-by-token (zero UI freeze during file writes), preserves up to **8 conversations** in host RAM (`~11–28 ms` swap), and keeps an **8-slot rolling prefix cache** so follow-ups, branch rollbacks, and re-serialized tool history don't re-prefill.
- **On-demand vision at 0 MiB VRAM:** Pass PNG/JPEG images via OpenAI `image_url`, Responses `input_image`, Anthropic base64, or CLI `--image`. The 23-layer ViT borrows temporary scratch space, releases it immediately (`+0.0 MiB` permanent VRAM), and reuses multi-turn image prefixes in **`0.0 ms`**.

---

## Quickstart

### Requirements
- **Linux x86_64** with an NVIDIA Blackwell GPU (`sm_120a`: RTX 5060 / 5070 / 5080 / 5090)
- **CUDA Toolkit 13.0+** (`13.4+` recommended for `L2::evict_first` hints), `g++`, **Zig 0.17.1+**, `libpng16`, `libjpeg`
- **Model weights:** [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (~20.2 GB safetensors)

### 1. Build & Download
```bash
git clone https://github.com/dialectmaterial/squid5060.git
cd squid5060

# Builds ReleaseFast binary at ./zig-out/bin/squid5060
zig build

# Download official Inkling-2-Mini weights
hf download thinkingmachines/Inkling-2-Mini --local-dir /path/to/inkling-2-mini
```

### 2. Start the OpenAI-Compatible Server
```bash
# 8 GB profile: lazy 32K->256K context, auto-sleep after 5 min idle, 4 parked chats
./zig-out/bin/squid5060 serve /path/to/inkling-2-mini \
  --host 127.0.0.1 --port 8000 \
  --ctx 262144 --idle-sleep 300 --slot-keep 4
```

Point any OpenAI-compatible client (Open WebUI, Pi, OpenCode, `curl`, Python SDK) at `http://127.0.0.1:8000/v1`.

### 3. Or Run One-Shot CLI (Text & Vision)
```bash
# Text generation
./zig-out/bin/squid5060 generate /path/to/inkling-2-mini \
  --prompt "Write a clean Python implementation of Dijkstra's shortest path algorithm." \
  --max-tokens 512 --mtp 7

# Image understanding (0 MiB permanent VRAM)
./zig-out/bin/squid5060 generate /path/to/inkling-2-mini \
  --image /path/to/screenshot.png \
  --prompt "Read the text and summarize what is shown in this image." \
  --max-tokens 256 --mtp 7

# Manual VRAM sleep / wake / status (GET never wakes the GPU)
curl http://127.0.0.1:8000/v1/vram
curl -X POST http://127.0.0.1:8000/v1/vram -H 'Content-Type: application/json' -d '{"action":"sleep"}'
```

---

## Benchmarks & Quality

Measured strictly single-stream (`c=1`, zero multi-user batching) on an **RTX 5060 Laptop 8 GB** (`sm_120a`, 26 SMs, 7.8 GiB usable VRAM, PCIe Gen5 x8) + **Intel Core Ultra 7 255H** (16 cores, DDR5, AC power), using the same OpenAI-SSE `bench.py` harness and continuous `nvidia-smi` peak VRAM monitoring:

| Benchmark (`c=1`, RTX 5060 Laptop 8 GB) | `llama.cpp` (`UD-Q3_K_XL`) | `squid5060` v3.1 | **`squid5060` v3.8** | **Gain vs `llama.cpp`** |
|---|---:|---:|---:|---:|
| **6-prompt decode median (`T=0.0` / `T=0.6` / `T=1.0`)** | `37.2 tok/s` | `82.4 / 80.0 / 80.9` | **`107.1 / 103.1 / 102.0 tok/s`** | **2.74× – 2.88×** |
| **Peak single-prompt decode (SSE / Spark-Bench)** | `37.9 tok/s` | `105.4 tok/s` | **`136.2` / `161.9 tok/s`** | **3.59× – 4.27×** |
| **Long-ctx Needle decode (`18.8K` / `30K` 3-depth)** | `32.1 tok/s` | `97.9 tok/s` | **`125.9` / `117.7 tok/s`** | **3.67× – 3.92×** |
| **Sampled code-edit decode (`11.4K`, `T=1.0`)** | — | — | **`142.7 tok/s`** *(+29.4% prompt-copy)* | **Zero bias** |
| **Cold prefill (`3.3K` / `12.9K` random filler)** | `174 / 168 tok/s` | `1,182 / 1,793 tok/s` | **`1,116 / 1,859 tok/s`** | **6.4× / 11.1×** |
| **Long-ctx Needle prefill (`24K`)** | `149 tok/s` *(160s)* | `2,025 tok/s` *(11.8s)* | **`2,061 tok/s`** *(`2,248` peak, **11.6s**)* | **13.8×** |
| **240K Needle retrieval (`240,104` tok in 8 GB)** | OOM / spill | — | **`PASS` @ `45.5 tok/s`** *(0 KV spill)* | **Fits in 8 GB** |
| **Peak / Idle-Sleep VRAM** | `7.35 GiB` / — | `7.47` / `0.22 GiB` | **`7.47 GiB` / `0.22 GiB`** *(~329 ms wake)* | **7.25 GiB freed** |

### Quality Verified — Not Quantized to Mush
- **Spark-Bench (`knapcio/inkling-2-mini-spark-bench`, 295 items):** **`286 / 295`** (`GSM8K 195/200`, `qeval 71/75`, `long-context 20/20` — median **`127.2 tok/s`**, peak **`161.9 tok/s`**).
- **Multi-depth Needle-in-a-Haystack:** `18.8K` (`3/3`), `29.9K` at `10% / 50% / 90%` depths (`3/3`), `128K` (`PASS @ 63.2 tok/s`), and `240K` (`PASS @ 45.5 tok/s`, lossless mid-session `32K → 64K` grow).
- **Exact FP64 KLD (`7,784` tokens):** `0.000000` self-determinism; **`mean_KLD = 0.009136`**, **`top1 = 96.97%`**, **`ΔPPL = -0.0088`** for full NVFP4 + Q4 KV + Hybrid AVX-VNNI + Blackwell TC MoE (actually slightly closer to BF16 reference perplexity than legacy FP16-accum GPU-only).

---

## Under the Hood: How 14.5 GB Fits in 8 GB at 100+ tok/s

### The Core Insight: Looped Experts + Parallel CPU/GPU Misses

`Inkling-2-Mini` has **24 physical layers**, but executes **48 logical visits**: layers `0–5` (prelude), layers `6–17` looped **3 times** (36 visits), and layers `18–23` (coda).

That geometry changes everything on an 8 GB card: **75% of all MoE compute hits only 50% of the physical experts.** Keeping the hottest ~1,470 of 3,072 experts resident in a `6.13 GiB` VRAM cache yields a **`93–96%` hit rate** inside the looped core.

For the remaining ~5–9% of experts that miss VRAM and live in pinned host RAM, `squid5060` refuses to bottleneck on the `24 GB/s` PCIe bus. Instead, it splits misses across **both** memory buses simultaneously — **12 CPU cores executing AVX-VNNI directly out of DDR5 (~52–60 GB/s)** in parallel with **GPU PCIe zero-copy (~24 GB/s)** — delivering **~75–82 GB/s** of combined miss bandwidth (`3.1×` PCIe alone) and reaching **`93.5%` of the physical hardware roofline** on `B=1` decode (`10.01 ms/tok`).

### Key Engineering Pillars

1. **Unified 3,072-Expert VRAM Cache & Built-in Warm Profile**
   - Packs all 24 layers' NVFP4 experts (converting L20–23 MXFP8 → NVFP4 on load) into uniform `4.27 MiB`, 256-byte-aligned slots (`1,470` slots at 32K; `1,215` at 256K).
   - Ships with a built-in multi-domain hot-expert profile (`src/hot_experts.zig`) bulk-DMA'd in ~270 ms so Prompt #1 starts warm even on a fresh install, then refined with aging LRFU (`decay=0.985`, hysteresis `1.50`), instant bulk-DMA post-prefill tail refill, and zero-race async promotion on a dedicated `copy_stream`.
2. **Hybrid AVX-VNNI CPU + Blackwell Tensor Core GPU MoE**
   - Persistent 12-thread (`6P+6E`) C++20 worker pool (`cuda/cpu_moe.cpp`) using Q8-activation NVFP4 `_mm256_dpbusd_avx_epi32` AVX-VNNI kernels, 32-row atomic work-stealing, `WAITPKG` (`umwait`) low-latency spin/nap (0% idle CPU), and a CUDA-Graph-safe mapped-memory doorbell with cost-balanced `2/3` miss splitting.
   - On GPU hits, `B=1..3` uses half-warp PDL+L2 kernels while `B>=4` dispatches Blackwell `mma.m16n8k16` Tensor Core MoE (`42` regs, **0 register spills** across all 551 audited kernels) with CUDA 13.4+ `L2::evict_first` streaming hints.
3. **5-Way Adaptive DSpark Speculation (`{1, 2, 3, 4, 8}`) + Sampled-Safe Prompt-Copy**
   - Embeds a corpus-frequency-ranked 32K draft shortlist (`src/draft_shortlist.bin`, `32,834` tokens) directly in the binary — saving ~245 MiB VRAM (+57 expert slots) and speeding up drafting ~4×.
   - Inside CUDA 13's `cudaGraphCondTypeSwitch`, selects verify width from `{1, 2, 3, 4, 8}` using hardware-calibrated step costs `{1.00, 1.45, 1.70, 2.05, 3.20}` (`gamma=1.50`, temperature slope `0.40`), PDL + early-exit Markov RNN drafting, and exact Leviathan/Chen rejection sampling with always-on `RSFIX` positive-mass fallback.
   - On long prompts (`>= 8K`), sampled-safe Prompt-Copy (64K history, 16-token exact match, exact `q=1.0` point-mass verification) adds **`+29.4%`** on verbatim code edits at `T=1.0` with zero distribution bias.
4. **1-Launch PDL Blackwell Cluster Attention & `sm_120a` Block-Scaled Prefill**
   - Single-launch `attn_cluster_decode_kernel` uses Blackwell thread-block clusters (`CS=4` local / `CS=8` global on 26-SM RTX 5060), DSMEM barriers, and Programmatic Dependent Launch (PDL) overlap across QKV, attention, norms, and DSpark, paired with 3 KB SMEM L1-cached LM-head MMA and 128-CTA PDL stochastic sampling.
   - Prefill uses native `sm_120a` `m16n8k64` block-scaled NVFP4 Tensor Core grouped MoE (`BK=256`, 128B coalesced PCIe cachelines, preloaded scales) + dynamic tail sizing and reverse-causal FlashAttention.
5. **Lazy `32K → 256K` Tail-Aliased Global KV & Shared-Loop KV**
   - Combines FP8 local SWA KV with training-free Last-Loop Shared Global KV (1,024-token native tail rings on Loops 1 & 2, shared bulk on Loop 3) and caps DSpark's global KV at 32K on 8 GB GPUs.
   - Global KV starts at `32K` (`0.32 GiB` runtime VRAM) and tail-aliases `64K → 128K → 256K` slabs onto the expert cache pool on demand with lossless mid-session host-bounce preservation — zero host KV spill even at `240K`.
6. **Agentic UX, LoopCD-Hidden & Zero-VRAM Vision**
   - **8-slot rolling prefix cache** (sub-4K 512-tok grid + rolling turn checkpoints + `ToolRawHistoryCache` immune to client JSON re-serialization), **live incremental SSE `tool_calls` streaming**, and **4–8 host-RAM parked sessions** (`11–28 ms` swap).
   - **Zero-overhead LoopCD-Hidden (`arXiv:2610.02185`):** Optional `--loopcd-omega` / `--loopcd-ref` contrastive decoding via tail-aliased scratch (`0.0 MiB` decode VRAM), keyed in prefix and parked caches.
   - **Lazy 0-VRAM Vision:** 23-layer BF16 ViT + `2×2` merge projector (`cuda/vision.cu`) borrowing prefill scratch + content-addressed negative synthetic image token IDs for `0.0 ms` Turn 2+ cache reuse.

---

## Configuration & CLI Reference

On GPUs with `< 15 GiB` VRAM, `squid5060` automatically enables the 8 GB Unified Expert Cache profile (`--mode nvfp4 --kv q4 --mtp 7 --spec-threshold 0.90 --embed-host --cpu-moe auto`). Larger Blackwell GPUs (`>= 15 GiB`, e.g. RTX 5090) automatically run full-VRAM mode with dynamic SM-aware grids.

<details>
<summary><strong>Click to expand CLI flags, environment overrides & server endpoints</strong></summary>

### CLI Flags

| Flag / Env | Default (`< 15 GiB` `serve`) | Description |
|---|---|---|
| `--version` / `-V` | — | Print `squid5060` version and exit |
| `--mode` | `nvfp4` (`fp8` on ≥15 GiB) | Weight precision: `nvfp4` \| `fp8` \| `exact` (`exact` requires `--no-expert-cache`) |
| `--kv` | `q4` (`auto` on ≥15 GiB) | Global KV cache format: `q4` \| `fp8` \| `bf16` |
| `--ctx` | `32768` (pass `262144` for lazy 256K) | Context ceiling; `serve` grows global KV lazily `32K → 64K → 128K → 256K` |
| `--mtp` | `7` (`0` = off) | DSpark speculative drafter (`7` = 5-way `{1,2,3,4,8}` adaptive switch + prompt-copy) |
| `--spec-threshold` | `0.90` (`0.80` on ≥15 GiB) | Adaptive verify confidence threshold (`0.0` = fixed `B=8`) |
| `--cpu-moe` | `auto` (`on` on <15 GiB) | Hybrid AVX-VNNI CPU+GPU miss execution: `auto` \| `on` \| `off` |
| `--idle-sleep` | `0` (disabled) | Seconds of inactivity before freeing ~7.25 GiB VRAM (`7.47 → 0.22 GiB`) |
| `--slot-keep` | `4` (max `8`) | LRU host-RAM parked conversations preserved across switches & sleep/wake |
| `--image` | none (`generate` only) | PNG or JPEG image path (`0 MiB` permanent VRAM) |
| `--temp` / `--top-p` / `--top-k` | `1.0` / `0.95` / `0` | GPU sampling defaults (overridable per-request, plus `chat_template_kwargs`) |
| `--loopcd-omega` / `--loopcd-ref` | `0.0` / `1` | LoopCD-Hidden contrastive strength & reference loop `1\|2` |
| `--api-key` / `--api-key-file` | `$SQUID_API_KEY` | Optional constant-time `Authorization: Bearer <key>` auth (mode-600 key file) |
| `--cache-mib` / `--cache-headroom-mib` | auto / `160` | Override VRAM expert cache budget or reserved desktop headroom |
| `--cache-file` / `--no-warm-cache` | `<model_dir>/.squid5060-warm-cache.bin` | Warm-cache LRFU snapshot path (falls back to built-in profile) or disable |
| `--ablation` | none | Optional 1.8 MB runtime-gated Fisher-LDA `.safetensors` (see [`tools/ablate.py`](tools/ablate.py)) |

### Selected Environment Overrides
- `SQUID_DRAFT_VOCAB=32k|49k|full` — Override embedded 32K DSpark draft shortlist.
- `SQUID_CPU_MOE_THREADS`, `SQUID_CPU_MOE_SPLIT` — Override hybrid CPU thread count (default `12 = 6P+6E`) or miss split ratio (default `auto`, `2/3` CPU).
- `SQUID_PCOPY_MIN_CTX` — Minimum context length for sampled-safe prompt-copy (default `8192`).
- `SQUID_LOCAL_KV_BF16=1` / `SQUID_SHARE_LOOP_KV=0` — Disable FP8 local SWA or last-loop global KV sharing.
- `SQUID_BLOCK_VERIFY=1` / `SQUID_PREFILL_Q64=1` — Opt-in experimental Sun et al. Block Verification / `Q_TOK=64` prefill attention.

### HTTP Endpoints
- `POST /v1/chat/completions` — OpenAI chat completions (SSE streaming, live `tool_calls` deltas, `reasoning_content`, `chat_template_kwargs`, OpenAI/Anthropic base64 images)
- `POST /v1/completions` — Raw text completions
- `GET /v1/models` — Model list & active context ceiling
- `GET /v1/vram` · `POST /v1/vram` — Inspect VRAM state (without waking) or manually `{"action":"sleep"|"wake"}`
- `GET /health` — Health, engine version, sleep state, and parked slot count

</details>

---

## Deep Dives, Attribution & License

- **Documentation:** [`THEORY.md`](THEORY.md) (RTX 5060 roofline & Phases 1–8 audit) · [`LITE.md`](LITE.md) (visual explainer) · [`LEARNINGS.md`](LEARNINGS.md) (optimization log) · [`CHANGELOG.md`](CHANGELOG.md) · [`deploy/squid5060.service`](deploy/squid5060.service) (systemd unit)
- **Attribution:** `squid5060` is forked from and tracks [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090), which builds on [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine), [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine), and [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill). Training-free last-loop KV sharing was inspired by [2DSD (SqueezeAILab)](https://github.com/SqueezeAILab/2DSD). Full attributions live in [`NOTICE`](NOTICE).
- **License:** Licensed under the **[Apache License, Version 2.0](LICENSE)**.
