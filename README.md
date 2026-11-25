# squid5060

**A single-binary, zero-Python TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible HTTP/SSE server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B looped MoE + native ViT vision) — engineered for 8 GB Blackwell GPUs (RTX 5060 Laptop / Desktop, `sm_120a`).**

Forked from [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090), `squid5060` v3.0 brings native NVFP4 Blackwell PTX inference, persistent CUDA Graphs, **Hybrid AVX2 FP32 CPU + GPU PCIe MoE miss execution**, **temperature-aware 4-way `{1, 2, 4, 8}` adaptive DSpark speculative decoding + Hybrid O(1) Prompt-Lookup**, **pre-warmed multi-domain expert cache**, **tail-aliased `B=1024` hybrid VRAM/PCIe Tensor Core prefill**, **FP8 local SWA + shared-loop Q4 global KV caches**, **auto idle-sleep (`7.5 GiB → 222 MiB` VRAM, 326 ms wake)**, **host-RAM multi-session conversation parking (~22 ms swap)**, **on-demand PNG/JPEG vision (`0.0 MiB` permanent VRAM)**, and **up to 256K context** into **< 7.5 GiB VRAM** — beating tuned `llama.cpp` (`UD-Q3_K_XL` + MoE-cache loop dedup) by **2.31× on single-stream decode (`85.8 tok/s` median, `96.8 tok/s` peak)** and **5.9×–9.4× on cold prefill**.

![squid5060 RTX 5060 Laptop 8 GB Benchmarks vs llama.cpp and v2.1](assets/benchmark-5060.png)

> **Upfront Attribution:** `squid5060` tracks [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090) (which builds on [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine), [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine), and [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill)). Training-free last-loop KV sharing was inspired by [2DSD (SqueezeAILab)](https://github.com/SqueezeAILab/2DSD). Full attributions live in [`NOTICE`](NOTICE).
>
> **Deep Dives (from upstream `squid5090`):** [`LITE.md`](LITE.md) · [`LEARNINGS.md`](LEARNINGS.md) · [`THEORY.md`](THEORY.md)

---

## Measured RTX 5060 Laptop (8 GB) Benchmarks vs. `llama.cpp`

Measured strictly single-stream (`c=1`, sequential — zero multi-user batching) on an **NVIDIA GeForce RTX 5060 Laptop GPU** (`sm_120a`, 26 SMs, 32 MiB L2, 7.8 GiB usable VRAM, PCIe Gen5 x8) + **Intel Core Ultra 7 255H** (16 cores, DDR5) using the exact same OpenAI-SSE `bench.py` harness (6 diverse prompts × 512 decode, cold 3.3K & 12.9K random-filler prefill, long-context Needle-in-a-Haystack retrieval, continuous `nvidia-smi` peak VRAM probe):

| Metric (`c=1`, RTX 5060 Laptop 8 GB) | `llama.cpp` `ink-dedup` (`UD-Q3_K_XL`, `T=0.6`) | `squid5060` v2.1 (GPU-only) | **`squid5060` v3.0 (`T=0.6` exact match)** | **`squid5060` v3.0 (`T=0.0` greedy)** | **v3.0 Gain vs. `llama.cpp`** |
|---|---:|---:|---:|---:|---:|
| **6-prompt `c=1` decode median** | `37.2 tok/s` | `58.6 tok/s` | **`85.8 tok/s`** *(up to `91.3`)* | **`83.1 tok/s`** *(peak **`96.8`**)* | **2.31× faster (+131%)** |
| **Long-ctx Needle decode (23.9K)** | `32.1 tok/s` | `68.4 / 71.1 tok/s` | **`70.4 tok/s`** *(battery)* | **`80.9 tok/s`** *(battery)* | **2.52× faster** |
| **Cold prefill (3.3K random filler)** | `174 tok/s` | `1,031 tok/s` | **`981 tok/s`** | **`1,031 tok/s`** | **5.64× – 5.92× faster** |
| **Cold prefill (12.9K random filler)** | `168 tok/s` | `1,319 tok/s` | **`1,319 tok/s`** | **`1,315 tok/s`** | **7.86× faster** |
| **Long-ctx Needle prefill** | `149 tok/s` *(204s)* | `1,398 tok/s` *(17.1s)* | **`1,398 tok/s`** *(17.1s)* | **`1,392 tok/s`** *(17.2s)* | **9.37× faster** |
| **Needle-in-a-Haystack accuracy** | `3/3` | `3/3` PASS | **`3/3` PASS** | **`3/3` PASS** | **100% exact** |
| **Peak VRAM / Idle-Sleep VRAM** | `7.35 GiB` / — | `7.47 GiB` / — | **`7.47 GiB` / `0.22 GiB`** *(326 ms wake)* | **`7.47 GiB` / `0.22 GiB`** | **7.25 GiB freed idle** |
| **Multi-chat swap (4 parked slots)** | — | cold re-prefill | **`22–27 ms`** *(1.8K–2.3K tok)* | **`22–27 ms`** | **Zero re-prefill** |
| **On-demand Vision (PNG/JPEG)** | — | — | **`158 ms`** *(128 tok, `+0.0 MiB` VRAM)* | **`158 ms`** | **0 MiB permanent VRAM** |
| **256K context (`--ctx 262144 --kv q4`)** | — | `1,192` experts | **`1,192` experts (`7.25 GiB` VRAM)** | — | **Fits 256K in 8 GB** |

- **Quality verified:** Exact FP64 KLD on `--raw` `tools/kld_corpus.txt` (3,026 tokens):
  - **Hybrid AVX2 FP32 CPU+GPU MoE (`--cpu-moe on` vs `--cpu-moe off`):** **`mean_KLD = 0.004581`, `med = 0.000146`, `top1 = 98.81%`, `ref_PPL = 2.1770`, `cand_PPL = 2.1760` (`dPPL = -0.0011` — slightly *lower* perplexity from exact FP32 CPU accumulation!)**.
  - **FP8 local SWA + Q4 global KV vs. BF16 local KV:** `mean_KLD = 0.009121`, `top1 = 97.20%`, `dPPL = -0.0298`.
  - **Training-free Last-Loop KV Sharing (`W=1,024` native tail window on Layers 11 & 17):** **`mean_KLD = 0.006537`, `med = 0.000000`, `top1 = 98.65%`, `dPPL = -0.0105`**, plus `3/3` verbatim 23.9K needle retrieval.
- **Stochastic sampling (`--temp 1.0 --top-p 0.95`):** Uses exact Leviathan/Chen rejection sampling with residual resampling inside the CUDA Graph (zero-overhead GPU sampling matching Inkling-2-Mini's official `generation_config.json`).

---

## Architecture: How `squid5060` v3.0 Hits 85.8 tok/s + Vision + Idle-Sleep in 8 GB VRAM

Full `Inkling-2-Mini` weights are ~14.54 GiB in NVFP4 — nearly double an 8 GB GPU's VRAM. On any GPU with `< 15 GiB` VRAM, `squid5060` automatically enables its **8 GB Unified Expert Cache** profile (`--mode nvfp4 --kv q4 --ctx 32768 --mtp 7 --spec-threshold 0.90 --embed-host --cpu-moe auto`) while keeping full-VRAM mode and dynamic SM-aware grids for larger Blackwell GPUs (`>= 15 GiB`):

1. **Hybrid AVX2+FMA+F16C CPU + GPU PCIe MoE Miss Execution (`+46%` Decode Speedup):**
   - On 8 GB GPUs, ~52% of Inkling-2-Mini's 3,072 NVFP4 experts live in pinned host RAM. Previously every cache miss serialized over the ~24 GB/s PCIe link.
   - `squid5060` v3.0 adds a persistent POSIX-pthread AVX2+FMA+F16C worker pool (`cuda/cpu_moe.cpp`) coordinated via a CUDA-Graph-safe mapped-memory doorbell (`moe_cpu_dispatch_kernel` / `moe_cpu_merge_kernel`). During decode (`B <= 8`), half of RAM misses (`split=1/2`) execute **simultaneously on CPU cores directly out of DDR5 RAM in exact FP32** (~35 GB/s, `0.129 ms/expert`, 0% CPU when idle via wake/nap CV) while the GPU runs VRAM hits + the other half over PCIe — pushing 6-prompt median decode from **58.6 → 85.8 tok/s** (`+46.4%`, **96.8 tok/s peak**).
2. **Unified 3,072-Expert VRAM Cache + Built-in Hot-Expert Profile (`~6.1–6.5 GiB`, 1,471–1,529 physical slots at 32K):**
   - All 24 physical layers' 3,072 experts (L0–19 native NVFP4 + L20–23 converted from MXFP8 → NVFP4 on load in 32-expert GPU chunks) are packed into a uniform `4,478,976 B` (`4.27 MiB`), 256-byte-aligned slot (`w13_data`, `w13_scales`, `w2_data`, `w2_scales`).
   - **Pre-warmed multi-domain profile + fixed LRFU warm-prior (`prior_max = 4.0`):** Even on a fresh install with no `.squid5060-warm-cache.bin` snapshot yet, `squid5060` bulk-DMAs its built-in blended hot-expert profile (`src/hot_experts.zig`) in ~270 ms at startup so Prompt #1 starts warm; subsequent runs refine and persist the snapshot.
   - **Zero-race async overlapped promotion:** Dedicated `copy_stream` overlaps H2D promotions with the next forward pass using 8 staging slots and 1-token victim quarantine while keeping persistent `cudaGraphExec_t` graphs completely intact.
3. **Power-Law Temperature-Aware 4-Way `{1, 2, 4, 8}` Adaptive Switch + Hybrid O(1) Prompt-Lookup:**
   - Inside CUDA 13's `cudaGraphCondTypeSwitch`, `z_adaptive_select_width_kernel` calibrates DSpark draft probabilities via `p_k = conf_k^gamma` (scaling `gamma` with `temperature` and `--spec-threshold`) and maximizes `E[accepted(w)] / cost(w)` over `{1, 2, 4, 8}` with 8 GB PCIe-calibrated step costs `{1.00, 1.30, 2.45, 4.50}` — dropping cleanly to `B=1` on uncertain steps while scaling to `B=4/8` on code and verbatim spans.
4. **Auto Idle-Sleep (`--idle-sleep <sec>` + `GET/POST /v1/vram`):**
   - After `<sec>` seconds without inference traffic (or manually via `POST /v1/vram {"action":"sleep"}`), `squid5060` parks the live conversation to host RAM, destroys CUDA graphs, frees the weight/runtime/expert-cache VRAM arenas (**`7.47 GiB → 0.22 GiB`**, freeing ~7.25 GiB for games/other apps), and naps CPU workers.
   - `GET /health` and `GET /v1/vram` inspect status without waking. The next chat/completion request (or `POST /v1/vram {"action":"wake"}`) re-allocates VRAM and H2D-restores dense weights + ~1,463 hottest experts from host RAM in **`~326 ms`** and restores the parked chat in **`~22 ms`**.
5. **Host-RAM Conversation Parking (`--slot-keep N`, default `4`):**
   - Keeps up to `N` (max 8) full conversations in LRU host-RAM slots (tokens, local SWA KV + sconv rings at both `decode_end` and `prefill_end`, plus global KV & DSpark KV) at **0.0 MiB extra VRAM**. Switching between multiple long coding/chat sessions restores in **~22–27 ms** instead of re-prefilling, and survives VRAM sleep/wake cycles.
6. **On-Demand Vision / Image Input (`0.0 MiB` Permanent VRAM):**
   - Runs Inkling-2-Mini's built-in **23-layer BF16 ViT + `2×2` merge projector** (`cuda/vision.cu`, `src/vision.zig`) on demand for OpenAI `content: [{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}]` requests and CLI `--image <path>`.
   - Lazily mmaps the 422 BF16 `model.visual.*` tensors from `model-00001-of-00001.safetensors`, borrows the existing prefill tail scratch + shared cuBLAS handle, executes 23 layers with 0-indexed interleaved axial 2D RoPE + FlashAttention (~158 ms for 128 visual tokens), and immediately returns all borrowed VRAM slots (`+0.0 MiB` permanent VRAM delta). Supports PNG (`libpng16`, RGBA checkerboard compositing) and JPEG (`libjpeg`) with automatic bicubic downscale to ≤1,024 visual tokens.
7. **Hybrid VRAM/PCIe `B=1024` Tensor Core Prefill + FP8 Local / Shared-Loop Q4 Global KV (256K Context in 8 GB):**
   - `p_wgemm_kernel<..., CACHED=true>` keeps `cp.async` for VRAM hits and uses coalesced `uint4` loads for PCIe misses (**19× speedup**, `981–1,398 tok/s` cold prefill) while borrowing `375.9 MiB` prefill scratch from evictable cache tail slots (`0.0 MiB` permanent decode overhead).
   - FP8 local SWA KV + training-free Last-Loop Shared Global KV (1,024-token native tail rings on Loops 1 & 2, shared bulk on Loop 3) cuts global KV banks `8 → 4`, fitting **256K context** with 1,192 cached experts inside 8 GB VRAM.

---

## Quickstart

### Requirements
- **OS:** Linux x86_64
- **GPU:** NVIDIA Blackwell GPU (`sm_120a`, e.g., RTX 5060 8 GB / RTX 5070 / RTX 5090)
- **CUDA Toolkit:** CUDA `13.0+` (`nvcc`, `libcudart`, `libcublas` — auto-detects `/opt/cuda` or `/usr/local/cuda`, overridable via `-Dcuda-path=...`)
- **System Libraries:** `libpng16`, `libjpeg` (for on-demand PNG/JPEG vision decoding)
- **Build Tools:** `g++`, **Zig `0.17.1+`**
- **Model:** [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (~20.2 GB safetensors)

### Build & Run
```bash
git clone https://github.com/dialectmaterial/squid5060.git
cd squid5060

# Build ReleaseFast binary at zig-out/bin/squid5060 (auto-detects /opt/cuda or /usr/local/cuda)
zig build

# Download official Inkling-2-Mini safetensors
hf download thinkingmachines/Inkling-2-Mini --local-dir /path/to/inkling-2-mini

# Start OpenAI-compatible HTTP/SSE server (8 GB profile + auto-sleep after 5 min idle + 4 parked sessions)
./zig-out/bin/squid5060 serve /path/to/inkling-2-mini \
  --host 127.0.0.1 --port 8000 --idle-sleep 300 --slot-keep 4

# Optional: bearer auth from a mode-600 key file (never passed in process argv)
./zig-out/bin/squid5060 serve /path/to/inkling-2-mini \
  --host 127.0.0.1 --port 8000 --idle-sleep 300 --slot-keep 4 --api-key-file /path/to/api-key

# Manual VRAM sleep / wake / status (does not wake on GET)
curl http://127.0.0.1:8000/v1/vram
curl -X POST http://127.0.0.1:8000/v1/vram -H 'Content-Type: application/json' -d '{"action":"sleep"}'
curl -X POST http://127.0.0.1:8000/v1/vram -H 'Content-Type: application/json' -d '{"action":"wake"}'

# One-shot CLI generation (text or image!)
./zig-out/bin/squid5060 generate /path/to/inkling-2-mini \
  --prompt "Write a clean Python implementation of Dijkstra's shortest path algorithm." \
  --max-tokens 512 --ctx 32768 --mtp 7 --spec-threshold 0.90

./zig-out/bin/squid5060 generate /path/to/inkling-2-mini \
  --image /path/to/screenshot.png \
  --prompt "Read the text and summarize what is shown in this image." \
  --max-tokens 256 --mtp 7

# 128K or 256K context scaling in 8 GB VRAM
./zig-out/bin/squid5060 longctx /path/to/inkling-2-mini --ctx 131072 --kv q4
./zig-out/bin/squid5060 longctx /path/to/inkling-2-mini --ctx 262144 --kv q4
```

*(Optional draft speedup: if a flat little-endian `int32` `draft_vocab_32k.bin` or `draft_vocab_49k.bin` token-ID table is placed in the model directory, `squid5060` automatically prunes the DSpark draft LM head from 200K → 32K, saving ~245 MiB VRAM (+57 cached experts) and speeding up drafting ~4×. Without it, or with `SQUID_DRAFT_VOCAB=full`, `squid5060` seamlessly uses the full 200K draft vocabulary.)*

### Key CLI Flags & Overrides

| Flag / Env | Default (`< 15 GiB` `serve`) | Description |
|---|---|---|
| `--mode` | `nvfp4` (`fp8` on ≥15 GiB) | Weight precision: `nvfp4` \| `fp8` \| `exact` (`exact` requires `--no-expert-cache`) |
| `--kv` | `q4` (`auto` on ≥15 GiB) | Global KV cache format: `q4` \| `fp8` \| `bf16` (`SQUID_LOCAL_KV_BF16=1` disables FP8 local SWA; `SQUID_SHARE_LOOP_KV=0` disables last-loop global KV sharing) |
| `--ctx` | `32768` (up to `262144` in 8 GB) | Context window size |
| `--mtp` | `7` (`0` = off) | DSpark speculative drafter (`7` = temperature-aware 4-way adaptive `{1,2,4,8}` CUDA switch + Prompt-Lookup) |
| `--spec-threshold` | `0.90` (`0.80` on ≥15 GiB) | Markov confidence threshold for adaptive verify width (`0.0` = fixed `B=8`) |
| `--cpu-moe` | `auto` (`on` on <15 GiB) | Hybrid AVX2 FP32 CPU+GPU miss execution: `auto` \| `on` \| `off` (`SQUID_CPU_MOE_SPLIT=1/2`, `SQUID_CPU_MOE_THREADS`) |
| `--idle-sleep` | `0` (disabled) | Seconds of inactivity before releasing ~7.25 GiB VRAM (`7.47 → 0.22 GiB`); auto-wakes in ~326 ms on next request |
| `--slot-keep` | `4` (max `8`) | LRU host-RAM parked conversations preserved across session switches and VRAM sleep/wake (`0` disables) |
| `--image` | none (`generate` only) | Path to a PNG or JPEG image to encode on-demand (`0 MiB` permanent VRAM); in `serve`, use OpenAI `image_url` base64 data URIs |
| `--temp` / `--top-p` / `--top-k` | `1.0` / `0.95` / `0` | GPU sampling & exact Leviathan/Chen rejection sampling defaults (overridable per-request) |
| `--api-key` / `--api-key-file` | `$SQUID_API_KEY` | Optional `Authorization: Bearer <key>` constant-time auth (rejects empty/oversized key files) |
| `--cache-mib` / `--cache-headroom-mib` | auto / `160` | Override VRAM expert cache size or reserved desktop headroom |
| `--cache-file` / `--no-warm-cache` | `<model_dir>/.squid5060-warm-cache.bin` | Warm-cache LRFU snapshot path (falls back to built-in multi-domain profile) or disable |
| `--ablation` | none | Optional 1.8 MB runtime-gated Fisher-LDA `.safetensors` (see [`tools/ablate.py`](tools/ablate.py)) |

---

## License & Documentation

- **[`LICENSE`](LICENSE)** / **[`NOTICE`](NOTICE)** — Apache-2.0 license and complete open-source attributions.
- **[`deploy/squid5060.service`](deploy/squid5060.service)** — Systemd service unit template.
- **[`LITE.md`](LITE.md)** / **[`LEARNINGS.md`](LEARNINGS.md)** / **[`THEORY.md`](THEORY.md)** — Architecture and roofline guides.
