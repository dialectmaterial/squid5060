# squid5060

**A single-binary, zero-Python TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible HTTP/SSE server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B looped MoE + native ViT vision) — engineered for 8 GB Blackwell GPUs (RTX 5060 Laptop / Desktop, `sm_120a`).**

Forked from [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090), `squid5060` v3.2 combines upstream Blackwell **`sm_120a` native block-scaled Tensor Core prefill MoE** (`m16n8k64` NVFP4 with 128B PCIe cachelines & preloaded scales), **1-launch PDL Cluster Attention** (`CS=4` local / `CS=8` global on 26-SM RTX 5060), **CUDA 13.4+ `L2::evict_first` streaming weight cache hints**, **3 KB SMEM L1-cached LM-head & DSpark MMA kernels**, **DSpark Local TC Split-K**, **embedded corpus-frequency 32K draft shortlist (`src/draft_shortlist.bin`)**, **128-CTA multi-block PDL stochastic sampling**, **post-`[DONE]` deferred checkpointing**, and **zero-overhead LoopCD-Hidden (`arXiv:2610.02185`)** with `squid5060`'s 8 GB architecture — **Hybrid AVX2 FP32 CPU + GPU PCIe MoE miss execution**, **temperature-aware 4-way `{1, 2, 4, 8}` adaptive DSpark speculative decoding**, **pre-warmed multi-domain expert cache**, **FP8 local SWA + shared-loop Q4 global KV caches**, **auto idle-sleep (`7.47 GiB → 0.22 GiB` VRAM, 326 ms wake)**, **host-RAM conversation parking (~22 ms swap)**, **on-demand PNG/JPEG vision (`0.0 MiB` permanent VRAM, `0.0 ms` multi-turn prefix-cache reuse)**, and **up to 256K context** in **< 7.5 GiB VRAM** — hitting **`2,061–2,248 tok/s` 24K prefill**, **`89.4–91.0 tok/s` 6-prompt `c=1` decode median**, and **`105.4 tok/s` peak single-stream decode**.

![squid5060 RTX 5060 Laptop 8 GB Benchmarks vs llama.cpp](assets/benchmark-5060.png)

> **Upfront Attribution:** `squid5060` tracks [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090) (which builds on [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine), [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine), and [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill)). Training-free last-loop KV sharing was inspired by [2DSD (SqueezeAILab)](https://github.com/SqueezeAILab/2DSD). Full attributions live in [`NOTICE`](NOTICE).
>
> **Deep Dives (from upstream `squid5090`):** [`LITE.md`](LITE.md) · [`LEARNINGS.md`](LEARNINGS.md) · [`THEORY.md`](THEORY.md)

---

## Measured RTX 5060 Laptop (8 GB) Benchmarks vs. `llama.cpp`

Measured strictly single-stream (`c=1`, sequential — zero multi-user batching) on an **NVIDIA GeForce RTX 5060 Laptop GPU** (`sm_120a`, 26 SMs, 32 MiB L2, 7.8 GiB usable VRAM, PCIe Gen5 x8) + **Intel Core Ultra 7 255H** (16 cores, DDR5) using the exact same OpenAI-SSE `bench.py` harness (6 diverse prompts × 512 decode, cold 3.3K & 12.9K random-filler prefill, long-context Needle-in-a-Haystack retrieval, continuous `nvidia-smi` peak VRAM probe):

| Metric (`c=1`, RTX 5060 Laptop 8 GB) | `llama.cpp` `ink-dedup` (`UD-Q3_K_XL`, `T=0.6`) | `squid5060` v3.0 | **`squid5060` v3.1 (`T=0.6` / `T=1.0`)** | **`squid5060` v3.1 (`T=0.0` greedy)** | **v3.1 Gain vs. `llama.cpp`** |
|---|---:|---:|---:|---:|---:|
| **Long-ctx Needle decode (23.9K)** | `32.1 tok/s` | `80.9 tok/s` | **`97.9 tok/s`** | **`97.9 tok/s`** | **3.05× faster (+205%)** |
| **Peak single-prompt decode (`c=1`)** | `37.9 tok/s` | `96.8 tok/s` | **`105.4 tok/s`** *(warm CLI `116.7`)* | **`93.8 tok/s`** *(warm CLI `115.7`)* | **2.78× faster (+178%)** |
| **6-prompt `c=1` decode median** | `37.2 tok/s` | `85.8 tok/s` | **`80.0 / 80.9 tok/s`** *(unbiased stoch.)* | **`82.4 tok/s`** | **2.15× – 2.22× faster** |
| **Short-prompt TTFT** | — | `245.3 ms` | **`229.5 ms`** | **`271.5 ms`** | **1.07× faster vs v3.0** |
| **Cold prefill (3.3K random filler)** | `174 tok/s` | `1,031 tok/s` | **`1,143 / 1,182 tok/s`** | **`1,167 tok/s`** | **6.57× – 6.79× faster** |
| **Cold prefill (12.9K random filler)** | `168 tok/s` | `1,319 tok/s` | **`1,792 / 1,777 tok/s`** | **`1,793 tok/s`** | **10.7× faster** |
| **Long-ctx Needle prefill (23.9K)** | `149 tok/s` *(160s)* | `1,398 tok/s` *(17.1s)* | **`2,025 tok/s`** *(**11.8s**)* | **`2,025 tok/s`** *(**11.8s**)* | **13.6× faster** |
| **Needle-in-a-Haystack accuracy** | `3/3` | `3/3` PASS | **`3/3` PASS** | **`3/3` PASS** | **100% exact** |
| **Peak VRAM / Idle-Sleep VRAM** | `7.35 GiB` / — | `7.47 GiB` / `0.22 GiB` | **`7.47 GiB` / `0.22 GiB`** *(329 ms wake)* | **`7.47 GiB` / `0.22 GiB`** | **7.25 GiB freed idle** |
| **Multi-chat swap (4 parked slots)** | — | `22–27 ms` | **`11–28 ms`** *(host-RAM LRU)* | **`11–28 ms`** | **Zero re-prefill** |
| **On-demand Vision (PNG/JPEG)** | — | `158 ms` (`+0.0 MiB`) | **`+0.0 MiB` VRAM** *(23-layer BF16 ViT)* | **`+0.0 MiB` VRAM** | **0 MiB permanent VRAM** |
| **256K context (`--ctx 262144 --kv q4`)** | — | `1,192` experts | **`1,192` experts (`7.25 GiB` VRAM)** | — | **Fits 256K in 8 GB** |

- **Quality verified:** Exact FP64 KLD on `--raw` `tools/kld_corpus.txt`:
  - **Self-determinism (`--kv bf16`, `--cpu-moe off`):** **`mean_KLD = 0.000000`, `max = 0.000000`, `top1 = 100.00%`**.
  - **v3.1 Q4 global + FP8 local SWA + shared-loop KV (`CS=4/8` PDL Cluster Attention) vs. BF16 KV:** **`mean_KLD = 0.007732`, `med = 0.001956`, `top1 = 97.88%`** (and **`mean_KLD = 0.008548`, `top1 = 97.62%`** with Hybrid AVX2 FP32 CPU+GPU MoE enabled), plus `3/3` verbatim 23.9K needle retrieval.
  - **Hybrid AVX2 FP32 CPU+GPU MoE (`--cpu-moe on` vs `--cpu-moe off` on 3,026 tokens):** **`mean_KLD = 0.004581`, `med = 0.000146`, `top1 = 98.81%`, `dPPL = -0.0011`**.
- **v3.2 upstream sync (`124afb8`, CUDA 13.4):** Adding `L2::evict_first` streaming weight cache hints, 3 KB SMEM L1-cached LM-head & DSpark MMA kernels, DSpark Local TC Split-K, embedded corpus-frequency 32K draft shortlist (`src/draft_shortlist.bin`, replacing naive `0..32700`), post-`[DONE]` deferred checkpointing, and lazy content-addressed 0-VRAM vision prefix caching (`0.0 ms` Turn 2+ ViT reuse + Anthropic `source.type=base64` support) lifted 6-prompt `c=1` decode median to **`89.4 / 90.9 / 91.0 tok/s`** (`T=0.6 / 1.0 / 0.0`, **`+10.5% to +12.4%`** over v3.1) and 24K needle prefill to **`2,061 tok/s` median (`2,248 tok/s` peak, `3/3` PASS)**, verified at **`0.000000` self-KLD** and **`0.011396` Q4 + Hybrid CPU-MoE KLD (`dPPL +0.0025`)** over 7,784 tokens.
- **Stochastic sampling (`--temp 1.0 --top-p 0.95`):** Uses 128-CTA multi-block PDL GPU sampling with exact Leviathan/Chen rejection sampling + residual resampling inside the CUDA Graph, and automatically disables Prompt-Lookup for stochastic sampling (`T > 0, top_k != 1`) to prevent rejection-sampling cliffs.

---

## Architecture: How `squid5060` v3.1 Hits 2,025 tok/s Prefill + 98 tok/s @ 24K in 8 GB VRAM

Full `Inkling-2-Mini` weights are ~14.54 GiB in NVFP4 — nearly double an 8 GB GPU's VRAM. On any GPU with `< 15 GiB` VRAM, `squid5060` automatically enables its **8 GB Unified Expert Cache** profile (`--mode nvfp4 --kv q4 --ctx 32768 --mtp 7 --spec-threshold 0.90 --embed-host --cpu-moe auto`) while keeping full-VRAM mode and dynamic SM-aware grids for larger Blackwell GPUs (`>= 15 GiB`):

1. **1-Launch PDL Blackwell Cluster Attention + 128-CTA Multi-Block PDL Sampling (`+42.8%` Long-Context Decode):**
   - Replaces the legacy multi-kernel decode attention + merge pipeline with a single `attn_cluster_decode_kernel` using Blackwell thread-block clusters, DSMEM barriers, and Programmatic Dependent Launch (PDL) overlap across QKV projection, attention, residual norms, and DSpark drafting.
   - Hardware-aware cluster sizing (`CS=4` for `window=512` local SWA and `CS=8` for global attention on 26-SM RTX 5060; `CS=16` global on ≥64-SM GPUs) with full BF16/FP8/Q4 KV and shared-loop global KV support — pushing 24K-context decode from **68.5 → 97.9 tok/s** (`+42.8%`).
   - Multi-block 128-CTA PDL LM-head stochastic sampler (`g_lm_samp`) + 576-thread zero-SMEM `float4` PDL DSpark & residual kernels.
2. **Native `sm_120a` Block-Scaled Tensor Core Prefill MoE + Fused 2D-Parallel Prefill (`2,025 tok/s`, `+38.6%`):**
   - `p_wgemm_blkscale_kernel<..., CACHED=true>` executes Blackwell native `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (and MXFP8 `m16n8k32`) directly with L2-resident FP32 `atomicAdd` on Down projection, `BK=256` full 128-byte coalesced PCIe cachelines, and once-per-CTA contiguous `uint4` weight-scale preloading (`-55%` MoE prefill latency on PCIe misses).
   - Paired with 2D-parallel fused SConv/KV store (`p_local_sconv_prep_store_kernel` / `p_global_attn_prep_store_kernel`), 7-launch batched DSpark commit, reverse-causal `exp2f` FlashAttention prefill, and `min_batch=64` on 8 GB GPUs (keeping short prompts `<64` tok on the `B=8` hybrid AVX2+GPU decode graphs with zero 90-slot tail eviction).
3. **Hybrid AVX2+FMA+F16C CPU + GPU PCIe MoE Miss Execution (`+46%` Decode Speedup):**
   - On 8 GB GPUs, ~52% of Inkling-2-Mini's 3,072 NVFP4 experts live in pinned host RAM. Previously every cache miss serialized over the ~24 GB/s PCIe link.
   - `squid5060` adds a persistent POSIX-pthread AVX2+FMA+F16C worker pool (`cuda/cpu_moe.cpp`) coordinated via a CUDA-Graph-safe mapped-memory doorbell (`moe_cpu_dispatch_kernel` / `moe_cpu_merge_kernel`). During decode (`B <= 8`), half of RAM misses (`split=1/2`) execute **simultaneously on CPU cores directly out of DDR5 RAM in exact FP32** (~35 GB/s, `0.129 ms/expert`, 0% CPU when idle via wake/nap CV) while the GPU runs VRAM hits + the other half over PCIe.
4. **Unified 3,072-Expert VRAM Cache + Built-in Hot-Expert Profile (`~6.1–6.5 GiB`, 1,471–1,529 physical slots at 32K):**
   - All 24 physical layers' 3,072 experts (L0–19 native NVFP4 + L20–23 converted from MXFP8 → NVFP4 on load in 32-expert GPU chunks) are packed into a uniform `4,478,976 B` (`4.27 MiB`), 256-byte-aligned slot (`w13_data`, `w13_scales`, `w2_data`, `w2_scales`).
   - **Pre-warmed multi-domain profile + fixed LRFU warm-prior (`prior_max = 4.0`):** Even on a fresh install with no `.squid5060-warm-cache.bin` snapshot yet, `squid5060` bulk-DMAs its built-in blended hot-expert profile (`src/hot_experts.zig`) in ~270 ms at startup so Prompt #1 starts warm; subsequent runs refine and persist the snapshot.
   - **Zero-race async overlapped promotion:** Dedicated `copy_stream` overlaps H2D promotions with the next forward pass using 8 staging slots and 1-token victim quarantine while keeping persistent `cudaGraphExec_t` graphs completely intact.
5. **Power-Law Temperature-Aware 4-Way `{1, 2, 4, 8}` Adaptive Switch + Zero-Overhead LoopCD-Hidden:**
   - Inside CUDA 13's `cudaGraphCondTypeSwitch`, `z_adaptive_select_width_kernel` calibrates DSpark draft probabilities via `p_k = conf_k^gamma` (scaling `gamma` with `temperature` and `--spec-threshold`) and maximizes `E[accepted(w)] / cost(w)` over `{1, 2, 4, 8}` with 8 GB PCIe-calibrated step costs `{1.00, 1.30, 2.45, 4.50}` — dropping cleanly to `B=1` on uncertain steps while scaling to `B=4/8` on code and verbatim spans.
   - **LoopCD-Hidden (`arXiv:2610.02185`):** Optional per-request or CLI `--loopcd-omega <f>` / `--loopcd-ref 1|2` hidden-state contrastive decoding (`h' = (1 + ω)·h_3 - ω·h_ref`) at `0.0 MiB` decode VRAM via tail-aliased scratch, properly keyed in both `PrefixCache` and `ParkedSlot` with zero CUDA graph rebuilds.
6. **Auto Idle-Sleep (`--idle-sleep <sec>` + `GET/POST /v1/vram`) & Host-RAM Conversation Parking (`--slot-keep N`):**
   - After `<sec>` seconds without inference traffic (or manually via `POST /v1/vram {"action":"sleep"}`), `squid5060` parks the live conversation to host RAM, destroys CUDA graphs, frees the weight/runtime/expert-cache VRAM arenas (**`7.47 GiB → 0.22 GiB`**, freeing ~7.25 GiB for games/other apps), and naps CPU workers; wakes in **`~326–329 ms`**.
   - Keeps up to `N` (default `4`, max `8`) full conversations in LRU host-RAM slots at **0.0 MiB extra VRAM**, swapping in **~11–28 ms** across multi-chat switches and VRAM sleep/wake cycles, with unbounded multi-turn tool-call history (`O(1)` `ToolCallIdMap`).
7. **On-Demand Vision (`0.0 MiB` Permanent VRAM, `0.0 ms` Multi-Turn Reuse) & 256K Context (`FP8` Local / Shared-Loop `Q4` Global KV):**
   - Lazily runs Inkling-2-Mini's **23-layer BF16 ViT + `2×2` merge projector** (`cuda/vision.cu`, `src/vision.zig`) for OpenAI `image_url`, Responses `input_image`, Anthropic `source.type=base64` PNG/JPEG requests and CLI `--image <path>` by borrowing the prefill tail scratch + shared cuBLAS handle, returning all borrowed slots immediately (`+0.0 MiB` permanent VRAM). Content-addressed negative synthetic token IDs (`SplitMix64` hash over image bytes) allow Turn 2+ follow-ups on the same image to hit the prefix cache and parked slots with **`0.0 ms` ViT re-encode**.
   - FP8 local SWA KV + training-free Last-Loop Shared Global KV (1,024-token native tail rings on Loops 1 & 2, shared bulk on Loop 3) cuts global KV banks `8 → 4`, fitting **256K context** with 1,192 cached experts inside `7.25 GiB` VRAM.

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

*(Built-in draft shortlist: `squid5060` embeds a corpus-frequency-ranked 32K shortlist [`src/draft_shortlist.bin`](src/draft_shortlist.bin) (`32,834` tokens including special tokens) directly into the binary — zero external files needed, saving ~245 MiB VRAM (+57 cached experts) and speeding up DSpark drafting ~4×. Set `SQUID_DRAFT_VOCAB=49k` for 49K rows or `SQUID_DRAFT_VOCAB=full` to force the full 200K draft vocabulary.)*

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
| `--temp` / `--top-p` / `--top-k` | `1.0` / `0.95` / `0` | 128-CTA PDL GPU sampling & exact Leviathan/Chen rejection sampling defaults (overridable per-request) |
| `--loopcd-omega` / `--loopcd-ref` | `0.0` / `1` | Zero-overhead LoopCD-Hidden (`arXiv:2610.02185`) contrastive strength & reference loop pass `1\|2` (also per-request JSON `loopcd_omega` / `loopcd_ref`) |
| `--api-key` / `--api-key-file` | `$SQUID_API_KEY` | Optional `Authorization: Bearer <key>` constant-time auth (rejects empty/oversized key files) |
| `--cache-mib` / `--cache-headroom-mib` | auto / `160` | Override VRAM expert cache size or reserved desktop headroom |
| `--cache-file` / `--no-warm-cache` | `<model_dir>/.squid5060-warm-cache.bin` | Warm-cache LRFU snapshot path (falls back to built-in multi-domain profile) or disable |
| `--ablation` | none | Optional 1.8 MB runtime-gated Fisher-LDA `.safetensors` (see [`tools/ablate.py`](tools/ablate.py)) |

---

## License & Documentation

- **[`LICENSE`](LICENSE)** / **[`NOTICE`](NOTICE)** — Apache-2.0 license and complete open-source attributions.
- **[`deploy/squid5060.service`](deploy/squid5060.service)** — Systemd service unit template.
- **[`LITE.md`](LITE.md)** / **[`LEARNINGS.md`](LEARNINGS.md)** / **[`THEORY.md`](THEORY.md)** — Architecture and roofline guides.
