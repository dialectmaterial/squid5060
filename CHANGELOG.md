# Changelog

All notable changes to `squid5060` are documented here.

## [v3.8.0] — 2026-11-26

### Added
- **Lazy `32K → 64K → 128K → 256K` Global KV:** Booting with `--ctx 262144` allocates only `32K` global KV initially (keeping all `1,470` hot experts resident in VRAM), tail-aliases larger KV tiers onto the expert cache pool on demand with lossless mid-session host-bounce preservation, and shrinks back to `32K` on fresh short chats. Verified `240K` (`240,104` tok) needle recall PASS at `45.5 tok/s` with 0 host KV spill.
- **DSpark 32K Global KV Cap on 8 GB GPUs:** Reclaims ~23–66 expert cache slots at `256K` context (`1,215` active experts).
- **Sampled-Safe Prompt-Copy (`SQUID_PCOPY_MIN_CTX=8192`):** Exact `q=1.0` point-mass verification over 64K history with 16-token exact match, lifting verbatim `11.4K` code-edit decode at `T=1.0, top_p=0.95` from `110.3 → 142.7 tok/s` (`+29.4%`) with zero distribution bias.
- **8-Slot Rolling Prefix Cache & `ToolRawHistoryCache`:** Sub-4K `512`-tok checkpoint grid (`84.2%` hit rate on `1,825`-tok few-shot prompts), rolling turn checkpoints (`97.7%` hit on branch rollbacks), and raw tool-call JSON ring replay (`97.6%` hit across client JSON re-serialization).
- **Live Incremental SSE `tool_calls` Streaming:** UTF-8-safe streaming JSON parser emits tool call headers and argument deltas token-by-token (`424` chunks verified, 0 duplicates).
- **CLI Versioning & Official License Metadata:** Added `--version` / `-V` CLI flag, `"version"` in `/health`, and canonical SPDX Apache-2.0 `LICENSE` formatting.

### Changed
- Always-on `RSFIX` positive-mass verifier fallback; Sun et al. Block Verification (`SQUID_BLOCK_VERIFY=1`) and `Q_TOK=64` prefill attention (`SQUID_PREFILL_Q64=1`) kept opt-in after empirical A/B profiling.

## [v3.7.0] — 2026-11-26

### Added
- **Hybrid AVX-VNNI CPU MoE (`cuda/cpu_moe.cpp`):** 12-thread (`6P+6E`) Q8-activation NVFP4 `_mm256_dpbusd_avx_epi32` worker pool with 32-row atomic work-stealing, `WAITPKG` (`umwait`), and cost-balanced `2/3` miss split (`~75–82 GB/s` combined DDR5 + PCIe miss bandwidth).
- **Blackwell Tensor Core Cached MoE (`B>=4`):** `mma.m16n8k16` cached MoE with 42 regs and **0 register spills** across all 551 audited kernels, plus PDL and selective `L2::evict_first` hints.
- **5-Way `{1, 2, 3, 4, 8}` Adaptive DSpark Switch:** Hardware-calibrated step costs `{1.00, 1.45, 1.70, 2.05, 3.20}` (`gamma=1.50`, `tslope=0.40`) + PDL and early-exit Markov RNN drafting — reaching **`103.1–107.1 tok/s`** 6-prompt median, **`136.2 tok/s`** peak SSE, and **`93.5%`** of the physical `B=1` hybrid roofline (`10.01 ms/tok`).
- **Instant Bulk-DMA Prefill Tail Refill & Dynamic Tail Sizing:** Warm-preserving prefill LRFU (`decay=0.985`) and `pb=128/256/512/1024` dynamic prefill tails.
- **Spark-Bench Quality Gate & `chat_template_kwargs`:** Verified **`286 / 295`** on `knapcio/inkling-2-mini-spark-bench` (`GSM8K 195/200`, `qeval 71/75`, `lc 20/20`, `127.2 tok/s` median / `161.9 tok/s` peak) and added `reasoning_effort` / `enable_thinking` support.

## [v3.2.0] — 2026-11-26

- Upstream `squid5090` (`124afb8`) sync for 8 GB RTX 5060: CUDA 13.4+ `L2::evict_first` streaming weight hints, 3 KB SMEM L1-cached LM-head & DSpark MMA kernels, DSpark local TC Split-K, embedded corpus-frequency 32K draft shortlist (`src/draft_shortlist.bin`), post-`[DONE]` deferred checkpointing, and lazy content-addressed 0-VRAM vision prefix caching (`0.0 ms` Turn 2+ ViT reuse + Anthropic base64 image support).

## [v3.1.0] — 2026-11-25

- Upstream `squid5090` sync: 1-launch PDL Blackwell Cluster Attention (`CS=4/8`), native `sm_120a` block-scaled Tensor Core prefill MoE (`2,025 tok/s` @ 24K), 128-CTA PDL multi-block stochastic sampler, zero-overhead LoopCD-Hidden (`arXiv:2610.02185`), and `O(1)` tool history tracking.

## [v3.0.0] — 2026-11-24

- Hybrid AVX2 FP32 CPU + GPU PCIe MoE miss execution (`85.8 tok/s` median), auto idle-sleep (`7.47 GiB → 0.22 GiB` VRAM, ~326 ms wake), host-RAM conversation parking (`--slot-keep 4`, `11–28 ms` swap), and on-demand 23-layer BF16 ViT vision (`0.0 MiB` permanent VRAM).

## [v2.1.0] — 2026-11-24

- Training-free Last-Loop KV Sharing (`256K` context in 8 GB VRAM) and dynamic SM-aware grids.

## [v1.0.0] — 2026-11-24

- Initial 8 GB RTX 5060 port of `tensorfish/squid5090`: unified 3,072-expert NVFP4 VRAM cache (`4.27 MiB` slots), aging LRFU with async overlapped `copy_stream` promotion, CTA-uniform dual-path PCIe `uint4` kernels (`23.9 GB/s`), built-in warm-cache snapshot, and `--api-key-file`.
