#pragma once

#include "config.hpp"
#include <cstdint>

namespace squid {

constexpr int CPU_MOE_MAX_TASKS = MTP_BLOCK_SIZE * TOP_K; // 64
constexpr int CPU_MOE_MAX_OUTS  = MTP_BLOCK_SIZE * TOP_K; // 64

// Zero-copy pinned mapped doorbell shared between GPU kernels and persistent AVX2 CPU worker pool.
// GPU writes task metadata + x_h + req_seq (release); CPU computes assigned RAM misses and writes
// out_buf + ack_seq (release); GPU waits on ack_seq (acquire) and scatters out_buf into moe_slot_out.
struct alignas(64) CpuMoeBell {
    volatile uint32_t req_seq;
    volatile uint32_t ack_seq;
    volatile uint32_t abort_flag;
    int32_t layer_idx;
    int32_t batch_size;
    int32_t n_tasks;
    int32_t n_outs;
    int32_t pad0;

    int32_t task_expert[CPU_MOE_MAX_TASKS];
    uint8_t task_token_mask[CPU_MOE_MAX_TASKS];
    uint8_t task_out_base[CPU_MOE_MAX_TASKS]; // starting index in out_buf for this task
    uint8_t out_tok[CPU_MOE_MAX_OUTS];        // token b in 0..B-1 for each output
    uint8_t out_slot[CPU_MOE_MAX_OUTS];       // slot s in 0..TOP_K-1 for each output
    float   out_rw[CPU_MOE_MAX_OUTS];         // router weight for each output

    alignas(64) uint16_t x_h[MTP_BLOCK_SIZE * HIDDEN];         // 36 KiB fp16 inputs
    alignas(64) float    out_buf[CPU_MOE_MAX_OUTS][HIDDEN];    // 576 KiB weighted expert outputs
};

// Device-resident VRAM scratch state for CUDA-Graph-safe CPU MoE dispatch (zero PCIe read when n_tasks == 0!).
struct alignas(64) CpuMoeDevState {
    uint32_t seq;
    uint32_t n_tasks;
    uint32_t n_outs;
    uint32_t enabled;       // 0 = disabled, 1 = active
    uint32_t split_num;     // cpu_take = (n_misses * split_num + split_round) / split_den
    uint32_t split_round;
    uint32_t split_den;
    uint32_t max_batch;     // max B for which CPU participates (e.g. 1, 2, 4, or 8)
    uint64_t skip_mask;     // bit u (B>1) or bit slot (B=1) = 1 if CPU computes it
};

struct CpuMoePool;

} // namespace squid

extern "C" {

// Create persistent AVX2 Q8-NVFP4 CPU worker pool.
// `host_experts[24]` are host pointers to packed [128, EXPERT_SLOT_BYTES] banks;
// `host_scale_w13[24*128]` and `host_scale_w2[24*128]` are host F32 per-expert outer scales.
void* squid_cpu_moe_create(
    void* h_bell,
    const uint8_t* const* host_experts,
    const float* host_scale_w13,
    const float* host_scale_w2,
    int32_t num_threads);

void squid_cpu_moe_wake(void* pool);
void squid_cpu_moe_nap(void* pool);
void squid_cpu_moe_destroy(void* pool);

} // extern "C"
