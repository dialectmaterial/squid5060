#pragma once

#include <cstdint>
#include <cstddef>

extern "C" {

// Pointers into mmap'd model-00001-of-00001.safetensors for all 422 BF16 ViT tensors.
struct CVisionLayerPtrs {
    const void* ln1_w;       // [1024]
    const void* ln1_b;       // [1024]
    const void* q_w;         // [1024, 1024]
    const void* q_b;         // [1024]
    const void* k_w;         // [1024, 1024]
    const void* k_b;         // [1024]
    const void* v_w;         // [1024, 1024]
    const void* v_b;         // [1024]
    const void* o_w;         // [1024, 1024]
    const void* o_b;         // [1024]
    const void* ls1_gamma;   // [1024]
    const void* ln2_w;       // [1024]
    const void* ln2_b;       // [1024]
    const void* fc_w;        // [4096, 1024]
    const void* fc_b;        // [4096]
    const void* proj_w;      // [1024, 4096]
    const void* proj_b;      // [1024]
    const void* ls2_gamma;   // [1024]
};

struct CVisionWeightsHost {
    const void* class_embedding;         // [1024]
    const void* cls_position_embedding;  // [1024]
    const void* patch_embedding_w;       // [1024, 1200]
    const void* position_embedding;      // [1024, 1024] (32x32x1024)
    const void* ln_pre_w;                // [1024]
    const void* ln_pre_b;                // [1024]
    CVisionLayerPtrs layers[23];
    const void* proj_in_w;               // [4096, 4096]
    const void* proj_out_w;              // [2304, 4096]
};

// Fast header-only probe of PNG or JPEG bytes (no pixel decode, zero GPU work):
// computes exact visual token count matching `squid_vision_encode_image` and a content hash
// for multi-turn prefix-cache reuse via negative synthetic token IDs.
// Returns 0 on success, non-zero error code on invalid header.
int32_t squid_vision_probe_image(
    const uint8_t* img_bytes,
    size_t img_len,
    int32_t max_tokens,
    int32_t* out_num_tokens,
    int32_t* out_cols,
    int32_t* out_rows,
    uint64_t* out_hash);

// Decode PNG or JPEG bytes and run Inkling-2-Mini's 23-layer ViT + projector on GPU using
// `scratch_dev` (`scratch_bytes` borrowed from prefill tail, or temporary cudaMalloc if null).
// Writes `out_num_tokens` (<= max_tokens) and `[out_num_tokens * 2304]` FP32 embeddings into `out_embeds_host`.
// Returns 0 on success, non-zero error code on decode/size failure.
int32_t squid_vision_encode_image(
    const uint8_t* img_bytes,
    size_t img_len,
    const CVisionWeightsHost* weights,
    void* scratch_dev,
    size_t scratch_bytes,
    int32_t max_tokens,
    float* out_embeds_host,
    int32_t* out_num_tokens,
    int32_t* out_cols,
    int32_t* out_rows,
    void* stream);

} // extern "C"
