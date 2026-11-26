// On-Demand Vision Encoder Glue for Inkling-2-Mini (0 MiB permanent VRAM).
// Lazily mmaps model-00001-of-00001.safetensors to resolve all 422 BF16 `model.visual.*`
// tensors and runs the 23-layer BF16 Tensor Core ViT + 2x2 merge projector via borrowed VRAM scratch.

const std = @import("std");
const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const safetensors = @import("safetensors.zig");
const tiger = @import("tiger.zig");

pub const image_token_id: i32 = 200054;
pub const max_image_tokens: u32 = 1024;
pub const max_images_per_req: usize = 4;

pub const ImageSpan = struct {
    prompt_pos: u32 = 0,
    num_tokens: u32 = 0,
    embeds: []f32 = &.{}, // [num_tokens * constants.hidden_dim] FP32 host slice
    raw_bytes: []const u8 = &.{}, // optional compressed PNG/JPEG bytes for lazy encoding after prefix-cache match
    embed_slot: []f32 = &.{}, // backing buffer for lazy encoding
    encoded: bool = false,
};

pub const ProbeResult = struct {
    num_tokens: u32,
    cols: u32,
    rows: u32,
    hash: u64,
};

pub const EncodeResult = struct {
    num_tokens: u32,
    cols: u32,
    rows: u32,
};

fn splitmix64_mix(z_in: u64) u64 {
    var z: u64 = (z_in ^ (z_in >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Content-addressed negative synthetic token ID for visual tile `k` of an image with `hash`.
/// Enables automatic 0 ms prefix-cache & parked-slot reuse across multi-turn vision chats.
pub fn image_token_id_for(hash: u64, k: u32) i32 {
    const v: u64 = splitmix64_mix(hash ^ ((@as(u64, k) +% 1) *% 0x9E3779B97F4A7C15)) >> 33;
    return @intCast(-1 - @as(i64, @intCast(v)));
}

pub fn probe_image_bytes(img_bytes: []const u8, max_tokens: u32) ?ProbeResult {
    if (img_bytes.len < 8) return null;
    const cap_toks: u32 = @min(max_tokens, max_image_tokens);
    var out_n: i32 = 0;
    var out_c: i32 = 0;
    var out_r: i32 = 0;
    var out_h: u64 = 0;
    const rc = cuda.squid_vision_probe_image(
        img_bytes.ptr,
        img_bytes.len,
        @intCast(cap_toks),
        &out_n,
        &out_c,
        &out_r,
        &out_h,
    );
    if (rc != 0 or out_n <= 0) return null;
    return .{
        .num_tokens = @intCast(out_n),
        .cols = @intCast(out_c),
        .rows = @intCast(out_r),
        .hash = out_h,
    };
}

pub fn encode_image_bytes(
    model_dir: []const u8,
    img_bytes: []const u8,
    scratch_dev: ?*anyopaque,
    scratch_bytes: usize,
    max_tokens: u32,
    out_embeds: []f32,
    stream: ?*anyopaque,
) ?EncodeResult {
    if (model_dir.len == 0 or img_bytes.len < 8) return null;
    const cap_toks: u32 = @min(max_tokens, max_image_tokens);
    if (out_embeds.len < @as(usize, cap_toks) * constants.hidden_dim) return null;

    var path_buf: [1024]u8 = undefined;
    const path_z = tiger.format_path_z(&path_buf, model_dir, "/model-00001-of-00001.safetensors");
    var sf: safetensors.SafetensorsFile = .{};
    if (!sf.open_file(path_z)) {
        tiger.print_stderr("[vision] Failed to open safetensors in '{s}'\n", .{model_dir});
        return null;
    }
    defer sf.close_file();

    var vw: cuda.CVisionWeightsHost = .{};
    vw.class_embedding = sf.data_ptr(sf.lookup_optional("model.visual.backbone.embeddings.class_embedding") orelse return null);
    vw.cls_position_embedding = sf.data_ptr(sf.lookup_optional("model.visual.backbone.embeddings.cls_position_embedding") orelse return null);
    vw.patch_embedding_w = sf.data_ptr(sf.lookup_optional("model.visual.backbone.embeddings.patch_embedding.weight") orelse return null);
    vw.position_embedding = sf.data_ptr(sf.lookup_optional("model.visual.backbone.embeddings.position_embedding") orelse return null);
    vw.ln_pre_w = sf.data_ptr(sf.lookup_optional("model.visual.backbone.ln_pre.weight") orelse return null);
    vw.ln_pre_b = sf.data_ptr(sf.lookup_optional("model.visual.backbone.ln_pre.bias") orelse return null);
    vw.proj_in_w = sf.data_ptr(sf.lookup_optional("model.visual.proj_in.weight") orelse return null);
    vw.proj_out_w = sf.data_ptr(sf.lookup_optional("model.visual.proj_out.weight") orelse return null);

    var nbuf: [160]u8 = undefined;
    var l: u32 = 0;
    while (l < 23) : (l += 1) {
        const L = &vw.layers[l];
        L.ln1_w = lookup_layer(&sf, &nbuf, l, "ln_1.weight") orelse return null;
        L.ln1_b = lookup_layer(&sf, &nbuf, l, "ln_1.bias") orelse return null;
        L.q_w = lookup_layer(&sf, &nbuf, l, "self_attn.q_proj.weight") orelse return null;
        L.q_b = lookup_layer(&sf, &nbuf, l, "self_attn.q_proj.bias") orelse return null;
        L.k_w = lookup_layer(&sf, &nbuf, l, "self_attn.k_proj.weight") orelse return null;
        L.k_b = lookup_layer(&sf, &nbuf, l, "self_attn.k_proj.bias") orelse return null;
        L.v_w = lookup_layer(&sf, &nbuf, l, "self_attn.v_proj.weight") orelse return null;
        L.v_b = lookup_layer(&sf, &nbuf, l, "self_attn.v_proj.bias") orelse return null;
        L.o_w = lookup_layer(&sf, &nbuf, l, "self_attn.out_proj.weight") orelse return null;
        L.o_b = lookup_layer(&sf, &nbuf, l, "self_attn.out_proj.bias") orelse return null;
        L.ls1_gamma = lookup_layer(&sf, &nbuf, l, "ls_1.gamma") orelse return null;
        L.ln2_w = lookup_layer(&sf, &nbuf, l, "ln_2.weight") orelse return null;
        L.ln2_b = lookup_layer(&sf, &nbuf, l, "ln_2.bias") orelse return null;
        L.fc_w = lookup_layer(&sf, &nbuf, l, "mlp.c_fc.weight") orelse return null;
        L.fc_b = lookup_layer(&sf, &nbuf, l, "mlp.c_fc.bias") orelse return null;
        L.proj_w = lookup_layer(&sf, &nbuf, l, "mlp.c_proj.weight") orelse return null;
        L.proj_b = lookup_layer(&sf, &nbuf, l, "mlp.c_proj.bias") orelse return null;
        L.ls2_gamma = lookup_layer(&sf, &nbuf, l, "ls_2.gamma") orelse return null;
    }

    var out_n: i32 = 0;
    var out_c: i32 = 0;
    var out_r: i32 = 0;
    const t0 = tiger.clock_monotonic_ns();
    const rc = cuda.squid_vision_encode_image(
        img_bytes.ptr,
        img_bytes.len,
        &vw,
        scratch_dev,
        scratch_bytes,
        @intCast(cap_toks),
        out_embeds.ptr,
        &out_n,
        &out_c,
        &out_r,
        stream,
    );
    if (rc != 0 or out_n <= 0) {
        tiger.print_stderr("[vision] squid_vision_encode_image failed rc={d}\n", .{rc});
        return null;
    }
    const ms: f64 = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1_000_000.0;
    tiger.print_stdout(
        "[vision] Encoded image -> {d}x{d} grid ({d} visual tokens) in {d:.1}ms (0 MiB permanent VRAM)\n",
        .{ out_c, out_r, out_n, ms },
    );
    return .{
        .num_tokens = @intCast(out_n),
        .cols = @intCast(out_c),
        .rows = @intCast(out_r),
    };
}

fn lookup_layer(sf: *const safetensors.SafetensorsFile, buf: []u8, layer_idx: u32, suffix: []const u8) ?*const anyopaque {
    const name = std.fmt.bufPrint(buf, "model.visual.backbone.layers.{d}.{s}", .{ layer_idx, suffix }) catch return null;
    const info = sf.lookup_optional(name) orelse return null;
    return sf.data_ptr(info);
}

/// Decode a `data:image/...;base64,...` data URI (or raw base64 string) into `out_buf`.
pub fn decode_base64_url(url: []const u8, out_buf: []u8) ?[]u8 {
    var payload = url;
    if (std.mem.startsWith(u8, payload, "data:")) {
        const comma = std.mem.indexOfScalar(u8, payload, ',') orelse return null;
        payload = payload[comma + 1 ..];
    }
    var acc: u32 = 0;
    var bits: u5 = 0;
    var out_len: usize = 0;
    for (payload) |ch| {
        if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n') continue;
        if (ch == '=') break;
        const val: u32 = if (ch >= 'A' and ch <= 'Z')
            ch - 'A'
        else if (ch >= 'a' and ch <= 'z')
            ch - 'a' + 26
        else if (ch >= '0' and ch <= '9')
            ch - '0' + 52
        else if (ch == '+' or ch == '-')
            62
        else if (ch == '/' or ch == '_')
            63
        else
            return null;
        acc = (acc << 6) | val;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (out_len >= out_buf.len) return null;
            out_buf[out_len] = @intCast((acc >> bits) & 0xFF);
            out_len += 1;
        }
    }
    if (out_len < 8) return null;
    return out_buf[0..out_len];
}
