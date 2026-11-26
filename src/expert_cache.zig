// TigerStyle Unified 8 GB VRAM MoE Expert Cache + Zero-Copy Pinned Host Fallback Manager.
// Deduplicates all 48 logical visits onto the 24 physical layers (3,072 physical experts total).
// Loop layers (6..17, visited 3x per token) naturally accumulate 3x touch weight and higher LRFU priority.
// Supports Phase 2A Prefill-Scratch Tail Aliasing: prefill borrows tail slots from d_cache_pool and
// returns them to a zero-eviction free list immediately after prefill so decode gets 100% of VRAM!
// Zero dynamic allocations post-init(); zero CUDA Graph invalidations on slot updates!

const std = @import("std");
const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const hot_experts = @import("hot_experts.zig");
const tiger = @import("tiger.zig");
const weights_mod = @import("weights.zig");

const total_experts: usize = constants.total_physical_experts; // 3072
const max_candidates_per_step: usize = 3072;
const async_staging_slots: usize = 8;

const PromoteCand = struct {
    eid: u16,
    score: u32,
};

const InflightPromote = struct {
    eid: u16 = 0,
    slot: u32 = 0,
};

pub const ExpertCache = struct {
    enabled: bool = false,
    async_promote: bool = true,
    use_lrfu: bool = true, // Exact global-min Aging-LFU with hysteresis (vs CLOCK)
    num_slots: u32 = 0,
    capacity_bytes: u64 = 0,
    kv_reserved_tail: u32 = 0,
    prefill_reserved_tail: u32 = 0,

    // Device allocations (1 contiguous cache pool + 12 KiB slot map + 12 KiB touch counts)
    d_cache_pool: ?[*]u8 = null,
    d_slot_map: ?[*]i32 = null,
    d_touch_counts: ?[*]u32 = null,

    // Dedicated async DMA stream + event for zero-stall overlapped promotion
    copy_stream: ?*anyopaque = null,
    copy_done_event: ?*anyopaque = null,

    // Pinned host mirror & telemetry readback (24 KiB total, zero CPU stall on async copies)
    h_slot_map: ?[*]i32 = null,
    h_touch_counts: ?[*]u32 = null,

    // Host replacement state (static Startup-allocated slices)
    slot_owner: []i32 = &.{},
    clock_bits: []u8 = &.{},
    slot_reserved: []bool = &.{}, // true if in ready_free / inflight / quarantine / prefill tail
    free_list: []u32 = &.{}, // unreserved, unowned slots returned from prefill tail (zero-eviction refill!)
    free_list_len: usize = 0,
    ewma_score: [total_experts]f32 = @splat(0.0),
    recent_misses: [total_experts]u16 = @splat(0),
    is_inflight: [total_experts]bool = @splat(false),
    clock_hand: u32 = 0,
    free_slots_used: u32 = 0,
    step_counter: u64 = 0,

    // Async overlap ring state (max async_staging_slots in each list)
    ready_free: [async_staging_slots]u32 = @splat(0),
    ready_free_len: usize = 0,
    quarantine: [async_staging_slots]u32 = @splat(0),
    quarantine_len: usize = 0,
    inflight: [async_staging_slots]InflightPromote = @splat(.{}),
    inflight_len: usize = 0,

    // Tunable policy parameters (overridable via env vars for hill-climbing)
    admit_threshold: u16 = 2,
    loop_clock_max: u8 = 6,
    nonloop_clock_max: u8 = 3,
    lrfu_decay: f32 = 0.998,
    lrfu_hysteresis: f32 = 1.50,
    lrfu_pf_decay: f32 = 0.985,
    lrfu_pf_scale: f32 = 6.0,
    loop_score_mult: f32 = 1.00,
    tail_refill: bool = true,
    warm_prior_max: f32 = 4.0,
    max_promotes_per_decode: u32 = 8,
    max_promotes_per_prefill_warm: u32 = 32,
    sleeping: bool = false,

    // Telemetry counters
    decode_accesses: u64 = 0,
    decode_hits: u64 = 0,
    decode_misses: u64 = 0,
    decode_promotions: u64 = 0,
    decode_evictions: u64 = 0,
    zone_accesses: [3]u64 = .{ 0, 0, 0 }, // 0=prelude(0..5), 1=loop(6..17), 2=coda(18..23)
    zone_hits: [3]u64 = .{ 0, 0, 0 },
    prefill_accesses: u64 = 0,
    prefill_hits: u64 = 0,
    prefill_misses: u64 = 0,
    prefill_promotions: u64 = 0,

    // Static scratch buffer for sorting promotion candidates (zero malloc in hot path!)
    cand_buf: [max_candidates_per_step]PromoteCand = undefined,

    // Optional warm-cache snapshot file path (12 KiB LRFU score snapshot -> ~0.25s bulk DMA warmup!)
    cache_file_buf: [1024]u8 = @splat(0),
    cache_file_len: usize = 0,

    pub fn init(
        self: *ExpertCache,
        weights: *weights_mod.ModelWeights,
        requested_cache_mib: u32,
        headroom_mib: u32,
        cache_file_path: []const u8,
        stream: ?*anyopaque,
    ) void {
        tiger.assert(!self.enabled);
        tiger.assert(weights.use_expert_cache);

        // Read optional env overrides for hill-climbing cache policies
        if (parse_env_u32("SQUID_CACHE_ADMIT")) |v| self.admit_threshold = @intCast(@min(@max(v, 1), 64));
        if (parse_env_u32("SQUID_CACHE_LOOP_MAX")) |v| self.loop_clock_max = @intCast(@min(@max(v, 1), 15));
        if (parse_env_u32("SQUID_CACHE_NONLOOP_MAX")) |v| self.nonloop_clock_max = @intCast(@min(@max(v, 1), 15));
        if (parse_env_u32("SQUID_CACHE_MAX_PROMOTE")) |v| self.max_promotes_per_decode = @min(v, 256);
        if (parse_env_u32("SQUID_CACHE_ASYNC")) |v| self.async_promote = (v != 0);
        if (parse_env_u32("SQUID_CACHE_DECAY")) |v| {
            if (v >= 800 and v <= 1000) self.lrfu_decay = @as(f32, @floatFromInt(v)) / 1000.0;
        }
        if (parse_env_u32("SQUID_CACHE_HYST")) |v| {
            if (v >= 100 and v <= 500) self.lrfu_hysteresis = @as(f32, @floatFromInt(v)) / 100.0;
        }
        if (parse_env_u32("SQUID_CACHE_PF_DECAY")) |v| {
            if (v >= 800 and v <= 1000) self.lrfu_pf_decay = @as(f32, @floatFromInt(v)) / 1000.0;
        }
        if (parse_env_u32("SQUID_CACHE_PF_SCALE")) |v| {
            if (v <= 500) self.lrfu_pf_scale = @as(f32, @floatFromInt(v)) / 10.0;
        }
        if (parse_env_u32("SQUID_CACHE_PF_PROMOTE")) |v| {
            self.max_promotes_per_prefill_warm = @min(v, 256);
        }
        if (parse_env_u32("SQUID_CACHE_LOOP_MULT")) |v| {
            if (v >= 100 and v <= 400) self.loop_score_mult = @as(f32, @floatFromInt(v)) / 100.0;
        }
        if (parse_env_u32("SQUID_CACHE_TAIL_REFILL")) |v| {
            self.tail_refill = (v != 0);
        }
        if (parse_env_u32("SQUID_WARM_PRIOR")) |v| {
            if (v >= 1 and v <= 1000) self.warm_prior_max = @as(f32, @floatFromInt(v)) / 10.0;
        }
        if (tiger.getenv("SQUID_CACHE_POLICY")) |p| {
            const ps = std.mem.span(p);
            if (std.mem.eql(u8, ps, "clock")) self.use_lrfu = false;
            if (std.mem.eql(u8, ps, "lrfu")) self.use_lrfu = true;
        }

        var free_vram: usize = 0;
        var total_vram: usize = 0;
        tiger.assert_cuda(cuda.squid_cuda_mem_get_info(&free_vram, &total_vram));

        const headroom_bytes: u64 = @as(u64, headroom_mib) * 1024 * 1024;
        const avail_bytes: u64 = if (free_vram > headroom_bytes) (free_vram - headroom_bytes) else 0;
        var budget_bytes: u64 = avail_bytes;
        if (requested_cache_mib > 0) {
            const req_bytes: u64 = @as(u64, requested_cache_mib) * 1024 * 1024;
            if (req_bytes < budget_bytes) budget_bytes = req_bytes;
        }

        const slot_bytes: u64 = constants.expert_slot_bytes;
        const slots: u32 = @intCast(@min(total_experts, budget_bytes / slot_bytes));
        if (slots < 32) {
            tiger.print_stderr(
                "[cache] ERROR: insufficient VRAM for expert cache ({d:.1} MiB free, headroom={d} MiB -> {d} slots, need >= 32 slots / 137 MiB). Increase --cache-mib or reduce --ctx.\n",
                .{ @as(f64, @floatFromInt(free_vram)) / (1024.0 * 1024.0), headroom_mib, slots },
            );
            std.process.exit(1);
        }

        const pool_bytes: u64 = @as(u64, slots) * slot_bytes;
        var raw_pool: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&raw_pool, @intCast(pool_bytes)));
        self.d_cache_pool = @ptrCast(raw_pool.?);

        var raw_smap: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&raw_smap, total_experts * @sizeOf(i32)));
        self.d_slot_map = @ptrCast(@alignCast(raw_smap.?));

        var raw_touch: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&raw_touch, total_experts * @sizeOf(u32)));
        self.d_touch_counts = @ptrCast(@alignCast(raw_touch.?));

        var raw_hsmap: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&raw_hsmap, total_experts * @sizeOf(i32), 0));
        self.h_slot_map = @ptrCast(@alignCast(raw_hsmap.?));

        var raw_htouch: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&raw_htouch, total_experts * @sizeOf(u32), 0));
        self.h_touch_counts = @ptrCast(@alignCast(raw_htouch.?));

        self.slot_owner = tiger.alloc_host_zeroed(i32, slots);
        self.clock_bits = tiger.alloc_host_zeroed(u8, slots);
        self.slot_reserved = tiger.alloc_host_zeroed(bool, slots);
        self.free_list = tiger.alloc_host_zeroed(u32, slots);
        self.free_list_len = 0;
        for (self.slot_owner) |*o| o.* = -1;

        tiger.assert_cuda(cuda.squid_cuda_stream_create(&self.copy_stream));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&self.copy_done_event, 1));
        if (self.async_promote and slots <= async_staging_slots + 16) {
            self.async_promote = false;
        }
        if (self.async_promote) {
            var k: usize = 0;
            while (k < async_staging_slots) : (k += 1) {
                const s: u32 = slots - 1 - @as(u32, @intCast(k));
                self.slot_reserved[s] = true;
                self.ready_free[k] = s;
            }
            self.ready_free_len = async_staging_slots;
        }

        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            self.h_slot_map.?[eid] = -1;
            self.h_touch_counts.?[eid] = 0;
            self.recent_misses[eid] = 0;
        }
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
            @ptrCast(self.d_slot_map.?),
            @ptrCast(self.h_slot_map.?),
            total_experts * @sizeOf(i32),
            stream,
        ));
        tiger.assert_cuda(cuda.squid_cuda_memset_async(
            @ptrCast(self.d_touch_counts.?),
            0,
            total_experts * @sizeOf(u32),
            stream,
        ));
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));

        self.num_slots = slots;
        self.capacity_bytes = pool_bytes;
        self.kv_reserved_tail = 0;
        self.prefill_reserved_tail = 0;
        self.clock_hand = 0;
        self.free_slots_used = 0;
        self.enabled = true;

        var l: u32 = 0;
        while (l < constants.layer_count) : (l += 1) {
            const base_off: usize = @as(usize, l) * constants.expert_count;
            const lw = &weights.layers[l];
            lw.w13.cache_pool = self.d_cache_pool;
            lw.w13.slot_map = self.d_slot_map.? + base_off;
            lw.w13.touch_counts = self.d_touch_counts.? + base_off;
            lw.w2.cache_pool = self.d_cache_pool;
            lw.w2.slot_map = self.d_slot_map.? + base_off;
            lw.w2.touch_counts = null;
        }

        const pool_mib: f64 = @as(f64, @floatFromInt(pool_bytes)) / (1024.0 * 1024.0);
        const free_mib: f64 = @as(f64, @floatFromInt(free_vram)) / (1024.0 * 1024.0);
        const cov_pct: f64 = 100.0 * @as(f64, @floatFromInt(slots)) / @as(f64, @floatFromInt(total_experts));
        tiger.print_stdout(
            "[moe-cache] Unified VRAM pool: {d:.1} MiB ({d}/{d} physical experts = {d:.1}%%, free_before={d:.1} MiB, policy={s}, async={d})\n",
            .{ pool_mib, slots, total_experts, cov_pct, free_mib, if (self.use_lrfu) "lrfu" else "clock", @as(u8, if (self.async_promote) 1 else 0) },
        );

        if (cache_file_path.len > 0 and cache_file_path.len + 1 < self.cache_file_buf.len) {
            @memcpy(self.cache_file_buf[0..cache_file_path.len], cache_file_path);
            self.cache_file_buf[cache_file_path.len] = 0;
            self.cache_file_len = cache_file_path.len;
            if (!self.load_warm_snapshot(weights, stream)) {
                self.load_builtin_profile(weights, stream);
            }
        }
    }

    /// Release all VRAM allocations owned by the expert cache while retaining live LRFU scores in host RAM.
    pub fn sleep_vram(self: *ExpertCache, weights: *weights_mod.ModelWeights, stream: ?*anyopaque) void {
        if (!self.enabled or self.sleeping) return;
        self.flush_async(stream);
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));
        if (self.d_cache_pool) |p| tiger.assert_cuda(cuda.squid_cuda_free(@ptrCast(p)));
        if (self.d_slot_map) |p| tiger.assert_cuda(cuda.squid_cuda_free(@ptrCast(p)));
        if (self.d_touch_counts) |p| tiger.assert_cuda(cuda.squid_cuda_free(@ptrCast(p)));
        self.d_cache_pool = null;
        self.d_slot_map = null;
        self.d_touch_counts = null;
        var l: u32 = 0;
        while (l < constants.layer_count) : (l += 1) {
            weights.layers[l].w13.cache_pool = null;
            weights.layers[l].w13.slot_map = null;
            weights.layers[l].w13.touch_counts = null;
            weights.layers[l].w2.cache_pool = null;
            weights.layers[l].w2.slot_map = null;
            weights.layers[l].w2.touch_counts = null;
        }
        self.sleeping = true;
    }

    /// Re-allocate VRAM pool and bulk-DMA-reload the hottest experts from live LRFU scores (~250 ms).
    pub fn wake_vram(self: *ExpertCache, weights: *weights_mod.ModelWeights, stream: ?*anyopaque) void {
        if (!self.enabled or !self.sleeping) return;
        const slots: u32 = self.num_slots;
        const pool_bytes: u64 = self.capacity_bytes;

        var raw_pool: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&raw_pool, @intCast(pool_bytes)));
        self.d_cache_pool = @ptrCast(raw_pool.?);

        var raw_smap: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&raw_smap, total_experts * @sizeOf(i32)));
        self.d_slot_map = @ptrCast(@alignCast(raw_smap.?));

        var raw_touch: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc(&raw_touch, total_experts * @sizeOf(u32)));
        self.d_touch_counts = @ptrCast(@alignCast(raw_touch.?));

        // Reset slot tracking and re-reserve async staging slots
        for (self.slot_owner) |*o| o.* = -1;
        @memset(self.clock_bits, 0);
        @memset(self.slot_reserved, false);
        self.free_list_len = 0;
        self.free_slots_used = 0;
        self.kv_reserved_tail = 0;
        self.prefill_reserved_tail = 0;
        self.clock_hand = 0;
        self.ready_free_len = 0;
        self.quarantine_len = 0;
        self.inflight_len = 0;
        @memset(&self.is_inflight, false);
        if (self.async_promote) {
            var k: usize = 0;
            while (k < async_staging_slots) : (k += 1) {
                const s: u32 = slots - 1 - @as(u32, @intCast(k));
                self.slot_reserved[s] = true;
                self.ready_free[k] = s;
            }
            self.ready_free_len = async_staging_slots;
        }

        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            self.h_slot_map.?[eid] = -1;
            self.h_touch_counts.?[eid] = 0;
        }
        tiger.assert_cuda(cuda.squid_cuda_memset_async(
            @ptrCast(self.d_touch_counts.?),
            0,
            total_experts * @sizeOf(u32),
            stream,
        ));

        var l: u32 = 0;
        while (l < constants.layer_count) : (l += 1) {
            const base_off: usize = @as(usize, l) * constants.expert_count;
            const lw = &weights.layers[l];
            lw.w13.cache_pool = self.d_cache_pool;
            lw.w13.slot_map = self.d_slot_map.? + base_off;
            lw.w13.touch_counts = self.d_touch_counts.? + base_off;
            lw.w2.cache_pool = self.d_cache_pool;
            lw.w2.slot_map = self.d_slot_map.? + base_off;
            lw.w2.touch_counts = null;
        }

        self.sleeping = false;
        self.bulk_load_from_scores(weights, stream, "wake");
    }

    pub fn deinit(self: *ExpertCache) void {
        if (!self.enabled) return;
        self.save_warm_snapshot();
        if (self.copy_done_event) |e| tiger.assert_cuda(cuda.squid_cuda_event_destroy(e));
        if (self.copy_stream) |s| tiger.assert_cuda(cuda.squid_cuda_stream_destroy(s));
        if (self.d_cache_pool) |p| tiger.assert_cuda(cuda.squid_cuda_free(@ptrCast(p)));
        if (self.d_slot_map) |p| tiger.assert_cuda(cuda.squid_cuda_free(@ptrCast(p)));
        if (self.d_touch_counts) |p| tiger.assert_cuda(cuda.squid_cuda_free(@ptrCast(p)));
        if (self.h_slot_map) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        if (self.h_touch_counts) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        if (self.slot_owner.len > 0) tiger.free_host_slice(i32, self.slot_owner);
        if (self.clock_bits.len > 0) tiger.free_host_slice(u8, self.clock_bits);
        if (self.slot_reserved.len > 0) tiger.free_host_slice(bool, self.slot_reserved);
        if (self.free_list.len > 0) tiger.free_host_slice(u32, self.free_list);
        self.* = .{};
    }

    pub inline fn usable_slots(self: *const ExpertCache) u32 {
        return self.num_slots - self.kv_reserved_tail - self.prefill_reserved_tail;
    }

    /// Reserve or release `n_kv` slots at the very end of `d_cache_pool` for lazy grow-on-demand global KV!
    /// Layout: [0 .. usable) active experts | [usable .. kv_start) prefill scratch | [kv_start .. num_slots) lazy KV slab.
    /// Must be called while prefill scratch is not borrowed (`prefill_reserved_tail == 0`).
    pub fn set_kv_reserved_tail(
        self: *ExpertCache,
        n_kv: u32,
        weights: *const weights_mod.ModelWeights,
        stream: ?*anyopaque,
    ) ?[*]u8 {
        tiger.assert(self.enabled);
        tiger.assert(self.prefill_reserved_tail == 0);
        tiger.assert(n_kv + 32 < self.num_slots);
        const slot_bytes: usize = @intCast(constants.expert_slot_bytes);
        if (n_kv == self.kv_reserved_tail) {
            if (n_kv == 0) return null;
            const kv_start: u32 = self.num_slots - n_kv;
            return self.d_cache_pool.? + @as(usize, kv_start) * slot_bytes;
        }

        self.flush_async(stream);
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));

        const old_kv_start: u32 = self.num_slots - self.kv_reserved_tail;
        const new_kv_start: u32 = self.num_slots - n_kv;
        const smap = self.h_slot_map.?;

        if (n_kv > self.kv_reserved_tail) {
            // Grow KV tail: reclaim [new_kv_start .. old_kv_start) from expert cache
            var map_dirty: bool = false;

            var rf_out: usize = 0;
            var rf_i: usize = 0;
            while (rf_i < self.ready_free_len) : (rf_i += 1) {
                if (self.ready_free[rf_i] < new_kv_start) {
                    self.ready_free[rf_out] = self.ready_free[rf_i];
                    rf_out += 1;
                }
            }
            self.ready_free_len = rf_out;

            var fl_out: usize = 0;
            var fl_i: usize = 0;
            while (fl_i < self.free_list_len) : (fl_i += 1) {
                if (self.free_list[fl_i] < new_kv_start) {
                    self.free_list[fl_out] = self.free_list[fl_i];
                    fl_out += 1;
                }
            }
            self.free_list_len = fl_out;

            if (self.free_slots_used > new_kv_start) {
                self.free_slots_used = new_kv_start;
            }

            var s: u32 = new_kv_start;
            while (s < old_kv_start) : (s += 1) {
                const owner = self.slot_owner[s];
                if (owner >= 0) {
                    smap[@intCast(owner)] = -1;
                    self.slot_owner[s] = -1;
                    map_dirty = true;
                }
                self.slot_reserved[s] = true;
            }
            self.kv_reserved_tail = n_kv;

            // Repopulate async staging slots right below new_kv_start so overlapped promotion stays active!
            if (self.async_promote and self.ready_free_len < async_staging_slots and new_kv_start > async_staging_slots + 16) {
                var cand_s: u32 = new_kv_start;
                while (cand_s > 0 and self.ready_free_len < async_staging_slots) {
                    cand_s -= 1;
                    if (self.slot_reserved[cand_s]) continue;
                    const owner = self.slot_owner[cand_s];
                    if (owner >= 0) {
                        smap[@intCast(owner)] = -1;
                        self.slot_owner[cand_s] = -1;
                        map_dirty = true;
                    }
                    // Also remove from free_list if present
                    var fi: usize = 0;
                    var fo: usize = 0;
                    while (fi < self.free_list_len) : (fi += 1) {
                        if (self.free_list[fi] != cand_s) {
                            self.free_list[fo] = self.free_list[fi];
                            fo += 1;
                        }
                    }
                    self.free_list_len = fo;
                    self.slot_reserved[cand_s] = true;
                    self.ready_free[self.ready_free_len] = cand_s;
                    self.ready_free_len += 1;
                }
            }

            if (map_dirty) {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                    @ptrCast(self.d_slot_map.?),
                    @ptrCast(smap),
                    total_experts * @sizeOf(i32),
                    stream,
                ));
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));
            }
            return self.d_cache_pool.? + @as(usize, new_kv_start) * slot_bytes;
        } else {
            // Shrink KV tail: return [old_kv_start .. new_kv_start) to free_list & bulk-refill hot experts!
            var s: u32 = old_kv_start;
            while (s < new_kv_start) : (s += 1) {
                tiger.assert(self.slot_owner[s] == -1);
                tiger.assert(self.slot_reserved[s]);
                if (self.async_promote and self.ready_free_len < async_staging_slots) {
                    self.ready_free[self.ready_free_len] = s;
                    self.ready_free_len += 1;
                } else {
                    self.slot_reserved[s] = false;
                    self.free_list[self.free_list_len] = s;
                    self.free_list_len += 1;
                }
            }
            self.kv_reserved_tail = n_kv;
            self.refill_free_list(weights, stream);
            if (n_kv == 0) return null;
            return self.d_cache_pool.? + @as(usize, new_kv_start) * slot_bytes;
        }
    }

    /// Borrow `n_tail` slots immediately before `kv_reserved_tail` as temporary prefill scratch.
    /// Unmaps any experts currently residing in those tail slots and uploads `d_slot_map` before returning.
    pub fn acquire_prefill_scratch(self: *ExpertCache, n_tail: u32, stream: ?*anyopaque) [*]u8 {
        tiger.assert(self.enabled);
        const kv_start: u32 = self.num_slots - self.kv_reserved_tail;
        tiger.assert(n_tail > 0 and n_tail + 16 < kv_start);
        const slot_bytes: usize = @intCast(constants.expert_slot_bytes);
        if (self.prefill_reserved_tail == n_tail) {
            const tail_start: u32 = kv_start - n_tail;
            return self.d_cache_pool.? + @as(usize, tail_start) * slot_bytes;
        }
        tiger.assert(self.prefill_reserved_tail == 0);

        // Drain all async promotions/quarantines and sync stream before reclaiming tail memory!
        self.flush_async(stream);
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));

        const tail_start: u32 = kv_start - n_tail;
        const smap = self.h_slot_map.?;
        var map_dirty: bool = false;

        // Compact ready_free to drop any staging slots inside [tail_start .. kv_start)
        var rf_out: usize = 0;
        var rf_i: usize = 0;
        while (rf_i < self.ready_free_len) : (rf_i += 1) {
            if (self.ready_free[rf_i] < tail_start) {
                self.ready_free[rf_out] = self.ready_free[rf_i];
                rf_out += 1;
            }
        }
        self.ready_free_len = rf_out;

        // Compact free_list to drop any free slots inside [tail_start .. kv_start)
        var fl_out: usize = 0;
        var fl_i: usize = 0;
        while (fl_i < self.free_list_len) : (fl_i += 1) {
            if (self.free_list[fl_i] < tail_start) {
                self.free_list[fl_out] = self.free_list[fl_i];
                fl_out += 1;
            }
        }
        self.free_list_len = fl_out;

        // Clamp bump watermark to tail_start so prefill promotions never enter the tail
        if (self.free_slots_used > tail_start) {
            self.free_slots_used = tail_start;
        }

        // Unmap and reserve every slot in [tail_start .. kv_start)
        var s: u32 = tail_start;
        while (s < kv_start) : (s += 1) {
            const owner = self.slot_owner[s];
            if (owner >= 0) {
                smap[@intCast(owner)] = -1;
                self.slot_owner[s] = -1;
                map_dirty = true;
            }
            self.slot_reserved[s] = true;
        }
        self.prefill_reserved_tail = n_tail;

        if (map_dirty) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                @ptrCast(self.d_slot_map.?),
                @ptrCast(smap),
                total_experts * @sizeOf(i32),
                stream,
            ));
        }

        return self.d_cache_pool.? + @as(usize, tail_start) * slot_bytes;
    }

    /// Release borrowed prefill scratch tail slots back to decode (zero-eviction refill via free_list!).
    pub fn release_prefill_scratch(self: *ExpertCache, stream: ?*anyopaque, already_synced: bool) void {
        if (!self.enabled or self.prefill_reserved_tail == 0) return;
        if (!already_synced) {
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));
        }
        const kv_start: u32 = self.num_slots - self.kv_reserved_tail;
        const tail_start: u32 = kv_start - self.prefill_reserved_tail;
        var s: u32 = tail_start;
        while (s < kv_start) : (s += 1) {
            tiger.assert(self.slot_owner[s] == -1);
            tiger.assert(self.slot_reserved[s]);
            if (self.async_promote and self.ready_free_len < async_staging_slots) {
                self.ready_free[self.ready_free_len] = s;
                self.ready_free_len += 1;
            } else {
                self.slot_reserved[s] = false;
                self.free_list[self.free_list_len] = s;
                self.free_list_len += 1;
            }
        }
        self.prefill_reserved_tail = 0;
    }

    inline fn eff_lrfu_score(self: *const ExpertCache, eid: usize) f32 {
        const sc: f32 = self.ewma_score[eid];
        if (self.loop_score_mult == 1.0) return sc;
        const layer: usize = eid / constants.expert_count;
        const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);
        return if (is_loop) (sc * self.loop_score_mult) else sc;
    }

    fn refill_free_list(self: *ExpertCache, weights: *const weights_mod.ModelWeights, stream: ?*anyopaque) void {
        if (!self.enabled or !self.tail_refill or self.free_list_len == 0) return;

        const smap = self.h_slot_map.?;
        var n_cands: usize = 0;
        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            if (smap[eid] >= 0 or self.is_inflight[eid]) continue;
            const sc: f32 = self.eff_lrfu_score(eid);
            if (std.math.isFinite(sc) and sc > 0.0) {
                const scaled: u32 = @intFromFloat(@min(sc * 10000.0, 4_000_000_000.0));
                self.cand_buf[n_cands] = .{ .eid = @intCast(eid), .score = if (scaled == 0) 1 else scaled };
                n_cands += 1;
            }
        }
        if (n_cands == 0) return;

        const to_fill: usize = @min(n_cands, self.free_list_len);
        self.sort_top_candidates(n_cands, to_fill);

        const pool_base = self.d_cache_pool.?;
        const slot_bytes: usize = @intCast(constants.expert_slot_bytes);
        var loaded: u32 = 0;
        var idx: usize = 0;
        while (idx < to_fill) : (idx += 1) {
            const e_id: usize = self.cand_buf[idx].eid;
            if (smap[e_id] >= 0) continue;
            const dst_slot: u32 = self.pop_unassigned_slot() orelse break;
            const layer: usize = e_id / constants.expert_count;
            const expert: usize = e_id % constants.expert_count;
            const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);
            const h_bank = weights.host_experts_h[layer].?;
            const src_ptr: *const anyopaque = @ptrCast(h_bank + expert * slot_bytes);
            const dst_ptr: *anyopaque = @ptrCast(pool_base + @as(usize, dst_slot) * slot_bytes);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(dst_ptr, src_ptr, slot_bytes, stream));
            smap[e_id] = @intCast(dst_slot);
            self.slot_owner[dst_slot] = @intCast(e_id);
            self.clock_bits[dst_slot] = if (is_loop) self.loop_clock_max else self.nonloop_clock_max;
            loaded += 1;
        }
        if (loaded > 0) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                @ptrCast(self.d_slot_map.?),
                @ptrCast(smap),
                total_experts * @sizeOf(i32),
                stream,
            ));
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));
            self.prefill_promotions += loaded;
        }
    }

    /// Release borrowed prefill tail slots AND immediately bulk-DMA-refill them with the hottest
    /// unmapped LRFU experts so decode starts Token #1 with 100% of VRAM slots resident!
    pub fn release_prefill_scratch_and_refill(
        self: *ExpertCache,
        weights: *const weights_mod.ModelWeights,
        stream: ?*anyopaque,
        already_synced: bool,
    ) void {
        self.release_prefill_scratch(stream, already_synced);
        self.refill_free_list(weights, stream);
    }

    /// Wait for any in-flight async promotions on copy_stream and publish them cleanly before prefill/reset.
    pub fn flush_async(self: *ExpertCache, main_stream: ?*anyopaque) void {
        if (!self.enabled or !self.async_promote) return;
        var map_dirty: bool = false;
        if (self.inflight_len > 0) {
            tiger.assert_cuda(cuda.squid_cuda_stream_wait_event(main_stream, self.copy_done_event));
            const smap = self.h_slot_map.?;
            var i: usize = 0;
            while (i < self.inflight_len) : (i += 1) {
                const inf = self.inflight[i];
                const eid: usize = inf.eid;
                const layer: usize = eid / constants.expert_count;
                const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);
                smap[eid] = @intCast(inf.slot);
                self.slot_owner[inf.slot] = @intCast(eid);
                self.clock_bits[inf.slot] = if (is_loop) 3 else 1;
                self.slot_reserved[inf.slot] = false;
                self.is_inflight[eid] = false;
                map_dirty = true;
            }
            self.inflight_len = 0;
        }
        if (self.quarantine_len > 0) {
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(main_stream));
            var q: usize = 0;
            while (q < self.quarantine_len) : (q += 1) {
                tiger.assert(self.ready_free_len < async_staging_slots);
                self.ready_free[self.ready_free_len] = self.quarantine[q];
                self.ready_free_len += 1;
            }
            self.quarantine_len = 0;
        }
        if (map_dirty) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                @ptrCast(self.d_slot_map.?),
                @ptrCast(self.h_slot_map.?),
                total_experts * @sizeOf(i32),
                main_stream,
            ));
        }
    }

    pub fn reset_stats(self: *ExpertCache) void {
        self.decode_accesses = 0;
        self.decode_hits = 0;
        self.decode_misses = 0;
        self.decode_promotions = 0;
        self.decode_evictions = 0;
        self.zone_accesses = .{ 0, 0, 0 };
        self.zone_hits = .{ 0, 0, 0 };
        self.prefill_accesses = 0;
        self.prefill_hits = 0;
        self.prefill_misses = 0;
        self.prefill_promotions = 0;
    }

    /// Zero touch counters on GPU (safe inside CUDA Graph capture!).
    pub fn record_zero_touches(self: *const ExpertCache, stream: ?*anyopaque) void {
        if (!self.enabled) return;
        tiger.assert_cuda(cuda.squid_cuda_memset_async(
            @ptrCast(self.d_touch_counts.?),
            0,
            total_experts * @sizeOf(u32),
            stream,
        ));
    }

    /// Enqueue async 12 KiB D2H readback of touch counts right before the existing decode sync!
    pub fn copy_touches_d2h_async(self: *const ExpertCache, stream: ?*anyopaque) void {
        if (!self.enabled) return;
        tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(
            @ptrCast(self.h_touch_counts.?),
            @ptrCast(self.d_touch_counts.?),
            total_experts * @sizeOf(u32),
            stream,
        ));
    }

    inline fn has_unassigned_slot(self: *const ExpertCache) bool {
        if (self.free_list_len > 0) return true;
        const usable: u32 = self.usable_slots();
        var cur: u32 = self.free_slots_used;
        while (cur < usable) : (cur += 1) {
            if (!self.slot_reserved[cur]) return true;
        }
        return false;
    }

    inline fn pop_unassigned_slot(self: *ExpertCache) ?u32 {
        if (self.free_list_len > 0) {
            self.free_list_len -= 1;
            const s = self.free_list[self.free_list_len];
            tiger.assert(!self.slot_reserved[s] and self.slot_owner[s] == -1);
            return s;
        }
        const usable: u32 = self.usable_slots();
        while (self.free_slots_used < usable) {
            const s = self.free_slots_used;
            self.free_slots_used += 1;
            if (!self.slot_reserved[s] and self.slot_owner[s] == -1) return s;
        }
        return null;
    }

    /// Process touch counts after a decode step (host has already synced `stream` or `done_event`) and promote hot misses.
    pub fn process_decode_step(self: *ExpertCache, weights: *const weights_mod.ModelWeights, n_tokens: u32, stream: ?*anyopaque) void {
        if (!self.enabled) return;
        self.step_counter += 1;

        const smap = self.h_slot_map.?;
        var map_dirty: bool = false;

        // Async Phase A: publish promotions launched at step t-1 & release quarantined victim slots!
        if (self.async_promote) {
            if (self.inflight_len > 0) {
                tiger.assert_cuda(cuda.squid_cuda_stream_wait_event(stream, self.copy_done_event));
                var i: usize = 0;
                while (i < self.inflight_len) : (i += 1) {
                    const inf = self.inflight[i];
                    const eid: usize = inf.eid;
                    const layer: usize = eid / constants.expert_count;
                    const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);
                    smap[eid] = @intCast(inf.slot);
                    self.slot_owner[inf.slot] = @intCast(eid);
                    self.clock_bits[inf.slot] = if (is_loop) 3 else 1;
                    self.slot_reserved[inf.slot] = false;
                    self.is_inflight[eid] = false;
                    map_dirty = true;
                }
                self.inflight_len = 0;
            }
            if (self.quarantine_len > 0) {
                var q: usize = 0;
                while (q < self.quarantine_len) : (q += 1) {
                    tiger.assert(self.ready_free_len < async_staging_slots);
                    self.ready_free[self.ready_free_len] = self.quarantine[q];
                    self.ready_free_len += 1;
                }
                self.quarantine_len = 0;
            }
        }

        const touches = self.h_touch_counts.?;
        var n_cands: usize = 0;
        const has_free: bool = self.has_unassigned_slot();

        // Global Aging-LFU decay scaled by accepted token count (so spec decode ages identically to B=1!)
        if (self.use_lrfu) {
            var dec: f32 = self.lrfu_decay;
            const nt: u32 = @max(n_tokens, 1);
            if (nt > 1) {
                var k: u32 = 1;
                while (k < nt) : (k += 1) dec *= self.lrfu_decay;
            }
            var i: usize = 0;
            while (i < total_experts) : (i += 1) {
                self.ewma_score[i] *= dec;
            }
        }

        // First pass: update scores & stats for all touched experts
        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            const cnt: u32 = touches[eid];
            if (cnt == 0) continue;

            const layer: u32 = @intCast(eid / constants.expert_count);
            const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);
            const zone: usize = if (layer < constants.loop_start_layer) 0 else if (layer < constants.loop_end_layer) 1 else 2;
            self.decode_accesses += cnt;
            self.zone_accesses[zone] += cnt;

            if (self.use_lrfu) {
                self.ewma_score[eid] += @floatFromInt(cnt);
            }

            const cslot: i32 = smap[eid];
            if (cslot >= 0) {
                self.decode_hits += cnt;
                self.zone_hits[zone] += cnt;
                if (!self.use_lrfu) {
                    const s: usize = @intCast(cslot);
                    const cmax: u8 = if (is_loop) self.loop_clock_max else self.nonloop_clock_max;
                    const bump: u8 = @intCast(@min(cnt, cmax));
                    const next_val: u16 = @as(u16, self.clock_bits[s]) + bump;
                    self.clock_bits[s] = @intCast(@min(next_val, cmax));
                }
            } else {
                self.decode_misses += cnt;
                if (self.is_inflight[eid]) continue;

                if (self.use_lrfu) {
                    const scaled: u32 = @intFromFloat(@min(self.ewma_score[eid] * 1000.0, 4_000_000_000.0));
                    self.cand_buf[n_cands] = .{
                        .eid = @intCast(eid),
                        .score = if (has_free) (scaled + 1) else scaled,
                    };
                    n_cands += 1;
                } else {
                    const add_m: u32 = @min(cnt, 32);
                    const next_m: u32 = @as(u32, self.recent_misses[eid]) + add_m;
                    self.recent_misses[eid] = @intCast(@min(next_m, 65535));

                    if (has_free or self.recent_misses[eid] >= self.admit_threshold) {
                        const weight_mult: u32 = if (is_loop) 2 else 1;
                        self.cand_buf[n_cands] = .{
                            .eid = @intCast(eid),
                            .score = @as(u32, self.recent_misses[eid]) * weight_mult + cnt,
                        };
                        n_cands += 1;
                    }
                }
            }
        }

        if (n_cands > 0) {
            const cold_startup: bool = (self.free_slots_used < self.usable_slots() / 2);
            if (self.async_promote and !cold_startup) {
                const max_p: usize = @min(n_cands, @min(self.max_promotes_per_decode, async_staging_slots));
                if (self.launch_async_promotions(weights, n_cands, max_p)) {
                    map_dirty = true;
                }
            } else {
                const max_p: usize = if (has_free)
                    @min(n_cands, @max(self.max_promotes_per_decode, 64))
                else
                    @min(n_cands, self.max_promotes_per_decode);
                if (self.promote_top_candidates(weights, n_cands, max_p, true, stream)) {
                    map_dirty = true;
                }
            }
        }

        if (map_dirty) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                @ptrCast(self.d_slot_map.?),
                @ptrCast(smap),
                total_experts * @sizeOf(i32),
                stream,
            ));
        }

        if ((self.step_counter & 63) == 0) {
            var i: usize = 0;
            while (i < total_experts) : (i += 1) {
                self.recent_misses[i] >>= 1;
            }
        }
    }

    /// Process touch counts after a batched prefill chunk to warm the VRAM cache with the prompt's hottest experts.
    pub fn process_prefill_chunk(self: *ExpertCache, weights: *const weights_mod.ModelWeights, chunk_len: u32, stream: ?*anyopaque) void {
        if (!self.enabled) return;
        self.flush_async(stream);

        const touches = self.h_touch_counts.?;
        const smap = self.h_slot_map.?;
        var n_cands: usize = 0;

        // Normalize prefill chunk touches and preserve warm decode LRFU priors across long prompts!
        const pf_scale: f32 = self.lrfu_pf_scale / @as(f32, @floatFromInt(@max(chunk_len, 1)));
        if (self.use_lrfu and self.lrfu_pf_decay < 1.0) {
            var i: usize = 0;
            while (i < total_experts) : (i += 1) {
                self.ewma_score[i] *= self.lrfu_pf_decay;
            }
        }

        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            const cnt: u32 = touches[eid];
            if (cnt == 0) continue;

            const layer: u32 = @intCast(eid / constants.expert_count);
            const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);
            self.prefill_accesses += cnt;

            if (self.use_lrfu) {
                self.ewma_score[eid] += @as(f32, @floatFromInt(cnt)) * pf_scale;
            }

            const cslot: i32 = smap[eid];
            if (cslot >= 0) {
                self.prefill_hits += cnt;
                const s: usize = @intCast(cslot);
                const cmax: u8 = if (is_loop) self.loop_clock_max else self.nonloop_clock_max;
                self.clock_bits[s] = cmax;
            } else {
                self.prefill_misses += cnt;
                self.cand_buf[n_cands] = .{
                    .eid = @intCast(eid),
                    .score = cnt,
                };
                n_cands += 1;
            }
        }

        if (n_cands > 0) {
            const usable: u32 = self.usable_slots();
            const free_approx: usize = self.free_list_len + (if (self.free_slots_used < usable) (usable - self.free_slots_used) else 0);
            const max_p: usize = if (free_approx > 0)
                @min(n_cands, @max(free_approx, self.max_promotes_per_prefill_warm))
            else
                @min(n_cands, self.max_promotes_per_prefill_warm);
            if (self.promote_top_candidates(weights, n_cands, max_p, false, stream)) {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                    @ptrCast(self.d_slot_map.?),
                    @ptrCast(smap),
                    total_experts * @sizeOf(i32),
                    stream,
                ));
            }
        }
    }

    fn launch_async_promotions(
        self: *ExpertCache,
        weights: *const weights_mod.ModelWeights,
        n_cands: usize,
        max_promote: usize,
    ) bool {
        if (max_promote == 0 or n_cands == 0) return false;
        self.sort_top_candidates(n_cands, max_promote);

        const cands = self.cand_buf[0..n_cands];
        const smap = self.h_slot_map.?;
        const pool_base = self.d_cache_pool.?;
        const slot_bytes: usize = @intCast(constants.expert_slot_bytes);
        var promoted: u32 = 0;
        var map_dirty: bool = false;

        var idx: usize = 0;
        while (idx < max_promote and self.inflight_len < async_staging_slots) : (idx += 1) {
            const eid: usize = cands[idx].eid;
            if (smap[eid] >= 0 or self.is_inflight[eid]) continue;

            // Prefer zero-eviction async refill from free_list / unused slots (e.g., released prefill tail)!
            var dst_slot: u32 = 0;
            if (self.pop_unassigned_slot()) |empty_slot| {
                dst_slot = empty_slot;
                self.slot_reserved[dst_slot] = true;
            } else {
                if (self.ready_free_len == 0) break;
                const vic_slot: u32 = if (self.use_lrfu)
                    (self.find_lrfu_victim(eid, true) orelse break)
                else
                    self.evict_clock_slot(true);

                self.ready_free_len -= 1;
                dst_slot = self.ready_free[self.ready_free_len];
                tiger.assert(self.slot_reserved[dst_slot]);

                self.slot_reserved[vic_slot] = true;
                tiger.assert(self.quarantine_len < async_staging_slots);
                self.quarantine[self.quarantine_len] = vic_slot;
                self.quarantine_len += 1;
                map_dirty = true;
            }

            const layer: usize = eid / constants.expert_count;
            const expert: usize = eid % constants.expert_count;
            const h_bank = weights.host_experts_h[layer].?;
            const src_ptr: *const anyopaque = @ptrCast(h_bank + expert * slot_bytes);
            const dst_ptr: *anyopaque = @ptrCast(pool_base + @as(usize, dst_slot) * slot_bytes);

            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(dst_ptr, src_ptr, slot_bytes, self.copy_stream));

            self.inflight[self.inflight_len] = .{ .eid = @intCast(eid), .slot = dst_slot };
            self.inflight_len += 1;
            self.is_inflight[eid] = true;
            self.recent_misses[eid] = 0;
            promoted += 1;
        }

        if (promoted > 0) {
            tiger.assert_cuda(cuda.squid_cuda_event_record(self.copy_done_event, self.copy_stream));
            self.decode_promotions += promoted;
        }
        return map_dirty;
    }

    fn sort_top_candidates(self: *ExpertCache, n_cands: usize, max_promote: usize) void {
        const cands = self.cand_buf[0..n_cands];
        if (max_promote < n_cands and max_promote <= 64) {
            var i: usize = 0;
            while (i < max_promote) : (i += 1) {
                var best_idx: usize = i;
                var j: usize = i + 1;
                while (j < n_cands) : (j += 1) {
                    if (cands[j].score > cands[best_idx].score) best_idx = j;
                }
                if (best_idx != i) {
                    const tmp = cands[i];
                    cands[i] = cands[best_idx];
                    cands[best_idx] = tmp;
                }
            }
        } else if (max_promote < n_cands) {
            std.mem.sort(PromoteCand, cands, {}, struct {
                fn lessThan(_: void, a: PromoteCand, b: PromoteCand) bool {
                    return a.score > b.score;
                }
            }.lessThan);
        }
    }

    fn promote_top_candidates(
        self: *ExpertCache,
        weights: *const weights_mod.ModelWeights,
        n_cands: usize,
        max_promote: usize,
        is_decode: bool,
        stream: ?*anyopaque,
    ) bool {
        if (max_promote == 0 or n_cands == 0) return false;
        self.sort_top_candidates(n_cands, max_promote);

        const cands = self.cand_buf[0..n_cands];
        const smap = self.h_slot_map.?;
        const pool_base = self.d_cache_pool.?;
        const slot_bytes: usize = @intCast(constants.expert_slot_bytes);
        var promoted: u32 = 0;

        var idx: usize = 0;
        while (idx < max_promote) : (idx += 1) {
            const eid: usize = cands[idx].eid;
            if (smap[eid] >= 0) continue;

            const dst_slot: u32 = self.acquire_slot(eid, is_decode) orelse break;
            const layer: usize = eid / constants.expert_count;
            const expert: usize = eid % constants.expert_count;
            const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);

            const h_bank = weights.host_experts_h[layer].?;
            const src_ptr: *const anyopaque = @ptrCast(h_bank + expert * slot_bytes);
            const dst_ptr: *anyopaque = @ptrCast(pool_base + @as(usize, dst_slot) * slot_bytes);

            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(dst_ptr, src_ptr, slot_bytes, stream));

            smap[eid] = @intCast(dst_slot);
            self.slot_owner[dst_slot] = @intCast(eid);
            self.clock_bits[dst_slot] = if (is_loop) 3 else 1;
            self.recent_misses[eid] = 0;
            promoted += 1;
        }

        if (promoted > 0) {
            if (is_decode) {
                self.decode_promotions += promoted;
            } else {
                self.prefill_promotions += promoted;
            }
            return true;
        }
        return false;
    }

    fn acquire_slot(self: *ExpertCache, cand_eid: usize, is_decode: bool) ?u32 {
        if (self.pop_unassigned_slot()) |s| return s;
        if (self.use_lrfu) {
            return self.find_lrfu_victim(cand_eid, is_decode);
        }
        return self.evict_clock_slot(is_decode);
    }

    fn find_lrfu_victim(self: *ExpertCache, cand_eid: usize, is_decode: bool) ?u32 {
        const usable: u32 = self.usable_slots();
        const cand_score: f32 = self.eff_lrfu_score(cand_eid);
        var min_score: f32 = 1e30;
        var min_slot: ?u32 = null;
        var s: u32 = 0;
        while (s < usable) : (s += 1) {
            if (self.slot_reserved[s]) continue;
            const owner = self.slot_owner[s];
            if (owner < 0) continue;
            const sc: f32 = self.eff_lrfu_score(@intCast(owner));
            if (sc < min_score) {
                min_score = sc;
                min_slot = s;
            }
        }
        const vic = min_slot orelse return null;
        if (cand_score <= min_score * self.lrfu_hysteresis) return null;

        const old_eid = self.slot_owner[vic];
        if (old_eid >= 0) {
            self.h_slot_map.?[@intCast(old_eid)] = -1;
            self.slot_owner[vic] = -1;
            if (is_decode) self.decode_evictions += 1;
        }
        return vic;
    }

    fn evict_clock_slot(self: *ExpertCache, is_decode: bool) u32 {
        const n: u32 = self.usable_slots();
        const smap = self.h_slot_map.?;
        if (self.clock_hand >= n) self.clock_hand = 0;
        var steps: u32 = 0;
        const max_steps: u32 = n * 16;
        while (steps < max_steps) : (steps += 1) {
            const s = self.clock_hand;
            self.clock_hand = if (s + 1 < n) (s + 1) else 0;
            if (self.slot_reserved[s]) continue;
            if (self.clock_bits[s] == 0) {
                const old_eid = self.slot_owner[s];
                if (old_eid >= 0) {
                    smap[@intCast(old_eid)] = -1;
                    self.slot_owner[s] = -1;
                    if (is_decode) self.decode_evictions += 1;
                }
                return s;
            }
            self.clock_bits[s] -= 1;
        }
        var fb: u32 = 0;
        while (fb < n) : (fb += 1) {
            const s = self.clock_hand;
            self.clock_hand = if (s + 1 < n) (s + 1) else 0;
            if (self.slot_reserved[s]) continue;
            const old_eid = self.slot_owner[s];
            if (old_eid >= 0) {
                smap[@intCast(old_eid)] = -1;
                self.slot_owner[s] = -1;
                if (is_decode) self.decode_evictions += 1;
            }
            return s;
        }
        @panic("expert_cache: no evictable slot available");
    }

    fn load_warm_snapshot(self: *ExpertCache, weights: *const weights_mod.ModelWeights, stream: ?*anyopaque) bool {
        if (self.cache_file_len == 0) return false;
        const fd = tiger.open(@ptrCast(&self.cache_file_buf), tiger.O_RDONLY);
        if (fd < 0) return false;
        defer _ = tiger.close(fd);

        const magic: []const u8 = "SQ5060C1";
        const expected_bytes: usize = 8 + total_experts * @sizeOf(f32);
        var buf: [8 + total_experts * @sizeOf(f32)]u8 = undefined;
        var got: usize = 0;
        var iters: u32 = 0;
        while (got < expected_bytes and iters < 64) : (iters += 1) {
            const n = tiger.read(fd, buf[got..].ptr, expected_bytes - got);
            if (n <= 0) break;
            got += @intCast(n);
        }
        if (got != expected_bytes or !std.mem.eql(u8, buf[0..8], magic)) return false;

        var max_sc: f32 = 0.0;
        var raw_scores: [total_experts]f32 = @splat(0.0);
        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            const bits = std.mem.readInt(u32, buf[8 + eid * 4 ..][0..4], .little);
            const sc: f32 = @bitCast(bits);
            if (std.math.isFinite(sc) and sc > 0.0) {
                raw_scores[eid] = sc;
                if (sc > max_sc) max_sc = sc;
            }
        }
        if (max_sc <= 0.0) return false;

        const scale: f32 = self.warm_prior_max / max_sc;
        eid = 0;
        while (eid < total_experts) : (eid += 1) {
            self.ewma_score[eid] = raw_scores[eid] * scale;
        }
        self.bulk_load_from_scores(weights, stream, "snapshot");
        return true;
    }

    fn load_builtin_profile(self: *ExpertCache, weights: *const weights_mod.ModelWeights, stream: ?*anyopaque) void {
        const inv_max: f32 = self.warm_prior_max / 65535.0;
        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            self.ewma_score[eid] = @as(f32, @floatFromInt(hot_experts.default_scores[eid])) * inv_max;
        }
        self.bulk_load_from_scores(weights, stream, "built-in profile");
    }

    fn bulk_load_from_scores(self: *ExpertCache, weights: *const weights_mod.ModelWeights, stream: ?*anyopaque, source_label: []const u8) void {
        const t0 = tiger.clock_monotonic_ns();
        var n_cands: usize = 0;
        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            const sc: f32 = self.ewma_score[eid];
            if (std.math.isFinite(sc) and sc > 0.0) {
                const scaled: u32 = @intFromFloat(@min(sc * 10000.0, 4_000_000_000.0));
                self.cand_buf[n_cands] = .{ .eid = @intCast(eid), .score = if (scaled == 0) 1 else scaled };
                n_cands += 1;
            }
        }
        if (n_cands == 0) return;

        const active_cap: u32 = if (self.async_promote) (self.num_slots - @as(u32, async_staging_slots)) else self.num_slots;
        const to_load: usize = @min(n_cands, active_cap);
        std.mem.sort(PromoteCand, self.cand_buf[0..n_cands], {}, struct {
            fn lessThan(_: void, a: PromoteCand, b: PromoteCand) bool {
                return a.score > b.score;
            }
        }.lessThan);

        const smap = self.h_slot_map.?;
        const pool_base = self.d_cache_pool.?;
        const slot_bytes: usize = @intCast(constants.expert_slot_bytes);
        var loaded: u32 = 0;
        var idx: usize = 0;
        while (idx < to_load) : (idx += 1) {
            const e_id: usize = self.cand_buf[idx].eid;
            const dst_slot: u32 = self.pop_unassigned_slot() orelse break;
            const layer: usize = e_id / constants.expert_count;
            const expert: usize = e_id % constants.expert_count;
            const is_loop: bool = (layer >= constants.loop_start_layer and layer < constants.loop_end_layer);
            const h_bank = weights.host_experts_h[layer].?;
            const src_ptr: *const anyopaque = @ptrCast(h_bank + expert * slot_bytes);
            const dst_ptr: *anyopaque = @ptrCast(pool_base + @as(usize, dst_slot) * slot_bytes);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(dst_ptr, src_ptr, slot_bytes, stream));
            smap[e_id] = @intCast(dst_slot);
            self.slot_owner[dst_slot] = @intCast(e_id);
            self.clock_bits[dst_slot] = if (is_loop) self.loop_clock_max else self.nonloop_clock_max;
            loaded += 1;
        }
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
            @ptrCast(self.d_slot_map.?),
            @ptrCast(smap),
            total_experts * @sizeOf(i32),
            stream,
        ));
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(stream));
        const ms: f64 = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1e6;
        tiger.print_stdout(
            "[moe-cache] Warm-loaded {d} experts from {s} in {d:.1}ms (prior_max={d:.1})\n",
            .{ loaded, source_label, ms, self.warm_prior_max },
        );
    }

    fn save_warm_snapshot(self: *const ExpertCache) void {
        if (self.cache_file_len == 0 or self.step_counter < 16) return;
        const fd = tiger.creat(@ptrCast(&self.cache_file_buf), 0o644);
        if (fd < 0) return;
        defer _ = tiger.close(fd);

        var buf: [8 + total_experts * @sizeOf(f32)]u8 = undefined;
        @memcpy(buf[0..8], "SQ5060C1");
        var max_sc: f32 = 0.0;
        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            if (self.ewma_score[eid] > max_sc) max_sc = self.ewma_score[eid];
        }
        const norm: f32 = if (max_sc > 0.0) (100.0 / max_sc) else 0.0;
        eid = 0;
        while (eid < total_experts) : (eid += 1) {
            const sc: f32 = self.ewma_score[eid] * norm;
            std.mem.writeInt(u32, buf[8 + eid * 4 ..][0..4], @bitCast(sc), .little);
        }
        tiger.write_fd_all(fd, &buf);
    }

    pub fn print_stats(self: *const ExpertCache, tag: []const u8) void {
        if (!self.enabled) return;
        const dec_pct: f64 = if (self.decode_accesses > 0)
            100.0 * @as(f64, @floatFromInt(self.decode_hits)) / @as(f64, @floatFromInt(self.decode_accesses))
        else
            0.0;
        const pf_pct: f64 = if (self.prefill_accesses > 0)
            100.0 * @as(f64, @floatFromInt(self.prefill_hits)) / @as(f64, @floatFromInt(self.prefill_accesses))
        else
            0.0;
        var z_slots: [3]u32 = .{ 0, 0, 0 };
        var eid: usize = 0;
        while (eid < total_experts) : (eid += 1) {
            if (self.h_slot_map.?[eid] >= 0) {
                const l: u32 = @intCast(eid / constants.expert_count);
                const z: usize = if (l < constants.loop_start_layer) 0 else if (l < constants.loop_end_layer) 1 else 2;
                z_slots[z] += 1;
            }
        }
        const resident_slots: u32 = z_slots[0] + z_slots[1] + z_slots[2];
        const z0: f64 = if (self.zone_accesses[0] > 0) 100.0 * @as(f64, @floatFromInt(self.zone_hits[0])) / @as(f64, @floatFromInt(self.zone_accesses[0])) else 0.0;
        const z1: f64 = if (self.zone_accesses[1] > 0) 100.0 * @as(f64, @floatFromInt(self.zone_hits[1])) / @as(f64, @floatFromInt(self.zone_accesses[1])) else 0.0;
        const z2: f64 = if (self.zone_accesses[2] > 0) 100.0 * @as(f64, @floatFromInt(self.zone_hits[2])) / @as(f64, @floatFromInt(self.zone_accesses[2])) else 0.0;
        tiger.print_stdout(
            "[moe-cache {s}] slots={d}/{d} (pre={d}/768 loop={d}/1536 coda={d}/768) | decode hit={d:.2}%% (pre={d:.1}%% loop={d:.1}%% coda={d:.1}%%) prom={d} evict={d} | prefill hit={d:.2}%%\n",
            .{
                tag,
                resident_slots,
                self.num_slots,
                z_slots[0],
                z_slots[1],
                z_slots[2],
                dec_pct,
                z0,
                z1,
                z2,
                self.decode_promotions,
                self.decode_evictions,
                pf_pct,
            },
        );
    }
};

fn parse_env_u32(name: [*:0]const u8) ?u32 {
    const val = tiger.getenv(name) orelse return null;
    return std.fmt.parseInt(u32, std.mem.span(val), 10) catch null;
}
