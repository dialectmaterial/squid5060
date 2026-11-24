// TigerStyle core primitives, always-on release assertions, monotonic timing,
// fixed-capacity ByteWriter, and minimal POSIX/libc C-ABI declarations.
// Zero dynamic allocations during request or token execution.

const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.os.tag == .linux and builtin.cpu.arch != .x86_64) {
        @compileError("squid5060 POSIX ABI constants are verified for linux x86_64");
    }
}

/// TigerStyle always-on assertion: remains active even in ReleaseFast builds.
/// Crashes loudly on invariant violation rather than silently corrupting state.
pub inline fn assert(condition: bool) void {
    if (!condition) {
        @branchHint(.cold);
        @panic("TigerStyle assertion failed");
    }
}

/// Always-on assertion for CUDA runtime status codes (`0 == cudaSuccess`).
pub inline fn assert_cuda(error_code: i32) void {
    if (error_code != 0) {
        @branchHint(.cold);
        log_error_i32("CUDA runtime call failed with error_code=", error_code);
        @panic("CUDA assertion failed");
    }
}

/// Align `value` upward to a power-of-two `alignment` boundary.
pub inline fn align_up_u64(value: u64, alignment: u64) u64 {
    assert(alignment > 0);
    assert((alignment & (alignment - 1)) == 0);
    const mask: u64 = alignment - 1;
    const result: u64 = (value + mask) & ~mask;
    assert(result >= value);
    assert((result & mask) == 0);
    return result;
}

pub inline fn min_u64(a: u64, b: u64) u64 {
    return if (a < b) a else b;
}

pub inline fn max_u64(a: u64, b: u64) u64 {
    return if (a > b) a else b;
}

pub inline fn min_i32(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}

pub inline fn max_i32(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}

/// FNV-1a 64-bit hash for flat static open-addressed tables.
pub inline fn hash_bytes_u64(bytes: []const u8) u64 {
    var hash: u64 = 14695981039346656037;
    for (bytes) |byte| {
        hash ^= @as(u64, byte);
        hash *%= 1099511628211;
    }
    return if (hash == 0) 1 else hash;
}

/// SplitMix64 mixer for 64-bit integer keys (e.g., BPE token pairs).
pub inline fn hash_pair_u64(key: u64) u64 {
    var mixed: u64 = key +% 0x9e3779b97f4a7c15;
    mixed = (mixed ^ (mixed >> 30)) *% 0xbf58476d1ce4e5b9;
    mixed = (mixed ^ (mixed >> 27)) *% 0x94d049bb133111eb;
    mixed = mixed ^ (mixed >> 31);
    return if (mixed == 0) 1 else mixed;
}

// --- POSIX / Libc C-ABI Declarations ---

pub const Timespec = extern struct {
    tv_sec: i64,
    tv_nsec: i64,
};

pub const SockaddrIn = extern struct {
    sin_family: u16,
    sin_port: u16,
    sin_addr: u32,
    sin_zero: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
};

comptime {
    if (@sizeOf(SockaddrIn) != 16) {
        @compileError("SockaddrIn must be exactly 16 bytes");
    }
}

pub const O_RDONLY: i32 = 0;
pub const SEEK_SET: i32 = 0;
pub const SEEK_END: i32 = 2;
pub const PROT_READ: i32 = 1;
pub const MAP_PRIVATE: i32 = 2;
pub const MAP_FAILED_ADDR: usize = std.math.maxInt(usize);

pub const CLOCK_REALTIME: i32 = 0;
pub const CLOCK_MONOTONIC: i32 = 1;

pub const AF_INET: i32 = 2;
pub const SOCK_STREAM: i32 = 1;
pub const SOL_SOCKET: i32 = 1;
pub const SO_REUSEADDR: i32 = 2;
pub const IPPROTO_TCP: i32 = 6;
pub const TCP_NODELAY: i32 = 1;
pub const MSG_NOSIGNAL: i32 = 0x4000;
pub const EINTR: i32 = 4;

pub const SIGINT: i32 = 2;
pub const SIGPIPE: i32 = 13;
pub const SIGTERM: i32 = 15;

pub const SignalHandler = ?*const fn (i32) callconv(.c) void;
pub const SIG_IGN: SignalHandler = @ptrFromInt(1);

pub extern "c" fn open(pathname: [*:0]const u8, flags: i32, ...) i32;
pub extern "c" fn close(fd: i32) i32;
pub extern "c" fn read(fd: i32, buffer: [*]u8, count: usize) isize;
pub extern "c" fn write(fd: i32, buffer: [*]const u8, count: usize) isize;
pub extern "c" fn lseek(fd: i32, offset: i64, whence: i32) i64;
pub extern "c" fn mmap(addr: ?*anyopaque, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) *anyopaque;
pub extern "c" fn munmap(addr: *anyopaque, length: usize) i32;
pub extern "c" fn malloc(size: usize) ?*anyopaque;
pub extern "c" fn calloc(count: usize, size: usize) ?*anyopaque;
pub extern "c" fn free(ptr: ?*anyopaque) void;
pub extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
pub extern "c" fn clock_gettime(clock_id: i32, tp: *Timespec) i32;
pub extern "c" fn socket(domain: i32, sock_type: i32, protocol: i32) i32;
pub extern "c" fn setsockopt(fd: i32, level: i32, optname: i32, optval: *const anyopaque, optlen: u32) i32;
pub extern "c" fn bind(fd: i32, addr: *const SockaddrIn, addrlen: u32) i32;
pub extern "c" fn listen(fd: i32, backlog: i32) i32;
pub extern "c" fn accept(fd: i32, addr: ?*SockaddrIn, addrlen: ?*u32) i32;
pub extern "c" fn recv(fd: i32, buffer: [*]u8, length: usize, flags: i32) isize;
pub extern "c" fn send(fd: i32, buffer: [*]const u8, length: usize, flags: i32) isize;
pub extern "c" fn inet_pton(af: i32, src: [*:0]const u8, dst: *u32) i32;
pub extern "c" fn signal(signum: i32, handler: SignalHandler) SignalHandler;
pub extern "c" fn __errno_location() *i32;

pub inline fn htons(value: u16) u16 {
    return @byteSwap(value);
}

pub inline fn get_errno() i32 {
    return __errno_location().*;
}

/// Monotonic wall clock in nanoseconds.
pub fn clock_monotonic_ns() u64 {
    var ts: Timespec = .{ .tv_sec = 0, .tv_nsec = 0 };
    const rc: i32 = clock_gettime(CLOCK_MONOTONIC, &ts);
    assert(rc == 0);
    assert(ts.tv_sec >= 0);
    assert(ts.tv_nsec >= 0 and ts.tv_nsec < 1_000_000_000);
    return @as(u64, @intCast(ts.tv_sec)) * 1_000_000_000 + @as(u64, @intCast(ts.tv_nsec));
}

/// UNIX epoch timestamp in seconds (for OpenAI JSON `created` fields).
pub fn epoch_seconds() i64 {
    var ts: Timespec = .{ .tv_sec = 0, .tv_nsec = 0 };
    const rc: i32 = clock_gettime(CLOCK_REALTIME, &ts);
    assert(rc == 0);
    assert(ts.tv_sec >= 0);
    return ts.tv_sec;
}

/// Allocate zero-initialized host memory at startup (never called on hot paths).
pub fn alloc_host_zeroed(comptime T: type, count: usize) []T {
    assert(count > 0);
    const raw: ?*anyopaque = calloc(count, @sizeOf(T));
    assert(raw != null);
    const typed: [*]T = @ptrCast(@alignCast(raw.?));
    return typed[0..count];
}

/// Free startup-allocated host memory during teardown.
pub fn free_host_slice(comptime T: type, slice: []T) void {
    assert(slice.len > 0);
    free(@ptrCast(slice.ptr));
}

// --- Zero-Allocation Fixed-Buffer Writer & Direct POSIX Stdout/Stderr ---

pub const ByteWriter = struct {
    buffer: []u8,
    length: usize,

    pub fn init(buffer: []u8) ByteWriter {
        assert(buffer.len > 0);
        return .{ .buffer = buffer, .length = 0 };
    }

    pub fn reset(self: *ByteWriter) void {
        assert(self.length <= self.buffer.len);
        self.length = 0;
    }

    pub fn slice(self: *const ByteWriter) []const u8 {
        assert(self.length <= self.buffer.len);
        return self.buffer[0..self.length];
    }

    pub fn remaining(self: *const ByteWriter) usize {
        assert(self.length <= self.buffer.len);
        return self.buffer.len - self.length;
    }

    pub fn append_slice(self: *ByteWriter, text: []const u8) void {
        assert(self.length <= self.buffer.len);
        assert(text.len <= self.buffer.len - self.length);
        @memcpy(self.buffer[self.length .. self.length + text.len], text);
        self.length += text.len;
    }

    pub fn try_append_slice(self: *ByteWriter, text: []const u8) bool {
        assert(self.length <= self.buffer.len);
        if (text.len > self.buffer.len - self.length) return false;
        @memcpy(self.buffer[self.length .. self.length + text.len], text);
        self.length += text.len;
        return true;
    }

    pub fn append_byte(self: *ByteWriter, byte: u8) void {
        assert(self.length < self.buffer.len);
        self.buffer[self.length] = byte;
        self.length += 1;
    }

    pub fn try_append_byte(self: *ByteWriter, byte: u8) bool {
        assert(self.length <= self.buffer.len);
        if (self.length >= self.buffer.len) return false;
        self.buffer[self.length] = byte;
        self.length += 1;
        return true;
    }

    pub fn append_fmt(self: *ByteWriter, comptime fmt: []const u8, args: anytype) void {
        assert(self.length <= self.buffer.len);
        const written = std.fmt.bufPrint(self.buffer[self.length..], fmt, args) catch {
            @panic("ByteWriter.append_fmt buffer overflow");
        };
        self.length += written.len;
        assert(self.length <= self.buffer.len);
    }

    pub fn append_json_escaped(self: *ByteWriter, text: []const u8) void {
        assert(self.length <= self.buffer.len);
        self.append_byte('"');
        const hex_digits = "0123456789abcdef";
        for (text) |byte| {
            switch (byte) {
                '"' => self.append_slice("\\\""),
                '\\' => self.append_slice("\\\\"),
                '\n' => self.append_slice("\\n"),
                '\r' => self.append_slice("\\r"),
                '\t' => self.append_slice("\\t"),
                0x08 => self.append_slice("\\b"),
                0x0c => self.append_slice("\\f"),
                else => {
                    if (byte < 0x20) {
                        var esc: [6]u8 = .{ '\\', 'u', '0', '0', hex_digits[(byte >> 4) & 0x0f], hex_digits[byte & 0x0f] };
                        self.append_slice(&esc);
                    } else {
                        self.append_byte(byte);
                    }
                },
            }
        }
        self.append_byte('"');
    }

    pub fn append_c_string_terminated(self: *ByteWriter, text: []const u8) [*:0]const u8 {
        const start = self.length;
        self.append_slice(text);
        self.append_byte(0);
        return @ptrCast(self.buffer[start..self.length].ptr);
    }
};

pub fn write_fd_all(fd: i32, bytes: []const u8) void {
    assert(fd >= 0);
    var sent: usize = 0;
    const iterations_max: usize = bytes.len + 64;
    var iteration: usize = 0;
    while (sent < bytes.len and iteration < iterations_max) : (iteration += 1) {
        const written: isize = write(fd, bytes[sent..].ptr, bytes.len - sent);
        if (written <= 0) {
            if (written < 0 and get_errno() == EINTR) continue;
            return;
        }
        sent += @intCast(written);
    }
}

pub fn print_stdout(comptime fmt: []const u8, args: anytype) void {
    var stack_buffer: [8192]u8 = undefined;
    const text = std.fmt.bufPrint(&stack_buffer, fmt, args) catch stack_buffer[0..];
    write_fd_all(1, text);
}

pub fn print_stderr(comptime fmt: []const u8, args: anytype) void {
    var stack_buffer: [4096]u8 = undefined;
    const text = std.fmt.bufPrint(&stack_buffer, fmt, args) catch stack_buffer[0..];
    write_fd_all(2, text);
}

fn log_error_i32(prefix: []const u8, value: i32) void {
    print_stderr("{s}{d}\n", .{ prefix, value });
}

/// Build a null-terminated path `<directory>/<suffix>` inside caller stack buffer.
pub fn format_path_z(buffer: []u8, directory: []const u8, suffix: []const u8) [*:0]const u8 {
    assert(buffer.len > directory.len + suffix.len + 1);
    assert(directory.len > 0);
    @memcpy(buffer[0..directory.len], directory);
    @memcpy(buffer[directory.len .. directory.len + suffix.len], suffix);
    buffer[directory.len + suffix.len] = 0;
    return @ptrCast(buffer[0 .. directory.len + suffix.len + 1].ptr);
}
