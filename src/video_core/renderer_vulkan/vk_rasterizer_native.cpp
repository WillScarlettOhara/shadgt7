// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-100: native mode (RYGT_NATIVE=1, ryGT builds). Draws, dispatches and copies are recorded
// with the native image registry and guest memory read in place, instead of the texture and
// buffer caches, which keep images and buffers in step with guest memory draw by draw and are
// most of the recorder's cost. Version 0 records on the recorder thread, in GPU order, into a
// secondary command buffer; before anything else goes into the scheduler's command buffer (a
// submit, a flip) that list is ended and run there, so fences, labels and presenting work as
// before. Images stay in the general layout. Synchronisation is by global memory barriers: when
// a pass starts with other targets, before a draw reads a picture written since the last
// barrier, and around dispatches and copies. Later versions record the game's render jobs on its
// own threads with this code (ryty_gpu/native/DESIGN.md, N4).

#include "video_core/renderer_vulkan/build_queue.h"
#include "video_core/renderer_vulkan/vk_rasterizer_native.h"
#include "video_core/renderer_vulkan/upload_memory_recovery.h"
#include "video_core/host_shaders/native_region_writeback_comp.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <sstream>
#include <mutex>
#include <set>
#include <string_view>
#include <tuple>

#include <fmt/ranges.h>
#include <magic_enum/magic_enum.hpp>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/diagnostics.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/perf_monitor.h"
#include "common/thread.h"
#include "common/ryty_config.h"
// (ryty_config.h brings in windows.h, whose `near` and `far` macros would rename variables here.)
#undef near
#undef far
#include "core/debug_state.h"
#include "video_core/renderdoc.h"
#include "core/guest_hash.h"
#include "core/memory.h"
#include "ryty_gpu/native/image_registry.h"
#include "ryty_gpu/native/job_walker.h"
#include "ryty_gpu/native/rhi/rhi_vulkan.h"
#include "ryty_gpu/ryty_diag.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/draw_pipe.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/renderer_vulkan/depth_attachment.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/resource_binding.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image_descriptor.h"
#include "video_core/texture_cache/sampler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {
vk::ImageViewType ConvertImageViewType(AmdGpu::ImageType type);
}

namespace Vulkan {
void RytyWatchWrites(std::uint64_t address, std::uint32_t ms, std::string label); // DIAG-067
void RytyWatchDump(std::string label);                                           // DIAG-193

namespace {

std::pair<u32, u64> DiagnosticPassOptions(u64 frame) {
    static const char* path = Common::DiagEnv("RYGT_DIAG_PASS_FILE");
    static u64 last_frame = ~u64{0};
    static unsigned flags = 0;
    static unsigned long long skip = 0;
    if (path && last_frame != frame) {
        last_frame = frame;
        if (auto* file = std::fopen(path, "r")) {
            std::fscanf(file, "%u %llx", &flags, &skip);
            std::fclose(file);
        }
    }
    return {flags, skip};
}

/// DIAG-193 (diagnosis, RYGT_DIAG_HW_WATCH=<hex address>,<seconds>): a hardware write watch over
/// that address from that many seconds on, dumped when DIAG-184 meets a texture there.
u64 HwWatchAddress() {
    static const u64 address = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_HW_WATCH");
        return env && *env ? u64(std::strtoull(env, nullptr, 16)) : u64{0};
    }();
    return address;
}

/// Logs the first few times and then every 1000th.
bool Sample(std::atomic<u32>& counter) {
    const u32 n = ++counter;
    return n <= 8 || n % 1000 == 0;
}

// What vk_rasterizer.cpp's ClassifyDraw decides, from the registers alone.
enum class Filter { Draw, EliminateFastClear, FmaskDecompress, Resolve, NoPrimitive, Copy };

Filter Classify(const AmdGpu::Regs& regs) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        return Filter::EliminateFastClear;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        return Filter::FmaskDecompress;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        return Filter::Resolve;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return Filter::NoPrimitive;
    }
    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const bool depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const bool stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        return Filter::Copy;
    }
    return Filter::Draw;
}

// As vk_rasterizer.cpp's MakeUserData.
Shader::PushData UserData(const AmdGpu::Regs& regs) {
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    push_data.quad_sample_mask = {regs.reg_array[0xa30e], regs.reg_array[0xa30f]}; // PA_SC_AA_MASK
    return push_data;
}

// As vk_rasterizer.cpp's GetDrawOffsets.
std::pair<u32, u32> DrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

// PERF-028's quad corners (vk_rasterizer.cpp).
constexpr std::array<u32, 6> QuadFirstVertex{0, 1, 2, 0, 2, 3};
constexpr std::array<u32, 6> QuadLastVertex{0, 1, 3, 1, 2, 3};
constexpr u64 MaxQuadIndexBytes = 16ULL << 20;

template <typename T>
u32 ExpandQuads(const T* in, u32 count, const std::optional<T> restart,
                const std::array<u32, 6>& corners, T* out) {
    u32 written = 0;
    if (!restart) {
        for (u32 quad = 0; quad + 4 <= count; quad += 4) {
            for (const u32 corner : corners) {
                out[written++] = in[quad + corner];
            }
        }
        return written;
    }
    std::array<T, 4> quad{};
    u32 filled = 0;
    for (u32 i = 0; i < count; ++i) {
        if (in[i] == *restart) {
            filled = 0;
            continue;
        }
        quad[filled++] = in[i];
        if (filled == 4) {
            for (const u32 corner : corners) {
                out[written++] = quad[corner];
            }
            filled = 0;
        }
    }
    return written;
}

// The texture cache's image usage, features and type (texture_cache/image.cpp), so native
// images are made as its images are.
vk::ImageUsageFlags UsageOf(const Instance& instance, const VideoCore::ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (!info.props.is_block) {
        if (instance.IsAttachmentFeedbackLoopLayoutSupported()) {
            usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
        }
        if (info.props.is_depth) {
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
        } else {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            if (info.num_samples == 1 || instance.IsMultisampleStorageImageSupported()) {
                usage |= vk::ImageUsageFlagBits::eStorage;
            }
        }
    } else {
        usage |= vk::ImageUsageFlagBits::eStorage;
    }
    return usage;
}

vk::FormatFeatureFlags2 FeaturesOf(vk::ImageUsageFlags usage) {
    vk::FormatFeatureFlags2 features{};
    if (usage & vk::ImageUsageFlagBits::eTransferSrc) {
        features |= vk::FormatFeatureFlagBits2::eTransferSrc;
    }
    if (usage & vk::ImageUsageFlagBits::eTransferDst) {
        features |= vk::FormatFeatureFlagBits2::eTransferDst;
    }
    if (usage & vk::ImageUsageFlagBits::eSampled) {
        features |= vk::FormatFeatureFlagBits2::eSampledImage;
    }
    if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
        features |= vk::FormatFeatureFlagBits2::eColorAttachment;
    }
    if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        features |= vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    }
    return features;
}

vk::ImageType TypeOf(AmdGpu::ImageType type) {
    return type == AmdGpu::ImageType::Color3D ? vk::ImageType::e3D : vk::ImageType::e2D;
}

u32 RhiUsage(vk::ImageUsageFlags usage) {
    u32 result = 0;
    if (usage & vk::ImageUsageFlagBits::eSampled) {
        result |= Native::Rhi::ImageUsageSampled;
    }
    if (usage & vk::ImageUsageFlagBits::eStorage) {
        result |= Native::Rhi::ImageUsageStorage;
    }
    if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
        result |= Native::Rhi::ImageUsageColorTarget;
    }
    if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        result |= Native::Rhi::ImageUsageDepthTarget;
    }
    if (usage & vk::ImageUsageFlagBits::eTransferSrc) {
        result |= Native::Rhi::ImageUsageTransferSrc;
    }
    if (usage & vk::ImageUsageFlagBits::eTransferDst) {
        result |= Native::Rhi::ImageUsageTransferDst;
    }
    return result | Native::Rhi::ImageUsageMutableFormat;
}

template <typename T>
T* Handle(u64 handle) {
    return reinterpret_cast<T*>(handle);
}

/// NAT-110: adds the time until the end of the scope to a counter.
// OPT-001: the phase timers read the clock on one draw in 16 (reading it in every phase of every
// draw was 7% of the recorder's time); the report divides by the draws timed.
thread_local bool t_timed = true;
struct PhaseClock {
    u64* total;
    std::chrono::steady_clock::time_point start{};
    explicit PhaseClock(u64& total_) : total{t_timed ? &total_ : nullptr} {
        if (total) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~PhaseClock() {
        if (total) {
            *total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
        }
    }
};

// DIAG-GT7-004: a piece of a dispatch's recorder time, on every dispatch (nothing when `total`
// is null: the same code for a draw).
struct PartClock {
    u64* total;
    std::chrono::steady_clock::time_point start{};
    explicit PartClock(u64* total_) : total{total_} {
        if (total) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~PartClock() {
        if (total) {
            *total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
        }
    }
};

// FIX-GT7-047: a list takes a chunk for itself even for a few KB, and keeps it until the GPU ran the
// list. GT7's race loading has about 350 lists waiting: at 8 MB a chunk, 2.8 GB of RAM the driver
// zeroes, for under 32 MB of data a frame. RYGT_NATIVE_CHUNK_MB=8: as before.
u64 ChunkSize() {
    static const u64 size = [] {
        const char* env = std::getenv("RYGT_NATIVE_CHUNK_MB");
        const long long mb = env ? std::strtoll(env, nullptr, 10) : 0;
        return mb > 0 ? u64(mb) << 20 : u64{1_MB};
    }();
    return size;
}
constexpr u32 SpliceAfter = 512; ///< commands per list before it is run and submitted

} // namespace

bool NativeModeRequested() {
    // NAT-160: the launcher's renderer choice (GPU.native_renderer in the game's settings, so
    // GTSport.exe started on its own follows it too); RYGT_NATIVE=1 or =0 overrides it.
    static const bool requested = [] {
        // FIX-230: logged, with where the choice came from (GT7 does not boot without it).
        if (const char* env = std::getenv("RYGT_NATIVE"); env && *env) {
            const bool on = env[0] == '1';
            LOG_WARNING(Render_Vulkan, "FIX-230: native renderer {} (RYGT_NATIVE={})",
                        on ? "on" : "off", env);
            return on;
        }
        const auto value = Common::RytyGameConfigValue("native_renderer");
#ifdef _WIN32
        // FIX-231: GT7 does not run on the old renderer (a tester's launches without the setting
        // hung or stopped at "protect region as write-only"), so GT7.exe always gets the native
        // renderer unless RYGT_NATIVE=0 says otherwise.
        wchar_t exe[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        const std::wstring name{exe};
        const std::wstring file = name.substr(name.find_last_of(L"\\/") + 1);
        if (Common::RytyGameConfigText().empty()) {
            // FIX-231: the launcher's settings were not found where the renderer looks.
            const std::wstring path = Common::RytyGameConfigPathW();
            std::string narrow(path.size() * 3 + 1, ' ');
            narrow.resize(std::max(0, WideCharToMultiByte(CP_UTF8, 0, path.c_str(),
                                                          static_cast<int>(path.size()),
                                                          narrow.data(),
                                                          static_cast<int>(narrow.size()),
                                                          nullptr, nullptr)));
            LOG_WARNING(Render_Vulkan,
                        "FIX-231: the launcher's settings file was not found at {} (start GT7 "
                        "with RyGT7.exe from its own folder); the game's defaults apply",
                        narrow);
        }
        if (_wcsicmp(file.c_str(), L"GT7.exe") == 0) {
            LOG_WARNING(Render_Vulkan, "FIX-231: native renderer on (GT7 always uses it; the "
                                       "settings say {})",
                        value ? *value : std::string{"nothing"});
            return true;
        }
#endif
        const bool on = value && *value == "true";
        LOG_WARNING(Render_Vulkan, "FIX-230: native renderer {} ({})", on ? "on" : "off",
                    value ? "native_renderer in user\\custom_configs\\CUSA03220.json"
                    : Common::RytyGameConfigText().empty()
                        ? "the launcher's settings file could not be read: default off"
                        : "no native_renderer setting: default off");
        return on;
    }();
    return requested;
}

namespace {
// OPT-013..015 on (RYGT_NATIVE_REUSE=0 off; DIAG-GT7-003 switches it while the game runs).
std::atomic<bool> g_reuse_on{[] {
    const char* env = std::getenv("RYGT_NATIVE_REUSE");
    return !(env && env[0] == '0');
}()};
// PERF-GT7-010 (on; RYGT_NATIVE_GPU_PRIORITY=0 off; DIAG-GT7-003 switches it while the game
// runs): the recorder and the command processor run above normal priority.
std::atomic<bool> g_boost_on{[] {
    const char* env = std::getenv("RYGT_NATIVE_GPU_PRIORITY");
    return !(env && env[0] == '0');
}()};
// PERF-GT7-013..016 (RYGT_NATIVE_NEXT=1).
std::atomic<bool> g_next_on{[] {
    const char* env = std::getenv("RYGT_NATIVE_NEXT");
    return env && env[0] == '1';
}()};
// PERF-GT7-018 (on; RYGT_NATIVE_RING=0 off; DIAG-GT7-003 switches it while the game runs).
std::atomic<bool> g_ring_on{[] {
    const char* env = std::getenv("RYGT_NATIVE_RING");
    return !(env && env[0] == '0');
}()};

// PERF-GT7-013: the memo tables' slot for a key of a few words (XXH3's cost showed at the
// ~12 lookups of every draw; a full key compare follows, so only the spread matters).
u64 MemoHash(const u64* words, size_t count, u64 seed) {
    u64 h = seed ^ 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < count; ++i) {
        h = (h ^ words[i]) * 0xFF51AFD7ED558CCDULL;
        h ^= h >> 32;
    }
    h *= 0xC4CEB9FE1A85EC53ULL;
    return h ^ (h >> 29);
}
} // namespace

bool NativeRenderer::NextOn() {
    return g_next_on.load(std::memory_order_relaxed);
}

bool NativeRenderer::ReuseOn() {
    return g_reuse_on.load(std::memory_order_relaxed);
}

bool NativeRenderer::RingOn() {
    return g_ring_on.load(std::memory_order_relaxed);
}

bool NativeRenderer::PreparedOn() {
    return PrepareEnabled();
}

bool NativeRenderer::BoostOn() {
    return g_boost_on.load(std::memory_order_relaxed);
}

void NativeRenderer::ApplyRecorderPriority() {
    // PERF-GT7-010: in GT7's missions the shader compiler's threads (the driver's, at normal
    // priority, each a whole core) and the game's job threads spinning on GPU labels ran while
    // the recorder, at ~30 ms of work for a 33 ms frame, ran 80% of a missed frame's interval.
    const bool boost = BoostOn();
    if (boost != recorder_boosted) {
        recorder_boosted = boost;
        Common::SetCurrentThreadPriority(boost ? Common::ThreadPriority::High
                                               : Common::ThreadPriority::Normal);
    }
    // PERF-GT7-018: the command thread records the native lists: it runs at the same priority
    // (set there, from the ring).
    if (CommandRing* ring = r.scheduler.CommandThreadRing()) {
        const bool command_boost = boost && RingOn();
        if (command_boost != command_boosted) {
            command_boosted = command_boost;
            ring->RecordControl([command_boost](Scheduler&) {
                Common::SetCurrentThreadPriority(command_boost ? Common::ThreadPriority::High
                                                               : Common::ThreadPriority::Normal);
            });
        }
    }
}

bool NativeRenderer::AbOn() {
    static const bool on = [] {
        const char* env = Common::DiagEnv("RYGT_NATIVE_AB");
        return env && env[0] == '1';
    }();
    return on;
}

std::string NativeRenderer::DispatchPartsText() const {
    // DIAG-GT7-004: us a dispatch by DIAG-188 part, its bindings and notes by piece, and the
    // FIX-203 copies before the work.
    const double count = double(std::max<u64>(stats.dispatches, 1));
    const auto per = [&](u64 ns) { return ns / 1000.0 / count; };
    u64 pieces = 0;
    for (const u64 ns : stats.dispatch_bind_ns) {
        pieces += ns;
    }
    const u64 bindings = stats.dispatch_part_ns[1];
    const auto& part = stats.dispatch_part_ns;
    const auto& bind = stats.dispatch_bind_ns;
    return fmt::format(
        "us in pipeline/bindings/images/barrier/record/notes {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/"
        "{:.2f}; bindings in begin/buffers/textures/copies before/fill/rest "
        "{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f} (region refreshes {:.2f}); notes in "
        "write-backs/GPU writes {:.2f}/{:.2f}; copies before {} ({:.1f} MB), buffers made {}, "
        "reused {}; image slots {} ({} from the memo, {:.1f} views each)",
        per(part[0]), per(part[1]), per(part[2]), per(part[3]), per(part[4]), per(part[5]),
        per(bind[0]), per(bind[1]), per(bind[2]), per(bind[3]), per(bind[4]),
        per(bindings > pieces ? bindings - pieces : 0), per(stats.dispatch_refresh_ns),
        per(stats.dispatch_note_ns[0]), per(stats.dispatch_note_ns[1]), stats.writeback_befores,
        stats.snapshot_bytes / 1048576.0, stats.snapshots_made, stats.snapshots_reused,
        stats.compute_texture_lookups, stats.compute_texture_hits,
        stats.compute_texture_lookups
            ? double(stats.compute_texture_bindings) / double(stats.compute_texture_lookups)
            : 0.0);
}

u32 NativeRenderer::AbMode() const {
    // A the defaults without PERF-GT7-018, B with it.
    return u32(ab_window % 2);
}

void NativeRenderer::ForgetBindings(Stream& s) {
    s.memo_layout = vk::PipelineLayout{};
    s.push_known = false;
    for (auto& memo : s.set_memos) {
        memo.valid = false;
    }
}

u64 NativeRenderer::ReuseGeneration() const {
    return registry->Generation() + unmap_generation.load(std::memory_order_relaxed) +
           buffer_generation;
}

std::pair<vk::Buffer, u64> NativeRenderer::UploadBytes(u32 slot, const void* data, u32 size,
                                                      u64 align) {
    Stream& s = stream;
    if (ReuseOn() && slot < s.upload_memos.size()) {
        UploadMemo& memo = s.upload_memos[slot];
        if (memo.list == s.serial && memo.bytes.size() == size &&
            std::memcmp(memo.bytes.data(), data, size) == 0) {
            ++stats.uploads_reused;
            return {memo.buffer, memo.offset};
        }
        const auto [buffer, offset, mapped] = Allocate(size, align);
        std::memcpy(mapped, data, size);
        memo.list = s.serial;
        memo.buffer = buffer;
        memo.offset = offset;
        memo.bytes.assign(static_cast<const u8*>(data), static_cast<const u8*>(data) + size);
        return {buffer, offset};
    }
    const auto [buffer, offset, mapped] = Allocate(size, align);
    std::memcpy(mapped, data, size);
    return {buffer, offset};
}

void NativeRenderer::DrawBindings::Clear() {
    writes.clear();
    write_sets.clear();
    buffers.clear();
    images.clear();
    push = {};
    sampled.clear();
    stored.clear();
    image_entries.clear(); // FIX-GT7-080
    gpu_writes.clear();
    fill_pattern.reset();
    fill_address = 0;
    fill_size = 0;
    writes_buffers = false;
    failed = false;
    diag_records = 0;
}

NativeRenderer::NativeRenderer(Rasterizer& rasterizer) : r{rasterizer} {
    device = Native::Rhi::CreateVulkanDevice(r.instance);
    ASSERT_MSG(device, "NAT-100: no native device");
    const VkSemaphore semaphore = r.scheduler.GetWorkSemaphore()->Handle();
    timeline = device->ImportFence(reinterpret_cast<u64>(semaphore));
    device->UseExternalTimeline(timeline.get(), [this] { return r.scheduler.CurrentTick(); });
    registry = std::make_unique<Native::ImageRegistry>(*device);
    registry->SetImageFactory([this](const VideoCore::ImageInfo& info, bool render_target) {
        return MakeImage(info, render_target);
    });
    registry->SetMemoryClamp([this](VAddr address, u64 size) -> u64 {
        // Mapped, and readable by the CPU (GPU-only memory is not: NAT-106).
        if (!r.memory->IsValidMapping(address)) {
            return 0;
        }
        const u64 mapped = r.memory->ClampRangeSize(address, size);
        return Native::ReadablePrefix(reinterpret_cast<const void*>(address), mapped);
    });
    registry->SetUploadSource([this](const Native::ImageEntry& entry,
                                     Native::Rhi::CommandList& list) {
        // NAT-107: textures are uploaded from guest memory as the GPU sees it, in GPU order.
        Native::ImageRegistry::UploadSource source;
        auto* direct = r.buffer_cache.GetDirectMemory();
        const VAddr address = entry.info.guest_address;
        const u64 size = entry.info.guest_size;
        if (!direct || size == 0) {
            return source;
        }
        // NAT-146: the GPU's bytes there are in a region (shader-written memory lives in video
        // memory): the upload reads the region.
        if (Region* region = RegionOver(address, size);
            region && address >= region->begin && address + size <= region->end &&
            (address - region->begin) % r.instance.StorageMinAlignment() == 0) {
            const u64 refreshed = stats.region_refreshes;
            RefreshRegion(*region);
            static const bool upload_barrier = [] {
                const char* env = std::getenv("RYGT_NATIVE_UPLOAD_BARRIER");
                return !(env && env[0] == '0');
            }();
            if (upload_barrier && stats.region_refreshes != refreshed) {
                // FIX-196: the CPU's bytes just copied into the region are there before the
                // upload reads it (its untiling dispatch or copy follows at once; the gathered
                // path below has its barrier). Without it, textures the CPU writes in GT7's
                // per-frame memory (a region: compute output lands there too) were uploaded
                // stale now and then. RYGT_NATIVE_UPLOAD_BARRIER=0: as before (diagnosis).
                const vk::MemoryBarrier2 copied{
                    .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eCopy |
                                    vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eTransferRead |
                                     vk::AccessFlagBits2::eShaderRead,
                };
                stream.cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                               .pMemoryBarriers = &copied});
            }
            auto wrapped = device->WrapBuffer(region->buffer->NativeHandle(), region->buffer->Size());
            source.buffer = wrapped.get();
            source.offset = address - region->begin;
            source.kind = 1;
            source.keep.push_back(std::move(wrapped));
            return source;
        }
        if (!direct->EnsureImported(address, size)) {
            return source;
        }
        if (const auto found = direct->FindBuffer(address, size);
            found && found->second % r.instance.StorageMinAlignment() == 0) {
            auto wrapped = device->WrapBuffer(
                reinterpret_cast<u64>(static_cast<VkBuffer>(found->first->Handle())),
                found->first->SizeBytes());
            source.buffer = wrapped.get();
            source.offset = found->second;
            source.kind = 2;
            source.keep.push_back(std::move(wrapped));
            return source;
        }
        auto gathered = device->CreateBuffer({.size = size,
                                              .usage = Native::Rhi::BufferUsageStorage |
                                                       Native::Rhi::BufferUsageTransferSrc |
                                                       Native::Rhi::BufferUsageTransferDst,
                                              .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                              .name = "native upload gather"});
        if (!gathered) {
            return source;
        }
        const CmdBuf cmd = CmdOf(list); // PERF-GT7-018
        const vk::Buffer dst{Handle<VkBuffer_T>(gathered->NativeHandle())};
        const bool all = direct->ForEachPart(
            address, size, [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
                cmd.copyBuffer(view.Handle(), dst,
                               vk::BufferCopy{.srcOffset = view_offset, .dstOffset = offset,
                                              .size = bytes});
            });
        if (all) {
            list.Barrier(Native::Rhi::ScopeTransfer,
                         Native::Rhi::ScopeCompute | Native::Rhi::ScopeTransfer);
            source.buffer = gathered.get();
            source.kind = 3;
        }
        source.keep.push_back(std::move(gathered));
        return source;
    });
    {
        // FIX-203: FIX-194's write-back of the words a draw or dispatch changed.
        const std::array<Native::Rhi::BindingSlot, 3> slots{{
            {0, Native::Rhi::BindingType::StorageBuffer, Native::Rhi::StageCompute},
            {1, Native::Rhi::BindingType::StorageBuffer, Native::Rhi::StageCompute},
            {2, Native::Rhi::BindingType::StorageBuffer, Native::Rhi::StageCompute},
        }};
        writeback_layout = device->CreatePipelineLayout({.bindings = slots,
                                                         .push_constant_bytes = sizeof(u32),
                                                         .push_constant_stages =
                                                             Native::Rhi::StageCompute});
        if (writeback_layout) {
            writeback_pipeline = device->CreateComputePipeline({
                .layout = writeback_layout.get(),
                .shader = {Native::Rhi::ShaderStage::Compute, NATIVE_REGION_WRITEBACK_COMP},
                .name = "native region write-back",
            });
        }
    }
    VideoCore::DirectMemory::HoldReleasesForSplice(); // FIX-205
    registry->SetUploadHook([this](Native::ImageEntry& entry) {
        // PERF-GT7-052: small textures in GT7's per-frame memory are checked each frame instead
        // (their pages hold the data the game rewrites every frame: watched, every one of those
        // writes faulted, 170,000-257,000 times in 10 s of a race).
        const u64 size = std::max<u64>(entry.info.guest_size, 1);
        if (PolledFrameTexture(entry.info.guest_address, size)) {
            std::scoped_lock lock{polled_mutex};
            polled_textures.emplace_back(entry.info.guest_address, size);
            return;
        }
        // NAT-105: CPU writes to a texture's memory from now on fault once, and the texture
        // is uploaded again.
        WatchPages(entry.info.guest_address, size);
    });
    gds = device->CreateBuffer({.size = 64_KB,
                                .usage = Native::Rhi::BufferUsageStorage |
                                         Native::Rhi::BufferUsageTransferSrc |
                                         Native::Rhi::BufferUsageTransferDst,
                                .memory = Native::Rhi::MemoryKind::Upload,
                                .name = "native gds"});
    ASSERT_MSG(gds && gds->Mapped(), "NAT-100: no GDS buffer");
    std::memset(gds->Mapped(), 0, gds->Size());
    if (AbOn()) {
        g_ring_on.store(false, std::memory_order_relaxed); // DIAG-GT7-003: window 0 is A
        RyDiag::SetProfileTag(1);
    }
    LOG_INFO(Render_Vulkan, "NAT-100: native mode: draws, dispatches and copies are recorded with "
                            "native images and guest memory read in place");
    LoadOneOffKinds(); // PERF-GT7-023
    if (const u64 address = HwWatchAddress(); address != 0) {
        const char* env = Common::DiagEnv("RYGT_DIAG_HW_WATCH");
        const char* comma = std::strchr(env, ',');
        const u32 delay = comma ? u32(std::strtoul(comma + 1, nullptr, 10)) : 10;
        std::thread([address, delay] {
            std::this_thread::sleep_for(std::chrono::seconds(delay));
            RytyWatchWrites(address, 1000 * 1000, "ring hw watch"); // DIAG-193
        }).detach();
    }
}

NativeRenderer::~NativeRenderer() {
    std::scoped_lock lock{mutex};
    SaveOneOffKinds(true); // PERF-GT7-023
    // PERF-GT7-018: commands still in the ring use this renderer's lists.
    r.scheduler.SyncCommandThread();
    const vk::Device vk_device = r.instance.GetDevice();
    for (const auto pool : free_pools) {
        vk_device.destroyDescriptorPool(pool);
    }
    for (const auto& [tick, pool] : retiring_pools) {
        vk_device.destroyDescriptorPool(pool);
    }
    for (const auto pool : stream.pools) {
        vk_device.destroyDescriptorPool(pool);
    }
    for (const auto pool : gpu_timing_pools) { // DIAG-GT7-001
        vk_device.destroyQueryPool(pool);
    }
    for (const auto& timing : gpu_timing_pending) {
        vk_device.destroyQueryPool(timing->pool);
    }
    views.clear();
    samplers.clear();
}

// GT7-UW: the display buffers' addresses (the presenter notes them as the game registers them).
namespace {
std::array<std::atomic<VAddr>, 16> g_display_addresses{};
} // namespace

void NoteDisplayAddress(VAddr address) {
    for (auto& slot : g_display_addresses) {
        VAddr expected = 0;
        if (slot.load(std::memory_order_relaxed) == address ||
            slot.compare_exchange_strong(expected, address)) {
            return;
        }
    }
}

static bool IsDisplayAddress(VAddr address) {
    for (const auto& slot : g_display_addresses) {
        const VAddr value = slot.load(std::memory_order_relaxed);
        if (value == 0) {
            return false;
        }
        if (value == address) {
            return true;
        }
    }
    return false;
}

float RytyDisplayAspect(); // ryty_game_patch.cpp (GTS-009)

bool NativeRenderer::HudSplitOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_HUD_SPLIT");
        return RytyDisplayAspect() > 0.0f && !(env && env[0] == '0');
    }();
    return on;
}

bool NativeRenderer::HudStreamOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_HUD_STREAM");
        return !(env && env[0] == '0');
    }();
    return on;
}

// Called with the HUD list swapped in as the stream.
void NativeRenderer::BeginHudStream() {
    Stream& s = stream;
    if (s.list) {
        return;
    }
    AcquireList(s); // PERF-GT7-018
    s.rendering = false;
    s.dynamic.Invalidate();
    s.pipeline = vk::Pipeline{};
    s.so_known = false; // SO-010
    s.vertex_state_known = false;
    s.commands = 0;
    s.run_program = 0; // PERF-GT7-043
    s.run_texels.Clear(); // PERF-GT7-057
    ++s.serial;        // OPT-015
    ForgetBindings(s); // OPT-014
    // It runs after the main list: what that list wrote is finished first.
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                              .pMemoryBarriers = &barrier});
}

// Runs the HUD list after the main one (in the same command buffer, at the same tick).
void NativeRenderer::SpliceHudStream(u64 tick) {
    Stream& s = hud_stream;
    if (!s.list) {
        return;
    }
    if (s.rendering) {
        s.cmd.endRendering();
        s.rendering = false;
    }
    s.list->End();
    r.scheduler.CommandBuffer().executeCommands(s.cmd);
    device->MarkExecuted(s.list, {timeline.get(), tick});
    {
        std::scoped_lock chunk_lock{chunk_mutex};
        for (auto& chunk : s.chunks) {
            chunk.retire_tick = tick;
            retiring_chunks.push_back(std::move(chunk));
        }
        for (const auto pool : s.pools) {
            retiring_pools.emplace_back(tick, pool);
        }
    }
    s.chunks.clear();
    s.chunk_used = 0;
    s.pools.clear();
    RetireSnapshots(s, tick); // PERF-GT7-007
    s.keep.clear();
    s.list = nullptr;
    s.cmd = CmdBuf{};
    ++s.serial;        // OPT-015: the list's upload memory retires with it
    ForgetBindings(s); // OPT-014
}

Native::ImageEntry* NativeRenderer::MakeHudPicture(const Native::ImageEntry& like) {
    auto* entry = new Native::ImageEntry{};
    entry->info = like.info;
    Native::MadeImage made = MakeImage(entry->info, true);
    if (!made.image) {
        delete entry;
        return nullptr;
    }
    entry->image = std::move(made.image);
    entry->backend_usage = made.usage;
    entry->backend_features = made.features;
    // New pictures start undefined; native mode keeps its pictures in the general layout.
    Stream& s = Begin();
    EndRendering();
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eNone,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())},
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                              .pImageMemoryBarriers = &barrier});
    LOG_INFO(Render_Vulkan, "GT7-UW: HUD picture for the display {:#x} ({}x{})",
             like.info.guest_address, like.info.size.width, like.info.size.height);
    return entry;
}

// Port of xacidr4inx's reflection fix (RyGT 0066): square colour images of six layers or more can
// be viewed as cube arrays. Not depth images (GT7's shadow cascades). FIX-GT7-084: what View checks
// before it makes a cube-array view.
static bool CubeCompatible(const Instance& instance, u32 width, u32 height, u32 layers, u32 samples,
                           bool volume, bool depth) {
    return instance.IsImageCubeArraySupported() && !volume && !depth && width == height &&
           layers >= 6 && samples == 1;
}

Native::MadeImage NativeRenderer::MakeImage(const VideoCore::ImageInfo& info, bool) {
    Native::MadeImage made{};
    if (info.pixel_format == vk::Format::eUndefined) {
        return made;
    }
    const Instance& instance = r.instance;
    vk::ImageCreateFlags flags{vk::ImageCreateFlagBits::eMutableFormat |
                               vk::ImageCreateFlagBits::eExtendedUsage};
    if (CubeCompatible(instance, info.size.width, info.size.height, info.resources.layers,
                       info.num_samples, info.props.is_volume, info.props.is_depth)) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (info.props.is_volume) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance.Is2dViewOf3dSupported()) {
            flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance.IsBlockTexelViewSupported()) {
        flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }
    vk::ImageUsageFlags usage = UsageOf(instance, info);
    vk::FormatFeatureFlags2 features = FeaturesOf(usage);
    vk::Format format = instance.GetSupportedFormat(info.pixel_format, features);
    const vk::ImageType type = TypeOf(info.type);
    auto properties =
        instance.GetPhysicalDevice().getImageFormatProperties2(vk::PhysicalDeviceImageFormatInfo2{
            .format = format,
            .type = type,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = usage,
            .flags = flags,
        });
    // FIX-232: an AMD driver refused BC6H images with storage use (about two million lookups
    // of a tester's session found no image: textures missing). Such an image is made without
    // the storage use (and the block-texel-view flag that goes with it): it is still sampled
    // and copied as the game does with it. RYGT_BLOCK_NO_STORAGE_FALLBACK=1 keeps the old way.
    static const bool fallback = [] {
        const char* env = std::getenv("RYGT_BLOCK_NO_STORAGE_FALLBACK");
        return !(env && env[0] == '1');
    }();
    if (properties.result != vk::Result::eSuccess && fallback &&
        (usage & vk::ImageUsageFlagBits::eStorage)) {
        const vk::ImageUsageFlags reduced = usage & ~vk::ImageUsageFlagBits::eStorage;
        const vk::ImageCreateFlags reduced_flags =
            flags & ~vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
        const vk::Format reduced_format =
            instance.GetSupportedFormat(info.pixel_format, FeaturesOf(reduced));
        auto retry = instance.GetPhysicalDevice().getImageFormatProperties2(
            vk::PhysicalDeviceImageFormatInfo2{
                .format = reduced_format,
                .type = type,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = reduced,
                .flags = reduced_flags,
            });
        if (retry.result == vk::Result::eSuccess) {
            static std::atomic<u32> logged{};
            if (Sample(logged)) {
                LOG_WARNING(Render_Vulkan,
                            "FIX-232: format {} ({}x{}) made without storage use (the driver "
                            "refuses it with storage)",
                            vk::to_string(format), info.size.width, info.size.height);
            }
            usage = reduced;
            flags = reduced_flags;
            format = reduced_format;
            features = FeaturesOf(reduced);
            properties = retry;
        }
    }
    if (properties.result != vk::Result::eSuccess) {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_ERROR(Render_Vulkan, "NAT-100: no image of format {} ({}x{}) at {:#x}",
                      vk::to_string(format), info.size.width, info.size.height,
                      info.guest_address);
        }
        return made;
    }
    const vk::ImageCreateInfo image_ci{
        .flags = flags,
        .imageType = type,
        .format = format,
        .extent{info.size.width, info.size.height, std::max(info.size.depth, 1u)},
        .mipLevels = std::max<u32>(info.resources.levels, 1),
        .arrayLayers = type == vk::ImageType::e3D ? 1u : std::max<u32>(info.resources.layers, 1),
        .samples = LiverpoolToVK::NumSamples(
            info.num_samples, properties.value.imageFormatProperties.sampleCounts),
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    Native::Rhi::ImageDesc desc{};
    desc.dim = type == vk::ImageType::e3D ? Native::Rhi::ImageDim::D3 : Native::Rhi::ImageDim::D2;
    desc.format = Native::Rhi::FormatFromVk(static_cast<u32>(format));
    desc.width = image_ci.extent.width;
    desc.height = image_ci.extent.height;
    desc.depth = image_ci.extent.depth;
    desc.mips = image_ci.mipLevels;
    desc.layers = image_ci.arrayLayers;
    desc.samples = static_cast<u32>(image_ci.samples);
    desc.usage = RhiUsage(usage);
    made.image = Native::Rhi::CreateVulkanImage(
        *device, static_cast<const VkImageCreateInfo&>(image_ci), desc);
    made.usage = static_cast<u32>(static_cast<VkImageUsageFlags>(usage));
    made.features = static_cast<u64>(static_cast<VkFormatFeatureFlags2>(features));
    return made;
}

vk::ImageView NativeRenderer::View(const Native::ImageEntry& entry,
                                   const VideoCore::ImageViewInfo& requested, ViewUse use) {
    std::scoped_lock lock{views_mutex};
    auto& records = views[&entry];
    for (const auto& record : records) {
        if (record.use == use && record.info == requested) {
            return *record.view;
        }
    }
    // As the texture cache's ImageView (texture_cache/image_view.cpp), on the native image.
    const auto& image_desc = entry.image->Desc();
    const vk::Image image{Handle<VkImage_T>(entry.image->NativeHandle())};
    const vk::Format image_format = static_cast<vk::Format>(image_desc.format);
    const bool volume = image_desc.dim == Native::Rhi::ImageDim::D3;
    VideoCore::ImageViewInfo info = requested;
    // A view never reaches past the image (a T# may name more levels or layers than the
    // layout the image was made for).
    const u32 mips = std::max<u32>(image_desc.mips, 1);
    const u32 layers = volume ? 1u : std::max<u32>(image_desc.layers, 1);
    info.range.base.level = std::min<u32>(info.range.base.level, mips - 1);
    info.range.extent.levels =
        std::clamp<u32>(info.range.extent.levels, 1, mips - info.range.base.level);
    info.range.base.layer = std::min<u32>(info.range.base.layer, layers - 1);
    info.range.extent.layers =
        std::clamp<u32>(info.range.extent.layers, 1, layers - info.range.base.layer);

    vk::ImageUsageFlags usage{entry.backend_usage};
    if (use != ViewUse::Storage) {
        usage &= ~vk::ImageUsageFlagBits::eStorage;
    }
    vk::ImageViewUsageCreateInfo usage_ci{.usage = usage};
    vk::ImageViewMinLodCreateInfoEXT min_lod_ci{};
    if (info.min_lod != 0 && r.instance.IsImageViewMinLodSupported()) {
        const float last_level =
            static_cast<float>(info.range.base.level + info.range.extent.levels - 1);
        min_lod_ci.minLod = std::min(static_cast<float>(info.min_lod) / 256.f, last_level);
        usage_ci.pNext = &min_lod_ci;
    }
    vk::Format format = info.format;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
    if (entry.info.props.is_depth) {
        // A depth image: attachments see depth (and stencil); a T# names R32/R16 for depth and
        // R8 for stencil.
        const vk::ImageAspectFlags all =
            entry.info.props.has_stencil
                ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
                : vk::ImageAspectFlagBits::eDepth;
        if (use == ViewUse::Attachment) {
            aspect = all;
        } else if (entry.info.props.has_stencil && !LiverpoolToVK::IsFormatDepthCompatible(format) &&
                   LiverpoolToVK::IsFormatStencilCompatible(format)) {
            aspect = vk::ImageAspectFlagBits::eStencil;
        } else {
            aspect = vk::ImageAspectFlagBits::eDepth;
        }
        format = image_format;
    } else {
        format = r.instance.GetSupportedFormat(format, vk::FormatFeatureFlags2{entry.backend_features});
    }
    vk::ImageViewType view_type = VideoCore::ConvertImageViewType(info.type);
    if (use == ViewUse::CubeSampled) {
        // FIX-GT7-084: the code samples a cube array (NativeCubeSampling), so the view is one:
        // whole cubes (VUID-VkImageViewCreateInfo-viewType-02961) of a picture made cube
        // compatible (image-01003). A range the picture cuts short keeps its whole cubes. Under
        // six faces, a volume, or a picture made without the flag, no view fits the code: the
        // slot is bound empty, as a failed view was, without the invalid call.
        const u32 cubes = info.range.extent.layers / 6;
        if (cubes == 0 || !CubeCompatible(r.instance, image_desc.width, image_desc.height,
                                          image_desc.layers, image_desc.samples, volume,
                                          entry.info.props.is_depth)) {
            static std::atomic<u32> logged{};
            if (Sample(logged)) {
                LOG_WARNING(Render_Vulkan,
                            "FIX-GT7-084: cube sample of {:#x} ({}x{}, {} layers from {} of {}) "
                            "bound empty: no cube-array view fits",
                            entry.info.guest_address, image_desc.width, image_desc.height,
                            info.range.extent.layers, info.range.base.layer, image_desc.layers);
            }
            return {};
        }
        info.range.extent.layers = cubes * 6;
        view_type = vk::ImageViewType::eCubeArray;
    }
    if (volume) {
        view_type = vk::ImageViewType::e3D;
    } else if (view_type == vk::ImageViewType::e3D) {
        view_type = info.range.extent.layers > 1 ? vk::ImageViewType::e2DArray
                                                 : vk::ImageViewType::e2D;
    } else if (view_type == vk::ImageViewType::e2D && info.range.extent.layers > 1) {
        view_type = vk::ImageViewType::e2DArray;
    }
    const vk::ImageViewCreateInfo view_ci{
        .pNext = &usage_ci,
        .image = image,
        .viewType = view_type,
        .format = format,
        .components = info.mapping == AmdGpu::IdentityMapping
                          ? vk::ComponentMapping{}
                          : LiverpoolToVK::ComponentMapping(info.mapping),
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = info.range.base.level,
            .levelCount = info.range.extent.levels,
            .baseArrayLayer = info.range.base.layer,
            .layerCount = info.range.extent.layers,
        },
    };
    auto [result, view] = r.instance.GetDevice().createImageViewUnique(view_ci);
    if (result != vk::Result::eSuccess) {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_ERROR(Render_Vulkan, "NAT-100: no view {} of image {:#x} ({})",
                      vk::to_string(format), entry.info.guest_address, vk::to_string(result));
        }
        return {};
    }
    const vk::ImageView handle = *view;
    records.push_back({requested, use, std::move(view)});
    return handle;
}

vk::Sampler NativeRenderer::SamplerFor(const AmdGpu::Sampler& original,
                                       AmdGpu::BorderColorBuffer border, bool is_depth) {
    auto sharp = original;
    if (is_depth && (DiagnosticPassOptions(frame).first & 1)) {
        sharp.depth_compare_func.Assign(AmdGpu::DepthCompare::Always);
    }
    // PERF-GT7-015 (C): the samplers this thread looked up last, by raw S#, without the lock and
    // the hash map (a draw binds ~4; samplers live as long as the renderer).
    struct CachedSampler {
        std::array<u64, 2> sharp;
        u64 extra;
        vk::Sampler sampler;
    };
    static thread_local std::array<CachedSampler, 256> cached{};
    static_assert(sizeof(AmdGpu::Sampler) == sizeof(std::array<u64, 2>));
    CachedSampler* cache = nullptr;
    std::array<u64, 2> raw;
    std::memcpy(raw.data(), &sharp, sizeof(raw));
    const bool custom = sharp.border_color_type == AmdGpu::BorderColor::Custom;
    const u64 extra = (is_depth ? 1u : 0u) | (custom ? u64(border.Address()) << 1 : 0);
    if (NextOn()) {
        cache = &cached[MemoHash(raw.data(), raw.size(), extra) & (cached.size() - 1)];
        if (cache->sampler && cache->sharp == raw && cache->extra == extra) {
            return cache->sampler;
        }
    }
    const vk::Sampler sampler = MakeSampler(sharp, border, is_depth);
    if (cache) {
        *cache = {raw, extra, sampler};
    }
    return sampler;
}

vk::Sampler NativeRenderer::MakeSampler(const AmdGpu::Sampler& sharp,
                                        AmdGpu::BorderColorBuffer border, bool is_depth) {
    const bool custom = sharp.border_color_type == AmdGpu::BorderColor::Custom;
    const u64 hash = XXH3_64bits(&sharp, sizeof(sharp)) ^
                     (is_depth ? 0x9E3779B97F4A7C15ULL : 0) ^
                     (custom ? u64(border.Address()) << 1 : 0);
    std::scoped_lock lock{sampler_mutex};
    if (const auto it = sampler_lookup.find(hash); it != sampler_lookup.end()) {
        return it->second;
    }
    // FIX-GT7-002: one Vulkan sampler per state the device sees. GT7's S#s differ in bits the
    // sampler doesn't use (performance hints, unused bits, the border colour table when the
    // colour isn't custom), and a sampler per raw S# passed the driver's limit (4000 on NVIDIA):
    // "Failed to create sampler: ErrorOutOfDeviceMemory" ended the game.
    AmdGpu::Sampler state{};
    state.clamp_x.Assign(sharp.clamp_x.Value());
    state.clamp_y.Assign(sharp.clamp_y.Value());
    state.clamp_z.Assign(sharp.clamp_z.Value());
    state.xy_mag_filter.Assign(sharp.xy_mag_filter.Value());
    state.xy_min_filter.Assign(sharp.xy_min_filter.Value());
    state.mip_filter.Assign(sharp.mip_filter.Value());
    if (AmdGpu::IsAnisoFilter(sharp.xy_mag_filter) || AmdGpu::IsAnisoFilter(sharp.xy_min_filter)) {
        state.max_aniso.Assign(sharp.max_aniso.Value());
    }
    if (is_depth) {
        state.depth_compare_func.Assign(sharp.depth_compare_func.Value());
    }
    state.border_color_type.Assign(sharp.border_color_type.Value());
    std::array<float, 4> custom_color{};
    if (custom) {
        state.border_color_ptr.Assign(sharp.border_color_ptr.Value());
        if (border.Address() != 0) {
            custom_color = border.Address<std::array<float, 4>*>()[sharp.border_color_ptr.Value()];
        }
    }
    const u32 limit = r.instance.GetMaxSamplerAllocationCount();
    if (samplers.size() >= limit / 4 * 3) {
        // Close to the limit: levels of detail in quarter steps.
        const auto quarter = [](float lod) { return std::round(lod * 4.0f) * 64.0f; };
        state.min_lod.Assign(u64(quarter(sharp.MinLod())));
        state.max_lod.Assign(u64(quarter(sharp.MaxLod())));
        state.lod_bias.Assign(u64(s64(quarter(sharp.LodBias()))) & 0x3fff);
    } else {
        state.min_lod.Assign(sharp.min_lod.Value());
        state.max_lod.Assign(sharp.max_lod.Value());
        state.lod_bias.Assign(sharp.lod_bias.Value());
    }
    // GFX-GT7-001 (the launcher's Texture sharpness, RYGT_TEXTURE_LOD_BIAS=<levels>): added to every
    // sampler's bias, negative for sharper textures. Compare samplers keep theirs: a bias sharpens
    // nothing in a shadow test.
    static const float texture_lod_bias = [this] {
        const char* env = std::getenv("RYGT_TEXTURE_LOD_BIAS");
        const float bias = env && *env ? std::strtof(env, nullptr) : 0.0f;
        if (!std::isfinite(bias) || bias == 0.0f) {
            return 0.0f;
        }
        LOG_INFO(Render_Vulkan, "GFX-GT7-001: LOD bias {} on every sampler, device limit +-{}",
                 bias, r.instance.MaxSamplerLodBias());
        return bias;
    }();
    if (texture_lod_bias != 0.0f && !is_depth) {
        // Vulkan bounds the bias's absolute value; sampler.cpp only bounds it from above.
        const float max_bias = r.instance.MaxSamplerLodBias();
        const float bias = std::clamp(state.LodBias() + texture_lod_bias, -max_bias, max_bias);
        state.lod_bias.Assign(u64(s64(std::round(bias * 256.0f))) & 0x3fff);
    }
    struct Key {
        AmdGpu::Sampler state;
        std::array<float, 4> color;
        u64 depth;
    };
    const Key key{state, custom_color, is_depth ? 1u : 0u};
    const u64 state_hash = XXH3_64bits(&key, sizeof(key));
    auto& slot = samplers[state_hash];
    if (!slot) {
        if (samplers.size() + 64 > limit && !samplers.empty()) {
            // At the limit: the last sampler made stands in (logged), rather than a failure.
            static std::atomic<u32> logged{};
            if (Sample(logged)) {
                LOG_ERROR(Render_Vulkan, "FIX-GT7-002: {} samplers, the device's limit is {}",
                          samplers.size(), limit);
            }
            samplers.erase(state_hash);
            return last_sampler;
        }
        slot = std::make_unique<VideoCore::Sampler>(r.instance, state, border, is_depth);
        last_sampler = slot->Handle();
    }
    if (sampler_lookup.size() > (1u << 20)) {
        sampler_lookup.clear();
    }
    sampler_lookup.emplace(hash, slot->Handle());
    return slot->Handle();
}

void NativeRenderer::AcquireList(Stream& s) {
    // PERF-GT7-018: with the scheduler's command thread, the list's commands go into its ring and
    // the command thread makes the Vulkan calls, the list's own (made, reset, begun, ended) too.
    // In races about a fifth of the recorder's time went to the driver recording these lists;
    // the caches still run here, in order, so nothing about what is recorded changes.
    if (CommandRing* ring = r.scheduler.CommandThreadRing(); ring && RingOn()) {
        if (Native::Rhi::CommandList* list = device->AcquireDeferredCommandList(ring)) {
            s.list = list;
            s.list->Begin();
            s.cmd = CmdBuf{ring, reinterpret_cast<const vk::CommandBuffer*>(list->NativeHandleSlot())};
            return;
        }
    }
    s.list = device->AcquireCommandList(Native::Rhi::CommandListLevel::Secondary);
    s.list->Begin();
    s.cmd = CmdBuf{vk::CommandBuffer{Handle<VkCommandBuffer_T>(s.list->NativeHandle())}};
}

CmdBuf NativeRenderer::CmdOf(Native::Rhi::CommandList& list) const {
    if (stream.list == &list) {
        return stream.cmd;
    }
    if (hud_stream.list == &list) {
        return hud_stream.cmd;
    }
    ASSERT_MSG(list.NativeHandleSlot() == nullptr,
               "PERF-GT7-018: commands for a deferred list of no stream");
    return CmdBuf{vk::CommandBuffer{Handle<VkCommandBuffer_T>(list.NativeHandle())}};
}

NativeRenderer::Stream& NativeRenderer::Begin() {
    Stream& s = stream;
    if (s.list) {
        return s;
    }
    if (r.instance.IsIntel()) {
        // FIX-188: Intel Arc (#65) records into the scheduler's own command buffer.
        r.scheduler.EndRendering();
        const vk::CommandBuffer direct = r.scheduler.DirectCommandBuffer();
        s.cmd = CmdBuf{direct};
        s.wrapped_list = device->WrapCommandBuffer(
            reinterpret_cast<u64>(static_cast<VkCommandBuffer>(direct)));
        s.list = s.wrapped_list.get();
    } else {
        AcquireList(s); // PERF-GT7-018
    }
    if (!s.lds_readers.empty()) {
        // PERF-GT7-060: a list ends through EndRendering, which records them; none may be left.
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_ERROR(Render_Vulkan, "PERF-GT7-060: {} reader passes left from an ended list",
                      s.lds_readers.size());
        }
        s.lds_readers.clear();
        s.lds_bytes = 0;
    }
    s.rendering = false;
    s.dynamic.Invalidate();
    s.pipeline = vk::Pipeline{};
    s.so_known = false; // SO-010
    s.vertex_state_known = false; // OPT-005
    s.commands = 0;
    s.run_program = 0; // PERF-GT7-043
    s.run_texels.Clear(); // PERF-GT7-057
    GpuTimeBegin(s); // DIAG-GT7-001: the first segment is the wait for earlier GPU work
    ++s.serial;        // OPT-015
    ForgetBindings(s); // OPT-014
    // What was recorded before this list (earlier lists, the presenter) is finished first.
    GlobalBarrier();
    MakePendingMirrors(); // NAT-150: behind the barrier the list starts with anyway
    return s;
}

void NativeRenderer::BeginAhead() {
    // PERF-GT7-008 (B): a list begun (command list, barrier, pending mirrors) cost a dispatch
    // 2.3 us on average in IB-3, the first work after each of the ~5,000 splices a second; it
    // is begun while the recorder waits for work instead. Not on Intel (FIX-188: the native
    // work goes in the scheduler's own command buffer there).
    if (!ReuseOn() || r.instance.IsIntel()) {
        return;
    }
    // PERF-GT7-018: a list begins at little cost when the command thread makes it; and the
    // command processor may record into the ring meanwhile (it drained the recorder to submit).
    if (RingOn() && r.scheduler.HasCommandThread()) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (stream.list) {
        return;
    }
    AmdGpu::RecorderParts::Scope part{AmdGpu::RecorderParts::Part::Ahead}; // DIAG-GT7-005
    Begin();
    ++stats.lists_ahead;
}

bool NativeRenderer::GpuTimeOn() {
    static const bool on = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_GPU_TIME");
        return env && env[0] == '1';
    }();
    return on;
}

void NativeRenderer::GpuTimeBegin(Stream& s) {
    if (!GpuTimeOn() || s.wrapped_list) {
        return;
    }
    vk::QueryPool pool{};
    if (!gpu_timing_pools.empty()) {
        pool = gpu_timing_pools.back();
        gpu_timing_pools.pop_back();
    } else {
        const auto [result, made] = r.instance.GetDevice().createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = GpuTimeQueries,
        });
        if (result != vk::Result::eSuccess) {
            return;
        }
        pool = made;
    }
    auto timing = std::make_unique<GpuTiming>();
    timing->pool = pool;
    timing->frame = frame;
    timing->cpu_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_timing_start).count();
    timing->segments.emplace_back();
    const auto& registry_stats = registry->GetStats();
    timing->uploads_at = registry_stats.uploads.load(std::memory_order_relaxed);
    timing->upload_bytes_at = registry_stats.upload_bytes.load(std::memory_order_relaxed);
    s.cmd.resetQueryPool(pool, 0, GpuTimeQueries);
    s.cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, timing->used++);
    s.gpu_timing = std::move(timing);
}

void NativeRenderer::GpuTimeMark(Stream& s) {
    GpuTiming* t = s.gpu_timing.get();
    if (!t || t->used + 1 >= GpuTimeQueries) {
        return; // (past the pool's size the rest of the list goes in its last segment)
    }
    const auto& registry_stats = registry->GetStats();
    const u64 uploads = registry_stats.uploads.load(std::memory_order_relaxed);
    const u64 upload_bytes = registry_stats.upload_bytes.load(std::memory_order_relaxed);
    GpuSegment& seg = t->segments.back();
    seg.uploads += u32(uploads - t->uploads_at);
    seg.upload_bytes += upload_bytes - t->upload_bytes_at;
    t->uploads_at = uploads;
    t->upload_bytes_at = upload_bytes;
    s.cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, t->pool, t->used++);
    t->segments.emplace_back();
}

void NativeRenderer::GpuTimeEnd(Stream& s, u64 tick) {
    GpuTiming* t = s.gpu_timing.get();
    if (!t) {
        return;
    }
    if (t->used + 1 < GpuTimeQueries) {
        GpuTimeMark(s); // the end, with an empty segment after it
        t->segments.pop_back();
    }
    t->tick = tick;
    t->cpu_end_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_timing_start).count();
    gpu_timing_pending.push_back(std::move(s.gpu_timing));
}

void NativeRenderer::GpuTimeCollect() {
    if (gpu_timing_pending.empty()) {
        return;
    }
    const double period_ms = double(r.instance.GetImageLimits().timestampPeriod) / 1e6;
    const vk::Device vk_device = r.instance.GetDevice();
    static thread_local std::vector<u64> stamps;
    while (!gpu_timing_pending.empty()) {
        GpuTiming& t = *gpu_timing_pending.front();
        // FIX-GT7-014: only once the list's submission is done. A pool used before can still
        // report that use's results until the GPU runs this list's reset: eSuccess alone took
        // them for this list's, and the pool went back to the free ones while the GPU had it.
        if (!r.scheduler.IsFree(t.tick)) {
            break; // later lists come after it
        }
        stamps.assign(t.used, 0);
        const vk::Result result = vk_device.getQueryPoolResults(
            t.pool, 0, t.used, stamps.size() * sizeof(u64), stamps.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        if (result == vk::Result::eNotReady) {
            break; // the GPU has not run it yet (later lists come after it)
        }
        if (result == vk::Result::eSuccess && t.used >= 2) {
            const double list_ms = double(stamps[t.used - 1] - stamps[0]) * period_ms;
            const double gap_ms = gpu_timing_prev_end != 0 && stamps[0] > gpu_timing_prev_end
                                      ? double(stamps[0] - gpu_timing_prev_end) * period_ms
                                      : 0.0;
            gpu_timing_prev_end = stamps[t.used - 1];
            // On the CPU: from the previous list's end to this one's start (a GPU gap about as
            // long means the GPU waited for the recorder; much longer, it ran other work).
            const double cpu_gap_ms =
                gpu_timing_prev_cpu_end != 0.0 ? (t.cpu_s - gpu_timing_prev_cpu_end) * 1000.0 : 0.0;
            gpu_timing_prev_cpu_end = t.cpu_end_s;
            auto& st = gpu_time_stats;
            ++st.lists;
            st.busy_ms += list_ms;
            st.max_list_ms = std::max(st.max_list_ms, list_ms);
            st.max_gap_ms = std::max(st.max_gap_ms, gap_ms);
            // The slowest segments of a slow list (or after a long gap), with what they held.
            std::vector<std::pair<double, u32>> segs;
            for (u32 i = 0; i + 1 < t.used && i < t.segments.size(); ++i) {
                segs.emplace_back(double(stamps[i + 1] - stamps[i]) * period_ms, i);
            }
            std::ranges::sort(segs, std::greater{});
            const bool slow = list_ms >= 20.0 || gap_ms >= 20.0 ||
                              (!segs.empty() && segs.front().first >= 10.0);
            if (slow) {
                ++st.slow;
                std::string text;
                for (size_t k = 0; k < std::min<size_t>(segs.size(), 4); ++k) {
                    const auto [ms, i] = segs[k];
                    const GpuSegment& g = t.segments[i];
                    text += fmt::format(
                        "; seg {}/{} {:.2f} ms: {} draws (max {} prims, ps {:#x}), {} dispatches "
                        "(max {} groups, cs {:#x}), {} uploads ({} KB)",
                        i, t.segments.size(), ms, g.draws, g.max_prims, g.max_prims_ps,
                        g.dispatches, g.max_groups, g.max_groups_cs, g.uploads,
                        g.upload_bytes / 1024);
                }
                u32 draws = 0, dispatches = 0;
                for (const GpuSegment& g : t.segments) {
                    draws += g.draws;
                    dispatches += g.dispatches;
                }
                LOG_WARNING(Render_Vulkan,
                            "DIAG-GT7-001: frame {} at {:.2f} s: list {:.2f} ms on the GPU "
                            "({} draws, {} dispatches, {} segments, recorded in {:.2f} ms), "
                            "started {:.2f} ms after the previous list ended on the GPU ({:.2f} "
                            "ms on the CPU){}",
                            t.frame, t.cpu_s, list_ms, draws, dispatches, t.segments.size(),
                            (t.cpu_end_s - t.cpu_s) * 1000.0, gap_ms, cpu_gap_ms, text);
            }
        }
        gpu_timing_pools.push_back(t.pool);
        gpu_timing_pending.pop_front();
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - gpu_timing_report >= std::chrono::seconds{10}) {
        auto& st = gpu_time_stats;
        LOG_WARNING(Render_Vulkan,
                    "DIAG-GT7-001 (10 s): {} lists, {:.0f} ms on the GPU in them, longest list "
                    "{:.2f} ms, longest gap between lists {:.2f} ms, {} slow lists logged",
                    st.lists, st.busy_ms, st.max_list_ms, st.max_gap_ms, st.slow);
        st = {};
        gpu_timing_report = now;
    }
}

void NativeRenderer::EndRendering() {
    // PERF-GT7-060: whatever ends the pass (a barrier, copies, a dispatch, another pass, the
    // list's end) comes after the run's reader passes.
    if (!stream.lds_readers.empty()) {
        FlushLdsReaders(1);
    }
    if (stream.rendering) {
        stream.cmd.endRendering();
        stream.rendering = false;
    }
}

void NativeRenderer::BeginRendering(const RenderState& state) {
    Stream& s = stream;
    if (s.rendering && s.render == state) {
        return;
    }
    EndRendering();
    // As Scheduler::BeginRendering.
    std::array<vk::RenderingAttachmentInfo, 8> colors;
    for (u32 i = 0; i < state.num_color_attachments; ++i) {
        const auto& cb = state.color_attachments[i];
        colors[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = vk::ImageLayout::eGeneral,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }
    const auto& db = state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth{
        .imageView = db.image_view,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{
                                         .depth = std::bit_cast<float>(db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil{
        .imageView = db.image_view,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };
    s.cmd.beginRendering(vk::RenderingInfo{
        .renderArea{.offset = {0, 0}, .extent = {state.width, state.height}},
        .layerCount = state.num_layers,
        .colorAttachmentCount = state.num_color_attachments,
        .pColorAttachments = colors.data(),
        .pDepthAttachment = db.has_depth ? &depth : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil : nullptr,
    });
    s.rendering = true;
    s.render = state;
    ++stats.passes;
}

void NativeRenderer::GlobalBarrier() {
    Stream& s = stream;
    EndRendering();
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    GpuTimeMark(s); // DIAG-GT7-001
    s.written.clear();
    s.buffers_written = false;
    s.attach_dirty.clear();
    s.read.clear();
    s.run_texels.Clear(); // PERF-GT7-057
    ++stats.barriers;
    FlushWritebacks();
}

void NativeRenderer::PassBarrier() {
    Stream& s = stream;
    EndRendering();
    constexpr auto stages = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                            vk::PipelineStageFlagBits2::eLateFragmentTests;
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = stages,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .dstStageMask = stages,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                         vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    s.attach_dirty.clear();
    ++stats.pass_barriers;
}

// FIX-150: GT Sport's small buffers from 0xf000000000 up are ones the GPU writes for the CPU
// to read back. RYGT_NATIVE_INPLACE_FROM=<hex address> moves the limit
// (ffffffffffffffff: every written range gets a region, as before).
static VAddr InplaceFrom() {
    static const VAddr from = [] {
        const char* env = std::getenv("RYGT_NATIVE_INPLACE_FROM");
        return env ? static_cast<VAddr>(std::strtoull(env, nullptr, 16)) : VAddr{0xf000000000};
    }();
    return from;
}

// FIX-GT7-041: a picture larger than NAT-112's drawn in the memory the CPU reads goes back in this
// submission, before the fence the CPU waits for. GT7 saves a car's thumbnail this way: it copies
// the finished 1920x1080 picture into two 1920x540 targets there, waits for the GPU and encodes
// them to PNG, so the saved thumbnail held what that memory held before: black. GT7 draws few such
// pictures: ten in a session with one customization. RYGT_NATIVE_WRITEBACK_CPU=0: as before.
void NativeRenderer::WritebackForCpu(const Native::ImageEntry* entry) {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_WRITEBACK_CPU");
        return !(env && env[0] == '0');
    }();
    if (!on || entry->info.guest_address < InplaceFrom() ||
        std::ranges::find(writebacks, entry) != writebacks.end()) {
        return;
    }
    writebacks.push_back(const_cast<Native::ImageEntry*>(entry));
    static std::atomic<u32> logged{};
    if (const u32 n = logged++; n < 64) {
        LOG_WARNING(Render_Vulkan, "FIX-GT7-041: picture {:#x} {}x{} goes back for the CPU ({} so far)",
                    entry->info.guest_address, entry->info.size.width, entry->info.size.height,
                    n + 1);
    }
}

// PERF-GT7-044: NAT-112 writes a small picture back to guest memory once a frame, when the frame
// ends, for the CPU to read (GT Sport's auto exposure), and not after every draw into it. GT7 draws
// ~100 such pictures a frame: on its pre-race screens ~210 write-backs a frame, nearly all tiled (a
// copy, a tiling dispatch and their barriers each) took 17% of the GPU's time in a GPU Trace.
// RYGT_NATIVE_WRITEBACK_EACH=1: after each draw, as before.
void NativeRenderer::QueueWriteback(const Native::ImageEntry* entry) {
    static const bool each = [] {
        const char* env = std::getenv("RYGT_NATIVE_WRITEBACK_EACH");
        return env && env[0] == '1';
    }();
    if (each) {
        if (std::ranges::find(writebacks, entry) == writebacks.end()) {
            writebacks.push_back(const_cast<Native::ImageEntry*>(entry));
        }
        return;
    }
    // Pictures of the same size at the same memory each write all of it: only the last one drawn
    // counts, and goes last.
    std::erase_if(frame_writebacks, [&](const Native::ImageEntry* other) {
        return other->info.guest_address == entry->info.guest_address &&
               other->info.guest_size == entry->info.guest_size;
    });
    frame_writebacks.push_back(const_cast<Native::ImageEntry*>(entry));
}

// PERF-GT7-044: work that writes memory a waiting picture covers comes after that picture, as the
// game ordered them; written back at the frame's end, the picture would overwrite the newer bytes.
// It goes at the next global barrier, before that work, and the bytes the work leaves stay right.
void NativeRenderer::WritebacksBefore(VAddr address, u64 size) {
    const auto meets = [&](const Native::ImageEntry* entry) {
        const VAddr begin = entry->info.guest_address;
        return address + std::max<u64>(size, 1) > begin && begin + entry->info.guest_size > address;
    };
    // FIX-GT7-040: a picture moved here when its frame ended counts too: unnoticed, it reached the
    // region only at the region's next use, over what this work writes.
    if (std::ranges::any_of(writebacks, meets)) {
        writebacks_due = true;
    }
    std::erase_if(frame_writebacks, [&](Native::ImageEntry* entry) {
        if (!meets(entry)) {
            return false;
        }
        if (std::ranges::find(writebacks, entry) == writebacks.end()) {
            writebacks.push_back(entry);
        }
        writebacks_due = true;
        return true;
    });
}

// FIX-GT7-040: what work writes in a region stays there, so a picture going back to guest memory
// first must reach the region first too. Left for the region's next use, it was copied in over the
// shader's bytes: GT7 fills its rain layer, whose memory also holds its reflection's small mips, and
// with no rain to draw it samples the fill right away. Those mips showed as a 64 KB block of huge
// values at every frame of a pre-race screen, under a large halo. A fill or a copy gets its region
// after this; a draw or dispatch has it already, and refreshes it again when this returns true.
// RYGT_NATIVE_WRITEBACK_REFRESH=0: as before.
bool NativeRenderer::SendWritebacksDue() {
    static const bool refresh = [] {
        const char* env = std::getenv("RYGT_NATIVE_WRITEBACK_REFRESH");
        return !(env && env[0] == '0');
    }();
    if (!refresh || !writebacks_due) {
        return false;
    }
    GlobalBarrier(); // the pictures go back here
    return true;
}

void NativeRenderer::WritebacksBefore(const DrawBindings& b) {
    for (const auto& [address, size] : b.gpu_writes) {
        WritebacksBefore(address, size);
    }
    if (!SendWritebacksDue()) {
        return;
    }
    for (const auto& [address, size] : b.gpu_writes) {
        if (Region* region = RegionOver(address, size)) {
            RefreshRegion(*region);
        }
    }
}

void NativeRenderer::FlushWritebacks() {
    if (writebacks.empty() || !stream.list) {
        return;
    }
    AmdGpu::RecorderParts::Scope recorder_part{AmdGpu::RecorderParts::Part::WriteBacks};
    struct Timed {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        u64& total;
        ~Timed() {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    } timed{.total = stats.writeback_ns}; // DIAG-188
    Stream& s = stream;
    auto* direct = r.buffer_cache.GetDirectMemory();
    const u64 alignment = r.instance.StorageMinAlignment();
    // PERF-GT7-045: a frame's ~190 pictures took 6 to 8 ms of GPU in a row at each frame's end, a
    // tiling dispatch writing guest memory a word at a time over PCIe, and the game's next compute
    // work waited behind them (with none, the time from the GPU's end to the next frame fell from
    // 11.5 to 7.8 ms). Now they are tiled in video memory, then copied to guest memory together.
    // A picture over the memory of one before it starts a new batch, so they land in the same
    // order. RYGT_NATIVE_WRITEBACK_BATCH=0: each in turn, as before.
    static const bool batched = [] {
        const char* env = std::getenv("RYGT_NATIVE_WRITEBACK_BATCH");
        return !(env && env[0] == '0');
    }();
    Native::DownloadBatch batch;
    std::vector<std::pair<VAddr, VAddr>> batch_ranges;
    bool use_batch = false;
    if (batched) {
        const u64 step_alignment = std::max<u64>(alignment, 256);
        u64 bytes = 0;
        for (const auto* entry : writebacks) {
            bytes += 2 * Common::AlignUp(u64(entry->info.guest_size), step_alignment);
        }
        use_batch = bytes != 0 && registry->BeginDownloads(batch, bytes, alignment);
    }
    for (auto* entry : writebacks) {
        const VAddr address = entry->info.guest_address;
        const u64 size = entry->info.guest_size;
        if (!direct || !direct->EnsureImported(address, size)) {
            continue;
        }
        if (use_batch && std::ranges::any_of(batch_ranges, [&](const auto& range) {
                return address < range.second && range.first < address + size;
            })) {
            registry->FinishDownloads(*s.list, batch);
            batch_ranges.clear();
        }
        const auto found = direct->FindBuffer(address, size);
        if (found && found->second % alignment != 0) {
            continue;
        }
        // FIX-GT7-041: guest memory is imported in pieces (2 MB there), and a 4 MB picture spans
        // three. It goes to a buffer first, then to each piece.
        std::unique_ptr<Native::Rhi::Buffer> out;
        bool downloaded = false;
        if (found) {
            out = device->WrapBuffer(
                reinterpret_cast<u64>(static_cast<VkBuffer>(found->first->Handle())),
                found->first->SizeBytes());
            downloaded = registry->RecordDownload(*s.list, *entry, *out, found->second,
                                                  use_batch ? &batch : nullptr);
            if (downloaded && use_batch) {
                batch_ranges.emplace_back(address, address + size);
            }
        } else {
            // This path copies back at once: what the batch holds goes first, in the order drawn.
            if (use_batch) {
                registry->FinishDownloads(*s.list, batch);
                batch_ranges.clear();
            }
            out = device->CreateBuffer({.size = size,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferSrc |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                        .name = "native write-back across views"});
            const auto ready = [&] {
                const vk::MemoryBarrier2 barrier{
                    .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .dstAccessMask =
                        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                };
                s.cmd.pipelineBarrier2(vk::DependencyInfo{
                    .memoryBarrierCount = 1,
                    .pMemoryBarriers = &barrier,
                });
            };
            if (out) {
                const vk::Buffer buffer{Handle<VkBuffer_T>(out->NativeHandle())};
                // The tiling leaves padding unwritten: the buffer starts with what guest memory
                // holds there, or the copy back would put uninitialized bytes in it.
                downloaded = direct->ForEachPart(
                    address, size,
                    [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
                        s.cmd.copyBuffer(view.Handle(), buffer,
                                         vk::BufferCopy{.srcOffset = view_offset,
                                                        .dstOffset = offset,
                                                        .size = bytes});
                    });
                if (downloaded) {
                    ready();
                    downloaded = registry->RecordDownload(*s.list, *entry, *out);
                }
                if (downloaded) {
                    ready();
                    downloaded = direct->ForEachPart(
                        address, size,
                        [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
                            s.cmd.copyBuffer(buffer, view.Handle(),
                                             vk::BufferCopy{.srcOffset = offset,
                                                            .dstOffset = view_offset,
                                                            .size = bytes});
                        });
                    ready(); // a later picture over this memory starts from these bytes
                }
            }
        }
        if (downloaded) {
            ++stats.writebacks;
            NoteHistory(address, size, 8, 0); // DIAG-187 (diagnosis)
            // NAT-146: regions over the picture's memory copy it in before their next use.
            if (Region* region = RegionOver(address, size)) {
                std::scoped_lock lock{region_dirty_mutex}; // FIX-GT7-020
                if (downloaded_pages.size() >= 1'000'000) {
                    downloaded_pages.clear(); // only compared again, as before FIX-GT7-020
                }
                for (VAddr page = Common::AlignDown(std::max<VAddr>(address, region->begin), 4096);
                     page < std::min<VAddr>(address + size, region->end); page += 4096) {
                    region->dirty_pages.push_back(page);
                    downloaded_pages.insert(page);
                }
            }
            // NAT-150: so do mirrors.
            for (auto it = FirstMirror(address);
                 it != mirrors.end() && it->second.begin < address + size; ++it) {
                Mirror& mirror = it->second;
                for (VAddr page = Common::AlignDown(std::max<VAddr>(address, mirror.begin), 4096);
                     page < std::min<VAddr>(address + size, mirror.end); page += 4096) {
                    mirror.dirty_pages.push_back(page);
                }
            }
        }
        if (out) {
            s.keep.push_back(std::move(out));
        }
    }
    if (use_batch) {
        registry->FinishDownloads(*s.list, batch);
    }
    writebacks.clear();
    writebacks_due = false;
    s.pipeline = vk::Pipeline{}; // the re-tiling binds its own pipeline
    s.so_known = false; // SO-010
    ForgetBindings(s);            // OPT-014: and its own bindings
    // What reads guest memory next (shaders, the CPU after the GPU) sees the pictures.
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}

void NativeRenderer::PrepareImages(std::span<const Native::ImageEntry* const> bound) {
    AmdGpu::RecorderParts::Scope part{AmdGpu::RecorderParts::Part::Uploads}; // DIAG-GT7-005
    const bool init = device->PendingImageInit();
    // PERF-109: the stale textures the work uses, with the uploads queued.
    const bool bound_stale = std::ranges::any_of(bound, [](const Native::ImageEntry* entry) {
        return entry && entry->from_cpu && entry->needs_upload.load(std::memory_order_relaxed);
    });
    const bool uploads = bound_stale || registry->HasPendingUploads();
    if (!init && !uploads) {
        return;
    }
    Stream& s = stream;
    EndRendering();
    if (init) {
        device->RecordPendingImageInit(*s.list);
    }
    if (uploads) {
        GlobalBarrier(); // NAT-107: uploads read guest memory as earlier GPU work left it
        registry->RecordUploads(
            *s.list, std::span<Native::ImageEntry* const>{
                         const_cast<Native::ImageEntry* const*>(bound.data()), bound.size()});
        s.pipeline = vk::Pipeline{}; // the uploads bind their own pipelines
        s.so_known = false; // SO-010
        ForgetBindings(s);            // OPT-014: and their own bindings
    }
}

void NativeRenderer::PrepareImages(const DrawBindings& b) {
    // PERF-109: the pictures the work samples and stores into.
    boost::container::small_vector<const Native::ImageEntry*, 32> bound;
    bound.insert(bound.end(), b.sampled.begin(), b.sampled.end());
    bound.insert(bound.end(), b.stored.begin(), b.stored.end());
    PrepareImages(std::span<const Native::ImageEntry* const>{bound.data(), bound.size()});
}

void NativeRenderer::ChunkFlushCheck() {
    // Long lists are run and submitted as they grow, so the GPU starts on them (as the old
    // path's SubmitChunkIfNeeded does).
    if (stream.commands >= SpliceAfter) {
        Splice();
        r.scheduler.Flush();
        return;
    }
    static const u32 feed_min = [] {
        const char* on = std::getenv("RYGT_FEED_GPU");
        if (on && on[0] == '0') {
            return 0u;
        }
        const char* env = std::getenv("RYGT_FEED_GPU_MIN");
        return std::max<u32>(env && *env ? u32(std::strtoul(env, nullptr, 10)) : 32u, 1u);
    }();
    if (feed_min == 0 || stream.commands < feed_min || (stream.commands & 15) != 0) {
        return;
    }
    const u64 tick = r.scheduler.CurrentTick();
    if (tick > 1 && r.scheduler.IsFree(tick - 1)) {
        Splice();
        r.scheduler.Flush();
    }
}

void NativeRenderer::TrimChunks() {
    // FIX-GT7-046: free chunks were kept for good. GT7's race loading takes many of them at once,
    // and they stayed free for the rest of the session while a frame uses far fewer. What stayed
    // free over the whole period goes back, a little at a time:
    // the newest are reused first, so the oldest are the ones not needed.
    // RYGT_NATIVE_CHUNK_TRIM=0: as before.
    static const bool trim = [] {
        const char* env = std::getenv("RYGT_NATIVE_CHUNK_TRIM");
        return !(env && env[0] == '0');
    }();
    constexpr u64 PeriodFrames = 150;
    const size_t KeptChunks = 128_MB / ChunkSize();
    const size_t AtMostChunks = 256_MB / ChunkSize();
    if (!trim || frame % PeriodFrames != 0) {
        return;
    }
    std::vector<Chunk> dropped;
    {
        std::scoped_lock lock{chunk_mutex};
        if (free_chunks_low > KeptChunks) {
            const auto count = std::min(free_chunks_low - KeptChunks, AtMostChunks);
            const auto first = free_chunks.begin();
            dropped.assign(std::make_move_iterator(first), std::make_move_iterator(first + count));
            free_chunks.erase(first, first + count);
        }
        free_chunks_low = free_chunks.size();
    }
}

NativeRenderer::Chunk NativeRenderer::TakeChunk(u64 size) {
    auto* semaphore = r.scheduler.GetWorkSemaphore();
    std::scoped_lock lock{chunk_mutex};
    if (free_chunks.empty() && !retiring_chunks.empty()) {
        semaphore->Refresh();
    }
    const u64 known = semaphore->KnownGpuTick();
    static const bool no_reuse = [] {
        const char* env = std::getenv("RYGT_NATIVE_NO_REUSE");
        return env && env[0] == '2'; // =2: chunks too
    }();
    static const u64 margin = [] { // =3: 16 GPU ticks later than they could be
        const char* env = std::getenv("RYGT_NATIVE_NO_REUSE");
        return env && env[0] == '3' ? u64{16} : u64{0};
    }();
    while (!no_reuse && !retiring_chunks.empty() &&
           retiring_chunks.front().retire_tick + margin <= known) {
        if (retiring_chunks.front().buffer->Size() == ChunkSize()) {
            free_chunks.push_back(std::move(retiring_chunks.front()));
        } else {
            device->DestroyCompletedBuffer(std::move(retiring_chunks.front().buffer));
        }
        retiring_chunks.pop_front();
    }
    if (size <= ChunkSize() && !free_chunks.empty()) {
        Chunk chunk = std::move(free_chunks.back());
        free_chunks.pop_back();
        free_chunks_low = std::min(free_chunks_low, free_chunks.size());
        return chunk;
    }
    if (size <= ChunkSize()) {
        free_chunks_low = 0;
    }
    const auto create = [&] {
        Chunk made;
        made.buffer = device->CreateBuffer({
            .size = std::max(size, ChunkSize()),
            .usage = Native::Rhi::BufferUsageStorage | Native::Rhi::BufferUsageUniform |
                     Native::Rhi::BufferUsageIndex | Native::Rhi::BufferUsageVertex |
                     Native::Rhi::BufferUsageIndirect | Native::Rhi::BufferUsageTransferSrc,
            .memory = Native::Rhi::MemoryKind::Upload,
            .name = "native upload",
        });
        return made;
    };
    Chunk chunk = create();
    if (!chunk.buffer || !chunk.buffer->Mapped()) {
        LOG_ERROR(Render_Vulkan,
                  "FIX-GT7-048: no upload memory for {} bytes ({} free, {} retiring chunks): "
                  "reclaiming completed allocations",
                  size, free_chunks.size(), retiring_chunks.size());
        device->DestroyCompletedBuffer(std::move(chunk.buffer));
        free_chunks_low = 0;
        chunk = RecoverUploadMemory(
            size, r.scheduler.CurrentTick(), !no_reuse, free_chunks, retiring_chunks,
            [&] {
                semaphore->Refresh();
                return semaphore->KnownGpuTick();
            },
            [&](u64 tick) { r.scheduler.Wait(tick); },
            [&](std::unique_ptr<Native::Rhi::Buffer> buffer) {
                device->DestroyCompletedBuffer(std::move(buffer));
            },
            [&] {
                device->Collect();
                return create();
            });
    }
    ASSERT_MSG(chunk.buffer && chunk.buffer->Mapped(), "NAT-100: no upload memory ({} bytes)",
               size);
    return chunk;
}

std::tuple<vk::Buffer, u64, u8*> NativeRenderer::Allocate(u64 size, u64 align) {
    Stream& s = stream;
    align = std::max<u64>(align, 16);
    if (s.chunks.empty() ||
        Common::AlignUp(s.chunk_used, align) + size > s.chunks.back().buffer->Size()) {
        s.chunks.push_back(TakeChunk(size));
        s.chunk_used = 0;
    }
    const auto& buffer = *s.chunks.back().buffer;
    const u64 offset = Common::AlignUp(s.chunk_used, align);
    s.chunk_used = offset + size;
    return {vk::Buffer{Handle<VkBuffer_T>(buffer.NativeHandle())}, offset, buffer.Mapped() + offset};
}

vk::DescriptorPool NativeRenderer::TakeDescriptorPool() {
    const vk::Device vk_device = r.instance.GetDevice();
    // DIAG-171 (diagnosis, RYGT_NATIVE_NO_REUSE=1, =2 upload chunks too): descriptor pools
    // are never used again (they pile up), to see whether reuse races the GPU.
    static const bool no_reuse = [] {
        const char* env = std::getenv("RYGT_NATIVE_NO_REUSE");
        return env && (env[0] == '1' || env[0] == '2');
    }();
    if (!no_reuse) {
        std::scoped_lock lock{chunk_mutex};
        const u64 known = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
        while (!retiring_pools.empty() && retiring_pools.front().first <= known) {
            std::ignore = vk_device.resetDescriptorPool(retiring_pools.front().second);
            free_pools.push_back(retiring_pools.front().second);
            retiring_pools.pop_front();
        }
        if (!free_pools.empty()) {
            const auto pool = free_pools.back();
            free_pools.pop_back();
            return pool;
        }
    }
    static constexpr std::array<vk::DescriptorPoolSize, 4> sizes{{
        {vk::DescriptorType::eStorageBuffer, 8192},
        {vk::DescriptorType::eSampledImage, 8192},
        {vk::DescriptorType::eStorageImage, 1024},
        {vk::DescriptorType::eSampler, 4096},
    }};
    const auto [result, pool] = vk_device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
        .maxSets = 512,
        .poolSizeCount = static_cast<u32>(sizes.size()),
        .pPoolSizes = sizes.data(),
    });
    ASSERT_MSG(result == vk::Result::eSuccess, "NAT-100: no descriptor pool");
    return pool;
}

NativeRenderer::Region* NativeRenderer::RegionOver(VAddr address, u64 size) {
    auto it = regions.upper_bound(address);
    if (it != regions.begin()) {
        Region& region = std::prev(it)->second;
        if (address < region.end) {
            return &region;
        }
    }
    if (it != regions.end() && it->second.begin < address + size) {
        return &it->second;
    }
    return nullptr;
}

NativeRenderer::Region* NativeRenderer::MakeRegion(VAddr address, u64 size, bool write) {
    auto* direct = r.buffer_cache.GetDirectMemory();
    // The new region takes in every region it overlaps (their GPU-written contents kept).
    VAddr begin = Common::AlignDown(address, 4096);
    VAddr end = Common::AlignUp(address + size, 4096);
    std::vector<Region> merged;
    bool refused;
    {
        std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
        for (const auto& [base, region] : regions) {
            if (region.begin < end && region.end > begin) {
                begin = std::min(begin, region.begin);
                end = std::max(end, region.end);
            }
        }
        // FIX-GT7-039: a range that can't be a region leaves the regions it reaches alone. In GT7's
        // rain a shader reads 2.17 GB up to the end of a pool, once a frame: that read used to erase
        // the ~26 regions under it, and with them what the GPU wrote there. They were made again
        // from guest memory each frame, 143 MB of copies a frame. A write of a size a region can
        // have still goes to guest memory in place, or to a smaller region: regions left over it
        // would keep their old bytes, so those are taken out.
        refused = !direct || end - begin > 1_GB || !r.IsMappedStart(begin) ||
                  !r.memory->IsValidGpuMapping(begin, end - begin);
        if (refused && !(write && size <= 1_GB)) {
            return nullptr;
        }
        for (auto it = regions.begin(); it != regions.end();) {
            if (it->second.begin < end && it->second.end > begin) {
                merged.push_back(std::move(it->second));
                it = regions.erase(it);
            } else {
                ++it;
            }
        }
    }
    ++buffer_generation; // OPT-003: regions are taken in or made
    buffer_log.Note(begin, end); // OPT-010
    if (refused) {
        return nullptr;
    }
    // GPU-only guest allocations need not have CPU-importable backing. They still need
    // storage for shader writes (including the terrain's persistent shadow map).
    const bool imported = direct->EnsureImported(begin, end - begin);
    Region region;
    region.begin = begin;
    region.end = end;
    region.buffer = device->CreateBuffer({.size = end - begin,
                                          .usage = Native::Rhi::BufferUsageStorage |
                                                   Native::Rhi::BufferUsageVertex |
                                                   Native::Rhi::BufferUsageIndex |
                                                   Native::Rhi::BufferUsageIndirect |
                                                   Native::Rhi::BufferUsageTransferSrc |
                                                   Native::Rhi::BufferUsageTransferDst,
                                          .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                          .name = "native region"});
    if (!region.buffer) {
        return nullptr;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(region.buffer->NativeHandle())};
    // Guest memory first, then what the GPU wrote into the regions taken in.
    if (imported) {
        direct->ForEachPart(begin, end - begin,
                        [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
                            s.cmd.copyBuffer(view.Handle(), dst,
                                             vk::BufferCopy{.srcOffset = view_offset,
                                                            .dstOffset = offset,
                                                            .size = bytes});
                        });
    } else {
        // Preserve readable CPU contents and initialize holes consistently with sparse guest
        // reads. Bound the staging allocation rather than snapshotting a whole large region,
        // and to a chunk: a larger allocation has a buffer of its own, not used again (FIX-GT7-051).
        for (u64 offset = 0; offset < end - begin;) {
            const u64 bytes = std::min<u64>({4_MB, ChunkSize(), end - begin - offset});
            const auto [upload, upload_offset, data] = Allocate(bytes, 4);
            r.memory->CopySparseMemory(begin + offset, data, bytes);
            s.cmd.copyBuffer(upload, dst, vk::BufferCopy{.srcOffset = upload_offset,
                                                        .dstOffset = offset,
                                                        .size = bytes});
            offset += bytes;
        }
    }
    for (auto& old : merged) {
        s.cmd.copyBuffer(vk::Buffer{Handle<VkBuffer_T>(old.buffer->NativeHandle())}, dst,
                         vk::BufferCopy{.srcOffset = 0,
                                        .dstOffset = old.begin - begin,
                                        .size = old.end - old.begin});
        s.keep.push_back(std::move(old.buffer)); // freed once the list ran
        // FIX-173: CPU writes to it not copied in yet are copied in at the new one's next use.
        region.dirty_pages.insert(region.dirty_pages.end(), old.dirty_pages.begin(),
                                  old.dirty_pages.end());
        region.cpu_pages.merge(old.cpu_pages); // FIX-194
        region.gpu_pages.merge(old.gpu_pages); // FIX-GT7-033
    }
    s.buffers_written = true;
    // CPU writes to it from now on are copied in before its next use.
    WatchPages(begin, end - begin);
    DropMirrors(begin, end - begin); // NAT-150: reads there use the region from now on
    ++stats.regions_made;
    regions_taken_in += merged.empty() ? 0 : 1; // FIX-GT7-090
    LOG_INFO(Render_Vulkan, "NAT-114: region {:#x}+{:#x} on the GPU ({} taken in)", begin,
             end - begin, merged.size());
    NoteHistory(begin, end - begin, 10, merged.size()); // DIAG-187 (diagnosis)
    std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
    auto [it, inserted] = regions.emplace(begin, std::move(region));
    return &it->second;
}

void NativeRenderer::SortRegionWrites() {
    if (!has_pending_region_writes.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    std::vector<std::pair<VAddr, VAddr>> writes;
    {
        std::scoped_lock lock{region_dirty_mutex};
        writes.swap(pending_region_writes);
    }
    const bool by_address = ReuseOn();
    for (const auto& [begin, end] : writes) {
        // PERF-GT7-006: from the region over the write's start, in address order (regions never
        // overlap), as PERF-GT7-003; each write walked all ~700 of GT7's regions (A still does).
        auto first = regions.begin();
        if (by_address) {
            first = regions.upper_bound(begin);
            if (first != regions.begin() && std::prev(first)->second.end > begin) {
                --first;
            }
        }
        for (auto it = first; it != regions.end() && (!by_address || it->second.begin < end);
             ++it) {
            Region& region = it->second;
            if (region.begin < end && region.end > begin) {
                for (VAddr page = std::max(begin, region.begin); page < std::min(end, region.end);
                     page += 4096) {
                    region.dirty_pages.push_back(page);
                }
            }
        }
        // FIX-173: copies of pages no region holds are not needed.
        if (end - begin <= 64_KB) {
            std::scoped_lock lock{region_dirty_mutex};
            for (VAddr page = begin; page < end; page += 4096) {
                if (!RegionOver(page, 4096)) {
                    page_snapshots.erase(page);
                }
            }
        }
        // NAT-150: mirrors copy the pages in before their next use, unless the CPU wrote that
        // granule in another recent frame too: then the mirror goes and the granule is read in
        // place for a while.
        for (auto it = FirstMirror(begin); it != mirrors.end() && it->second.begin < end;) {
            Mirror& mirror = it->second;
            const VAddr from = std::max(begin, mirror.begin);
            const VAddr to = std::min(end, mirror.end);
            bool dynamic = false;
            for (u64 granule = from >> 16; granule <= (to - 1) >> 16; ++granule) {
                const auto [written, first] = mirror_writes.try_emplace(granule, frame);
                if (!first && written->second != frame && frame - written->second < 600) {
                    dynamic = true;
                    dynamic_granules[granule] = frame;
                }
                written->second = frame;
            }
            // FIX-184 (diagnosis, RYGT_NATIVE_MIRROR_NODROP=1): CPU-written mirrors stay (their
            // pages are copied in at the next use as usual).
            static const bool no_drop = [] {
                const char* env = std::getenv("RYGT_NATIVE_MIRROR_NODROP");
                return env && env[0] == '1';
            }();
            if (dynamic && no_drop) {
                dynamic = false;
            }
            if (dynamic) {
                mirror_bytes -= mirror.end - mirror.begin;
                ++stats.mirrors_dropped;
                ++buffer_generation; // OPT-003
                buffer_log.Note(mirror.begin, mirror.end); // OPT-010
                std::unique_lock map_lock{buffer_map_mutex};
                it = mirrors.erase(it); // freed once the GPU is done with it
                continue;
            }
            for (VAddr page = from; page < to; page += 4096) {
                mirror.dirty_pages.push_back(page);
            }
            ++it;
        }
    }
    if (mirror_writes.size() > 1'000'000) {
        mirror_writes.clear();
    }
}

std::map<VAddr, NativeRenderer::Mirror>::iterator NativeRenderer::FirstMirror(VAddr begin) {
    auto it = mirrors.upper_bound(begin);
    if (it != mirrors.begin() && std::prev(it)->second.end > begin) {
        --it;
    }
    return it;
}

NativeRenderer::Mirror* NativeRenderer::MirrorHolding(VAddr address, u64 size) {
    auto it = mirrors.upper_bound(address);
    if (it == mirrors.begin()) {
        return nullptr;
    }
    Mirror& mirror = std::prev(it)->second;
    return address >= mirror.begin && address + size <= mirror.end ? &mirror : nullptr;
}

bool NativeRenderer::IsDynamic(VAddr begin, VAddr end) {
    constexpr u64 DynamicFrames = 600; // then the granule may be mirrored again
    if (dynamic_granules.empty()) {
        return false;
    }
    for (u64 granule = begin >> 16; granule <= (end - 1) >> 16; ++granule) {
        const auto it = dynamic_granules.find(granule);
        if (it == dynamic_granules.end()) {
            continue;
        }
        if (frame - it->second < DynamicFrames) {
            return true;
        }
        dynamic_granules.erase(it);
    }
    return false;
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::MirrorBuffer(VAddr address, u64 size) {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRRORS");
        return !(env && env[0] == '0');
    }();
    if (!enabled || size == 0 || size > 64_MB) {
        return std::nullopt;
    }
    SortRegionWrites();
    Mirror* mirror = MirrorHolding(address, size);
    if (mirror && !mirror->dirty_pages.empty() && !stream.list) {
        Begin(); // a new list makes the pending mirrors first (they can take this one in)
        mirror = MirrorHolding(address, size);
    }
    if (mirror) {
        RefreshMirror(*mirror);
        mirror->used_frame = frame;
        ++stats.mirror_binds;
        memo_kind = MemoKind::Mirror; // OPT-003
        memo_mirror = mirror;
        return std::pair{vk::Buffer{Handle<VkBuffer_T>(mirror->buffer->NativeHandle())},
                         u64{address - mirror->begin}};
    }
    // Not mirrored: a range bound in an earlier frame too gets a mirror at the start of the next
    // list (read in place until then).
    const VAddr begin = Common::AlignDown(address, 4096);
    // FIX-180: mirrors reach 256 KB past the range they are made for (RYGT_NATIVE_MIRROR_SLACK=
    // <KB> sets it, 0 as before). With mirrors cut at the range's end, Tokyo's world geometry
    // went missing for single frames in bursts, the sky colour showing through (Lance's
    // "vertex sky explosions"; 6-19 such frames a 450 s race, none with mirrors off). Mirror
    // contents matched guest memory on both the CPU (FIX-176) and the GPU (FIX-179) side and no
    // sampled draw read past its V#s (FIX-181), so the exact cause is still open; with the slack
    // the race had none (1 smaller jump in 13,000 frames), at the same speed (47-51 fps), with
    // about 105 bigger mirrors instead of 800.
    static const u64 slack = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRROR_SLACK");
        return (env && *env ? std::strtoull(env, nullptr, 10) : u64{256}) * 1024;
    }();
    const u64 reach = slack != 0 ? r.memory->ClampRangeSize(address, size + slack) : size;
    const VAddr end = Common::AlignUp(address + std::max(size, reach), 4096);
    if (mirror_candidates.size() > 65536) {
        mirror_candidates.clear(); // per-frame data at ever-new addresses
    }
    const auto [it, inserted] = mirror_candidates.try_emplace(begin, MirrorCandidate{end, frame});
    MirrorCandidate& candidate = it->second;
    if (inserted) {
        return std::nullopt;
    }
    candidate.end = std::max(candidate.end, end);
    if (candidate.frame == frame) {
        return std::nullopt;
    }
    candidate.frame = frame; // looked at again next frame at the earliest
    if (IsDynamic(begin, candidate.end)) {
        return std::nullopt;
    }
    pending_mirrors.emplace_back(begin, candidate.end);
    mirror_candidates.erase(it);
    return std::nullopt;
}

void NativeRenderer::RefreshMirror(Mirror& mirror) {
    if (mirror.dirty_pages.empty()) {
        return;
    }
    std::vector<VAddr> pages;
    pages.swap(mirror.dirty_pages);
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(mirror.buffer->NativeHandle())};
    std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    for (const VAddr page : pages) {
        if (page < mirror.begin || page >= mirror.end) {
            continue;
        }
        direct->ForEachPart(page, std::min<u64>(4096, mirror.end - page),
                            [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                u64 bytes) {
                                s.cmd.copyBuffer(view.Handle(), dst,
                                                 vk::BufferCopy{.srcOffset = view_offset,
                                                                .dstOffset = page - mirror.begin +
                                                                             offset,
                                                                .size = bytes});
                            });
        WatchPages(page, std::min<u64>(4096, mirror.end - page));
        const u64 g = (page - mirror.begin) / MirrorGranule; // FIX-176
        if (g < mirror.hashes.size()) {
            mirror.hashes[g] = 0;
        }
    }
    s.buffers_written = true;
    stats.mirror_refresh_pages += pages.size();
}

void NativeRenderer::MakePendingMirrors() {
    if (pending_mirrors.empty() || !stream.list) {
        return;
    }
    std::vector<std::pair<VAddr, VAddr>> pending;
    pending.swap(pending_mirrors);
    ++buffer_generation; // OPT-003: mirrors are made, taken in or evicted
    std::ranges::sort(pending);
    std::vector<std::pair<VAddr, VAddr>> ranges; // overlapping or touching ranges joined
    for (const auto& [begin, end] : pending) {
        if (!ranges.empty() && begin <= ranges.back().second) {
            ranges.back().second = std::max(ranges.back().second, end);
        } else {
            ranges.emplace_back(begin, end);
        }
    }
    static const u64 budget = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRROR_MB");
        return (env ? std::strtoull(env, nullptr, 10) : 1024) * 1_MB;
    }();
    auto* direct = r.buffer_cache.GetDirectMemory();
    Stream& s = stream;
    bool copied = false;
    bool evicted = false;
    for (auto [begin, end] : ranges) {
        // FIX-185 (diagnosis, RYGT_NATIVE_MIRROR_REGION_GAP=<KB>): no mirror within that distance
        // of a region.
        static const u64 region_gap = [] {
            const char* env = std::getenv("RYGT_NATIVE_MIRROR_REGION_GAP");
            return env && *env ? std::strtoull(env, nullptr, 10) * 1024 : u64{0};
        }();
        if (!direct || RegionOver(begin, end - begin) || IsDynamic(begin, end) ||
            (region_gap != 0 && RegionOver(begin > region_gap ? begin - region_gap : 0,
                                           end - begin + 2 * region_gap))) {
            continue; // shaders write there (the region holds it), or the CPU does
        }
        // The new mirror takes in the mirrors it overlaps.
        u64 taken = 0;
        for (auto it = FirstMirror(begin); it != mirrors.end() && it->second.begin < end; ++it) {
            begin = std::min(begin, it->second.begin);
            end = std::max(end, it->second.end);
            taken += it->second.end - it->second.begin;
        }
        if (end - begin > 256_MB) {
            continue;
        }
        if (mirror_bytes - taken + (end - begin) > budget && !evicted) {
            // Mirrors unused for two seconds go first.
            evicted = true;
            std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
            for (auto it = mirrors.begin(); it != mirrors.end();) {
                if (it->second.used_frame + 120 < frame &&
                    !(it->second.begin < end && it->second.end > begin)) {
                    mirror_bytes -= it->second.end - it->second.begin;
                    ++stats.mirrors_evicted;
                    buffer_log.Note(it->second.begin, it->second.end);
                    it = mirrors.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (mirror_bytes - taken + (end - begin) > budget) {
            ++stats.mirror_over_budget;
            continue;
        }
        if (!direct->EnsureImported(begin, end - begin)) {
            continue;
        }
        auto buffer = device->CreateBuffer({.size = end - begin,
                                            .usage = Native::Rhi::BufferUsageStorage |
                                                     Native::Rhi::BufferUsageVertex |
                                                     Native::Rhi::BufferUsageIndex |
                                                     Native::Rhi::BufferUsageTransferSrc |
                                                     Native::Rhi::BufferUsageTransferDst,
                                            .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                            .name = "native mirror"});
        if (!buffer) {
            continue;
        }
        {
            std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
            for (auto it = FirstMirror(begin); it != mirrors.end() && it->second.begin < end;) {
                mirror_bytes -= it->second.end - it->second.begin;
                it = mirrors.erase(it); // freed once the GPU is done with it
            }
        }
        buffer_log.Note(begin, end); // OPT-010
        const vk::Buffer dst{Handle<VkBuffer_T>(buffer->NativeHandle())};
        direct->ForEachPart(begin, end - begin,
                            [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                u64 bytes) {
                                s.cmd.copyBuffer(view.Handle(), dst,
                                                 vk::BufferCopy{.srcOffset = view_offset,
                                                                .dstOffset = offset,
                                                                .size = bytes});
                            });
        // CPU writes to it from now on are copied in before its next use.
        WatchPages(begin, end - begin);
        NoteHistory(begin, end - begin, 12, 0); // DIAG-187 (diagnosis)
        mirror_bytes += end - begin;
        ++stats.mirrors_made;
        stats.mirror_made_bytes += end - begin;
        std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
        mirrors.emplace(begin, Mirror{.begin = begin,
                                      .end = end,
                                      .buffer = std::move(buffer),
                                      .dirty_pages = {},
                                      .used_frame = frame});
        copied = true;
    }
    if (copied) {
        s.buffers_written = true; // the next draw waits for the copies
    }
}

void NativeRenderer::DropMirrors(VAddr address, u64 size) {
    for (auto it = FirstMirror(address); it != mirrors.end() && it->second.begin < address + size;) {
        mirror_bytes -= it->second.end - it->second.begin;
        ++buffer_generation; // OPT-003
        buffer_log.Note(it->second.begin, it->second.end); // OPT-010
        std::unique_lock map_lock{buffer_map_mutex};
        it = mirrors.erase(it); // freed once the GPU is done with it
    }
}

void NativeRenderer::RefreshRegion(Region& region) {
    SortRegionWrites();
    std::vector<VAddr> pages;
    pages.swap(region.dirty_pages);
    if (pages.empty()) {
        return;
    }
    const auto refresh_start = std::chrono::steady_clock::now();
    struct Timed {
        std::chrono::steady_clock::time_point start;
        u64& total;
        ~Timed() {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    } timed{refresh_start, stats.region_refresh_ns};
    stats.region_refresh_pages += pages.size();
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(region.buffer->NativeHandle())};
    std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    const auto copy_part = [&](VAddr from, u64 size) {
        return direct->ForEachPart(from, size,
                                   [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                       u64 bytes) {
                                       s.cmd.copyBuffer(
                                           view.Handle(), dst,
                                           vk::BufferCopy{.srcOffset = view_offset,
                                                          .dstOffset = from - region.begin + offset,
                                                          .size = bytes});
                                   });
    };
    const auto copy_in = [&](VAddr from, u64 size) {
        if (!copy_part(from, size) && size > 4096) {
            // PERF-104: a run of pages with one not imported stops at it; the pages after it come
            // in one by one, as they did before runs were joined.
            for (VAddr page = from; page < from + size; page += 4096) {
                copy_part(page, std::min<u64>(4096, from + size - page));
            }
        }
    };
    std::erase_if(pages, [&](VAddr page) { return page < region.begin || page >= region.end; });
    // FIX-173: each page as it was before the CPU wrote it (taken at the write fault).
    // PERF-104: all under one lock, and each run of pages watched again at once (one protect).
    std::vector<std::unique_ptr<std::array<u8, 4096>>> befores(pages.size());
    std::vector<bool> opened_ahead(pages.size()); // FIX-GT7-006
    {
        std::scoped_lock lock{region_dirty_mutex};
        for (size_t k = 0; k < pages.size(); ++k) {
            if (const auto it = page_snapshots.find(pages[k]); it != page_snapshots.end()) {
                befores[k] = std::move(it->second);
                page_snapshots.erase(it);
            }
            opened_ahead[k] = ahead_pages.erase(pages[k]) != 0;
            // FIX-GT7-020: a page a picture's download wrote is copied in whole. The compare
            // below reads guest memory now, before the GPU runs the download (recorded in this
            // list or one not run yet): with a copy from before, it found nothing changed there
            // and the region kept stale bytes where the picture went. PERF-GT7-030 took such
            // copies of thousands of pages a second in GT7 races, where pictures and CPU data
            // share the per-frame memory: long black triangles on some frames.
            if (downloaded_pages.erase(pages[k]) != 0) {
                if (befores[k]) {
                    ++downloaded_with_copy;
                    downloaded_ahead += opened_ahead[k] ? 1 : 0;
                }
                befores[k].reset();
                opened_ahead[k] = false;
            }
        }
    }
    // NAT-145: only the pages copied in are watched again (the region was watched whole
    // when made; re-watching all of it at each refresh cost ~9 us a draw with big regions).
    // FIX-173: watched before guest memory is compared, so a CPU write from now on faults
    // (and takes a new copy) instead of slipping in between the compare and the watch.
    static const bool batch = [] {
        const char* env = std::getenv("RYGT_NATIVE_REFRESH_BATCH"); // PERF-104 off: =0
        return !(env && env[0] == '0');
    }();
    for (size_t i = 0; i < pages.size();) {
        size_t j = i + 1;
        while (batch && j < pages.size() && pages[j] == pages[j - 1] + 4096) {
            ++j;
        }
        WatchPages(pages[i], std::min<VAddr>(pages[j - 1] + 4096, region.end) - pages[i]);
        i = j;
    }
    // PERF-104: pages copied in whole go in runs (one copy for consecutive pages).
    VAddr whole_begin = 0;
    VAddr whole_end = 0;
    const auto flush_whole = [&] {
        if (whole_end > whole_begin) {
            copy_in(whole_begin, whole_end - whole_begin);
        }
        whole_begin = whole_end = 0;
    };
    for (size_t k = 0; k < pages.size(); ++k) {
        const VAddr page = pages[k];
        const u64 page_bytes = std::min<u64>(4096, region.end - page);
        std::unique_ptr<std::array<u8, 4096>> before = std::move(befores[k]);
        // FIX-194: the compare only finds what the CPU wrote where guest memory held what the
        // region holds. A page the CPU didn't write lately may hold what the GPU wrote since
        // (in the region only): GT7 reuses its per-frame memory for compute output one frame
        // and UI vertices a few frames later, and vertices equal to the stale bytes guest memory
        // still had there (zeros, white) were never copied in, so the title's picture, its
        // blurred copy and other widgets were drawn from the compute output: missing or black
        // in one frame of four or five. Such a page is copied in whole.
        static const bool region_writeback = RegionWritebackOn();
        // (diagnosis) RYGT_NATIVE_REGION_WHOLE=1: every page the CPU wrote is copied in whole.
        static const bool always_whole = [] {
            const char* env = std::getenv("RYGT_NATIVE_REGION_WHOLE");
            return env && env[0] == '1';
        }();
        const auto known = region.cpu_pages.find(page);
        const bool recent = known != region.cpu_pages.end() &&
                            frame - known->second <= RegionRecentFrames;
        // FIX-GT7-006: a page PERF-GT7-030 opened with a fault next to it may not have been
        // written: copied whole, guest memory would go over what the GPU wrote there (in the
        // region only, as no write-back follows a page the CPU didn't write lately). FaultAhead
        // opens only pages the CPU wrote lately; one that stopped being so meanwhile is still
        // compared, and without its copy from before it copies nothing in.
        if (opened_ahead[k] && !recent && region_writeback && !always_whole && !before) {
            ++stats.region_partial_pages;
            continue;
        }
        // FIX-GT7-033: a page the CPU wrote lately is compared only if the GPU wrote it since its
        // last copy. Otherwise the compare only risks the page: a piece the region got wrong stays
        // wrong for as long as the CPU writes the same value. GT7 builds its constants in a heap
        // of such pages: the flag of its sky table's blit stayed 0 in the region at some addresses
        // while guest memory held 1, the table lost 3 of its 4 rows whenever a frame's constants
        // landed there, and the sky lost its clouds and the scene its light for that frame.
        // RYGT_NATIVE_REGION_COMPARE=1: as before.
        static const bool compare_all = [] {
            const char* env = std::getenv("RYGT_NATIVE_REGION_COMPARE");
            return env && env[0] == '1';
        }();
        const bool gpu_wrote = region.gpu_pages.erase(page) != 0;
        const bool compare =
            !always_whole &&
            (!region_writeback || (recent ? gpu_wrote || compare_all : opened_ahead[k]));
        // FIX-GT7-018: a page opened ahead counts as written lately only once the CPU changed it
        // (below). Counted at each refresh, every page of a chunk with one write stayed "written
        // lately" for good: opened ahead again and again, and its GPU writes written back.
        if (!opened_ahead[k]) {
            region.cpu_pages[page] = frame;
        }
        std::array<u8, 4096> now;
        if (!compare || !before ||
            !Core::RytyMemory::GuardedCopy(now.data(), reinterpret_cast<const void*>(page),
                                           page_bytes)) {
            if (whole_end != page || !batch) {
                flush_whole();
                whole_begin = page;
            }
            whole_end = page + page_bytes;
            ++stats.region_whole_pages;
            region.cpu_pages[page] = frame; // FIX-GT7-018: copied in whole, as written
            continue;
        }
        // The runs of 16-byte pieces the CPU changed (gaps under 64 bytes joined).
        constexpr u64 Piece = 16;
        u64 run_begin = 0, run_end = 0, changed = 0;
        bool in_run = false;
        for (u64 at = 0; at < page_bytes; at += Piece) {
            const u64 n = std::min(Piece, page_bytes - at);
            if (std::memcmp(now.data() + at, before->data() + at, n) == 0) {
                continue;
            }
            if (in_run && at - run_end < 64) {
                run_end = at + n;
                continue;
            }
            if (in_run) {
                copy_in(page + run_begin, run_end - run_begin);
                changed += run_end - run_begin;
            }
            in_run = true;
            run_begin = at;
            run_end = at + n;
        }
        if (in_run) {
            copy_in(page + run_begin, run_end - run_begin);
            changed += run_end - run_begin;
        }
        if (changed != 0) {
            region.cpu_pages[page] = frame; // FIX-GT7-018
        }
        ++stats.region_partial_pages;
        stats.region_partial_bytes += changed;
    }
    flush_whole();
    {
        // FIX-GT7-006: FaultAhead's view of cpu_pages (each page refreshed here was set above).
        std::scoped_lock lock{region_dirty_mutex};
        if (cpu_recent_pages.size() >= 1'000'000) {
            cpu_recent_pages.clear(); // only fewer pages opened ahead until they come again
        }
        for (const VAddr page : pages) {
            if (const auto it = region.cpu_pages.find(page); it != region.cpu_pages.end()) {
                cpu_recent_pages[page] = it->second;
            }
        }
    }
    s.buffers_written = true;
    ++stats.region_refreshes;
    if (stats.region_partial_pages != 0 && Sample(logged_partial)) {
        LOG_INFO(Render_Vulkan, "FIX-173: region pages copied in part: {} ({:.1f} KB of {:.1f} KB)",
                 stats.region_partial_pages, stats.region_partial_bytes / 1024.0,
                 stats.region_partial_pages * 4.0);
    }
}

bool NativeRenderer::OneOffDiagOn() {
    static const bool on = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_ONE_OFF_REDRAW");
        return (RyDiag::Enabled() && !(env && env[0] == '0')) || OneOffDiagAll();
    }();
    return on;
}

bool NativeRenderer::OneOffDiagAll() {
    // RYGT_DIAG_ONE_OFF_REDRAW=1: also the draws that found their pipeline built (they would
    // have waited for it), so any race start counts, not only one whose pipelines are new.
    static const bool all = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_ONE_OFF_REDRAW");
        return env && env[0] == '1';
    }();
    return all;
}

bool NativeRenderer::OneOffLearnOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_ONE_OFF_LEARN");
        return !(env && env[0] == '0');
    }();
    return on;
}

bool NativeRenderer::OneOffTrackOn() {
    return OneOffLearnOn() || OneOffDiagOn();
}

namespace {
// PERF-GT7-023: pictures of a kind redrawn after their wait before it stops waiting (the 128x128
// colour and 256x128 depth pictures of GT7's races had 34 and 137 in five runs, none sampled
// first).
constexpr u32 OneOffSafeCount = 8;
// At one address, 4 (the per-frame targets there come back at the same address run after run).
constexpr u32 OneOffPlaceSafeCount = 4;

bool OneOffKindExempt(u32 safe, u32 harmful) {
    return harmful == 0 && safe >= OneOffSafeCount;
}

bool OneOffPlaceExempt(u32 safe, u32 harmful) {
    return harmful == 0 && safe >= OneOffPlaceSafeCount;
}

// The family of a kind: its format and tiling, whatever its size.
template <typename Kind>
Kind FamilyOf(Kind kind) {
    kind.pitch = kind.height = kind.slices = kind.samples = 0;
    return kind;
}

std::filesystem::path OneOffKindsPath() {
    const auto& serial = Common::Singleton<Common::ElfInfo>::Instance()->GameSerial();
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
           fmt::format("{}.one_off_kinds.txt", serial);
}
} // namespace

size_t NativeRenderer::OneOffKindHash::operator()(const OneOffKind& kind) const noexcept {
    const std::array<u64, 4> words{u64(kind.depth) << 32 | kind.pitch,
                                   u64(kind.height) << 32 | kind.slices,
                                   u64(kind.samples) << 32 | kind.format,
                                   u64(kind.number_format) << 32 | kind.tile};
    return MemoHash(words.data(), words.size(), 0x198);
}

size_t NativeRenderer::OneOffPlaceHash::operator()(const OneOffPlace& place) const noexcept {
    return OneOffKindHash{}(place.kind) ^ (u64(place.address) * 0x9E3779B97F4A7C15ULL);
}

NativeRenderer::OneOffKind NativeRenderer::KindOf(const OneOffHit& hit) {
    return {.depth = hit.depth,
            .pitch = hit.pitch,
            .height = hit.height,
            .slices = hit.slices,
            .samples = hit.samples,
            .format = hit.format,
            .number_format = hit.number_format,
            .tile = hit.tile};
}

bool NativeRenderer::OneOffExempt(const OneOffHit& hit) const {
    if (!OneOffLearnOn()) {
        return false;
    }
    const OneOffKind kind = KindOf(hit);
    if (const auto it = one_off_kinds.find(kind);
        it != one_off_kinds.end() && OneOffKindExempt(it->second.safe, it->second.harmful)) {
        return true;
    }
    if (const auto it = one_off_places.find({hit.address, FamilyOf(kind)});
        it != one_off_places.end() && OneOffKindExempt(it->second.safe, it->second.harmful)) {
        return true;
    }
    const auto it = one_off_places.find({hit.address, kind});
    return it != one_off_places.end() && OneOffPlaceExempt(it->second.safe, it->second.harmful);
}

bool NativeRenderer::OneOffKnownHarmful() const {
    // FIX-GT7-121: a picture of a kind (or at a place) already seen sampled before its redraw
    // (PERF-GT7-023's harmful count) is one the game draws once and keeps: skipped at FIX-GT7-102/
    // 109's deadline, it stays wrong for good. GT7's car shadows (1024x1024 maps drawn once a
    // race) showed as dark rectangles under every car after a cold race start: ~20 of their draws
    // had given up at the 12 ms deadline. Such a draw waits for its pipeline whatever the budget;
    // the compile pack and the pre-translation remove the wait. RYGT_ONE_OFF_HARMFUL_WAIT=0: the
    // deadline for every kind, as before.
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_ONE_OFF_HARMFUL_WAIT");
        return !(env && env[0] == '0');
    }();
    if (!enabled || !OneOffLearnOn()) {
        return false;
    }
    for (const OneOffHit& hit : one_off_hits) {
        const OneOffKind kind = KindOf(hit);
        if (const auto it = one_off_kinds.find(kind);
            it != one_off_kinds.end() && it->second.harmful != 0) {
            return true;
        }
        if (const auto it = one_off_places.find({hit.address, kind});
            it != one_off_places.end() && it->second.harmful != 0) {
            return true;
        }
        if (const auto it = one_off_places.find({hit.address, FamilyOf(kind)});
            it != one_off_places.end() && it->second.harmful != 0) {
            return true;
        }
    }
    return false;
}

void NativeRenderer::LoadOneOffKinds() {
    if (!OneOffLearnOn()) {
        return;
    }
    std::ifstream file{OneOffKindsPath()};
    std::string line;
    u32 exempt = 0;
    u32 places_exempt = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream in{line};
        std::string what;
        VAddr address = 0;
        if (!(in >> what)) {
            continue;
        }
        if (what == "at") { // a place: "at <address> <kind ...>"
            std::string where;
            if (!(in >> where >> what)) {
                continue;
            }
            address = std::strtoull(where.c_str(), nullptr, 16);
        }
        OneOffKind kind{};
        OneOffKindStats seen{};
        if (!(in >> kind.pitch >> kind.height >> kind.slices >> kind.samples >> kind.format >>
              kind.number_format >> kind.tile >> seen.safe >> seen.harmful)) {
            continue;
        }
        kind.depth = what == "depth";
        if (address != 0) {
            one_off_places[{address, kind}] = seen;
            places_exempt += OneOffPlaceExempt(seen.safe, seen.harmful) ? 1 : 0;
        } else {
            one_off_kinds[kind] = seen;
            exempt += OneOffKindExempt(seen.safe, seen.harmful) ? 1 : 0;
        }
    }
    LOG_INFO(Render_Vulkan,
             "PERF-GT7-023: {} kinds of one-off pictures known, {} of them no longer waited for; "
             "{} places, {} of them no longer waited for",
             one_off_kinds.size(), exempt, one_off_places.size(), places_exempt);
}

void NativeRenderer::SaveOneOffKinds(bool wait) {
    if (!OneOffLearnOn() || !one_off_kinds_changed) {
        return;
    }
    one_off_kinds_changed = false;
    std::string text = "# PERF-GT7-023: FIX-198 picture kinds (kind pitch height slices samples "
                       "format number_format tile safe harmful),\n# and places (at address kind "
                       "... safe harmful)\n";
    const auto kind_text = [](const OneOffKind& kind, const OneOffKindStats& seen) {
        return fmt::format("{} {} {} {} {} {} {} {} {} {}\n", kind.depth ? "depth" : "color",
                           kind.pitch, kind.height, kind.slices, kind.samples, kind.format,
                           kind.number_format, kind.tile, seen.safe, seen.harmful);
    };
    for (const auto& [kind, seen] : one_off_kinds) {
        text += kind_text(kind, seen);
    }
    for (const auto& [place, seen] : one_off_places) {
        text += fmt::format("at {:#x} ", place.address) + kind_text(place.kind, seen);
    }
    auto write = [path = OneOffKindsPath(), text = std::move(text)] {
        static std::mutex write_mutex;
        std::scoped_lock lk{write_mutex};
        auto temp = path;
        temp += ".tmp";
        {
            std::ofstream out{temp, std::ios::binary | std::ios::trunc};
            out << text;
            if (!out) {
                return;
            }
        }
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
    };
    if (wait) {
        write();
    } else {
        std::thread(std::move(write)).detach(); // a cold file open costs the recorder ~5 ms
    }
}

void NativeRenderer::WatchOneOffs(const AmdGpu::Regs& regs, bool skipped) {
    if (!OneOffTrackOn()) {
        return;
    }
    u64 ps_hash = 0;
    if (regs.stage_enable.IsStageEnabled(u32(Shader::HwStage::Fragment))) {
        if (const auto* pgm = regs.ProgramForStage(u32(Shader::HwStage::Fragment));
            pgm && pgm->Address<u32*>()) {
            ps_hash = AmdGpu::GetParams(*pgm).hash;
        }
    }
    for (const OneOffHit& hit : one_off_hits) {
        auto [it, first] = one_off_watch.try_emplace(hit.address);
        if (first) {
            if (one_off_watch.size() > 4096) {
                one_off_watch.erase(it);
                continue;
            }
            it->second.hit = hit;
            it->second.frame = frame;
            it->second.ps_hash = ps_hash;
            it->second.skipped = skipped;
        }
        it->second.skipped &= skipped;
        ++it->second.waits;
        if (it->second.frame == frame) {
            ++it->second.draws;
        }
    }
    one_off_hits.clear();
}

void NativeRenderer::CheckOneOffs() {
    if (!OneOffTrackOn() || one_off_watch.empty()) {
        return;
    }
    std::erase_if(one_off_watch, [&](const auto& item) {
        const auto& [address, watch] = item;
        const Native::ImageEntry* entry = registry->FindExisting(
            address, [](const Native::ImageEntry& e) { return !e.from_cpu && e.image != nullptr; });
        const u64 written = entry ? entry->written_frame.load(std::memory_order_relaxed) : 0;
        const u64 sampled = entry ? entry->sampled_frame.load(std::memory_order_relaxed) : 0;
        size_t outcome;
        if (written > watch.frame) {
            const u64 after = written - watch.frame;
            outcome = after <= 1 ? 0 : after <= 4 ? 1 : after <= 16 ? 2 : after <= 64 ? 3 : 4;
        } else if (sampled > watch.frame) {
            outcome = 5;
        } else if (frame - watch.frame > 300) {
            outcome = 6;
        } else {
            return false; // still watched
        }
        const auto& hit = watch.hit;
        if (OneOffLearnOn()) {
            // PERF-GT7-023: what its kind learns. A picture whose draws were all skipped proves
            // nothing when redrawn (nothing waited), only when sampled first.
            const bool was_exempt = OneOffExempt(hit);
            auto& seen = one_off_kinds[KindOf(hit)];
            auto& here = one_off_places[{address, KindOf(hit)}];
            auto& family = one_off_places[{address, FamilyOf(KindOf(hit))}];
            if (outcome <= 4 && !watch.skipped) {
                ++seen.safe;
                ++here.safe;
                ++family.safe;
                one_off_kinds_changed = true;
            } else if (outcome == 5) {
                ++seen.harmful;
                ++here.harmful;
                ++family.harmful;
                if (was_exempt) {
                    LOG_WARNING(Render_Vulkan,
                                "PERF-GT7-023: picture {:#x} ({} {}x{} format {}/{}) sampled "
                                "before its redraw{}: it waits again",
                                address, hit.depth ? "depth" : "color", hit.pitch, hit.height,
                                hit.format, hit.number_format,
                                watch.skipped ? " after a skipped draw" : "");
                }
                one_off_kinds_changed = true;
            }
        }
        if (!OneOffDiagOn()) {
            return true;
        }
        const size_t why = hit.never ? 0 : 1;
        ++one_off_stats.pictures[why][outcome];
        one_off_stats.waits[why][outcome] += watch.waits;
        static constexpr std::array<const char*, 7> Outcomes{
            "redrawn", "redrawn", "redrawn", "redrawn", "redrawn", "sampled first", "neither"};
        LOG_WARNING(Render_Vulkan,
                    "DIAG-GT7-008 picture {:#x} {}{} {}x{} slices {} samples {} format {}/{} tile "
                    "{} ps {:#x} {} first frame {} draws {} waits {}: {} after {} frames",
                    address, hit.depth ? "depth" : "color", hit.depth ? "" : std::to_string(hit.cb),
                    hit.pitch, hit.height, hit.slices, hit.samples, hit.format, hit.number_format,
                    hit.tile, watch.ps_hash, hit.never ? "never" : "gap", watch.frame, watch.draws,
                    watch.waits, Outcomes[outcome],
                    outcome <= 4 ? written - watch.frame
                                 : (outcome == 5 ? sampled - watch.frame : frame - watch.frame));
        return true;
    });
}

std::string NativeRenderer::OneOffText() {
    u64 total = 0;
    for (const auto& row : one_off_stats.pictures) {
        for (const u64 n : row) {
            total += n;
        }
    }
    if (total == 0 && one_off_watch.empty()) {
        return {};
    }
    const auto row = [&](size_t why) {
        const auto& p = one_off_stats.pictures[why];
        const auto& w = one_off_stats.waits[why];
        return fmt::format("redrawn after 1 frame {} ({} waits), 2-4 {} ({}), 5-16 {} ({}), 17-64 {} "
                           "({}), 65-300 {} ({}); sampled in a later frame before any redraw {} "
                           "({}); neither in 300 frames {} ({})",
                           p[0], w[0], p[1], w[1], p[2], w[2], p[3], w[3], p[4], w[4], p[5], w[5],
                           p[6], w[6]);
    };
    std::string text = fmt::format("pictures FIX-198 waited for, never drawn: {}; not drawn for 64 "
                                   "frames: {}; {} still watched",
                                   row(0), row(1), one_off_watch.size());
    one_off_stats = {};
    return text;
}

// FIX-GT7-109: a frame's wait for one-off pipelines in play (0: as at loads).
static u64 PlayBudgetNs() {
    static const u64 ns = [] {
        u64 ms = 12;
        if (const char* env = std::getenv("RYGT_ONE_OFF_PLAY_BUDGET_MS"); env && *env) {
            ms = std::strtoull(env, nullptr, 10);
        }
        return ms * 1000000ull;
    }();
    return ns;
}

bool NativeRenderer::OneOffBudgetLeft() {
    // FIX-GT7-102: on a first GT7 race load (every shader new) one frame lasted 185 s: about
    // 5,000 draws into ~20 new pictures waited for their pipelines (FIX-198), one after the
    // other, as each draw's GPU work had to finish before the command stream went on. DIAG-GT7-008
    // then saw nearly all of those pictures redrawn the very next frame (0 or 1 not redrawn within
    // 300 frames, 3 or 4 sampled before their redraw). A frame now waits that long at most; its
    // other one-off draws are skipped and show as their pipelines arrive, as in keep-going mode.
    // In play the pipelines are usually known and the budget is not reached.
    static const u64 budget_ns = [] {
        u64 ms = 40; // FIX-GT7-109: 250 before, with waits as long as a compile
        if (const char* env = std::getenv("RYGT_ONE_OFF_BUDGET_MS"); env && *env) {
            ms = std::strtoull(env, nullptr, 10);
        }
        return ms * 1000000ull;
    }();
    if (budget_ns == 0) {
        return true;
    }
    if (one_off_budget_frame != frame) {
        one_off_budget_frame = frame;
        one_off_wait_ns_frame = 0;
        // FIX-GT7-109: in play (the game at 15 frames a second or more) a frame's one-off draws
        // wait RYGT_ONE_OFF_PLAY_BUDGET_MS in all (12 by default), never for a whole compile:
        // a new track's race start had frames of 400-940 ms (translations and pipelines of
        // one-off pictures), races 60-120 ms ones. Loads (slow frames) keep the budget above.
        // The game's own frame count, over half a second or more.
        const auto now = std::chrono::steady_clock::now();
        const u64 game_frames = Common::GetGameFrameTotals().frames;
        if (one_off_rate_since == std::chrono::steady_clock::time_point{}) {
            one_off_rate_since = now;
            one_off_rate_frames = game_frames;
        } else if (now - one_off_rate_since >= std::chrono::milliseconds{500}) {
            const double seconds = std::chrono::duration<double>(now - one_off_rate_since).count();
            // 15 frames a second, as FIX-GT7-109's race state: a slower PC races at 20-25.
            one_off_interactive = double(game_frames - one_off_rate_frames) / seconds >= 15.0 ||
                                  g_game_in_play.load(std::memory_order_relaxed);
            one_off_rate_since = now;
            one_off_rate_frames = game_frames;
        }
    }
    return one_off_wait_ns_frame < (one_off_interactive ? PlayBudgetNs() : budget_ns);
}

std::chrono::steady_clock::time_point NativeRenderer::OneOffDeadline() {
    // FIX-GT7-109: every wait ends at the frame's budget (in play or at a load), so no frame
    // waits a whole compile (a race start had frames of 170-940 ms). RYGT_ONE_OFF_BUDGET_MS=0:
    // unbounded at loads, as before FIX-GT7-102.
    static const u64 load_ns = [] {
        u64 ms = 40;
        if (const char* env = std::getenv("RYGT_ONE_OFF_BUDGET_MS"); env && *env) {
            ms = std::strtoull(env, nullptr, 10);
        }
        return ms * 1000000ull;
    }();
    const u64 budget = one_off_interactive ? PlayBudgetNs() : load_ns;
    if (budget == 0) {
        return std::chrono::steady_clock::time_point::max();
    }
    const u64 left = budget > one_off_wait_ns_frame ? budget - one_off_wait_ns_frame : 0;
    return std::chrono::steady_clock::now() + std::chrono::nanoseconds{left};
}

bool NativeRenderer::DrawMustNotSkip(const AmdGpu::Regs& regs) {
    // FIX-198: GPU.compile_mode 2 (keep going) skips a draw while its new pipeline builds, so
    // the game never stops for a compile: what it draws shows a frame or so later, as it's drawn
    // again every frame. A draw into a picture that isn't drawn every frame (one drawn once and
    // kept: a texture the game renders for later frames, a cached shadow map, the first frame of
    // a new picture) would leave it missing for good, so such a draw waits for its pipeline. A
    // picture counts as drawn every frame when one of the last four frames drew into it.
    // RYGT_NATIVE_WAIT_ONE_OFF=0 skips them too.
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_WAIT_ONE_OFF");
        return !(env && env[0] == '0');
    }();
    if (!enabled) {
        one_off_hits.clear();
        return false;
    }
    // DIAG-220 (diagnosis, RYGT_DIAG_ONE_OFF=1): why targets count as one-off: never drawn
    // into, or how many frames since they were last drawn into.
    static const bool diag = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_ONE_OFF");
        return env && env[0] == '1';
    }();
    const auto note = [&](u64 gap) {
        if (!diag) {
            return;
        }
        static std::array<u32, 6> buckets{}; // never, 5-8, 9-16, 17-64, 65-600, more
        static u64 report_frame = 0;
        const size_t i = gap == 0 ? 0 : gap <= 8 ? 1 : gap <= 16 ? 2 : gap <= 64 ? 3 : gap <= 600 ? 4 : 5;
        ++buckets[i];
        if (frame >= report_frame + 300) {
            report_frame = frame;
            LOG_WARNING(Render_Vulkan,
                        "DIAG-220: one-off targets in 300 frames: never drawn {}, last drawn 5-8 "
                        "frames ago {}, 9-16 {}, 17-64 {}, 65-600 {}, more {}",
                        buckets[0], buckets[1], buckets[2], buckets[3], buckets[4], buckets[5]);
            buckets = {};
        }
    };
    one_off_hits.clear(); // DIAG-GT7-008
    OneOffHit described{};
    const auto drawn_lately = [&](VAddr address) {
        const Native::ImageEntry* entry = registry->FindExisting(
            address, [](const Native::ImageEntry& e) { return !e.from_cpu && e.image != nullptr; });
        if (!entry) {
            note(0);
            if (OneOffTrackOn()) {
                described.address = address;
                described.never = true;
                one_off_hits.push_back(described);
            }
            return OneOffExempt(described); // never drawn into; PERF-GT7-023: unless its kind
        }
        const u64 last = entry->written_frame.load(std::memory_order_relaxed);
        const u64 earlier =
            last < frame ? last : entry->written_frame_before.load(std::memory_order_relaxed);
        // PERF-165: a picture drawn into within the last 64 frames is redrawn regularly (GT7
        // redraws some every 9-64 frames: DIAG-220 counted up to 983 such draws in 300 frames,
        // and they waited for new pipelines as one-off pictures); a skipped draw shows at its next
        // redraw. Never-drawn pictures still wait. RYGT_NATIVE_ONE_OFF_FRAMES=4: as before.
        static const u64 window = [] {
            const char* env = std::getenv("RYGT_NATIVE_ONE_OFF_FRAMES");
            return env && *env ? std::strtoull(env, nullptr, 10) : u64{64};
        }();
        const bool lately = earlier != 0 && frame - earlier <= window;
        if (!lately) {
            note(earlier == 0 ? 0 : frame - earlier);
            if (OneOffTrackOn()) {
                described.address = address;
                described.never = earlier == 0;
                one_off_hits.push_back(described);
            }
        }
        return lately || OneOffExempt(described); // PERF-GT7-023
    };
    if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Disable) {
        for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
            const auto& col_buf = regs.color_buffers[cb];
            if (OneOffTrackOn() && col_buf) {
                described = {.depth = false,
                             .cb = cb,
                             .pitch = col_buf.Pitch(),
                             .height = col_buf.Height(),
                             .slices = col_buf.NumSlices(),
                             .samples = col_buf.NumSamples(),
                             .format = u32(col_buf.GetDataFmt()),
                             .number_format = u32(col_buf.GetNumberFmt()),
                             .tile = u32(col_buf.GetTileMode())};
            }
            if (col_buf && regs.color_target_mask.GetMask(cb) != 0 &&
                !drawn_lately(col_buf.Address())) {
                return true;
            }
        }
    }
    if (OneOffTrackOn() && regs.depth_buffer.DepthValid()) {
        const auto& db = regs.depth_buffer;
        described = {.depth = true,
                     .pitch = db.Pitch(),
                     .height = db.Height(),
                     .slices = regs.depth_view.NumSlices(),
                     .samples = db.NumSamples(),
                     .format = u32(db.z_info.format),
                     .tile = u32(db.GetTileMode())};
    }
    if (regs.depth_control.depth_enable && regs.depth_control.depth_write_enable &&
        regs.depth_buffer.DepthValid() && !drawn_lately(regs.depth_buffer.DepthAddress())) {
        return true;
    }
    return false;
}

static const std::vector<std::pair<VAddr, VAddr>>& InplaceRanges();

/// PERF-GT7-052 (RYGT_POLL_FRAME_TEXTURES=0: watched as before): a texture of 256 KB or less in a
/// listed in-place range (GT7's per-frame memory) is checked each frame, not write-watched.
bool NativeRenderer::PolledFrameTexture(VAddr address, u64 size) {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_POLL_FRAME_TEXTURES");
        return !(env && env[0] == '0');
    }();
    if (!enabled || size > 256_KB) {
        return false;
    }
    for (const auto& [begin, end] : InplaceRanges()) {
        if (address >= begin && address + size <= end) {
            return true;
        }
    }
    return false;
}

/// FIX-194: written ranges up to this size have their recently CPU-written pages written back.
constexpr u64 MaxRegionWriteback = 4_MB;

/// FIX-172 (diagnosis, RYGT_NATIVE_INPLACE_RANGES=<hex address>+<hex size>,...): written ranges
/// meeting one of these stay in guest memory too.
static const std::vector<std::pair<VAddr, VAddr>>& InplaceRanges() {
    static const std::vector<std::pair<VAddr, VAddr>> inplace_ranges = [] {
        std::vector<std::pair<VAddr, VAddr>> list;
        // FIX-GT7-029: GT7's per-frame memory ends at 0x208600000 or 0x208800000, past the range
        // the launcher lists. A compute write there makes a small region; the unbounded reads,
        // which all end at that limit, meet it and grow it over most of the per-frame memory
        // (NAT-114): every CPU write then faults, ~100,000 protection calls in 2 s, and the frame
        // rate drops for the rest of the session. With a list (GT7's launcher), the tail stays in
        // guest memory too; RYGT_NATIVE_INPLACE_EXTRA, in the same form, replaces it (empty: none).
        const char* const ranges = std::getenv("RYGT_NATIVE_INPLACE_RANGES");
        const char* extra = std::getenv("RYGT_NATIVE_INPLACE_EXTRA");
        if (!extra && ranges) {
            extra = "208000000+800000";
        }
        for (const char* source : {ranges, extra}) {
            for (const char* at = source; at && *at;) {
                char* end = nullptr;
                const VAddr begin = std::strtoull(at, &end, 16);
                if (end == at || *end != '+') {
                    break;
                }
                at = end + 1;
                const u64 length = std::strtoull(at, &end, 16);
                list.emplace_back(begin, begin + std::max<u64>(length, 1));
                at = *end == ',' ? end + 1 : end;
            }
        }
        return list;
    }();
    return inplace_ranges;
}

// FIX-GT7-034: a shader's write to a listed range that could not be compared stays in the region.
static void NoteListedWriteSkipped(VAddr from, VAddr to) {
    static std::atomic<u32> logged{};
    if (const u32 n = logged++; n < 20) {
        LOG_WARNING(Render_Vulkan, "FIX-GT7-034: shader write {:#x}-{:#x} in a listed range not written back", from, to);
    }
}

// FIX-GT7-034: RYGT_NATIVE_INPLACE_WRITEBACK=0: what the GPU writes in listed ranges under a region
// stays there.
static bool InplaceWritebackOn() {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_INPLACE_WRITEBACK");
        return !(env && env[0] == '0');
    }();
    return enabled;
}

void NativeRenderer::SnapshotBeforeWrite(VAddr address, u64 size) {
    // FIX-203: FIX-194 wrote back every page of a written range the CPU wrote lately, but a
    // shader's storage buffer covers more than it writes: GT7's compute passes bind 875 KB of
    // its per-frame memory, where the CPU also writes descriptor tables. The write-back runs
    // when the GPU gets to it, after the CPU may have written the page again, and put the
    // region's older copy of the CPU's bytes over the new ones (DIAG-176 counted ~1000 such
    // writes a race). The pages are copied before the work, and the write-back compares
    // (native_region_writeback.comp): only the words the work changed go to guest memory.
    // RYGT_NATIVE_PRECISE_WRITEBACK=0: whole pages, as before.
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_PRECISE_WRITEBACK");
        return RegionWritebackOn() && !(env && env[0] == '0');
    }();
    if (!enabled || !writeback_pipeline || size == 0 || size > MaxRegionWriteback) {
        return;
    }
    Region* region = RegionOver(address, size);
    if (!region || address < region->begin || address + size > region->end) {
        return;
    }
    // FIX-GT7-034: what the GPU writes in a listed range is written back whatever the CPU did.
    const bool listed = InplaceWritebackOn() &&
                        std::ranges::any_of(InplaceRanges(), [&](const auto& listed_range) {
                            return address < listed_range.second && listed_range.first < address + size;
                        });
    if (!listed && region->cpu_pages.empty()) {
        return;
    }
    const VAddr begin = Common::AlignDown(address, 4096);
    const VAddr end = std::min<VAddr>(Common::AlignUp(address + size, 4096), region->end);
    bool recent = listed; // only pages the CPU wrote lately are written back (FIX-194)
    for (VAddr page = begin; page < end && !recent; page += 4096) {
        const auto it = region->cpu_pages.find(page);
        recent = it != region->cpu_pages.end() && frame - it->second <= RegionRecentFrames;
    }
    if (!recent) {
        return;
    }
    // PERF-GT7-007 (B): from the pool of copies the GPU is done with (A makes one each time).
    const bool pooled = ReuseOn();
    auto copy = pooled ? TakeSnapshotBuffer(end - begin)
                       : device->CreateBuffer({.size = end - begin,
                                               .usage = Native::Rhi::BufferUsageStorage |
                                                        Native::Rhi::BufferUsageTransferDst,
                                               .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                               .name = "native write-back before"});
    if (!copy) {
        return;
    }
    stats.snapshots_made += pooled ? 0 : 1;
    stats.snapshot_bytes += end - begin;
    Stream& s = Begin();
    GlobalBarrier(); // the region as the work before (and the copy-ins) left it
    s.cmd.copyBuffer(vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                     vk::Buffer{Handle<VkBuffer_T>(copy->NativeHandle())},
                     vk::BufferCopy{.srcOffset = begin - region->begin,
                                    .dstOffset = 0,
                                    .size = end - begin});
    // The work's writes wait for the copy's reads.
    const vk::MemoryBarrier2 copied{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &copied});
    ++stats.writeback_befores;
    written_befores.push_back({.address = address,
                               .size = size,
                               .begin = begin,
                               .end = end,
                               .copy = std::move(copy),
                               .pooled = pooled});
}

void NativeRenderer::DropWrittenBefores() {
    for (auto& before : written_befores) {
        // freed once the GPU used it (PERF-GT7-007: back to the pool then)
        (before.pooled ? stream.pooled : stream.keep).push_back(std::move(before.copy));
    }
    written_befores.clear();
}

std::unique_ptr<Native::Rhi::Buffer> NativeRenderer::TakeSnapshotBuffer(u64 size) {
    // PERF-GT7-007: by size class (powers of two from 64 KB; FIX-194 copies at most 4 MB).
    const u64 class_size = std::max<u64>(std::bit_ceil(size), 64_KB);
    auto* semaphore = r.scheduler.GetWorkSemaphore();
    if (!retiring_snapshots.empty() &&
        retiring_snapshots.front().retire_tick > semaphore->KnownGpuTick()) {
        semaphore->Refresh();
    }
    const u64 known = semaphore->KnownGpuTick();
    while (!retiring_snapshots.empty() && retiring_snapshots.front().retire_tick <= known) {
        free_snapshots.push_back(std::move(retiring_snapshots.front()));
        retiring_snapshots.pop_front();
    }
    for (size_t i = 0; i < free_snapshots.size(); ++i) {
        if (free_snapshots[i].buffer->Size() == class_size) {
            auto buffer = std::move(free_snapshots[i].buffer);
            if (i + 1 != free_snapshots.size()) {
                free_snapshots[i] = std::move(free_snapshots.back());
            }
            free_snapshots.pop_back();
            ++stats.snapshots_reused;
            return buffer;
        }
    }
    auto buffer = device->CreateBuffer({.size = class_size,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                        .name = "native write-back before"});
    if (buffer) {
        snapshot_pool_bytes += class_size;
        ++stats.snapshots_made;
    }
    return buffer;
}

void NativeRenderer::RetireSnapshots(Stream& s, u64 tick) {
    // PERF-GT7-007: the copies the list read go back to the pool once the GPU ran it; past
    // 256 MB in the pool (copies in flight and free), they are freed as before.
    constexpr u64 PoolLimit = 256_MB;
    for (auto& buffer : s.pooled) {
        if (snapshot_pool_bytes > PoolLimit) {
            snapshot_pool_bytes -= buffer->Size();
            s.keep.push_back(std::move(buffer));
            continue;
        }
        retiring_snapshots.push_back({std::move(buffer), tick});
    }
    s.pooled.clear();
}

bool NativeRenderer::WriteBackChangedWords(Stream& s, const Region& region,
                                          const WrittenBefore& before, VAddr run_from,
                                          VAddr run_to) {
    auto* direct = r.buffer_cache.GetDirectMemory();
    const u64 alignment = r.instance.StorageMinAlignment();
        bool compared = true;
        direct->ForEachPart(
            run_from, run_to - run_from,
            [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
                if (view_offset % alignment != 0 || bytes % 16 != 0) {
                    compared = false;
                    return;
                }
                auto guest = device->WrapBuffer(
                    reinterpret_cast<u64>(static_cast<VkBuffer>(view.Handle())),
                    view.SizeBytes());
                const std::array<Native::Rhi::Binding, 3> bindings{{
                    {.slot = 0,
                     .type = Native::Rhi::BindingType::StorageBuffer,
                     .buffer = region.buffer.get(),
                     .offset = run_from - region.begin + offset,
                     .size = bytes},
                    {.slot = 1,
                     .type = Native::Rhi::BindingType::StorageBuffer,
                     .buffer = before.copy.get(),
                     .offset = run_from - before.begin + offset,
                     .size = bytes},
                    {.slot = 2,
                     .type = Native::Rhi::BindingType::StorageBuffer,
                     .buffer = guest.get(),
                     .offset = view_offset,
                     .size = bytes},
                }};
                const u32 count = static_cast<u32>(bytes / 16);
                s.list->BindPipeline(writeback_pipeline.get());
                s.list->PushBindings(bindings);
                s.list->PushConstants(
                    0, std::span<const u8>{reinterpret_cast<const u8*>(&count),
                                           sizeof(count)});
                s.list->Dispatch((count + 63) / 64, 1, 1);
                s.keep.push_back(std::move(guest));
            });
        s.pipeline = vk::Pipeline{}; // the compare binds its own pipeline
        s.so_known = false; // SO-010
        ForgetBindings(s); // OPT-014
    return compared;
}

void NativeRenderer::NoteRegionWrite(VAddr address, u64 size, const WrittenBefore* before,
                                     bool shader_write) {
    // FIX-194: where the GPU writes a region over pages the CPU wrote lately (the pages FIX-173
    // compares), guest memory gets the bytes too, right behind the write, so it holds what the
    // region does at the next compare (GT Sport's cockpit light, written every third frame in
    // a page the CPU writes every frame, is copied out 64 bytes at a time). Bigger writes leave
    // those pages to be copied in whole at the CPU's next write instead. Pages the CPU didn't
    // write lately get nothing: they are copied in whole at its next write anyway, and
    // GPU-only memory keeps NAT-144's speed. RYGT_NATIVE_REGION_WRITEBACK=0: as before.
    static const bool enabled = RegionWritebackOn();
    if (!enabled || size == 0 || regions.empty()) {
        return;
    }
    constexpr u64 MaxWriteback = MaxRegionWriteback;
    auto* direct = r.buffer_cache.GetDirectMemory();
    const VAddr end = address + size;
    const u64 alignment = r.instance.StorageMinAlignment();
    // PERF-GT7-003: from the region over the write's start, in address order (regions never
    // overlap); every write walked them all.
    auto first = regions.upper_bound(address);
    if (first != regions.begin() && std::prev(first)->second.end > address) {
        --first;
    }
    for (auto it = first; it != regions.end() && it->second.begin < end; ++it) {
        Region& region = it->second;
        if (region.end <= address) {
            continue;
        }
        const VAddr from = std::max(address, region.begin);
        const VAddr to = std::min(end, region.end);
        // FIX-GT7-034: a listed range stays in guest memory because the game's CPU reads what the
        // GPU writes there, but a neighbour's region can grow over it (FIX-172 only stops a write
        // there making one). GT7's visibility states are such a range: its CPU only reads their
        // pages, so only the 512 states in a page it also writes reached it, and the models past
        // them in the frame's list were not drawn. The bytes written go to guest memory as they
        // would have.
        if (InplaceWritebackOn() && direct && to - from <= MaxWriteback) {
            for (const auto& [listed_begin, listed_end] : InplaceRanges()) {
                const VAddr copy_from = std::max(from, listed_begin);
                const VAddr copy_to = std::min(to, listed_end);
                if (copy_from >= copy_to) {
                    continue;
                }
                // A shader's storage buffer covers more than it writes, and the copy runs when the
                // GPU gets to it, after the CPU may have written there again: only the words the
                // shader changed go back (FIX-203). A copy or a fill writes all of its range.
                if (shader_write) {
                    if (!before || !before->copy) {
                        NoteListedWriteSkipped(copy_from, copy_to);
                        continue;
                    }
                    const VAddr run_from = Common::AlignDown(copy_from, 4096);
                    const VAddr run_to = std::min(Common::AlignUp(copy_to, 4096), before->end);
                    if (run_from < before->begin || run_from >= run_to ||
                        !direct->EnsureImported(run_from, run_to - run_from)) {
                        NoteListedWriteSkipped(copy_from, copy_to);
                        continue;
                    }
                    Stream& s = Begin();
                    GlobalBarrier(); // behind the write
                    if (WriteBackChangedWords(s, region, *before, run_from, run_to)) {
                        stats.region_writeback_bytes += run_to - run_from;
                        ++stats.precise_writebacks;
                    } else {
                        NoteListedWriteSkipped(copy_from, copy_to);
                    }
                    s.buffers_written = true;
                    ++stats.region_writebacks;
                    continue;
                }
                if (!direct->EnsureImported(copy_from, copy_to - copy_from)) {
                    continue;
                }
                Stream& s = Begin();
                GlobalBarrier(); // behind the write
                const vk::Buffer src{Handle<VkBuffer_T>(region.buffer->NativeHandle())};
                direct->ForEachPart(copy_from, copy_to - copy_from,
                                    [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                        u64 bytes) {
                                        s.cmd.copyBuffer(
                                            src, view.Handle(),
                                            vk::BufferCopy{.srcOffset = copy_from - region.begin +
                                                                        offset,
                                                           .dstOffset = view_offset,
                                                           .size = bytes});
                                    });
                s.buffers_written = true;
                stats.region_writeback_bytes += copy_to - copy_from;
                ++stats.region_writebacks;
            }
        }
        if (region.cpu_pages.empty()) {
            continue;
        }
        // The pages the CPU wrote lately, in order.
        std::vector<VAddr> pages;
        const VAddr first_page = Common::AlignDown(from, 4096);
        if ((to - first_page) / 4096 <= region.cpu_pages.size()) {
            for (VAddr page = first_page; page < to; page += 4096) {
                const auto it = region.cpu_pages.find(page);
                if (it != region.cpu_pages.end() && frame - it->second <= RegionRecentFrames) {
                    pages.push_back(page);
                }
            }
        } else {
            for (const auto& [page, written] : region.cpu_pages) {
                if (page + 4096 > from && page < to && frame - written <= RegionRecentFrames) {
                    pages.push_back(page);
                }
            }
            std::ranges::sort(pages);
        }
        if (pages.empty()) {
            continue;
        }
        if (to - from > MaxWriteback || !direct) {
            std::scoped_lock lock{region_dirty_mutex}; // FIX-GT7-006
            for (const VAddr page : pages) {
                region.cpu_pages.erase(page); // copied in whole at the next CPU write
                cpu_recent_pages.erase(page);
            }
            stats.region_pages_forgotten += pages.size();
            continue;
        }
        region.gpu_pages.insert(pages.begin(), pages.end()); // FIX-GT7-033
        Stream& s = Begin();
        GlobalBarrier(); // behind the write
        const vk::Buffer src{Handle<VkBuffer_T>(region.buffer->NativeHandle())};
        for (size_t i = 0; i < pages.size();) {
            size_t j = i + 1;
            while (j < pages.size() && pages[j] == pages[j - 1] + 4096) {
                ++j;
            }
            // FIX-203: with the pages from before the work, only the words it changed.
            if (before && before->copy && pages[i] >= before->begin &&
                pages[j - 1] + 4096 <= before->end && pages[j - 1] + 4096 <= region.end &&
                direct->EnsureImported(pages[i], pages[j - 1] + 4096 - pages[i])) {
                const VAddr run_from = pages[i];
                const VAddr run_to = pages[j - 1] + 4096;
                const bool compared = WriteBackChangedWords(s, region, *before, run_from, run_to);
                if (compared) {
                    stats.region_writeback_bytes += run_to - run_from;
                    ++stats.precise_writebacks;
                    NoteHistory(run_from, run_to - run_from, 9, 1); // DIAG-187 (diagnosis)
                    i = j;
                    continue;
                }
            }
            const VAddr run_from = std::max(from, pages[i]);
            const VAddr run_to = std::min(to, pages[j - 1] + 4096);
            if (run_from < run_to && direct->EnsureImported(run_from, run_to - run_from)) {
                direct->ForEachPart(run_from, run_to - run_from,
                                    [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                        u64 bytes) {
                                        s.cmd.copyBuffer(
                                            src, view.Handle(),
                                            vk::BufferCopy{.srcOffset = run_from - region.begin +
                                                                        offset,
                                                           .dstOffset = view_offset,
                                                           .size = bytes});
                                    });
                stats.region_writeback_bytes += run_to - run_from;
                NoteHistory(run_from, run_to - run_from, 9, 0); // DIAG-187 (diagnosis)
                if (WritebackRaceDiag()) {
                    // DIAG-176: pending until the GPU ran this list.
                    std::scoped_lock race_lock{writeback_race_mutex};
                    pending_writebacks.push_back(
                        {run_from, run_to, r.scheduler.CurrentTick() + 1, frame});
                }
            } else {
                std::scoped_lock lock{region_dirty_mutex}; // FIX-GT7-006
                for (size_t k = i; k < j; ++k) {
                    region.cpu_pages.erase(pages[k]);
                    cpu_recent_pages.erase(pages[k]);
                }
            }
            i = j;
        }
        s.buffers_written = true; // what reads guest memory next waits for the copies
        ++stats.region_writebacks;
    }
}

bool NativeRenderer::WritebackRaceDiag() {
    static const bool on = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_WRITEBACK_RACE");
        return env && env[0] == '1';
    }();
    return on;
}

bool NativeRenderer::RegionWritebackOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_REGION_WRITEBACK");
        return !(env && env[0] == '0');
    }();
    return on;
}

u64 NativeRenderer::SnapshotBelow() {
    static const u64 below = [] {
        const char* env = std::getenv("RYGT_NATIVE_SNAPSHOT_BELOW");
        return env && *env ? std::strtoull(env, nullptr, 0) : u64{0};
    }();
    return below;
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::GuestBuffer(VAddr address, u64 size,
                                                                      bool write, bool mirror) {
    if (size == 0) {
        return std::nullopt;
    }
    PhaseClock guest_clock{stats.guest_buffer_ns}; // NAT-145
    ++stats.guest_buffer_calls;
    last_mirror = nullptr; // OPT-004
    // OPT-003: the range's last result, when nothing it depends on changed.
    const u32 flags = u32(write) | (u32(mirror) << 1);
    const std::array<u64, 3> memo_key{address, size, flags};
    // PERF-GT7-013 (C): a lighter hash than XXH3 (the slot is checked against the whole key).
    const u64 slot = NextOn() ? MemoHash(memo_key.data(), memo_key.size(), 0)
                              : XXH3_64bits(memo_key.data(), sizeof(memo_key));
    BufferMemo& memo = buffer_memos[slot & (buffer_memos.size() - 1)];
    // PERF-GT7-047: a region or mirror made or dropped elsewhere leaves this range's as it was
    // (the generation moves ~9 times a frame on GT7's pre-race screens). Not for what is read in
    // place: a later lookup may give it a mirror (NAT-150).
    // RYGT_NATIVE_BUFFER_REVALIDATE=0: any change drops every memo, as before.
    if (memo.address == address && memo.size == size && memo.flags == flags &&
        memo.generation != buffer_generation) {
        static const bool revalidate = [] {
            const char* env = std::getenv("RYGT_NATIVE_BUFFER_REVALIDATE");
            return !(env && env[0] == '0');
        }();
        const std::pair<VAddr, VAddr> range{address, address + size};
        if (revalidate && (memo.region || memo.mirror) &&
            memo.hard_generation == buffer_hard_generation &&
            buffer_log.Seq() - memo.log_seq <= 64 &&
            buffer_log.Untouched(memo.log_seq, {&range, 1})) {
            memo.generation = buffer_generation;
            memo.log_seq = buffer_log.Seq();
        }
    }
    if (memo.generation == buffer_generation && memo.address == address && memo.size == size &&
        memo.flags == flags) {
        if (memo.region || memo.mirror) {
            SortRegionWrites(); // CPU writes since: pages to copy in (may drop a mirror)
        }
        const bool fresh =
            memo.generation == buffer_generation &&
            (!memo.region || memo.region->dirty_pages.empty()) &&
            (!memo.mirror || memo.mirror->dirty_pages.empty());
        if (fresh) {
            if (memo.region) {
                ++stats.region_binds;
            } else if (memo.mirror) {
                ++stats.mirror_binds;
                memo.mirror->used_frame = frame;
            }
            last_bound = memo.bound;
            last_mirror = memo.mirror; // OPT-004
            ++stats.buffer_memo_hits;
            return std::pair{memo.buffer, memo.offset};
        }
    }
    memo_kind = MemoKind::None;
    memo_region = nullptr;
    memo_mirror = nullptr;
    const auto result = GuestBufferLookup(address, size, write, mirror);
    if (result && memo_kind == MemoKind::Mirror) {
        last_mirror = memo_mirror; // OPT-004
    }
    if (result && memo_kind != MemoKind::None) {
        memo = BufferMemo{
            .address = address,
            .size = size,
            .flags = flags,
            .generation = buffer_generation,
            .hard_generation = buffer_hard_generation,
            .log_seq = buffer_log.Seq(),
            .buffer = result->first,
            .offset = result->second,
            .region = memo_region,
            .mirror = memo_mirror,
            .bound = last_bound,
        };
    }
    return result;
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::GuestBufferLookup(VAddr address, u64 size,
                                                                            bool write,
                                                                            bool mirror) {
    last_bound = Bound::Region;
    // NAT-114: ranges in (or reaching into) a region use it.
    if (Region* region = RegionOver(address, size)) {
        if (address < region->begin || address + size > region->end) {
            region = MakeRegion(address, size, write);
        }
        if (region) {
            RefreshRegion(*region);
            ++stats.region_binds;
            memo_kind = MemoKind::Region; // OPT-003
            memo_region = region;
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
    }
    // NAT-144: every range shaders write lives in video memory (a region), not in guest memory
    // over PCIe (the GPU was 85% busy waiting on PCIe, 55% with this; the race went from 25 to
    // 33-39 FPS). What the GPU writes there doesn't reach guest memory: GT Sport's CPU doesn't
    // read it back (the old renderer runs with buffer readbacks off too); small render targets
    // still go back (NAT-112). RYGT_NATIVE_WRITE_REGIONS=0: written ranges in guest memory.
    static const bool write_regions = [] {
        const char* env = std::getenv("RYGT_NATIVE_WRITE_REGIONS");
        return !(env && env[0] == '0');
    }();
    // FIX-150: in a region the CPU never saw them (no rain drops on the windshield); written ranges
    // there stay in guest memory.
    const VAddr inplace_from = InplaceFrom();
    const bool listed_in_place = std::ranges::any_of(InplaceRanges(), [&](const auto& range) {
        return address < range.second && address + size > range.first;
    });
    if (write && write_regions && size <= 256_MB && address < inplace_from && !listed_in_place) {
        if (Region* region = MakeRegion(address, size, true)) {
            ++stats.region_binds;
            memo_kind = MemoKind::Region; // OPT-003
            memo_region = region;
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
    }
    // FIX-170 (diagnosis, RYGT_NATIVE_SNAPSHOT_BELOW=<bytes>): small read-only shader buffers
    // (constants) are copied when the draw is recorded instead of read in place by the GPU later.
    if (!mirror && !write && size <= SnapshotBelow()) {
        last_bound = Bound::InPlace;
        return SnapshotRange(address, size);
    }
    if (mirror && !write) {
        if (const auto mirrored = MirrorBuffer(address, size)) {
            last_bound = Bound::Mirror; // NAT-150
            return mirrored;
        }
    }
    last_bound = Bound::InPlace;
    if (write) {
        DropMirrors(address, size); // NAT-150: the GPU writes guest memory there
    }
    if (auto* direct = r.buffer_cache.GetDirectMemory();
        direct && direct->EnsureImported(address, size)) {
        if (const auto found = direct->FindBuffer(address, size)) {
            if (!mirror) {
                memo_kind = MemoKind::InPlace; // OPT-003 (vertices and indices keep looking:
            }                                  // they become mirrors when bound again)
            return std::pair{found->first->Handle(), found->second};
        }
    }
    if (!write) {
        if (const auto gathered = GatherOnGpu(address, size)) {
            return gathered;
        }
    }
    if (write) {
        if (Region* region = MakeRegion(address, size, true)) {
            ++stats.region_binds;
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
        ++stats.unbound;
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-101: written range {:#x}+{:#x} is not in one imported view; bound "
                        "empty",
                        address, size);
        }
        return std::nullopt;
    }
    if (size > 64_MB) {
        ++stats.unbound;
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-101: read range {:#x}+{:#x} is not in one imported view and too "
                        "large to copy; bound empty",
                        address, size);
        }
        return std::nullopt;
    }
    return SnapshotRange(address, size);
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::SnapshotRange(VAddr address, u64 size) {
    // A snapshot of what the range holds now, in the list's upload memory (zeros where the CPU
    // can't read it: NAT-106).
    const auto [buffer, offset, data] = Allocate(size, r.instance.StorageMinAlignment());
    const u64 readable = Native::ReadablePrefix(reinterpret_cast<const void*>(address), size);
    // FIX-160: the game can release part of the range meanwhile (zeros there then).
    if (readable != 0 &&
        !Core::RytyMemory::GuardedCopy(data, reinterpret_cast<const void*>(address), readable)) {
        r.memory->CopySparseMemory(address, data, readable);
    }
    if (readable < size) {
        std::memset(data + readable, 0, size - readable);
    }
    ++stats.copied_reads;
    stats.copied_bytes += size;
    return std::pair{buffer, offset};
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::GatherOnGpu(VAddr address, u64 size) {
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct || size > 256_MB || !direct->EnsureImported(address, size)) {
        return std::nullopt;
    }
    auto gathered = device->CreateBuffer({.size = size,
                                          .usage = Native::Rhi::BufferUsageStorage |
                                                   Native::Rhi::BufferUsageVertex |
                                                   Native::Rhi::BufferUsageIndex |
                                                   Native::Rhi::BufferUsageIndirect |
                                                   Native::Rhi::BufferUsageTransferDst,
                                          .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                          .name = "native gather"});
    if (!gathered) {
        return std::nullopt;
    }
    Stream& s = Begin();
    // Copies run outside rendering, after what earlier work wrote; the draw waits for them
    // (the next barrier: they count as written buffers).
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(gathered->NativeHandle())};
    const bool all = direct->ForEachPart(
        address, size, [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
            s.cmd.copyBuffer(view.Handle(), dst,
                             vk::BufferCopy{.srcOffset = view_offset, .dstOffset = offset,
                                            .size = bytes});
        });
    s.buffers_written = true;
    s.keep.push_back(std::move(gathered));
    if (!all) {
        return std::nullopt;
    }
    ++stats.gathers;
    stats.gathered_bytes += size;
    return std::pair{dst, u64{0}};
}

bool NativeRenderer::TargetsFromMemo(const AmdGpu::Regs& regs, const GraphicsPipeline& pipeline,
                                     Targets& out) {
    TargetsMemo& m = targets_memo;
    const auto& key = pipeline.GetGraphicsKey();
    if (!m.valid ||
        m.generation != registry->Generation() + unmap_generation.load(std::memory_order_relaxed) ||
        m.mrt_mask != key.mrt_mask || m.color_mode != u32(regs.color_control.mode) ||
        m.target_mask != regs.color_target_mask.raw) {
        return false;
    }
    for (u32 cb = 0; cb < m.num_colors; ++cb) {
        if (std::memcmp(&m.color_buffers[cb], &regs.color_buffers[cb], sizeof(AmdGpu::ColorBuffer)) !=
                0 ||
            m.cb_extents[cb].raw != r.liverpool->DrawCbExtent(cb).raw) {
            return false;
        }
    }
    const bool depth_used = (regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
                            (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid());
    if (depth_used != m.depth_used) {
        return false;
    }
    if (depth_used &&
        (m.depth_flags != (u32(regs.depth_control.depth_enable) |
                           (u32(regs.depth_control.stencil_enable) << 1) |
                           (u32(regs.depth_control.depth_write_enable) << 2)) ||
         std::memcmp(&m.depth_buffer, &regs.depth_buffer, sizeof(AmdGpu::DepthBuffer)) != 0 ||
         std::memcmp(&m.depth_view, &regs.depth_view, sizeof(AmdGpu::DepthView)) != 0 ||
         m.htile != regs.depth_htile_data_base.GetAddress() ||
         m.db_extent.raw != r.liverpool->DrawDbExtent().raw)) {
        return false;
    }
    // The pictures, views and ranges are the ones the lookups would give; what each draw does
    // to them (pending rewrites, depth twins, clears of their metadata, their last use) is done
    // as without the memo.
    out = m.targets;
    for (u32 cb = 0; cb < m.num_colors; ++cb) {
        auto* entry = const_cast<Native::ImageEntry*>(out.colors[cb]);
        if (!entry) {
            continue;
        }
        entry->last_use.store(frame, std::memory_order_relaxed);
        ApplyRewrite(*entry); // NAT-120
        auto& attachment = out.state.color_attachments[cb];
        const bool is_clear = m.cmasks[cb] != 0 && cleared_meta.erase(m.cmasks[cb]) != 0;
        attachment.clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(regs.color_buffers[cb]).color.uint32
                     : std::array<u32, 4>{};
        attachment.is_clear = is_clear;
    }
    if (auto* entry = const_cast<Native::ImageEntry*>(out.depth)) {
        entry->last_use.store(frame, std::memory_order_relaxed);
        ApplyRewrite(*entry); // NAT-120
        SyncDepthTwin(*entry); // FIX-212
        bool meta_clear = false;
        if (m.htile != 0) {
            meta_clear = cleared_meta.erase(m.htile) != 0;
            if (!meta_clear && m.htile_meta && !entry->from_cpu &&
                entry->write_version.load(std::memory_order_relaxed) == 0) {
                meta_clear = true; // FIX-156
                ++stats.meta_clears;
            }
        }
        const auto aspects = DepthStencilClearAspects(regs.depth_buffer, regs.depth_control,
                                                      regs.depth_render_control, meta_clear);
        const bool depth_clear = bool(aspects & vk::ImageAspectFlagBits::eDepth);
        const bool stencil_clear = bool(aspects & vk::ImageAspectFlagBits::eStencil);
        auto& db = out.state.depth_stencil_attachment;
        if (db.has_depth) {
            db.clear_value[0] = depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            db.depth_clear = depth_clear;
        }
        if (db.has_stencil) {
            db.clear_value[1] = stencil_clear ? regs.stencil_clear : 0u;
            db.stencil_clear = stencil_clear;
        }
    }
    ++stats.targets_reused;
    return true;
}

bool NativeRenderer::ResolveTargets(const AmdGpu::Regs& regs, const GraphicsPipeline& pipeline,
                                    Targets& out) {
    // OPT-013: not with watched pictures or frame dumps (their logs and notes are per lookup).
    const bool memo = ReuseOn() && Native::WatchedTextures().empty() && !r.FrameDumpActive();
    if (memo && TargetsFromMemo(regs, pipeline, out)) {
        return true;
    }
    // Read before the lookups: a picture they make moves it on, and the next draw looks again.
    const u64 generation =
        registry->Generation() + unmap_generation.load(std::memory_order_relaxed);
    bool htile_meta = false;
    const auto& key = pipeline.GetGraphicsKey();
    RenderState& state = out.state;
    state = {};
    out.colors = {};
    out.depth = nullptr;
    u32 width = r.instance.GetMaxFramebufferWidth();
    u32 height = r.instance.GetMaxFramebufferHeight();
    u32 num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = static_cast<u16>(std::bit_width(key.mrt_mask));
    const bool skip_cb = regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        auto& attachment = state.color_attachments[cb];
        attachment = {};
        attachment.image_layout = vk::ImageLayout::eGeneral;
        const auto& col_buf = regs.color_buffers[cb];
        if (skip_cb || !col_buf || !regs.color_target_mask.GetMask(cb) ||
            (key.mrt_mask & (1u << cb)) == 0) {
            continue;
        }
        const VideoCore::TextureCache::ImageDesc desc{col_buf, r.liverpool->DrawCbExtent(cb)};
        Native::ImageEntry* entry = registry->Find(desc.info, true, frame);
        if (!entry) {
            ++stats.null_images;
            continue;
        }
        const vk::ImageView view = View(*entry, desc.view_info, ViewUse::Attachment);
        if (!view) {
            continue;
        }
        ApplyRewrite(*entry); // NAT-120
        if (Native::WatchedTexture(entry->info.guest_address, entry->info.guest_address + 1) &&
            Native::WatchSample()) {
            LOG_WARNING(Render_Vulkan, "FIX-153 watch: draw into target {:#x} {}x{} layers {} "
                        "({:#x} bytes), view layers {}+{} mip {}",
                        entry->info.guest_address, entry->info.size.width,
                        entry->info.size.height, entry->info.resources.layers,
                        entry->info.guest_size, desc.view_info.range.base.layer,
                        desc.view_info.range.extent.layers, desc.view_info.range.base.level);
        }
        out.colors[cb] = entry;
        out.color_ranges[cb] = {.mip = desc.view_info.range.base.level,
                                .mips = 1,
                                .layer = desc.view_info.range.base.layer,
                                .layers = desc.view_info.range.extent.layers}; // FIX-208
        // FIX-154: the draw covers the target it names, not the larger picture it may share
        // (a clear of the small one must leave the rest alone).
        const u32 mip = desc.view_info.range.base.level;
        width = std::min<u32>(width, std::max(std::min(entry->info.size.width,
                                                       desc.info.size.width) >> mip, 1u));
        height = std::min<u32>(height, std::max(std::min(entry->info.size.height,
                                                         desc.info.size.height) >> mip, 1u));
        num_layers = std::min<u32>(num_layers, desc.view_info.range.extent.layers);
        const VAddr cmask = col_buf.CmaskAddress();
        bool is_clear = false;
        if (cmask != 0) {
            known_meta.insert(cmask);
            is_clear = cleared_meta.erase(cmask) != 0;
        }
        attachment.image_view = view;
        attachment.clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf).color.uint32
                     : std::array<u32, 4>{};
        attachment.is_clear = is_clear;
    }
    auto& db = state.depth_stencil_attachment;
    db = {};
    db.image_layout = vk::ImageLayout::eGeneral;
    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const VAddr htile = regs.depth_htile_data_base.GetAddress();
        const VideoCore::TextureCache::ImageDesc desc{regs.depth_buffer, regs.depth_view,
                                                      regs.depth_control, htile,
                                                      r.liverpool->DrawDbExtent()};
        Native::ImageEntry* entry = registry->Find(desc.info, true, frame);
        const vk::ImageView view =
            entry ? View(*entry, desc.view_info, ViewUse::Attachment) : vk::ImageView{};
        if (view) {
            ApplyRewrite(*entry); // NAT-120
            SyncDepthTwin(*entry); // FIX-212
            if (regs.depth_buffer.StencilValid() && entry->info.props.has_stencil) {
                auto& owners = stencil_owners[regs.depth_buffer.StencilAddress()];
                if (std::ranges::find(owners, entry) == owners.end()) {
                    owners.push_back(entry);
                }
            }
            bool meta_clear = false;
            bool fresh = false;
            htile_meta = desc.info.meta_info.htile_addr != 0; // OPT-013
            if (htile != 0) {
                known_meta.insert(htile);
                meta_clear = cleared_meta.erase(htile) != 0;
                // FIX-156: a depth buffer with HTILE the GPU never drew into starts out cleared,
                // as in the old renderer (a new depth image's HTILE counts as cleared). The game
                // fast-clears the HTILE before the first pass that uses the buffer, while that
                // memory is not yet known as metadata. GT Sport draws the track from above once
                // while a race loads, and the cockpit's night lighting reads a map made of that
                // depth: started from the zeros of fresh memory, every fragment failed the depth
                // test, the map came out empty, and the cockpit went dark but for the odd frame.
                if (!meta_clear && desc.info.meta_info.htile_addr != 0 && !entry->from_cpu &&
                    entry->write_version.load(std::memory_order_relaxed) == 0) {
                    meta_clear = fresh = true;
                    ++stats.meta_clears;
                }
            }
            const auto aspects =
                DepthStencilClearAspects(regs.depth_buffer, regs.depth_control,
                                         regs.depth_render_control, meta_clear);
            const bool depth_clear = bool(aspects & vk::ImageAspectFlagBits::eDepth);
            const bool stencil_clear = bool(aspects & vk::ImageAspectFlagBits::eStencil);
            if (Native::WatchedTexture(entry->info.guest_address, entry->info.guest_address + 1) &&
                Native::WatchSample()) {
                const auto& dc = regs.depth_control;
                LOG_WARNING(Render_Vulkan, "FIX-156 watch: depth pass {:#x} {}x{} htile {:#x} "
                            "(tile surface {}) version {} meta clear {}{} -> depth clear {} ({}) "
                            "stencil clear {} | test {} write {} func {} stencil {} | render "
                            "control clear {}/{} (frame {})",
                            entry->info.guest_address, entry->info.size.width,
                            entry->info.size.height, htile,
                            u32(regs.depth_buffer.z_info.tile_surface_enable),
                            entry->write_version.load(), meta_clear, fresh ? " (fresh)" : "",
                            depth_clear, regs.depth_clear, stencil_clear, u32(dc.depth_enable),
                            u32(dc.depth_write_enable), u32(dc.depth_func),
                            u32(dc.stencil_enable),
                            u32(regs.depth_render_control.depth_clear_enable),
                            u32(regs.depth_render_control.stencil_clear_enable), frame);
            }
            db.image_view = view;
            if (regs.depth_buffer.DepthValid()) {
                db.clear_value[0] = depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
                db.has_depth = true;
                db.depth_clear = depth_clear;
            }
            if (regs.depth_buffer.StencilValid() && entry->info.props.has_stencil) {
                db.clear_value[1] = stencil_clear ? regs.stencil_clear : 0u;
                db.has_stencil = true;
                db.stencil_clear = stencil_clear;
            }
            width = std::min<u32>(width, std::min(entry->info.size.width, desc.info.size.width));
            height =
                std::min<u32>(height, std::min(entry->info.size.height, desc.info.size.height));
            num_layers = std::min<u32>(num_layers, desc.view_info.range.extent.layers);
            out.depth = entry;
        } else {
            ++stats.null_images;
        }
    }
    state.width = static_cast<u16>(std::min<u32>(width, 0xffff));
    state.height = static_cast<u16>(std::min<u32>(height, 0xffff));
    state.num_layers = static_cast<u16>(
        num_layers == std::numeric_limits<u16>::max() ? 1u : std::max(num_layers, 1u));
    TargetsMemo& m = targets_memo; // OPT-013
    m.valid = memo;
    if (memo) {
        m.generation = generation;
        m.mrt_mask = key.mrt_mask;
        m.color_mode = u32(regs.color_control.mode);
        m.target_mask = regs.color_target_mask.raw;
        m.num_colors = state.num_color_attachments;
        for (u32 cb = 0; cb < m.num_colors; ++cb) {
            m.color_buffers[cb] = regs.color_buffers[cb];
            m.cb_extents[cb] = r.liverpool->DrawCbExtent(cb);
            m.cmasks[cb] = regs.color_buffers[cb].CmaskAddress();
        }
        m.depth_used = (regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
                       (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid());
        m.depth_flags = u32(regs.depth_control.depth_enable) |
                        (u32(regs.depth_control.stencil_enable) << 1) |
                        (u32(regs.depth_control.depth_write_enable) << 2);
        m.depth_buffer = regs.depth_buffer;
        m.depth_view = regs.depth_view;
        m.htile = regs.depth_htile_data_base.GetAddress();
        m.db_extent = r.liverpool->DrawDbExtent();
        m.htile_meta = htile_meta;
        m.targets = out;
    }
    return true;
}

void NativeRenderer::BindStagesStable(std::span<const Shader::Info* const> stages,
                                      const AmdGpu::Regs& regs, bool compute, DrawBindings& b) {
    // FIX-GT7-090: the buffers of a draw or dispatch are looked up one after the other, and a
    // written range reaching past its region makes a bigger one that takes the old one in
    // (MakeRegion): an earlier binding of the same work still names the old region's buffer,
    // which is kept only until the list has run. Its writes went there and were lost. GT7's
    // crowd: cs 0x5972569e writes its section list and, in the same page, its group table; the
    // first frame the game puts them at a new address, the list's binding made a region (0x3000)
    // and the table's binding a bigger one taking it in (0x4000), so the first of the three
    // dispatches wrote its sections into the old buffer. The scatter (cs 0x7dd8d7b6) then read
    // the stale sections from guest memory (an index of -64), ~313 crowd instances kept
    // garbage matrices, and vs 0xb1e0c79f drew them as huge triangles at the near plane in the
    // depth pre-pass: one frame black under the pre-race HUD (triangles, bands, sky, whole
    // scene). Found by Codex's replays (occlusion queries per draw, the crowd buffers copied
    // after the scatter). The bindings are made again once nothing more is taken in (they all
    // name the merged region then). RYGT_NATIVE_STABLE_BINDINGS=0: as before.
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_STABLE_BINDINGS");
        return !(env && env[0] == '0');
    }();
    for (u32 attempt = 0;; ++attempt) {
        const u64 taken_before = regions_taken_in;
        BindStages(stages, regs, compute, b);
        if (!on || regions_taken_in == taken_before || attempt == 3) {
            return;
        }
        ++stage_rebinds;
        static std::atomic<u32> logged{};
        if (const u32 n = logged++; n < 20 || n % 1000 == 0) {
            LOG_INFO(Render_Vulkan, "FIX-GT7-090: a region taken in while binding a {} (frame {}): "
                     "bound again ({} so far)", compute ? "dispatch" : "draw", frame, n + 1);
        }
        b.Clear();
        b.push = UserData(regs);
    }
}

void NativeRenderer::BindStages(std::span<const Shader::Info* const> stages,
                                const AmdGpu::Regs& regs, bool compute, DrawBindings& b) {
    DropWrittenBefores(); // FIX-203: a skipped draw's copies are not the next one's
    const u64 nulls_before = stats.null_images; // DIAG-170
    const auto reasons_before = stats.null_reasons;
    size_t total = 0;
    for (const auto* stage : stages) {
        if (stage) {
            total += stage->buffers.size() + stage->images.size() + stage->samplers.size();
        }
    }
    b.writes.reserve(total);
    Shader::Backend::Bindings binding{};
    std::array<StageResourceBindings, Shader::MaxStageTypes> plans{};
    for (u32 i = 0; i < stages.size(); ++i) {
        const auto* stage = stages[i];
        if (!stage) {
            continue;
        }
        // SHD-002: each stage binds in its own set, from its own start.
        plans[i] = PlanStageResourceBindings(*stage,
                                             Shader::Backend::StageStart(u32(stage->sw_stage)));
        auto buffer_binding = plans[i].buffers;
        PhaseClock clock{stats.phase_ns[2]};
        PartClock part{compute ? &stats.dispatch_bind_ns[1] : nullptr}; // DIAG-GT7-004
        BindBuffers(*stage, regs, compute, buffer_binding, b);
        binding = plans[i].next;
    }
    for (u32 i = 0; i < stages.size(); ++i) {
        if (const auto* stage = stages[i]) {
            auto texture_binding = plans[i].textures;
            PhaseClock clock{stats.phase_ns[3]};
            PartClock part{compute ? &stats.dispatch_bind_ns[2] : nullptr}; // DIAG-GT7-004
            BindTextures(*stage, regs, texture_binding, b);
        }
    }
    // FIX-203: the region pages the storage buffers may write, as the work will find them
    // (after the copy-ins its bindings recorded).
    {
        PartClock part{compute ? &stats.dispatch_bind_ns[3] : nullptr}; // DIAG-GT7-004
        for (const auto& [address, size] : b.gpu_writes) {
            SnapshotBeforeWrite(address, size);
        }
    }
    // DIAG-169 (diagnosis, RYGT_WATCH_DRAW_TEXTURE=<w>x<h>[,<w>x<h>...]): for draws sampling a
    // picture of that size, what guest memory holds now under each buffer they read.
    static const std::vector<std::pair<u32, u32>> watch_sizes = [] {
        std::vector<std::pair<u32, u32>> list;
        const char* env = Common::DiagEnv("RYGT_WATCH_DRAW_TEXTURE");
        for (const char* at = env; at && *at;) {
            char* end = nullptr;
            const u32 w = u32(std::strtoul(at, &end, 10));
            if (end == at || *end != 'x') {
                break;
            }
            const u32 h = u32(std::strtoul(end + 1, &end, 10));
            list.emplace_back(w, h);
            at = *end == ',' ? end + 1 : end;
        }
        return list;
    }();
    probe_buffers.clear();
    probe_bound_hashes.clear();
    for (const auto* stage : stages) {
        if (stage) {
            history_program = stage->pgm_hash; // DIAG-187: the last stage's
        }
    }
    for (const auto* stage : stages) {
        if (stage && !compute) {
            NoteProgramTextures(stage->pgm_hash, b); // DIAG-184 (diagnosis)
        }
    }
    // DIAG-233 (diagnosis, RYGT_DIAG_STAGE_TEXTURES=<hex program hash>): every texture of the draws
    // using that program, once a frame (400 lines at most).
    static const u64 diag233_program = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_STAGE_TEXTURES");
        return env && *env ? u64(std::strtoull(env, nullptr, 16)) : u64{0};
    }();
    static u32 diag233_lines = 0;
    static u64 diag233_frame = ~u64{0};
    if (diag233_program != 0 && diag233_lines < 400 && diag233_frame != frame &&
        std::ranges::any_of(stages, [](const auto* stage) {
            return stage && stage->pgm_hash == diag233_program;
        })) {
        diag233_frame = frame;
        for (const auto* sampled : b.sampled) {
            if (!sampled || diag233_lines >= 400) {
                continue;
            }
            ++diag233_lines;
            LOG_WARNING(Render_Vulkan, "DIAG-233: frame {} {} {:#x}+{:#x} {}x{} mips {} layers {} {} "
                        "made frame {}",
                        frame, sampled->from_cpu ? "texture" : "target",
                        sampled->info.guest_address, sampled->info.guest_size,
                        sampled->info.size.width, sampled->info.size.height,
                        sampled->info.resources.levels, sampled->info.resources.layers,
                        vk::to_string(sampled->info.pixel_format), sampled->made_frame);
            if (sampled->info.resources.levels > 1 && diag233_lines < 400) {
                std::string mips;
                for (u32 m = 0; m < sampled->info.resources.levels; ++m) {
                    const auto& mip = sampled->info.mips_layout[m];
                    mips += fmt::format(" {}:{:#x}+{:#x}p{}", m, mip.offset, mip.size, mip.pitch);
                }
                ++diag233_lines;
                LOG_WARNING(Render_Vulkan, "DIAG-233: {:#x} tile {} array {} pitch {} micro mips {}:{}",
                            sampled->info.guest_address, u32(sampled->info.tile_mode),
                            u32(sampled->info.array_mode), sampled->info.pitch,
                            sampled->info.micro_tiled_mips, mips);
            }
        }
    }
    if (!diag_program_textures.empty() && diag185_lines < 200) {
        diag_what.clear(); // DIAG-185: the work's programs
        for (const auto* stage : stages) {
            if (stage) {
                diag_what += fmt::format(" {:#x}", stage->pgm_hash);
            }
        }
    }
    if (DiagFrame()) {
        // DIAG-178: this work's programs.
        diag_what.clear();
        for (const auto* stage : stages) {
            if (stage) {
                diag_what += fmt::format(" {:#x}", stage->pgm_hash);
            }
        }
    }
    if (strip_frame == frame) {
        // DIAG-173: what this draw or dispatch is, for the strip's pixel history.
        strip_what.clear();
        for (const auto* stage : stages) {
            if (stage) {
                strip_what += fmt::format(" {:#x}", stage->pgm_hash);
            }
        }
        strip_what += " |";
        for (const auto* entry : b.sampled) {
            strip_what += fmt::format(" {:#x}:{}x{}l{}{}", entry->info.guest_address,
                                      entry->info.size.width, entry->info.size.height,
                                      entry->info.resources.layers,
                                      entry->from_cpu ? "t" : "");
        }
    }
    // RYGT_PROBE_AFTER=<seconds>: DIAG-169/170 only from then on (the game runs at full speed
    // until the scene to look at).
    static const auto watch_from = [] {
        const char* env = Common::DiagEnv("RYGT_PROBE_AFTER");
        return std::chrono::steady_clock::now() +
               std::chrono::seconds(env && *env ? std::atoi(env) : 0);
    }();
    if (!watch_sizes.empty() && !compute && std::chrono::steady_clock::now() >= watch_from) {
        const Native::ImageEntry* hit = nullptr;
        for (const auto* entry : b.sampled) {
            for (const auto& [w, h] : watch_sizes) {
                if (entry && entry->info.size.width == w && entry->info.size.height == h) {
                    hit = entry;
                }
            }
        }
        if (hit) {
            std::string text;
            for (const auto* stage : stages) {
                if (!stage) {
                    continue;
                }
                u32 index = 0;
                for (const auto& desc : stage->buffers) {
                    const u32 buffer_index = index++;
                    if (desc.IsSpecial()) {
                        continue;
                    }
                    const auto vsharp = desc.GetSharp(*stage);
                    const u64 size = std::min<u64>(vsharp.GetSize(), 512);
                    std::array<u32, 128> words{};
                    const bool read =
                        size != 0 &&
                        Core::RytyMemory::GuardedCopy(words.data(),
                                                      reinterpret_cast<const void*>(
                                                          VAddr(vsharp.base_address)),
                                                      size);
                    // The whole buffer's hash in guest memory, and whether a region (the GPU's
                    // copy) holds it, so draws of different frames compare.
                    const u64 whole = std::min<u64>(vsharp.GetSize(), 256_KB);
                    std::vector<u8> all(whole);
                    const bool read_all =
                        whole != 0 && Core::RytyMemory::GuardedCopy(
                                          all.data(),
                                          reinterpret_cast<const void*>(VAddr(vsharp.base_address)),
                                          whole);
                    bool in_region = false;
                    {
                        std::shared_lock map_lock{buffer_map_mutex};
                        in_region = RegionOver(VAddr(vsharp.base_address),
                                               std::max<u64>(vsharp.GetSize(), 1)) != nullptr;
                    }
                    text += fmt::format(" | {:#x} b{} {:#x}+{:#x}{} h{:016x}:", stage->pgm_hash,
                                        buffer_index, VAddr(vsharp.base_address),
                                        vsharp.GetSize(), in_region ? " region" : "",
                                        read_all ? XXH3_64bits(all.data(), whole) : 0);
                    if (read && probe_buffers.size() < 2) {
                        probe_buffers.emplace_back(VAddr(vsharp.base_address), size); // DIAG-170
                        probe_bound_hashes += fmt::format(
                            " {:016x}", XXH3_64bits(words.data(), size));
                    }
                    for (u64 w = 0; read && w < std::min<u64>(size, 64) / 4; ++w) {
                        text += fmt::format(" {:08x}", words[w]);
                    }
                }
            }
            static const bool quiet = [] {
                const char* env = Common::DiagEnv("RYGT_PROBE_AT");
                return env && *env; // DIAG-170 says what it needs
            }();
            if (!quiet) {
                std::string sampled_text;
                for (const auto* entry : b.sampled) {
                    if (entry) {
                        sampled_text += fmt::format(" {:#x}{}", entry->info.guest_address,
                                                    entry->from_cpu ? "t" : "");
                    }
                }
                LOG_WARNING(Render_Vulkan, "DIAG-169: draw sampling {:#x} {}x{} (frame {}) images{}{}",
                            hit->info.guest_address, hit->info.size.width,
                            hit->info.size.height, frame, sampled_text, text);
            }
        }
        probe_sampled = hit; // DIAG-170
        if (hit) {
            probe_bound = fmt::format("{} empty", stats.null_images - nulls_before);
            for (size_t k = 0; k < stats.null_reasons.size(); ++k) {
                if (stats.null_reasons[k] != reasons_before[k]) {
                    probe_bound += fmt::format(" (reason {})", k);
                }
            }
            for (const auto& image : b.images) {
                probe_bound += fmt::format(" {:x}", u64(static_cast<VkImageView>(image.imageView)) & 0xffffff);
            }
            for (const auto* entry : b.sampled) {
                probe_bound += fmt::format(" [{:#x} {}x{}{}]", entry->info.guest_address,
                                           entry->info.size.width, entry->info.size.height,
                                           entry->from_cpu ? " tex" : "");
            }
        }
    }
}

void NativeRenderer::ProbeDraw(const Targets& targets, const AmdGpu::Regs& regs) {
    // DIAG-170 (diagnosis, RYGT_PROBE_AT=<x>,<y>[,<y2>] with RYGT_WATCH_DRAW_TEXTURE): after
    // each draw DIAG-169 matches, 16x16 blocks of its first colour target at (x, y) and, in
    // targets that tall, (x, y2), read back from the GPU and logged with the frame (one line
    // per frame says which display buffer it shows), to see which draw of a glitched frame
    // went wrong.
    static const std::array<u32, 3> at = [] {
        std::array<u32, 3> a{};
        if (const char* env = Common::DiagEnv("RYGT_PROBE_AT"); env && *env) {
            std::sscanf(env, "%u,%u,%u", &a[0], &a[1], &a[2]);
        }
        return a;
    }();
    const Native::ImageEntry* sampled = std::exchange(probe_sampled, nullptr);
    if (!sampled || (at[0] == 0 && at[1] == 0) || !targets.colors[0] ||
        !targets.colors[0]->image || draw_probes.size() >= 4096) {
        return;
    }
    const Native::ImageEntry& target = *targets.colors[0];
    const auto& desc = target.image->Desc();
    const u32 bits = target.info.num_bits;
    {
        // The state the draw was recorded with.
        const auto& d = stream.dynamic;
        probe_bound += fmt::format(" | pipe {:x}", u64(static_cast<VkPipeline>(stream.pipeline)) & 0xffffff);
        for (const auto& v : d.viewports) {
            probe_bound += fmt::format(" vp {}/{} {}x{} {}..{}", v.x, v.y, v.width, v.height,
                                       v.minDepth, v.maxDepth);
        }
        for (const auto& sc : d.scissors) {
            probe_bound += fmt::format(" sc {}/{} {}x{}", sc.offset.x, sc.offset.y,
                                       sc.extent.width, sc.extent.height);
        }
        probe_bound += fmt::format(" z {}{} st {} cull {} wm {:x} fb {} rd {} att {:x}/{} {}x{} cl {}",
                                   d.depth_test_enabled, d.depth_write_enabled,
                                   d.stencil_test_enabled, u32(d.cull_mode),
                                   u32(d.color_write_masks[0]), u32(d.feedback_loop_enabled),
                                   d.rasterizer_discard_enable,
                                   u64(static_cast<VkImageView>(targets.state.color_attachments[0].image_view)) & 0xffffff,
                                   targets.state.num_color_attachments, targets.state.width,
                                   targets.state.height,
                                   targets.state.color_attachments[0].is_clear);
        // The depth test as the registers have it.
        const auto& vp = regs.viewports[0];
        probe_bound += fmt::format(
            " | zfunc {} cmp {} zclip near/far off {}/{} clamp off {} zscale {} zoffset {} "
            "zmin {} zmax {} depth att {:x}",
            u32(regs.depth_control.depth_func), u32(stream.dynamic.depth_compare_op),
            u32(regs.clipper_control.zclip_near_disable),
            u32(regs.clipper_control.zclip_far_disable),
            u32(regs.depth_render_override.disable_viewport_clamp), vp.zscale, vp.zoffset,
            regs.viewport_depths[0].zmin, regs.viewport_depths[0].zmax,
            u64(static_cast<VkImageView>(targets.state.depth_stencil_attachment.image_view)) &
                0xffffff);
    }
    if ((bits != 32 && bits != 64) || desc.width < at[0] + 16) {
        return;
    }
    if (probe_frame != frame) {
        probe_frame = frame;
        probe_draw = 0;
    }
    std::array<u32, 2> rows{at[1], at[2] != 0 && desc.height >= at[2] + 16 ? at[2] : at[1]};
    if (desc.height < rows[0] + 16) {
        return;
    }
    const u64 block = 16 * 16 * (bits / 8);
    constexpr u64 Extra = 2 * 512 + 4096 + 2 * 512 + 1024; // buffers, sampled, vertices, depth
    auto buffer = device->CreateBuffer({.size = block * 2 + Extra,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::Readback,
                                        .name = "draw probe"});
    if (!buffer || !buffer->Mapped()) {
        return;
    }
    Begin();
    GlobalBarrier();
    for (u32 i = 0; i < 2; ++i) {
        stream.cmd.copyImageToBuffer(
            vk::Image{Handle<VkImage_T>(target.image->NativeHandle())}, vk::ImageLayout::eGeneral,
            vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
            vk::BufferImageCopy{.bufferOffset = block * i,
                                .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                  .mipLevel = 0,
                                                  .baseArrayLayer = 0,
                                                  .layerCount = 1},
                                .imageOffset = {s32(at[0]), s32(rows[i]), 0},
                                .imageExtent = {16, 16, 1}});
    }
    // What the draw's buffers held in guest memory when it ran (as reads in place see them),
    // and a corner of what it sampled.
    u32 extra = 0;
    std::array<u32, 2> lengths{};
    const u32 texture_bytes = sampled->info.props.is_block ? 256 : 16 * 16 * sampled->info.num_bits / 8;
    if (auto* direct = r.buffer_cache.GetDirectMemory()) {
        for (u32 i = 0; i < probe_buffers.size(); ++i) {
            const auto [address, size] = probe_buffers[i];
            if (direct->EnsureImported(address, size)) {
                if (const auto found = direct->FindBuffer(address, size)) {
                    stream.cmd.copyBuffer(found->first->Handle(),
                                          vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
                                          vk::BufferCopy{.srcOffset = found->second,
                                                         .dstOffset = block * 2 + 512 * i,
                                                         .size = std::min<u64>(size, 512)});
                    extra |= 1u << i;
                    lengths[i] = u32(std::min<u64>(size, 512));
                }
            }
        }
    }
    if (sampled->image && sampled->image->Desc().width >= 16 &&
        sampled->image->Desc().height >= 16 && texture_bytes <= 4096) {
        stream.cmd.copyImageToBuffer(
            vk::Image{Handle<VkImage_T>(sampled->image->NativeHandle())},
            vk::ImageLayout::eGeneral, vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
            vk::BufferImageCopy{.bufferOffset = block * 2 + 1024,
                                .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                  .mipLevel = 0,
                                                  .baseArrayLayer = 0,
                                                  .layerCount = 1},
                                .imageOffset = {s32(std::min<u32>(at[0] / 2, sampled->image->Desc().width - 16) & ~15u),
                                                s32(std::min<u32>(at[1] / 2, sampled->image->Desc().height - 16) & ~15u), 0},
                                .imageExtent = {16, 16, 1}});
        extra |= 4;
    }
    std::array<u32, 2> vertex_lengths{};
    std::string vertex_recorded;
    for (u32 i = 0; i < probe_vertices.size(); ++i) {
        const auto& pv = probe_vertices[i];
        stream.cmd.copyBuffer(pv.buffer, vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
                              vk::BufferCopy{.srcOffset = pv.offset,
                                             .dstOffset = block * 2 + 1024 + 4096 + 512 * i,
                                             .size = pv.size});
        vertex_lengths[i] = u32(pv.size);
        vertex_recorded += fmt::format(" v{:#x}+{:#x} how {} {:016x} [{:08x} {:08x} {:08x} {:08x}]",
                                       pv.base, pv.size, pv.how, pv.hash, pv.head[0], pv.head[1],
                                       pv.head[2], pv.head[3]);
    }
    probe_bound_hashes += vertex_recorded;
    // Its depth buffer there (a 32-bit float depth aspect).
    bool depth = false;
    if (const auto* z = targets.depth; z && z->image && z->info.props.is_depth &&
        z->info.num_bits == 32 && z->image->Desc().width >= at[0] + 16 &&
        z->image->Desc().height >= rows[0] + 16) {
        stream.cmd.copyImageToBuffer(
            vk::Image{Handle<VkImage_T>(z->image->NativeHandle())}, vk::ImageLayout::eGeneral,
            vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
            vk::BufferImageCopy{.bufferOffset = block * 2 + 1024 + 4096 + 1024,
                                .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                                  .mipLevel = 0,
                                                  .baseArrayLayer = 0,
                                                  .layerCount = 1},
                                .imageOffset = {s32(at[0]), s32(rows[0]), 0},
                                .imageExtent = {16, 16, 1}});
        depth = true;
    }
    GlobalBarrier();
    draw_probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, frame, probe_draw++,
                           sampled->info.guest_address, sampled->info.size.width,
                           sampled->info.size.height, target.info.guest_address,
                           target.info.size.width, target.info.size.height, bits, rows,
                           std::move(probe_bound), probe_bound_hashes, extra, lengths,
                           texture_bytes, vertex_lengths, depth});
}

void NativeRenderer::StripDraw(const Targets& targets) {
    if (strip_frame == frame && targets.colors[0]) {
        StripImage(*targets.colors[0], false);
    }
}

void NativeRenderer::StripDispatch(const DrawBindings& b) {
    if (strip_frame != frame) {
        return;
    }
    for (const auto* stored : b.stored) {
        if (stored) {
            StripImage(*stored, true);
        }
    }
}

void NativeRenderer::StripImage(const Native::ImageEntry& target, bool dispatch) {
    // DIAG-173 (diagnosis, RYGT_PROBE_STRIP=<y> RYGT_PROBE_TARGET=<hex address>; armed for one
    // frame by a file named probe.now next to the user folder): after every draw into that
    // target (and dispatch storing to it), rows y..y+7 of it are read back, so each pixel of the
    // strip has its history (which draw wrote it last, and what it wrote). Written with the
    // frame's screenshot to user\log\strip_<frame>.bin/.txt. RYGT_PROBE_TARGET=1: every
    // picture of at least 1280x720; =2: of at least 640x360 (y scaled to its height, as a
    // 1080-line picture's).
    static const u32 strip_y = [] {
        const char* env = Common::DiagEnv("RYGT_PROBE_STRIP");
        return env && *env ? u32(std::atoi(env)) : ~0u;
    }();
    static const VAddr strip_target = [] {
        const char* env = Common::DiagEnv("RYGT_PROBE_TARGET");
        return env && *env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (strip_y == ~0u || strip_frame != frame || !target.image || target.info.props.is_depth) {
        return;
    }
    const auto& desc = target.image->Desc();
    // =3: every picture at least 8 rows tall.
    const u32 min_width = strip_target == 1 ? 1280 : strip_target == 2 ? 640 : 1;
    const u32 min_height = strip_target == 1 ? 720 : strip_target == 2 ? 360 : 8;
    if (strip_target > 3 ? target.info.guest_address != strip_target
                         : (desc.width < min_width || desc.height < min_height)) {
        return;
    }
    const u32 bits = target.info.num_bits;
    const u32 y = strip_target <= 3
                      ? std::min<u32>(u32(u64(strip_y) * desc.height / 1080), desc.height - 8)
                      : strip_y;
    if (bits % 8 != 0 || bits == 0 || bits > 128 || target.info.props.is_block ||
        desc.height < y + 8) {
        return;
    }
    const u64 bytes = u64(desc.width) * 8 * (bits / 8);
    auto buffer = device->CreateBuffer({.size = bytes,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::Readback,
                                        .name = "strip probe"});
    if (!buffer || !buffer->Mapped()) {
        return;
    }
    Begin();
    GlobalBarrier();
    stream.cmd.copyImageToBuffer(
        vk::Image{Handle<VkImage_T>(target.image->NativeHandle())}, vk::ImageLayout::eGeneral,
        vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
        vk::BufferImageCopy{.bufferOffset = 0,
                            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                              .mipLevel = 0,
                                              .baseArrayLayer = 0,
                                              .layerCount = 1},
                            .imageOffset = {0, s32(y), 0},
                            .imageExtent = {desc.width, 8, 1}});
    GlobalBarrier();
    strip_probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1,
                            u32(strip_probes.size()),
                            fmt::format(" @{:#x}:{}x{}b{}:{} y{}{}", target.info.guest_address,
                                        desc.width, desc.height, bits,
                                        vk::to_string(static_cast<vk::Format>(desc.format)), y,
                                        dispatch ? " cs" : "") +
                                strip_what});
}

void NativeRenderer::StripFrameEnd() {
    // DIAG-173: arming (probe.now, checked every 30 flips) and writing out.
    static u32 checks = 0;
    if (strip_frame == ~u64{0} && ++checks % 30 == 0) {
        std::error_code error;
        const auto trigger =
            Common::FS::GetUserPath(Common::FS::PathType::UserDir).parent_path() / "probe.now";
        if (std::filesystem::remove(trigger, error)) {
            strip_frame = frame; // the frame recorded from now on (FindDisplay counted it)
            strip_screenshot = true;
            LOG_WARNING(Render_Vulkan, "DIAG-173: strip of frame {}", strip_frame);
        }
        return;
    }
    if (strip_frame == ~u64{0}) {
        return;
    }
    if (strip_screenshot && frame == strip_frame + 1) {
        // This flip shows the frame the strips were taken of.
        VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
        strip_screenshot = false;
    }
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    if (frame <= strip_frame + 1 || strip_probes.empty() || strip_probes.back().tick > done) {
        if (frame > strip_frame + 600) {
            strip_probes.clear(); // never finished
            strip_frame = ~u64{0};
        }
        return;
    }
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
    const auto base = fmt::format("strip_{}", strip_frame);
    std::string index;
    if (std::FILE* file = std::fopen((dir / (base + ".bin")).string().c_str(), "wb")) {
        for (const auto& strip : strip_probes) {
            std::fwrite(strip.buffer->Mapped(), 1, strip.buffer->Size(), file);
            index += fmt::format("{}{}\n", strip.draw, strip.what);
        }
        std::fclose(file);
    }
    if (std::FILE* file = std::fopen((dir / (base + ".txt")).string().c_str(), "w")) {
        std::fwrite(index.data(), 1, index.size(), file);
        std::fclose(file);
    }
    LOG_WARNING(Render_Vulkan, "DIAG-173: {} strips of frame {} written to {}", strip_probes.size(),
                strip_frame, (dir / base).string());
    strip_probes.clear();
    strip_frame = ~u64{0};
}

void NativeRenderer::ProbeDispatch(const Shader::Info& stage, const DrawBindings& b,
                                   bool after) {
    // DIAG-175 (diagnosis, RYGT_PROBE_DISPATCH=<hex program hash>|all,
    // RYGT_PROBE_DISPATCH_TARGET=<hex address>): for those dispatches, texel (0,0) of the image
    // it stores to before and after it (mip 0: every layer; mips 1-7: the first 8 layers), the
    // center texel of the first 6 layers of every mip of the first 2 images it samples, and its
    // flattened user data, logged once the GPU ran it.
    static const char* program_env = Common::DiagEnv("RYGT_PROBE_DISPATCH");
    static const bool any_program = program_env && std::string_view{program_env} == "all";
    static const u64 program =
        program_env && *program_env ? u64(std::strtoull(program_env, nullptr, 16)) : u64{0};
    static const VAddr only = [] {
        const char* env = Common::DiagEnv("RYGT_PROBE_DISPATCH_TARGET");
        return env && *env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if ((!any_program && (program == 0 || stage.pgm_hash != program)) || b.stored.empty() ||
        dispatch_probes.size() >= 64) {
        return;
    }
    const auto* entry = b.stored[0];
    if (!entry || !entry->image || entry->info.num_bits != 32 || entry->info.props.is_depth ||
        (only && entry->info.guest_address != only)) {
        return;
    }
    const u32 layers = std::max<u32>(entry->image->Desc().layers, 1);
    const u32 mips = std::min<u32>(std::max<u32>(entry->image->Desc().mips, 1), 8);
    const u32 few = std::min<u32>(layers, 8);
    const u64 section = (u64(layers) + u64(mips - 1) * few) * 4;
    const auto copy_texels = [&](const Native::ImageEntry& image, u64 offset, u32 mip,
                                 u32 count, bool center) {
        const auto& desc = image.image->Desc();
        stream.cmd.copyImageToBuffer(
            vk::Image{Handle<VkImage_T>(image.image->NativeHandle())}, vk::ImageLayout::eGeneral,
            vk::Buffer{Handle<VkBuffer_T>(dispatch_probes.back().buffer->NativeHandle())},
            vk::BufferImageCopy{
                .bufferOffset = offset,
                .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                  .mipLevel = mip,
                                  .baseArrayLayer = 0,
                                  .layerCount = count},
                .imageOffset = {center ? s32(std::max<u32>(desc.width >> mip, 1) / 2) : 0,
                                center ? s32(std::max<u32>(desc.height >> mip, 1) / 2) : 0, 0},
                .imageExtent = {1, 1, 1}});
    };
    if (!after) {
        auto buffer = device->CreateBuffer({.size = section * 2 + 2 * 8 * 6 * 4,
                                            .usage = Native::Rhi::BufferUsageStorage |
                                                     Native::Rhi::BufferUsageTransferDst,
                                            .memory = Native::Rhi::MemoryKind::Readback,
                                            .name = "dispatch probe"});
        if (!buffer || !buffer->Mapped()) {
            return;
        }
        std::memset(buffer->Mapped(), 0xee, buffer->Size());
        std::string what = fmt::format("program {:#x} dims {}x{}x{} flat",
                                       stage.pgm_hash, r.liverpool->DrawCsRegs().dim_x,
                                       r.liverpool->DrawCsRegs().dim_y,
                                       r.liverpool->DrawCsRegs().dim_z);
        for (size_t i = 0; i < stage.flattened_ud_buf.size() && i < 48; ++i) {
            what += fmt::format(" {:x}", stage.flattened_ud_buf[i]);
        }
        dispatch_probes.push_back({std::move(buffer), 0, layers, mips,
                                   entry->info.guest_address, std::move(what)});
        GlobalBarrier();
        auto& text = dispatch_probes.back().what;
        text += " | sampled";
        for (size_t i = 0; i < b.sampled.size() && i < 2; ++i) {
            const auto* sampled = b.sampled[i];
            if (!sampled || !sampled->image) {
                text += " null";
                continue;
            }
            const auto& desc = sampled->image->Desc();
            text += fmt::format(" {:#x}:{}x{}l{}m{}b{}{}{} v{}", sampled->info.guest_address,
                                desc.width, desc.height, desc.layers, desc.mips,
                                sampled->info.num_bits, sampled->from_cpu ? "t" : "",
                                sampled == entry ? "=stored" : "",
                                sampled->write_version.load());
            if (sampled->info.num_bits != 32 || sampled->info.props.is_depth) {
                continue;
            }
            for (u32 mip = 0; mip < std::min<u32>(std::max<u32>(desc.mips, 1), 8); ++mip) {
                copy_texels(*sampled, section * 2 + (i * 8 + mip) * 24, mip,
                            std::min<u32>(std::max<u32>(desc.layers, 1), 6), true);
            }
        }
    } else if (dispatch_probes.empty() || dispatch_probes.back().tick != 0 ||
               dispatch_probes.back().address != entry->info.guest_address) {
        return;
    }
    GlobalBarrier();
    const u64 base = after ? section : 0;
    copy_texels(*entry, base, 0, layers, false);
    for (u32 mip = 1; mip < mips; ++mip) {
        copy_texels(*entry, base + (u64(layers) + u64(mip - 1) * few) * 4, mip, few, false);
    }
    GlobalBarrier();
    if (after) {
        dispatch_probes.back().tick = r.scheduler.CurrentTick() + 1;
    }
}

void NativeRenderer::ReadDispatchProbes() {
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = dispatch_probes.begin(); it != dispatch_probes.end();) {
        if (it->tick == 0 || it->tick > done) {
            ++it;
            continue;
        }
        it->buffer->Invalidate();
        const u32* words = reinterpret_cast<const u32*>(it->buffer->Mapped());
        const u32 few = std::min<u32>(it->layers, 8);
        const u64 count_per = u64(it->layers) + u64(it->mips - 1) * few;
        std::string changed, nonzero;
        u32 total = 0;
        for (u32 mip = 0; mip < it->mips; ++mip) {
            const u32 n = mip == 0 ? it->layers : few;
            const u64 at = mip == 0 ? 0 : u64(it->layers) + u64(mip - 1) * few;
            u32 count = 0, nz = 0;
            for (u32 l = 0; l < n; ++l) {
                const u32 before = words[at + l];
                const u32 now = words[count_per + at + l];
                nz += now != 0;
                if (before != now) {
                    if (++total <= 10) {
                        changed += fmt::format(" m{}l{}:{:x}->{:x}", mip, l, before, now);
                    }
                    ++count;
                }
            }
            nonzero += fmt::format(" m{}:{}/{}{}", mip, nz, n,
                                   count ? fmt::format("(+{})", count) : std::string{});
        }
        std::string sampled;
        for (u32 i = 0; i < 2; ++i) {
            sampled += " /";
            for (u32 mip = 0; mip < 8; ++mip) {
                const u32* texels = words + count_per * 2 + (i * 8 + mip) * 6;
                if (texels[0] == 0xeeeeeeee) {
                    break;
                }
                u32 nz = 0;
                for (u32 l = 0; l < 6; ++l) {
                    nz += texels[l] != 0 && texels[l] != 0xeeeeeeee;
                }
                sampled += fmt::format(" m{}:{}", mip, nz);
            }
        }
        LOG_WARNING(Render_Vulkan, "DIAG-175: dispatch into {:#x} ({} layers, {} mips): {} "
                    "changed{} | non-zero after{} | sampled non-zero{} | {}", it->address,
                    it->layers, it->mips, total, changed, nonzero, sampled, it->what);
        it = dispatch_probes.erase(it);
    }
}

void NativeRenderer::ProbeRefresh(const Native::ImageEntry& texture, const std::string& what) {
    static const std::pair<u32, u32> size = [] {
        std::pair<u32, u32> wh{};
        if (const char* env = Common::DiagEnv("RYGT_DIAG_REFRESH_READ"); env && *env) {
            std::sscanf(env, "%ux%u", &wh.first, &wh.second);
        }
        return wh;
    }();
    if (size.first == 0 || texture.info.size.width != size.first ||
        texture.info.size.height != size.second || texture.info.num_bits != 32 ||
        refresh_probes.size() >= 64) {
        return;
    }
    const u64 bytes = u64(size.first) * size.second * 4;
    auto buffer = device->CreateBuffer({.size = bytes,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::Readback,
                                        .name = "refresh probe"});
    if (!buffer || !buffer->Mapped()) {
        return;
    }
    GlobalBarrier();
    stream.cmd.copyImageToBuffer(
        vk::Image{Handle<VkImage_T>(texture.image->NativeHandle())}, vk::ImageLayout::eGeneral,
        vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
        vk::BufferImageCopy{.bufferOffset = 0,
                            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                              .mipLevel = 0,
                                              .baseArrayLayer = 0,
                                              .layerCount = 1},
                            .imageOffset = {0, 0, 0},
                            .imageExtent = {size.first, size.second, 1}});
    GlobalBarrier();
    refresh_probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, frame,
                              fmt::format("{:#x} {}x{} {} <-{}", texture.info.guest_address,
                                          size.first, size.second,
                                          vk::to_string(texture.info.pixel_format), what)});
}

void NativeRenderer::CompareRegion(VAddr base, u64 size, u32 stride, vk::Buffer buffer,
                                   u64 offset, u64 program, bool always) {
    // DIAG-211 (diagnosis, RYGT_DIAG_REGION_COMPARE=1 with SHADGT_WATCH_SHADERS): each distinct
    // region buffer a watched shader reads (up to 24 a frame), copied from the region right
    // where the draw reads it, compared with guest memory as it was when the draw was recorded.
    static const bool on = Common::DiagEnv("RYGT_DIAG_REGION_COMPARE") != nullptr;
    // RYGT_DIAG_REGION_COMPARE=4: indirect arguments only (nothing else queued to crowd them out).
    static const bool args_only = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_REGION_COMPARE");
        return env && env[0] == '4';
    }();
    if (!on || size == 0 || size > 512_KB || (args_only && !always) ||
        region_compares.size() >= (always ? 2048u : 256u)) {
        return;
    }
    static u64 seen_frame = ~u64{0};
    static std::vector<VAddr> seen;
    if (seen_frame != frame) {
        seen_frame = frame;
        seen.clear();
    }
    // RYGT_DIAG_REGION_COMPARE=3: every 10th frame, records of 144 bytes (GT7's crowd
    // instances) are copied at every draw that reads them, each compared with the frame's first.
    static const bool every_draw = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_REGION_COMPARE");
        return env && env[0] == '3';
    }();
    const bool every = every_draw && stride == 144 && frame % 10 == 0;
    if (!always && !every &&
        (seen.size() >= 24 || std::ranges::find(seen, base) != seen.end())) {
        return;
    }
    if (!always && !every) {
        seen.push_back(base);
    }
    std::vector<u8> cpu(size);
    if (!Core::RytyMemory::GuardedCopy(cpu.data(), reinterpret_cast<const void*>(base), size)) {
        return;
    }
    auto readback = device->CreateBuffer({.size = size,
                                          .usage = Native::Rhi::BufferUsageStorage |
                                                   Native::Rhi::BufferUsageTransferDst,
                                          .memory = Native::Rhi::MemoryKind::Readback,
                                          .name = "region compare"});
    if (!readback || !readback->Mapped()) {
        return;
    }
    Begin();
    GlobalBarrier(); // the region's copy-ins are done
    stream.cmd.copyBuffer(buffer, vk::Buffer{Handle<VkBuffer_T>(readback->NativeHandle())},
                          vk::BufferCopy{.srcOffset = offset, .dstOffset = 0, .size = size});
    GlobalBarrier();
    // Per page: frames since the CPU last wrote it (FIX-194's cpu_pages), '-' if not known.
    std::string pages;
    if (Region* region = RegionOver(base, size)) {
        for (VAddr page = Common::AlignDown(base, 4096); page < base + size; page += 4096) {
            const auto it = region->cpu_pages.find(page);
            pages += it == region->cpu_pages.end() ? std::string{" -"}
                                                   : fmt::format(" {}", frame - it->second);
        }
    }
    region_compares.push_back({std::move(readback), r.scheduler.CurrentTick() + 1, frame, base,
                               size, std::max<u32>(stride, 16), std::move(cpu),
                               fmt::format("program {:#x}, pages since CPU write:{}", program,
                                           pages),
                               always});
}

void NativeRenderer::CaptureImage(const Native::ImageEntry& entry, u64 at_frame,
                                  const std::string& what) {
    // DIAG-212 (diagnosis): mip 0 layer 0 of a 32-bit picture, copied where the work stands now.
    if (!entry.image || entry.info.num_bits != 32 || captures.size() >= 12) {
        return;
    }
    const auto& desc = entry.image->Desc();
    const u64 bytes = u64(desc.width) * desc.height * 4;
    auto buffer = device->CreateBuffer({.size = bytes,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::Readback,
                                        .name = "frame capture"});
    if (!buffer || !buffer->Mapped()) {
        return;
    }
    Begin();
    GlobalBarrier();
    stream.cmd.copyImageToBuffer(
        vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())}, vk::ImageLayout::eGeneral,
        vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
        vk::BufferImageCopy{.bufferOffset = 0,
                            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                              .mipLevel = 0,
                                              .baseArrayLayer = 0,
                                              .layerCount = 1},
                            .imageOffset = {0, 0, 0},
                            .imageExtent = {desc.width, desc.height, 1}});
    GlobalBarrier();
    captures.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, at_frame, what,
                        u32(desc.width), u32(desc.height), static_cast<u32>(desc.format)});
}

void NativeRenderer::WriteCaptures() {
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = captures.begin(); it != captures.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        it->buffer->Invalidate();
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                          fmt::format("cap_{}_{}_{}x{}_{}.bin", it->frame, it->what, it->width,
                                      it->height,
                                      vk::to_string(static_cast<vk::Format>(it->format)));
        if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
            std::fwrite(it->buffer->Mapped(), 1, it->buffer->Size(), file);
            std::fclose(file);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-212: frame {} {} written to {}", it->frame, it->what,
                    path.string());
        it = captures.erase(it);
    }
}

namespace {
// DIAG-237 (diagnosis): flattened dwords of hull shaders as copied when the draw was recorded.
struct StaleFlat {
    u64 program;
    u64 frame;
    std::vector<std::pair<VAddr, u32>> words;
};
std::vector<StaleFlat> g_stale_flats;
} // namespace

void NativeRenderer::NoteStaleFlats(const Shader::Info& stage) {
    // DIAG-237 (diagnosis, RYGT_DIAG_STALE_FLATS=1): at the end of the frame, whether guest memory
    // still holds what hull shaders' flattened constants were copied from when recorded.
    static const bool on = Common::DiagEnv("RYGT_DIAG_STALE_FLATS") != nullptr;
    if (!on || stage.sw_stage != Shader::SwStage::TessellationControl || g_stale_flats.size() >= 64) {
        return;
    }
    const auto& src = stage.flattened_ud_src;
    const auto& flat = stage.flattened_ud_buf;
    StaleFlat note{stage.pgm_hash, frame, {}};
    for (size_t i = 0; i < src.size() && i < flat.size(); ++i) {
        if (src[i] != 0) {
            note.words.emplace_back(src[i], flat[i]);
        }
    }
    g_stale_flats.push_back(std::move(note));
}

void NativeRenderer::CheckStaleFlats() {
    static u32 logged = 0;
    for (const auto& note : g_stale_flats) {
        u32 changed = 0;
        std::string first;
        for (const auto& [address, value] : note.words) {
            u32 now = 0;
            if (Core::RytyMemory::GuardedCopy(&now, reinterpret_cast<const void*>(address), 4) &&
                now != value) {
                if (changed < 6) {
                    first += fmt::format(" {:#x}:{:08x}->{:08x}", address, value, now);
                }
                ++changed;
            }
        }
        if (logged < 300 && (changed != 0 || logged % 50 == 0)) {
            ++logged;
            LOG_WARNING(Render_Vulkan, "DIAG-237: frame {} hs {:#x}: {} of {} flattened dwords "
                        "changed in memory since recorded{}",
                        note.frame, note.program, changed, note.words.size(), first);
        }
    }
    g_stale_flats.clear();
}

void NativeRenderer::CompareFlatSources(const Shader::Info& stage) {
    // DIAG-211 (diagnosis, RYGT_DIAG_REGION_COMPARE=1 with SHADGT_WATCH_SHADERS): every 10th
    // frame, each run of flattened dwords read from inside a region, as the region holds it right
    // where the draw runs, against the dwords the CPU walk copied.
    static const bool on = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_REGION_COMPARE");
        return env != nullptr && env[0] != '4';
    }();
    const auto& src = stage.flattened_ud_src;
    const auto& flat = stage.flattened_ud_buf;
    if (!on || frame % 10 != 0 || src.size() != flat.size()) {
        return;
    }
    for (size_t i = 0; i < src.size() && region_compares.size() < 256;) {
        size_t j = i + 1;
        while (j < src.size() && src[j] != 0 && src[j] == src[j - 1] + 4) {
            ++j;
        }
        const VAddr at = src[i];
        const u64 bytes = u64(j - i) * 4;
        Region* region = at != 0 ? RegionOver(at, bytes) : nullptr;
        if (region && at >= region->begin && at + bytes <= region->end) {
            auto readback = device->CreateBuffer({.size = bytes,
                                                  .usage = Native::Rhi::BufferUsageStorage |
                                                           Native::Rhi::BufferUsageTransferDst,
                                                  .memory = Native::Rhi::MemoryKind::Readback,
                                                  .name = "flat compare"});
            if (readback && readback->Mapped()) {
                Begin();
                GlobalBarrier();
                stream.cmd.copyBuffer(
                    vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                    vk::Buffer{Handle<VkBuffer_T>(readback->NativeHandle())},
                    vk::BufferCopy{.srcOffset = at - region->begin, .dstOffset = 0, .size = bytes});
                GlobalBarrier();
                std::vector<u8> walked(bytes);
                std::memcpy(walked.data(), flat.data() + i, bytes);
                region_compares.push_back(
                    {std::move(readback), r.scheduler.CurrentTick() + 1, frame, at, bytes, 4,
                     std::move(walked),
                     fmt::format("flat dwords {}..{} of program {:#x} (walked, not memory)", i,
                                 j - 1, stage.pgm_hash)});
            }
        }
        i = j;
    }
}

void NativeRenderer::ReadRegionCompares() {
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    static u64 same = 0, differ = 0;
    for (auto it = region_compares.begin(); it != region_compares.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        it->buffer->Invalidate();
        const u8* gpu = static_cast<const u8*>(it->buffer->Mapped());
        const u64 records = it->size / it->stride;
        u64 bad = 0;
        s64 first = -1;
        for (u64 i = 0; i < records; ++i) {
            if (std::memcmp(gpu + i * it->stride, it->cpu.data() + i * it->stride, it->stride) != 0) {
                bad += 1;
                first = first < 0 ? s64(i) : first;
            }
        }
        if (it->always) {
            // Indirect arguments: each draw's (index count, instances, first index, base vertex,
            // first instance) as the GPU read them, '*' where guest memory held other words.
            std::string draws;
            for (u64 i = 0; i < std::min<u64>(records, 16); ++i) {
                std::array<u32, 5> g{}, c{};
                const u64 at = i * it->stride;
                const u64 n = std::min<u64>(20, it->size - at);
                std::memcpy(g.data(), gpu + at, n);
                std::memcpy(c.data(), it->cpu.data() + at, n);
                draws += fmt::format(" [{} {} {} {} {}{}]", g[0], g[1], g[2], g[3], g[4],
                                     g == c ? "" : "*");
            }
            LOG_WARNING(Render_Vulkan, "DIAG-211: frame {} arguments {:#x} ({} draws, {} differ "
                        "from memory):{}; {}", it->frame, it->address, records, bad, draws,
                        it->what);
            it = region_compares.erase(it);
            continue;
        }
        // RYGT_DIAG_REGION_COMPARE=3: each later read of the frame against its first one.
        static const bool every_draw = [] {
            const char* env = Common::DiagEnv("RYGT_DIAG_REGION_COMPARE");
            return env && env[0] == '3';
        }();
        if (every_draw && it->stride == 144 && it->frame % 10 == 0) {
            static std::map<std::pair<u64, VAddr>, std::pair<std::vector<u8>, u32>> firsts;
            auto [first_read, inserted] =
                firsts.try_emplace({it->frame, it->address}, std::vector<u8>{}, 0u);
            ++first_read->second.second;
            if (inserted) {
                first_read->second.first.assign(gpu, gpu + it->size);
            } else if (first_read->second.first.size() == it->size) {
                const u8* base_bytes = first_read->second.first.data();
                u64 moved = 0;
                std::string text;
                for (u64 i = 0; i < records; ++i) {
                    if (std::memcmp(gpu + i * 144, base_bytes + i * 144, 144) == 0) {
                        continue;
                    }
                    if (++moved <= 3) {
                        text += fmt::format(" rec {}:", i);
                        u32 listed = 0;
                        for (u64 d = 0; d < 36 && listed < 6; ++d) {
                            u32 now, then;
                            std::memcpy(&now, gpu + i * 144 + d * 4, 4);
                            std::memcpy(&then, base_bytes + i * 144 + d * 4, 4);
                            if (now != then) {
                                ++listed;
                                text += fmt::format(" d{} {:08x}->{:08x}", d, then, now);
                            }
                        }
                    }
                }
                if (moved != 0) {
                    LOG_WARNING(Render_Vulkan, "DIAG-211: frame {} {:#x} read {} of the frame: {} of "
                                "{} records differ from its first read:{}", it->frame,
                                it->address, first_read->second.second, moved, records, text);
                }
            }
            while (firsts.size() > 64) {
                firsts.erase(firsts.begin());
            }
        }
        // RYGT_DIAG_REGION_COMPARE=2: records of 144 bytes (GT7's crowd instances) as the GPU
        // read them, every 10th frame: dwords 33 and 34 (the SH probe index) and a short hash.
        static const bool dump_records = [] {
            const char* env = Common::DiagEnv("RYGT_DIAG_REGION_COMPARE");
            return env && env[0] == '2';
        }();
        if (dump_records && it->stride == 144 && it->frame % 10 == 0) {
            std::string text;
            for (u64 i = 0; i < records; ++i) {
                u32 d33, d34;
                std::memcpy(&d33, gpu + i * 144 + 33 * 4, 4);
                std::memcpy(&d34, gpu + i * 144 + 34 * 4, 4);
                text += fmt::format(" {:x},{:x},{:04x}", d33, d34,
                                    u32(XXH3_64bits(gpu + i * 144, 128) & 0xffff));
            }
            LOG_WARNING(Render_Vulkan, "DIAG-211: frame {} records {:#x} ({}):{}", it->frame,
                        it->address, records, text);
        }
        if (bad == 0) {
            ++same;
        } else {
            ++differ;
            // Which dwords of the first differing records differ, region against memory.
            std::string dwords;
            u32 shown = 0;
            for (u64 i = u64(first); i < records && shown < 3; ++i) {
                const u64 base_at = i * it->stride;
                if (std::memcmp(gpu + base_at, it->cpu.data() + base_at, it->stride) == 0) {
                    continue;
                }
                ++shown;
                dwords += fmt::format(" rec {}:", i);
                u32 listed = 0;
                for (u64 d = 0; d < it->stride / 4 && listed < 8; ++d) {
                    u32 g, c;
                    std::memcpy(&g, gpu + base_at + d * 4, 4);
                    std::memcpy(&c, it->cpu.data() + base_at + d * 4, 4);
                    if (g != c) {
                        ++listed;
                        dwords += fmt::format(" d{} {:08x}/{:08x}", d, g, c);
                    }
                }
            }
            LOG_WARNING(Render_Vulkan, "DIAG-211: frame {} {:#x} differing dwords (region/memory):{}",
                        it->frame, it->address, dwords);
            const u64 at = u64(first) * it->stride;
            const auto words = [&](const u8* bytes) {
                std::array<u32, 4> w{};
                std::memcpy(w.data(), bytes + at, std::min<u64>(16, it->size - at));
                return fmt::format("{:08x} {:08x} {:08x} {:08x}", w[0], w[1], w[2], w[3]);
            };
            LOG_WARNING(Render_Vulkan, "DIAG-211: frame {} {:#x}+{:#x} stride {}: {} of {} records "
                        "differ, first {} (region {} | memory {}); {}; so far {} buffers same, {} "
                        "differ", it->frame, it->address, it->size, it->stride, bad, records,
                        first, words(gpu), words(it->cpu.data()), it->what, same, differ);
        }
        it = region_compares.erase(it);
    }
}

void NativeRenderer::ReadRefreshProbes() {
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = refresh_probes.begin(); it != refresh_probes.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        it->buffer->Invalidate();
        const u64 count = it->buffer->Size() / 4;
        const float* values = reinterpret_cast<const float*>(it->buffer->Mapped());
        u64 non_finite = 0, huge = 0, zero = 0;
        float lo = std::numeric_limits<float>::max(), hi = 0.0f;
        for (u64 i = 0; i < count; ++i) {
            const float v = values[i];
            if (!std::isfinite(v)) {
                ++non_finite;
                continue;
            }
            huge += std::abs(v) > 1e6f;
            zero += v == 0.0f;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (it->what.starts_with("vertex floats")) {
            // DIAG-208: as floats, 4 to a group.
            std::string text;
            for (u64 i = 0; i < std::min<u64>(count, 512); ++i) {
                text += fmt::format("{}{:g}", i % 4 == 0 ? " |" : " ", values[i]);
            }
            LOG_WARNING(Render_Vulkan, "DIAG-208: frame {} {}:{}", it->frame, it->what, text);
            it = refresh_probes.erase(it);
            continue;
        }
        if (it->what.starts_with("region words")) {
            const u32* words = reinterpret_cast<const u32*>(values);
            std::string text;
            for (u64 i = 0; i < std::min<u64>(it->buffer->Size() / 4, 8); ++i) {
                text += fmt::format(" {:08x}", words[i]);
            }
            LOG_WARNING(Render_Vulkan, "DIAG-187: frame {} {}:{}", it->frame, it->what, text);
            it = refresh_probes.erase(it);
            continue;
        }
        if (it->what.starts_with("GPU picture")) {
            const u16* halves = reinterpret_cast<const u16*>(values);
            std::string text;
            for (u64 i = 0; i < std::min<u64>(it->buffer->Size() / 2, 16); ++i) {
                text += fmt::format(" {:04x}", halves[i]);
            }
            LOG_WARNING(Render_Vulkan, "DIAG-184: frame {} {}: halves{}", it->frame, it->what, text);
            it = refresh_probes.erase(it);
            continue;
        }
        static u32 logged = 0;
        if (++logged <= 400 || huge != 0 || non_finite != 0) {
            LOG_WARNING(Render_Vulkan, "DIAG-179: frame {} {}: min {} max {}, {} huge, {} Inf/NaN, "
                        "{} zero of {}; first {:08x} {:08x} {:08x} {:08x}", it->frame, it->what, lo,
                        hi, huge, non_finite, zero, count,
                        reinterpret_cast<const u32*>(values)[0],
                        reinterpret_cast<const u32*>(values)[1],
                        reinterpret_cast<const u32*>(values)[2],
                        reinterpret_cast<const u32*>(values)[3]);
        }
        it = refresh_probes.erase(it);
    }
}

void NativeRenderer::ProbeVolume() {
    static const VAddr address = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_VOLUME");
        return env && *env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (address == 0) {
        return;
    }
    if (volume_probe && volume_tick != 0 &&
        volume_tick <= r.scheduler.GetWorkSemaphore()->KnownGpuTick()) {
        volume_probe->Invalidate();
        const u16* halves = reinterpret_cast<const u16*>(volume_probe->Mapped());
        const auto to_float = [](u16 h) {
            const u32 e = (h >> 10) & 0x1f, m = h & 0x3ff;
            const float v = e == 0 ? std::ldexp(float(m), -24)
                                   : std::ldexp(1.0f + float(m) / 1024.0f, int(e) - 15);
            return (h & 0x8000) ? -v : v;
        };
        std::string text;
        u64 bad_total = 0;
        for (u32 z = 0; z < volume_depth; ++z) {
            u32 bad = 0, zero = 0;
            float hi = 0.0f;
            for (u32 i = 0; i < volume_width * volume_height; ++i) {
                const u16* texel = halves + (u64(z) * volume_width * volume_height + i) * 4;
                bool texel_bad = false, texel_zero = true;
                for (u32 c = 0; c < 3; ++c) {
                    const u16 h = texel[c];
                    if (((h >> 10) & 0x1f) == 31) {
                        texel_bad = true;
                        continue;
                    }
                    const float v = std::abs(to_float(h));
                    texel_bad |= v > 64.0f;
                    texel_zero &= v == 0.0f;
                    hi = std::max(hi, v);
                }
                bad += texel_bad;
                zero += texel_zero;
            }
            bad_total += bad;
            if (bad != 0 || z % 16 == 0 || z == volume_depth - 1) {
                text += fmt::format(" z{}:{}bad/{}zero/max{:.2f}", z, bad, zero, hi);
            }
        }
        LOG_WARNING(Render_Vulkan, "DIAG-180: frame {} volume {:#x} {}x{}x{}: {} bad texels;{}",
                    frame, address, volume_width, volume_height, volume_depth, bad_total, text);
        // The whole volume, once, to compare runs (user\log\volume_<frame>.bin).
        static bool written = false;
        if (!written && frame >= 2000) {
            written = true;
            const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                              fmt::format("volume_{}.bin", frame);
            if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
                std::fwrite(halves, 1, u64(volume_width) * volume_height * volume_depth * 8, file);
                std::fclose(file);
            }
        }
        volume_tick = 0;
    }
    if (frame % 300 != 0 || volume_tick != 0) {
        return;
    }
    Native::ImageEntry* entry = registry->FindExisting(
        address, [](const Native::ImageEntry& e) { return !e.from_cpu && e.image != nullptr; });
    if (!entry || entry->info.num_bits != 64) {
        return;
    }
    const auto& desc = entry->image->Desc();
    volume_width = desc.width;
    volume_height = desc.height;
    volume_depth = std::max<u32>(desc.depth, 1);
    const u64 bytes = u64(volume_width) * volume_height * volume_depth * 8;
    if (!volume_probe || volume_probe->Size() < bytes) {
        volume_probe = device->CreateBuffer({.size = bytes,
                                             .usage = Native::Rhi::BufferUsageStorage |
                                                      Native::Rhi::BufferUsageTransferDst,
                                             .memory = Native::Rhi::MemoryKind::Readback,
                                             .name = "volume probe"});
    }
    if (!volume_probe || !volume_probe->Mapped()) {
        return;
    }
    Begin();
    GlobalBarrier();
    stream.cmd.copyImageToBuffer(
        vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())}, vk::ImageLayout::eGeneral,
        vk::Buffer{Handle<VkBuffer_T>(volume_probe->NativeHandle())},
        vk::BufferImageCopy{.bufferOffset = 0,
                            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                              .mipLevel = 0,
                                              .baseArrayLayer = 0,
                                              .layerCount = 1},
                            .imageOffset = {0, 0, 0},
                            .imageExtent = {volume_width, volume_height, volume_depth}});
    GlobalBarrier();
    volume_tick = r.scheduler.CurrentTick() + 1;
}

void NativeRenderer::NoteProgramTextures(u64 program, const DrawBindings& b) {
    static const u64 watched = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_PROGRAM_TEXTURES");
        return env && *env ? u64(std::strtoull(env, nullptr, 16)) : u64{0};
    }();
    if (watched == 0 || program != watched) {
        return;
    }
    for (const auto* sampled : b.sampled) {
        if (sampled && sampled->info.guest_size <= 4096 &&
            std::ranges::find(diag_program_textures, sampled) == diag_program_textures.end() &&
            diag_program_textures.size() < 16) {
            diag_program_textures.push_back(const_cast<Native::ImageEntry*>(sampled));
            PrintHistory(sampled->info.guest_address,
                         std::max<u64>(sampled->info.guest_size, 1)); // DIAG-187
            if (const u64 watched_at = HwWatchAddress();
                watched_at >= sampled->info.guest_address &&
                watched_at < sampled->info.guest_address + sampled->info.guest_size) {
                RytyWatchDump(fmt::format("texture {:#x} (frame {})", sampled->info.guest_address,
                                          frame)); // DIAG-193
            }
            // DIAG-195 (diagnosis): the other pictures over its memory, and when they were used.
            registry->ForEachOverlapping(
                sampled->info.guest_address,
                sampled->info.guest_address + std::max<u64>(sampled->info.guest_size, 1),
                [&](Native::ImageEntry& other) {
                    if (&other == sampled) {
                        return;
                    }
                    LOG_WARNING(Render_Vulkan, "DIAG-195: {:#x} 4x4 lies in {} {:#x}+{:#x} {}x{} {} "
                                "mips {} layers {}, made frame {}, last used frame {}",
                                sampled->info.guest_address, other.from_cpu ? "texture" : "target",
                                other.info.guest_address, other.info.guest_size,
                                other.info.size.width, other.info.size.height,
                                vk::to_string(other.info.pixel_format), other.info.resources.levels,
                                other.info.resources.layers, other.made_frame,
                                other.last_use.load());
                });
            LOG_WARNING(Render_Vulkan, "DIAG-192: {:#x}: tile mode {} array mode {} pitch {} "
                        "tiled {} guest size {:#x} mips {} alt tile {}",
                        sampled->info.guest_address, u32(sampled->info.tile_mode),
                        u32(sampled->info.array_mode), sampled->info.pitch,
                        u32(sampled->info.props.is_tiled), sampled->info.guest_size,
                        sampled->info.resources.levels, u32(sampled->info.alt_tile));
            // DIAG-192 (diagnosis): all of its memory, as halves, 32 bytes a line.
            std::array<u16, 16> line{};
            for (u64 at = 0; at < std::min<u64>(sampled->info.guest_size, 1024); at += 32) {
                if (!Core::RytyMemory::GuardedCopy(
                        line.data(),
                        reinterpret_cast<const void*>(sampled->info.guest_address + at), 32)) {
                    break;
                }
                std::string text;
                for (const u16 half : line) {
                    text += fmt::format(" {:04x}", half);
                }
                LOG_WARNING(Render_Vulkan, "DIAG-192: {:#x}+{:03x}:{}", sampled->info.guest_address,
                            at, text);
            }
            LOG_WARNING(Render_Vulkan, "DIAG-184: frame {} program {:#x} samples {} {:#x} {}x{} "
                        "{}; memory {}", frame, program, sampled->from_cpu ? "texture" : "target",
                        sampled->info.guest_address, sampled->info.size.width,
                        sampled->info.size.height, vk::to_string(sampled->info.pixel_format),
                        Native::WatchedBytes(sampled->info.guest_address));
        }
    }
}

static bool WriteHistoryOn() {
    static const bool on = Common::DiagEnv("RYGT_DIAG_WRITE_HISTORY") != nullptr;
    return on;
}

void NativeRenderer::NoteHistory(VAddr begin, u64 size, u32 kind, u64 program) {
    if (!WriteHistoryOn()) {
        return;
    }
    constexpr u64 Capacity = 1ull << 22;
    // RYGT_DIAG_HISTORY_RANGES=<hex address>+<hex size>,...: notes over these are kept from the
    // start too (the ring holds the last few hundred frames).
    const auto& kept_ranges = HistoryRanges();
    std::scoped_lock lock{write_history_mutex};
    if (write_history.empty()) {
        write_history.resize(Capacity);
    }
    write_history[write_history_count++ % Capacity] = {begin, size, frame, program, kind};
    for (const auto& [from, to] : kept_ranges) {
        if (begin < to && begin + std::max<u64>(size, 1) > from && kept_history.size() < 100000) {
            kept_history.push_back({begin, size, frame, program, kind});
            break;
        }
    }
}

const std::vector<std::pair<VAddr, VAddr>>& NativeRenderer::HistoryRanges() {
    static const std::vector<std::pair<VAddr, VAddr>> kept_ranges = [] {
        std::vector<std::pair<VAddr, VAddr>> list;
        const char* env = Common::DiagEnv("RYGT_DIAG_HISTORY_RANGES");
        for (const char* at = env; at && *at;) {
            char* end = nullptr;
            const VAddr from = std::strtoull(at, &end, 16);
            if (end == at || *end != '+') {
                break;
            }
            at = end + 1;
            const u64 length = std::strtoull(at, &end, 16);
            list.emplace_back(from, from + std::max<u64>(length, 1));
            at = *end == ',' ? end + 1 : end;
        }
        return list;
    }();
    return kept_ranges;
}

void NativeRenderer::WatchHistoryRanges() {
    // DIAG-187: the kept ranges' pages are watched every frame, so every CPU write there faults
    // (and is noted, with where it wrote) from the start.
    if (!WriteHistoryOn()) {
        return;
    }
    for (const auto& [from, to] : HistoryRanges()) {
        if (r.IsMapped(from, to - from)) {
            WatchPages(from, to - from);
        }
    }
    // DIAG-214 (diagnosis): a frame dump also prints the kept ranges' histories.
    static u64 printed_at = 0;
    if (r.FrameDumpCollecting() && frame > printed_at + 30) {
        printed_at = frame;
        for (const auto& [from, to] : HistoryRanges()) {
            PrintHistory(from, to - from);
        }
    }
}

void NativeRenderer::PrintHistory(VAddr begin, u64 size) {
    if (!WriteHistoryOn()) {
        return;
    }
    static constexpr std::array<const char*, 15> kinds{
        "draw into target", "depth target",     "image store",         "storage buffer",
        "fill",             "copy",             "inline write",        "CPU write fault",
        "target written back", "region written back", "region made",   "region dropped",
        "mirror made",      "draw came in for target", "draw came in for depth"};
    std::scoped_lock lock{write_history_mutex};
    u32 kept_lines = 0;
    for (const WriteNote& n : kept_history) {
        if (n.begin < begin + size && n.begin + std::max<u64>(n.size, 1) > begin &&
            kept_lines++ < 200) {
            LOG_WARNING(Render_Vulkan, "DIAG-187:   (from the start) frame {} {} {:#x}+{:#x} "
                        "program {:#x}", n.frame, n.kind < kinds.size() ? kinds[n.kind] : "?",
                        n.begin, n.size, n.program);
        }
    }
    const u64 capacity = write_history.size();
    const u64 kept = std::min<u64>(write_history_count, capacity);
    std::vector<const WriteNote*> hits; // newest first
    for (u64 i = 0; i < kept; ++i) {
        const WriteNote& note = write_history[(write_history_count - 1 - i) % capacity];
        if (note.begin < begin + size && note.begin + std::max<u64>(note.size, 1) > begin) {
            hits.push_back(&note);
        }
    }
    LOG_WARNING(Render_Vulkan, "DIAG-187: frame {} {:#x}+{:#x}: {} of the last {} noted writes "
                "over it (since frame {})", frame, begin, size, hits.size(), kept,
                kept ? write_history[(write_history_count - kept) % capacity].frame : 0);
    for (u64 i = 0; i < hits.size(); ++i) {
        if (i >= 40 && i + 20 < hits.size()) {
            continue; // the newest 40 and the oldest 20
        }
        const WriteNote& n = *hits[i];
        LOG_WARNING(Render_Vulkan, "DIAG-187:   frame {} {} {:#x}+{:#x} program {:#x}", n.frame,
                    n.kind < kinds.size() ? kinds[n.kind] : "?", n.begin, n.size, n.program);
    }
}

void NativeRenderer::NoteProgramTextureWrite(VAddr address, u64 size, const char* kind) {
    if (diag_program_textures.empty() || diag185_lines >= 200) {
        return;
    }
    for (const auto* entry : diag_program_textures) {
        const VAddr begin = entry->info.guest_address;
        const VAddr end = begin + std::max<u64>(entry->info.guest_size, 1);
        if (address < end && address + size > begin) {
            ++diag185_lines;
            LOG_WARNING(Render_Vulkan, "DIAG-185: frame {} {} {:#x}+{:#x} over texture {:#x} "
                        "{}x{} ({}); memory {}", frame, kind, address, size, begin,
                        entry->info.size.width, entry->info.size.height, diag_what,
                        Native::WatchedBytes(begin));
            return;
        }
    }
}

void NativeRenderer::NoteProgramTextureWrites(const Targets* targets, const DrawBindings& b) {
    if (diag_program_textures.empty()) {
        return;
    }
    if (targets) {
        for (const auto* color : targets->colors) {
            if (color) {
                NoteProgramTextureWrite(color->info.guest_address,
                                        std::max<u64>(color->info.guest_size, 1),
                                        "draw into target");
            }
        }
        if (targets->depth) {
            NoteProgramTextureWrite(targets->depth->info.guest_address,
                                    std::max<u64>(targets->depth->info.guest_size, 1),
                                    "depth target");
        }
    }
    for (const auto* stored : b.stored) {
        if (stored) {
            NoteProgramTextureWrite(stored->info.guest_address,
                                    std::max<u64>(stored->info.guest_size, 1), "image store");
        }
    }
    for (const auto& [address, size] : b.gpu_writes) {
        NoteProgramTextureWrite(address, size, "storage buffer");
    }
}

void NativeRenderer::ReportProgramTextures() {
    if (diag_program_textures.empty() || frame % 120 != 0) {
        return;
    }
    for (auto* entry : diag_program_textures) {
        // The picture on the GPU too (DIAG-179's read-back, logged with its words).
        if (entry->image && entry->info.num_bits == 64 && refresh_probes.size() < 64) {
            const auto& desc = entry->image->Desc();
            const u64 bytes = u64(desc.width) * desc.height * 8;
            auto buffer = device->CreateBuffer({.size = bytes,
                                                .usage = Native::Rhi::BufferUsageStorage |
                                                         Native::Rhi::BufferUsageTransferDst,
                                                .memory = Native::Rhi::MemoryKind::Readback,
                                                .name = "program texture probe"});
            if (buffer && buffer->Mapped()) {
                Begin();
                GlobalBarrier();
                stream.cmd.copyImageToBuffer(
                    vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())},
                    vk::ImageLayout::eGeneral,
                    vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
                    vk::BufferImageCopy{
                        .bufferOffset = 0,
                        .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                          .mipLevel = 0,
                                          .baseArrayLayer = 0,
                                          .layerCount = 1},
                        .imageOffset = {0, 0, 0},
                        .imageExtent = {desc.width, desc.height, 1}});
                GlobalBarrier();
                refresh_probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, frame,
                                          fmt::format("GPU picture of {:#x} {}x{}",
                                                      entry->info.guest_address, desc.width,
                                                      desc.height)});
            }
        }
        LOG_WARNING(Render_Vulkan, "DIAG-184: frame {} {:#x} {}x{}: memory {}, uploads {}, kind {}, "
                    "needs upload {}, memory as uploaded {}, version {}, last use {}", frame,
                    entry->info.guest_address, entry->info.size.width, entry->info.size.height,
                    Native::WatchedBytes(entry->info.guest_address), entry->total_uploads.load(),
                    entry->upload_kind, entry->needs_upload.load(),
                    entry->content_hash != 0 ? (registry->MemoryUnchanged(*entry) ? "yes" : "NO")
                                             : "no hash",
                    entry->write_version.load(), entry->last_use.load());
    }
}

void NativeRenderer::CheckSmallTextures() {
    static const bool enabled = Common::DiagEnv("RYGT_DIAG_STALE_SMALL") != nullptr;
    if (!enabled || frame % 120 != 0) {
        return;
    }
    u32 checked = 0, stale = 0;
    static u32 logged = 0;
    registry->ForEachOverlapping(0, ~VAddr{0}, [&](Native::ImageEntry& entry) {
        if (!entry.from_cpu || entry.info.guest_size > 64_KB || entry.content_hash == 0 ||
            entry.needs_upload.load(std::memory_order_relaxed) ||
            entry.last_use.load(std::memory_order_relaxed) + 2 < frame) {
            return;
        }
        ++checked;
        if (registry->MemoryUnchanged(entry)) {
            return;
        }
        ++stale;
        if (logged++ < 60) {
            LOG_WARNING(Render_Vulkan, "DIAG-182: frame {} texture {:#x} {}x{} {} ({:#x} bytes) "
                        "differs from its upload, not marked; memory {}", frame,
                        entry.info.guest_address, entry.info.size.width, entry.info.size.height,
                        vk::to_string(entry.info.pixel_format), entry.info.guest_size,
                        Native::WatchedBytes(entry.info.guest_address));
        }
    });
    LOG_INFO(Render_Vulkan, "DIAG-182: frame {}: {} small textures in use checked, {} differ from "
             "their upload unmarked", frame, checked, stale);
}

void NativeRenderer::ReadDrawProbes() {
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = draw_probes.begin(); it != draw_probes.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        const u8* bytes = it->buffer->Mapped();
        std::string text;
        for (u32 i = 0; i < 2; ++i) {
            std::array<double, 4> sum{};
            for (u32 t = 0; t < 256; ++t) {
                if (it->bits == 32) {
                    u32 word;
                    std::memcpy(&word, bytes + (i * 256 + t) * 4, 4);
                    sum[0] += (word & 1023) / 1023.0;
                    sum[1] += ((word >> 10) & 1023) / 1023.0;
                    sum[2] += ((word >> 20) & 1023) / 1023.0;
                    sum[3] += (word >> 30) / 3.0;
                } else {
                    std::array<u16, 4> half;
                    std::memcpy(half.data(), bytes + (i * 256 + t) * 8, 8);
                    for (u32 c = 0; c < 4; ++c) {
                        const u32 e = (half[c] >> 10) & 31, m = half[c] & 1023;
                        const float v = e == 0    ? std::ldexp(float(m), -24)
                                        : e == 31 ? 0.0f
                                                  : std::ldexp(float(m | 1024), int(e) - 25);
                        sum[c] += (half[c] & 0x8000) ? -v : v;
                    }
                }
            }
            u32 first_word = 0;
            std::memcpy(&first_word, bytes + i * 256 * (it->bits / 8), 4);
            text += fmt::format(" y{}: {:.3f} {:.3f} {:.3f} a {:.3f} [{:08x}]", it->y[i],
                                sum[0] / 256, sum[1] / 256, sum[2] / 256, sum[3] / 256,
                                first_word);
            if (it->y[1] == it->y[0]) {
                break;
            }
        }
        const u64 block = 16 * 16 * (it->bits / 8);
        std::string ran;
        for (u32 i = 0; i < 2; ++i) {
            if (it->extra & (1u << i)) {
                ran += fmt::format(" {:016x}", XXH3_64bits(bytes + block * 2 + 512 * i,
                                                            it->lengths[i]));
            }
        }
        if (it->extra & 4) {
            ran += fmt::format(" tex {:016x}", XXH3_64bits(bytes + block * 2 + 1024,
                                                            it->texture_bytes));
        }
        for (u32 i = 0; i < 2; ++i) {
            if (it->vertex_lengths[i] != 0) {
                const u8* at = bytes + block * 2 + 1024 + 4096 + 512 * i;
                std::array<u32, 4> head{};
                std::memcpy(head.data(), at, std::min<u32>(it->vertex_lengths[i], 16));
                ran += fmt::format(" v {:016x} [{:08x} {:08x} {:08x} {:08x}]",
                                   XXH3_64bits(at, it->vertex_lengths[i]), head[0], head[1],
                                   head[2], head[3]);
            }
        }
        if (it->depth) {
            std::array<float, 256> z;
            std::memcpy(z.data(), bytes + block * 2 + 1024 + 4096 + 1024, sizeof(z));
            const auto [lo, hi] = std::ranges::minmax(z);
            ran += fmt::format(" depth {} ({}..{})", z[0], lo, hi);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-170: frame {} draw {} sampling {:#x} {}x{} into {:#x} "
                    "{}x{}:{} | {} | recorded{} ran{}", it->frame, it->draw, it->sampled,
                    it->sampled_width, it->sampled_height, it->target, it->width, it->height,
                    text, it->bound, it->recorded, ran);
        it = draw_probes.erase(it);
    }
}

void NativeRenderer::BindBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs, bool compute,
                                 Shader::Backend::Bindings& binding, DrawBindings& b) {
    const u64 alignment = r.instance.StorageMinAlignment();
    // DIAG-150 (SHADGT_WATCH_SHADERS=<hex,...>): a watched shader's buffers, how they are bound
    // and the first bytes guest memory holds there, and its flattened user data.
    if (Rasterizer::ExposureLogOn()) {
        r.ExposureLogShader(stage, "native"); // FIX-153 (diagnosis)
    }
    static const bool watching = Common::DiagEnv("SHADGT_WATCH_SHADERS") != nullptr;
    const bool watched = watching && IsWatchedShader(stage.pgm_hash);
    static std::atomic<u64> watched_binds{};
    const u64 watched_bind = watched ? ++watched_binds : 0;
    // SHADGT_WATCH_SHADERS_ALL=1: every bind of the watched shaders, not a sample.
    static const bool log_all = Common::DiagEnv("SHADGT_WATCH_SHADERS_ALL") != nullptr;
    const bool log_watched =
        watched && (log_all || watched_bind <= 8 || watched_bind % 300 == 0);
    if (log_watched) {
        std::string flat;
        for (size_t i = 0; i < std::min<size_t>(stage.flattened_ud_buf.size(), 192); ++i) {
            flat += fmt::format(" {:08x}", stage.flattened_ud_buf[i]);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-150: native shader {:#x} use {}: {} dwords flattened:{}",
                    stage.pgm_hash, watched_bind, stage.flattened_ud_buf.size(), flat);
        // FIX-153 (diagnosis): where the walker read each run of flattened dwords.
        std::string sources;
        const auto& src = stage.flattened_ud_src;
        for (size_t i = 0; i < src.size();) {
            size_t j = i + 1;
            while (j < src.size() && src[j] != 0 && src[j] == src[j - 1] + 4) {
                ++j;
            }
            if (src[i] != 0) {
                sources += fmt::format(" [{}..{}] {:#x}", i, j - 1, src[i]);
            }
            i = j;
        }
        LOG_WARNING(Render_Vulkan, "DIAG-150: native shader {:#x} use {} sources:{}",
                    stage.pgm_hash, watched_bind, sources);
    }
    auto* direct = stage.uses_direct ? r.buffer_cache.GetDirectMemory() : nullptr;
    // OPT-015: the stage set's upload memos (flattened user data, clip planes, D3 table).
    const u32 upload_slot = Shader::Backend::StageSet(u32(stage.sw_stage)) * 3;
    // D3's table: page table L1 and zero page, then base, size and valid of each buffer read
    // through it.
    static thread_local std::vector<std::array<u32, 4>> direct_entries;
    direct_entries.assign(stage.buffers.size(), {});
    constexpr u64 GpuAddressLimit = 1ULL << 40;
    u32 index = 0;
    for (const auto& desc : stage.buffers) {
        const u32 buffer_index = index++;
        vk::DescriptorBufferInfo info{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        if (desc.IsSpecial()) {
            switch (desc.buffer_type) {
            case Shader::BufferType::GdsBuffer:
                info = {Handle<VkBuffer_T>(gds->NativeHandle()), 0, gds->Size()};
                break;
            case Shader::BufferType::Flatbuf: {
                const u32 size = static_cast<u32>(stage.flattened_ud_buf.size() * sizeof(u32));
                if (size != 0) {
                    const auto [buffer, offset] = UploadBytes(
                        upload_slot + 0, stage.flattened_ud_buf.data(), size, alignment); // OPT-015
                    info = {buffer, offset, size};
                }
                if (watched) {
                    CompareFlatSources(stage); // DIAG-211 (diagnosis)
                }
                NoteStaleFlats(stage); // DIAG-237 (diagnosis)
                break;
            }
            case Shader::BufferType::ClipPlanes:
                if (regs.clipper_control.user_clip_plane_enable != 0) {
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const auto [buffer, offset] = UploadBytes(upload_slot + 1, planes.data(),
                                                              sizeof(planes), alignment); // OPT-015
                    info = {buffer, offset, sizeof(planes)};
                }
                break;
            case Shader::BufferType::DirectTable: {
                const u32 words = static_cast<u32>(4 + stage.buffers.size() * 4);
                static thread_local std::vector<u32> table_words;
                table_words.assign(words, 0);
                u32* const table = table_words.data();
                if (auto* memory = r.buffer_cache.GetDirectMemory()) {
                    const u64 l1 = memory->L1Address();
                    const u64 zero = memory->ZeroPageAddress();
                    table[0] = u32(l1);
                    table[1] = u32(l1 >> 32);
                    table[2] = u32(zero);
                    table[3] = u32(zero >> 32);
                    for (size_t i = 0; i < stage.buffers.size(); ++i) {
                        std::memcpy(&table[4 + i * 4], direct_entries[i].data(), 16);
                    }
                }
                const auto [buffer, offset] = UploadBytes(upload_slot + 2, table,
                                                          words * sizeof(u32), alignment); // OPT-015
                info = {buffer, offset, words * sizeof(u32)};
                break;
            }
            case Shader::BufferType::SharedMemory: {
                const auto& cs_program = r.liverpool->DrawCsRegs();
                const u64 lds_size =
                    std::max<u64>(u64(cs_program.SharedMemSize()) * cs_program.NumWorkgroups(), 16);
                // GCN LDS is undefined when a workgroup starts: one GPU-only buffer serves every
                // dispatch (they are separated by barriers).
                if (!lds || lds->Size() < lds_size) {
                    lds = device->CreateBuffer({
                        .size = std::max<u64>(Common::AlignUp(lds_size, 1_MB), 4_MB),
                        .usage = Native::Rhi::BufferUsageStorage,
                        .memory = Native::Rhi::MemoryKind::DeviceLocal,
                        .name = "native lds",
                    });
                }
                if (lds) {
                    info = {Handle<VkBuffer_T>(lds->NativeHandle()), 0, lds_size};
                }
                break;
            }
            default: {
                static std::atomic<u32> logged{};
                if (Sample(logged)) {
                    LOG_WARNING(Render_Vulkan,
                                "NAT-101: shader {:#x} uses buffer type {}, not supported in "
                                "native mode; bound empty",
                                stage.pgm_hash, u32(desc.buffer_type));
                }
                break;
            }
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            const u64 base = vsharp.base_address;
            const u64 sharp_size = vsharp.GetSize();
            // FIX-031 / FIX-036 / FIX-046: V#s that are not buffers are bound empty.
            const bool impossible =
                vsharp.num_records != UINT32_MAX &&
                (sharp_size >= GpuAddressLimit || base + sharp_size > GpuAddressLimit ||
                 (sharp_size >= 1_GB && base != 0 && !r.memory->IsValidMapping(base)));
            const bool unmapped =
                !impossible && base != 0 && sharp_size != 0 && !r.IsMappedStart(base);
            const bool bindable = base != 0 && sharp_size != 0 && !impossible && !unmapped;
            u64 size = bindable ? r.memory->ClampRangeSize(base, sharp_size) : 0;
            // FIX-GT7-067: a read-only V# whose records run past the memory the game committed
            // is cut where that memory stops being readable. At a race load GT7 binds its
            // per-frame ring at 0x200800000 from each draw's start to the end of a ~110 MB
            // reservation, 48 MB of it committed. Not importable whole, each such range was
            // copied into upload memory at every bind, zeros past the committed part (up to
            // 64 MB each, 78 GB in 10 s, until "NAT-100: no upload memory" ended the game),
            // or bound empty past 64 MB. The console can't read past the cut either. Only from
            // 4 MB (those were 33-106 MB), so that other bindings don't pay the lookups.
            // RYGT_NATIVE_READABLE_CUT=0: as before.
            static const bool readable_cut = [] {
                const char* env = std::getenv("RYGT_NATIVE_READABLE_CUT");
                return !(env && env[0] == '0');
            }();
            // FIX-GT7-068: a range unreadable from its first page (IsMappedStart takes any address
            // in the guest arena, committed or not) had no prefix to cut to: it kept its size and
            // was copied, all zeros, into upload memory at every bind (32 MB each, as before
            // FIX-GT7-067). It is bound empty instead: the null descriptor reads zeros too, as
            // the copy did, and as the NAT-101 ranges over 64 MB already are.
            bool unreadable = false;
            if (bindable && readable_cut && !desc.is_written && size >= 4_MB) {
                if (auto* const memory = r.buffer_cache.GetDirectMemory();
                    memory && !RegionOver(base, size) && !memory->EnsureImported(base, size)) {
                    const u64 readable =
                        Native::ReadablePrefix(reinterpret_cast<const void*>(base), size);
                    if (readable == 0) {
                        unreadable = true;
                        ++stats.unbound;
                        static std::atomic<u32> empty_logged{};
                        if (Sample(empty_logged)) {
                            LOG_WARNING(Render_Vulkan,
                                        "FIX-GT7-068: read range {:#x}+{:#x} unreadable from its "
                                        "start; bound empty (reads give zeros)",
                                        base, size);
                        }
                    } else if (readable < size) {
                        static std::atomic<u32> cut_logged{};
                        if (Sample(cut_logged)) {
                            LOG_WARNING(Render_Vulkan,
                                        "FIX-GT7-067: read range {:#x}+{:#x} cut to its "
                                        "readable {:#x}",
                                        base, size, readable);
                        }
                        size = readable;
                    }
                }
            }
            if (bindable && !unreadable) {
                // FIX-174 (diagnosis, RYGT_NATIVE_WATCH_BUFFER=<hex address>+<hex size>): the
                // shaders that bind a buffer over that range (the first ones, then now and then).
                static const std::pair<VAddr, VAddr> watch_buffer = [] {
                    const char* env = Common::DiagEnv("RYGT_NATIVE_WATCH_BUFFER");
                    if (!env || !*env) {
                        return std::pair<VAddr, VAddr>{};
                    }
                    char* end = nullptr;
                    const VAddr at = std::strtoull(env, &end, 16);
                    const u64 length = end && *end == '+' ? std::strtoull(end + 1, nullptr, 16) : 1;
                    return std::pair<VAddr, VAddr>{at, at + std::max<u64>(length, 1)};
                }();
                if (watch_buffer.first != 0 && base < watch_buffer.second &&
                    base + size > watch_buffer.first) {
                    static std::atomic<u32> watch_logged{};
                    if (Sample(watch_logged)) {
                        LOG_WARNING(Render_Vulkan,
                                    "FIX-174 watch: {} shader {:#x} binds buffer {} {:#x}+{:#x} "
                                    "(frame {})",
                                    stage.sw_stage == Shader::SwStage::Compute ? "compute" : "draw",
                                    stage.pgm_hash, desc.is_written ? "write" : "read", base,
                                    size, frame);
                    }
                }
                if (vsharp.GetStride() == 144 && !desc.is_written) {
                    b.diag_records = base; // DIAG-214 (diagnosis)
                }
                const auto log_buffer = [&](const char* how) {
                    if (!log_watched) {
                        return;
                    }
                    std::array<u32, 4> head{};
                    const bool read = Core::RytyMemory::GuardedCopy(head.data(),
                                                        reinterpret_cast<const void*>(base),
                                                        std::min<u64>(size, sizeof(head)));
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-150: frame {} native shader {:#x} use {} buffer {} {} "
                                "{:#x}+{:#x} stride {} records {:#x}: {}; memory {}",
                                frame, stage.pgm_hash, watched_bind, buffer_index,
                                desc.is_written ? "write" : "read", base, size, vsharp.GetStride(),
                                vsharp.num_records, how,
                                read ? fmt::format("{:08x} {:08x} {:08x} {:08x}", head[0], head[1],
                                                   head[2], head[3])
                                     : std::string{"unreadable"});
                };
                static const u64 mirror_storage = [] {
                    const char* env = std::getenv("RYGT_NATIVE_MIRROR_STORAGE");
                    return env && *env ? std::strtoull(env, nullptr, 10) * 1_MB : u64{64} * 1_MB;
                }();
                const bool mirrored_read = !desc.is_written && size <= mirror_storage &&
                                           size > SnapshotBelow() &&
                                           vsharp.num_records != UINT32_MAX;
                bool mirror_bound = mirrored_read;
                std::optional<std::pair<vk::Buffer, u64>> in_view;
                if (mirrored_read && !RegionOver(base, size) && !MirrorHolding(base, size)) {
                    std::ignore = MirrorBuffer(base, size);
                    mirror_bound = false;
                    if (direct && direct->EnsureImported(base, size)) {
                        if (const auto view = direct->FindBuffer(base, size)) {
                            in_view = std::pair{view->first->Handle(), view->second};
                            last_bound = Bound::InPlace;
                        }
                    }
                }
                if (direct && desc.IsDirectEligible() && size < (u64{1} << 32) &&
                    !mirror_bound && !in_view &&
                    (desc.is_written || size > SnapshotBelow()) && // FIX-170
                    !RegionOver(base, size) && direct->EnsureImported(base, size)) {
                    // D3: read in place through the page table; nothing bound.
                    direct_entries[buffer_index] = {u32(base), u32(base >> 32), u32(size), 1u};
                    if (InPlaceSampled()) {
                        in_place.Note(2, base, size); // NAT-142
                    }
                    log_buffer("in place through the page table (D3)");
                } else if (const auto found = in_view ? in_view : GuestBuffer(base, size,
                                                                            desc.is_written,
                                                                            mirror_bound)) {
                    log_buffer(last_bound == Bound::Region   ? "region"
                               : last_bound == Bound::Mirror ? "mirror"
                                                             : "in place");
                    if (watched && last_bound == Bound::Region && !desc.is_written) {
                        CompareRegion(base, size, vsharp.GetStride(), found->first, found->second,
                                      stage.pgm_hash); // DIAG-211 (diagnosis)
                    }
                    const u64 aligned = Common::AlignDown(found->second, alignment);
                    const u64 adjust = found->second - aligned;
                    b.push.AddOffset(binding.buffer, static_cast<u32>(adjust));
                    info = {found->first, aligned, size + adjust};
                    b.writes_buffers |= desc.is_written;
                    if (InPlaceSampled()) {
                        in_place.Note(BoundKind(desc.is_written ? 4 : 3), base, size); // NAT-142
                    }
                    if (desc.is_written) {
                        b.gpu_writes.emplace_back(base, size); // NAT-119
                    }
                }
            }
        }
        b.buffers.push_back(info);
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
        b.writes.push_back(vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &b.buffers.back(),
        });
        ++binding.buffer;
    }
    (void)compute;
}

size_t NativeRenderer::TextureMemoSlot(const std::array<u64, 4>& raw_sharp, u32 memo_flags) const {
    // PERF-GT7-013 (C): a lighter hash than XXH3 (the slot is checked against the whole key).
    const u64 hash = NextOn() ? MemoHash(raw_sharp.data(), raw_sharp.size(), memo_flags)
                              : XXH3_64bits_withSeed(raw_sharp.data(), sizeof(raw_sharp), memo_flags);
    return hash & (texture_memos.size() - 1);
}

void NativeRenderer::BindTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                                  Shader::Backend::Bindings& binding, DrawBindings& b) {
    if (NextOn() && stage.images.size() > 1 && !r.FrameDumpCollecting()) {
        // PERF-GT7-014 (C): the stage's texture memo slots are fetched into the cache first;
        // each slot's first touch was the costliest line of BindTextures (a 1.4 MB table).
        for (const auto& image_desc : stage.images) {
            const auto tsharp = image_desc.GetSharp(stage);
            if (tsharp.Address() == 0) {
                continue;
            }
            std::array<u64, 4> raw_sharp{};
            std::memcpy(raw_sharp.data(), &tsharp, sizeof(raw_sharp));
            const u32 memo_flags = u32(image_desc.is_written) | (u32(image_desc.is_array) << 1) |
                                   (u32(image_desc.is_depth) << 2) |
                                   (u32(image_desc.mip_fallback_mode) << 3) |
                                   ((image_desc.constant_mip_index & 0xff) << 6) |
                                   (image_desc.NumBindings(stage) << 16);
            const char* line =
                reinterpret_cast<const char*>(&texture_memos[TextureMemoSlot(raw_sharp, memo_flags)]);
            _mm_prefetch(line, _MM_HINT_T0);
            _mm_prefetch(line + 64, _MM_HINT_T0);
        }
    }
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        {
            // DIAG-186 (diagnosis, with RYGT_DIAG_PROGRAM_TEXTURES=<hash>): where that program's
            // small textures' descriptors were read (guest address of the T#'s first dword), and
            // whether GPU work writes there (a region), every 120 frames.
            static const u64 watched = [] {
                const char* env = Common::DiagEnv("RYGT_DIAG_PROGRAM_TEXTURES");
                return env && *env ? u64(std::strtoull(env, nullptr, 16)) : u64{0};
            }();
            if (watched != 0 && stage.pgm_hash == watched && frame % 120 == 0 &&
                u64(tsharp.width + 1) * (tsharp.height + 1) <= 64) {
                const u32 at = image_desc.sharp_fetch.offsets[0];
                const u64 src = at < stage.flattened_ud_src.size() ? stage.flattened_ud_src[at] : 0;
                const bool region = src != 0 && RegionOver(src, 32) != nullptr;
                LOG_WARNING(Render_Vulkan, "DIAG-186: frame {} program {:#x} T# {:#x} {}x{} from "
                            "flat dword {} read at {:#x} ({}); T# {:08x} {:08x} {:08x} {:08x}",
                            frame, stage.pgm_hash, tsharp.Address(), tsharp.width + 1,
                            tsharp.height + 1, at, src,
                            src == 0 ? "user data" : region ? "in a GPU-written region" : "guest memory",
                            reinterpret_cast<const u32*>(&tsharp)[0],
                            reinterpret_cast<const u32*>(&tsharp)[1],
                            reinterpret_cast<const u32*>(&tsharp)[2],
                            reinterpret_cast<const u32*>(&tsharp)[3]);
                // DIAG-187: what wrote the descriptor there, and what the GPU holds there.
                static u32 histories = 0;
                if (src != 0 && histories < 4) {
                    ++histories;
                    PrintHistory(src, 32);
                    Region* over = RegionOver(src, 32);
                    if (over && src >= over->begin && src + 32 <= over->end &&
                        refresh_probes.size() < 64) {
                        auto buffer = device->CreateBuffer(
                            {.size = 32,
                             .usage = Native::Rhi::BufferUsageStorage |
                                      Native::Rhi::BufferUsageTransferDst,
                             .memory = Native::Rhi::MemoryKind::Readback,
                             .name = "descriptor probe"});
                        if (buffer && buffer->Mapped()) {
                            Begin();
                            GlobalBarrier();
                            stream.cmd.copyBuffer(
                                vk::Buffer{Handle<VkBuffer_T>(over->buffer->NativeHandle())},
                                vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
                                vk::BufferCopy{.srcOffset = src - over->begin,
                                               .dstOffset = 0,
                                               .size = 32});
                            GlobalBarrier();
                            refresh_probes.push_back(
                                {std::move(buffer), r.scheduler.CurrentTick() + 1, frame,
                                 fmt::format("region words of the T# read at {:#x}", src)});
                        }
                    }
                }
            }
        }
        const bool cube_sampling =
            image_desc.post_op == Shader::SharpFetchPostOp::NativeCubeSampling;
        const u32 num_bindings = image_desc.NumBindings(stage);
        const u32 first = static_cast<u32>(b.images.size());
        const bool storage = image_desc.is_written;
        const bool compute_stage = stage.sw_stage == Shader::SwStage::Compute; // PERF-GT7-009
        if (compute_stage) {
            ++stats.compute_texture_lookups;
            stats.compute_texture_bindings += num_bindings;
        }
        // FIX-GT7-080: which descriptor of a compute stage each picture is bound to (SameTexelKey).
        const auto note_entry = [&](const Native::ImageEntry* entry) {
            if (!compute_stage) {
                return;
            }
            const size_t index = size_t(&image_desc - stage.images.data());
            if (b.image_entries.size() <= index) {
                b.image_entries.resize(index + 1, nullptr);
            }
            b.image_entries[index] = entry;
        };
        const auto add_write = [&] {
            b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
            b.writes.push_back(vk::WriteDescriptorSet{
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = binding.unified,
                .dstArrayElement = 0,
                .descriptorCount = num_bindings,
                .descriptorType =
                    storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
                .pImageInfo = &b.images[first],
            });
            binding.unified += num_bindings;
        };
        // DIAG-213 (diagnosis, SHADGT_WATCH_SHADERS with RYGT_DIAG_IMAGE_SLOTS=1): every 30th
        // frame, each image slot of a watched shader: its T# and what it was bound to.
        static const bool slot_log = Common::DiagEnv("RYGT_DIAG_IMAGE_SLOTS") != nullptr &&
                                     Common::DiagEnv("SHADGT_WATCH_SHADERS") != nullptr;
        const bool log_slot = slot_log && frame % 30 == 0 && IsWatchedShader(stage.pgm_hash);
        // PERF-GT7-006: the slot's text is only made when it is logged (A, the reference, still
        // makes it for every slot, as before).
        const bool make_text = log_slot || !ReuseOn();
        const auto slot_text = [&](const auto& make_bound) {
            if (!make_text) {
                return;
            }
            const std::string bound = make_bound();
            if (log_slot) {
                LOG_WARNING(Render_Vulkan, "DIAG-213: frame {} program {:#x} slot {} T# {:#x} {}x{} "
                            "fmt {}/{} type {} levels {}-{} layers {}-{}{} -> {}", frame,
                            stage.pgm_hash, first, tsharp.Address(), tsharp.width + 1,
                            tsharp.height + 1, u32(tsharp.GetDataFmt()), u32(tsharp.GetNumberFmt()),
                            u32(tsharp.GetType()), u32(tsharp.base_level), u32(tsharp.last_level),
                            u32(tsharp.base_array), u32(tsharp.last_array),
                            image_desc.is_written ? " (store)" : "", bound);
            }
        };
        const auto bind_null = [&](u32 reason) {
            for (u32 i = 0; i < num_bindings; ++i) {
                b.images.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral});
            }
            ++stats.null_images;
            ++stats.null_reasons[reason];
            slot_text([&] { return fmt::format("NULL (reason {})", reason); }); // DIAG-213
            add_write();
        };
        // PERF-GT7-006: an empty T# (no address or format: ~0.8 a draw in GT7) is bound empty
        // before the memo is hashed; the memo never holds one.
        if (ReuseOn() && (tsharp.Address() == 0 ||
                          tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid ||
                          !magic_enum::enum_contains(tsharp.GetDataFmt()) ||
                          !magic_enum::enum_contains(tsharp.GetNumberFmt()))) {
            ++stats.texture_lookups;
            bind_null(0);
            continue;
        }
        // OPT-002: the picture and views this T# resolved to last time.
        ++stats.texture_lookups;
        TextureMemo* memo = nullptr;
        std::array<u64, 4> raw_sharp{};
        static_assert(sizeof(tsharp) == sizeof(raw_sharp));
        std::memcpy(raw_sharp.data(), &tsharp, sizeof(raw_sharp));
        const u32 memo_flags = u32(storage) | (u32(image_desc.is_array) << 1) |
                               (u32(image_desc.is_depth) << 2) |
                               (u32(image_desc.mip_fallback_mode) << 3) |
                               ((image_desc.constant_mip_index & 0xff) << 6) |
                               (num_bindings << 16) | (u32(cube_sampling) << 31);
        if (num_bindings <= TextureMemoViews && stage.pgm_hash != 0xaa3822a3 &&
            !r.FrameDumpCollecting()) {
            memo = &texture_memos[TextureMemoSlot(raw_sharp, memo_flags)];
            // PERF-GT7-047: a picture made or dropped elsewhere leaves this one as it was:
            // up to ~300,000 lookups per 10 s on GT7's rain race went through Find again.
            // RYGT_NATIVE_TEXTURE_REVALIDATE=0: any change drops every memo, as before.
            if (memo->entry && memo->flags == memo_flags && memo->sharp == raw_sharp &&
                memo->generation != registry->Generation() + unmap_generation) {
                static const bool revalidate = [] {
                    const char* env = std::getenv("RYGT_NATIVE_TEXTURE_REVALIDATE");
                    return !(env && env[0] == '0');
                }();
                const u64 registry_now = registry->Generation();
                const u64 unmap_now = unmap_generation.load(std::memory_order_acquire);
                const std::pair<VAddr, VAddr> range{memo->range_begin, memo->range_end};
                if (revalidate && memo->unmap_generation == unmap_now &&
                    registry_now - memo->registry_generation <= 256 &&
                    registry->Untouched(memo->registry_generation, {&range, 1})) {
                    memo->generation = registry_now + unmap_now;
                    memo->registry_generation = registry_now;
                }
            }
            if (memo->entry && memo->flags == memo_flags && memo->sharp == raw_sharp &&
                memo->generation == registry->Generation() + unmap_generation) {
                Native::ImageEntry* entry = memo->entry;
                entry->last_use.store(frame, std::memory_order_relaxed);
                if (Native::WatchedTexture(entry->info.guest_address,
                                           entry->info.guest_address + 1)) {
                    WatchBind(*entry, tsharp, storage, true, stage.pgm_hash); // FIX-157
                }
                SeqBind(*entry, tsharp, storage, stage.pgm_hash); // DIAG-208 (diagnosis)
                ApplyRewrite(*entry); // NAT-120
                SyncDepthTwin(*entry); // FIX-212
                if (!storage) {
                    if (entry->from_cpu && registry->RewrittenSmall(*entry)) {
                        ++stats.small_rewrites; // FIX-202
                    }
                    if (entry->from_cpu) {
                        registry->LayersRewritten(*entry, frame); // FIX-221
                    }
                    RefreshSources(*entry); // NAT-104 (FIX-197: targets too)
                } else if (PartSyncOn()) {
                    RefreshSources(*entry); // FIX-210: a store counts as writing all of it
                }
                for (u32 i = 0; i < num_bindings; ++i) {
                    b.images.push_back({VK_NULL_HANDLE, memo->views[i], vk::ImageLayout::eGeneral});
                }
                (storage ? b.stored : b.sampled).push_back(entry);
                note_entry(entry); // FIX-GT7-080
                slot_text([&] {
                    return fmt::format("{} {:#x} {}x{} v{} (memo)",
                                       entry->from_cpu ? "texture" : "target",
                                       entry->info.guest_address, entry->info.size.width,
                                       entry->info.size.height, entry->write_version.load());
                }); // DIAG-213
                add_write();
                ++stats.texture_memo_hits;
                stats.compute_texture_hits += compute_stage ? 1 : 0;
                continue;
            }
        }
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            bind_null(0);
            continue;
        }
        if (!r.memory->IsValidGpuMapping(tsharp.Address(), 0)) {
            bind_null(1);
            continue;
        }
        if (LiverpoolToVK::TrySurfaceFormat(data_fmt, num_fmt) == vk::Format::eUndefined) {
            bind_null(2);
            continue;
        }
        if (const auto geometry_error =
                VideoCore::CheckImageDescriptorGeometry(tsharp, r.instance.GetImageLimits());
            geometry_error != VideoCore::ImageDescriptorGeometryError::None) {
            // FIX-GT7-076 (diagnosis): the first rejected sample counts, to tell a garbage T#
            // from a real 16-sample texture.
            if (geometry_error == VideoCore::ImageDescriptorGeometryError::SampleCount) {
                static std::atomic<u32> sample_rejects{0};
                if (sample_rejects.fetch_add(1, std::memory_order_relaxed) < 8) {
                    std::array<u64, 4> words{};
                    std::memcpy(words.data(), &tsharp, sizeof(words));
                    LOG_WARNING(Render_Vulkan,
                                "FIX-GT7-076: T# with {} samples bound empty, shader {:#x}, "
                                "address {:#x}, {}x{} type {} tile {} fmt {}/{}, words {:016x} "
                                "{:016x} {:016x} {:016x}",
                                tsharp.NumSamples(), stage.pgm_hash, tsharp.Address(),
                                tsharp.width + 1, tsharp.height + 1, u32(tsharp.GetType()),
                                u32(tsharp.GetTileMode()), u32(tsharp.GetDataFmt()),
                                u32(tsharp.GetNumberFmt()), words[0], words[1], words[2],
                                words[3]);
                }
            }
            bind_null(3);
            continue;
        }
        const VideoCore::TextureCache::ImageDesc desc{tsharp, image_desc};
        // FIX-046: an image must start inside the guest address space. FIX-081: GT Sport's grass
        // shader reads garbage T#s from one slot when the texture is unused; from there only an
        // image the game already made is bound.
        if (!r.IsMappedStart(desc.info.guest_address)) {
            bind_null(4);
            continue;
        }
        if (stage.pgm_hash == 0xaa3822a3 && image_desc.sharp_fetch.offsets[0] == 109 &&
            !registry->HasImageAt(desc.info.guest_address)) {
            bind_null(5);
            continue;
        }
        Native::ImageEntry* entry = registry->Find(desc.info, storage, frame);
        if (!entry) {
            bind_null(6);
            continue;
        }
        if (Native::WatchedTexture(desc.info.guest_address, desc.info.guest_address + 1)) {
            WatchBind(*entry, tsharp, storage, false, stage.pgm_hash); // FIX-157 (diagnosis)
        }
        SeqBind(*entry, tsharp, storage, stage.pgm_hash); // DIAG-208 (diagnosis)
        ApplyRewrite(*entry); // NAT-120 (NAT-133: textures too)
        SyncDepthTwin(*entry); // FIX-212
        if (!storage && r.FrameDumpCollecting()) {
            // FIX-152 (diagnosis): RYGT_FRAME_DUMP_ALL_TEXTURES=1 dumps uploaded ones too.
            static const bool dump_all = [] {
                const char* env = Common::DiagEnv("RYGT_FRAME_DUMP_ALL_TEXTURES");
                return env && env[0] == '1';
            }();
            bool gpu_written = dump_all || !entry->from_cpu;
            if (!gpu_written) {
                std::scoped_lock lock{entry->sources_mutex};
                gpu_written = !entry->sources.empty();
            }
            if (gpu_written) {
                auto key = Rasterizer::FrameDumpTextureKey(desc.info); // NAT-123
                if (std::ranges::find(dump_textures, key, &decltype(dump_textures)::value_type::first) ==
                    dump_textures.end()) {
                    dump_textures.emplace_back(std::move(key), entry);
                }
            }
        }
        if (!storage) {
            if (entry->from_cpu && registry->RewrittenSmall(*entry)) {
                ++stats.small_rewrites; // FIX-202
            }
            if (entry->from_cpu) {
                registry->LayersRewritten(*entry, frame); // FIX-221
            }
            // (part of the textures' time; reported apart; OPT-001: on timed draws only)
            PhaseClock sources_clock{stats.phase_ns[4]};
            RefreshSources(*entry); // NAT-104 (FIX-197: targets too)
        } else if (PartSyncOn()) {
            RefreshSources(*entry); // FIX-210: a store counts as writing all of it
        }
        const auto fallback = image_desc.mip_fallback_mode;
        for (u32 i = 0; i < num_bindings; ++i) {
            auto view_info = desc.view_info;
            if (fallback == Shader::MipStorageFallbackMode::ConstantIndex) {
                view_info.range.base.level += image_desc.constant_mip_index;
                view_info.range.extent.levels = 1;
            } else if (fallback == Shader::MipStorageFallbackMode::DynamicIndex) {
                view_info.range.base.level += i;
                view_info.range.extent.levels = 1;
            }
            const vk::ImageView view =
                View(*entry, view_info, storage        ? ViewUse::Storage
                                        : cube_sampling ? ViewUse::CubeSampled
                                                        : ViewUse::Sampled);
            b.images.push_back({VK_NULL_HANDLE, view, vk::ImageLayout::eGeneral});
        }
        if (memo) {
            // OPT-002: remembered for the next draws reading this T# the same way.
            memo->sharp = raw_sharp;
            memo->flags = memo_flags;
            memo->entry = entry;
            for (u32 i = 0; i < num_bindings; ++i) {
                memo->views[i] = b.images[first + i].imageView;
            }
            memo->registry_generation = registry->Generation();
            memo->unmap_generation = unmap_generation.load(std::memory_order_acquire);
            memo->generation = memo->registry_generation + memo->unmap_generation;
            // PERF-GT7-047: what Find looked at, and the picture it answered with.
            memo->range_begin = std::min(desc.info.guest_address, entry->info.guest_address);
            memo->range_end =
                std::max(desc.info.guest_address + std::max<u64>(desc.info.guest_size, 1),
                         entry->info.guest_address + std::max<u64>(entry->info.guest_size, 1));
        }
        (storage ? b.stored : b.sampled).push_back(entry);
        note_entry(entry); // FIX-GT7-080
        slot_text([&] {
            return fmt::format("{} {:#x} {}x{} v{}", entry->from_cpu ? "texture" : "target",
                               entry->info.guest_address, entry->info.size.width,
                               entry->info.size.height, entry->write_version.load());
        }); // DIAG-213
        add_write();
    }
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                regs.ta_bc_base.Address() == 0)) {
            ssharp = AmdGpu::Sampler{};
        }
        // DIAG-235 (diagnosis, RYGT_DIAG_HS_SAMPLERS=1): hull shader samplers, 40 at most.
        static const bool diag235 = Common::DiagEnv("RYGT_DIAG_HS_SAMPLERS") != nullptr;
        static u32 diag235_lines = 0;
        if (diag235 && stage.sw_stage == Shader::SwStage::TessellationControl && diag235_lines < 40) {
            ++diag235_lines;
            LOG_WARNING(Render_Vulkan, "DIAG-235: hs {:#x} S# unnorm {} mip_filter {} xy_min {} "
                        "xy_mag {} min_lod {} max_lod {} lod_bias {}",
                        stage.pgm_hash, u32(ssharp.force_unnormalized.Value()),
                        u32(ssharp.mip_filter.Value()), u32(ssharp.xy_min_filter.Value()),
                        u32(ssharp.xy_mag_filter.Value()), ssharp.MinLod(), ssharp.MaxLod(),
                        u32(ssharp.lod_bias.Value()));
        }
        b.images.push_back({SamplerFor(ssharp, regs.ta_bc_base, sampler.is_depth), VK_NULL_HANDLE,
                            vk::ImageLayout::eGeneral});
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
        b.writes.push_back(vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &b.images.back(),
        });
    }
}

void NativeRenderer::PushBindings(const Pipeline& pipeline, vk::PipelineBindPoint bind_point,
                                  DrawBindings& b) {
    Stream& s = stream;
    const CmdBuf cmdbuf{s.cmd};
    const vk::PipelineLayout pipeline_layout = pipeline.GetLayout();
    // OPT-014: draws in a row with the same pipeline layout keep what the list bound for it
    // (sets and push constants stay bound across pipeline binds of one layout, and only these
    // draws bind graphics state in the list).
    const bool memo =
        ReuseOn() && bind_point == vk::PipelineBindPoint::eGraphics && !s.wrapped_list;
    if (memo && pipeline_layout != s.memo_layout) {
        ForgetBindings(s);
        s.memo_layout = pipeline_layout;
    }
    const auto stage_flags =
        pipeline.IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    if (memo && s.push_known && std::memcmp(&s.last_push, &b.push, sizeof(b.push)) == 0) {
        ++stats.push_constants_skipped;
    } else {
        cmdbuf.pushConstants(pipeline_layout, stage_flags, 0u, sizeof(b.push), &b.push);
        if (memo) {
            s.last_push = b.push;
            s.push_known = true;
        }
    }
    if (b.writes.empty()) {
        return;
    }
    // SHD-002: each stage's writes go to its own set; one set may be pushed.
    std::array<boost::container::small_vector<vk::WriteDescriptorSet, 32>,
               Shader::Backend::MaxDescriptorSets>
        by_set;
    for (size_t i = 0; i < b.writes.size(); ++i) {
        by_set[i < b.write_sets.size() ? b.write_sets[i] : 0].push_back(b.writes[i]);
    }
    const vk::Device vk_device = r.instance.GetDevice();
    const u64 generation = memo ? ReuseGeneration() : 0;
    static thread_local std::vector<u64> signature;
    for (u32 index = 0; index < by_set.size(); ++index) {
        auto& writes = by_set[index];
        if (writes.empty()) {
            continue;
        }
        SetMemo* set_memo = memo && index < s.set_memos.size() ? &s.set_memos[index] : nullptr;
        if (set_memo) {
            // OPT-014: what the writes bind, descriptor by descriptor.
            signature.clear();
            for (const auto& write : writes) {
                signature.push_back((u64(write.dstBinding) << 32) |
                                    (u64(write.descriptorType) << 8) | write.descriptorCount);
                for (u32 i = 0; i < write.descriptorCount; ++i) {
                    if (write.pBufferInfo) {
                        const auto& info = write.pBufferInfo[i];
                        signature.push_back(
                            reinterpret_cast<u64>(static_cast<VkBuffer>(info.buffer)));
                        signature.push_back(info.offset);
                        signature.push_back(info.range);
                    } else if (write.pImageInfo) {
                        const auto& info = write.pImageInfo[i];
                        signature.push_back(
                            reinterpret_cast<u64>(static_cast<VkSampler>(info.sampler)));
                        signature.push_back(
                            reinterpret_cast<u64>(static_cast<VkImageView>(info.imageView)));
                    }
                }
            }
            if (set_memo->valid && set_memo->generation == generation &&
                set_memo->signature == signature) {
                ++(index == pipeline.PushSet() ? stats.pushes_skipped : stats.sets_reused);
                continue; // still bound
            }
        }
        if (index == pipeline.PushSet()) {
            cmdbuf.pushDescriptorSetKHR(bind_point, pipeline_layout, index, writes);
            if (set_memo) {
                set_memo->valid = true;
                set_memo->generation = generation;
                set_memo->signature.swap(signature);
            }
            continue;
        }
        const vk::DescriptorSetLayout layout = pipeline.SetLayout(index);
        vk::DescriptorSet set{};
        for (u32 attempt = 0; attempt < 2 && !set; ++attempt) {
            if (s.pools.empty() || attempt == 1) {
                s.pools.push_back(TakeDescriptorPool());
            }
            vk::DescriptorSetAllocateInfo alloc{
                .descriptorPool = s.pools.back(),
                .descriptorSetCount = 1,
                .pSetLayouts = &layout,
            };
            VkDescriptorSet raw{};
            if (vk_device.allocateDescriptorSets(&alloc,
                                                 reinterpret_cast<vk::DescriptorSet*>(&raw)) ==
                vk::Result::eSuccess) {
                set = raw;
            }
        }
        if (!set) {
            static std::atomic<u32> logged{};
            if (Sample(logged)) {
                LOG_ERROR(Render_Vulkan, "NAT-100: no descriptor set for a draw");
            }
            continue;
        }
        for (auto& write : writes) {
            write.dstSet = set;
        }
        vk_device.updateDescriptorSets(writes, {});
        cmdbuf.bindDescriptorSets(bind_point, pipeline_layout, index, set, {});
        ++stats.sets_written;
        if (set_memo) {
            set_memo->valid = true;
            set_memo->generation = generation;
            set_memo->signature.swap(signature);
        }
    }
}

bool NativeRenderer::MirrorKind(char kind) {
    // FIX-177 (diagnosis, RYGT_NATIVE_MIRRORS=v or i): only vertex or only index ranges mirrored.
    static const char only = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRRORS");
        return env && (env[0] == 'v' || env[0] == 'i') ? env[0] : '\0';
    }();
    return only == '\0' || only == kind;
}

void NativeRenderer::ResolveVertexBuffers(const GraphicsPipeline& pipeline,
                                          const AmdGpu::Regs& regs, VertexIndexState& v) {
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(v.attributes, v.input_bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    // OPT-004: a draw's attributes mostly read one mesh (~12 V#s a draw in GT Sport races): a
    // range inside the mirror the previous attribute was bound from, while it is there with
    // nothing to copy in, is bound from it without a lookup (what the lookup would return).
    Mirror* near = nullptr;
    u64 near_generation = 0;
    const bool sampled = InPlaceSampled();
    static const bool probing = [] {
        const char* env = Common::DiagEnv("RYGT_PROBE_AT");
        return env && *env; // DIAG-170
    }();
    probe_vertices.clear();
    for (u32 index = 0; index < guest_buffers.size(); ++index) {
        const auto& guest = guest_buffers[index];
        std::optional<std::pair<vk::Buffer, u64>> found;
        u64 size = 0;
        const VAddr base = guest.base_address;
        const u64 sharp_size = guest.GetSize();
        // FIX-178 (diagnosis, RYGT_NATIVE_VERTEX_NEAR=0): every attribute looked up.
        static const bool near_on = [] {
            const char* env = std::getenv("RYGT_NATIVE_VERTEX_NEAR");
            return !(env && env[0] == '0');
        }();
        bool via_near = false; // DIAG-170
        if (near_on && near && !sampled && base != 0 && sharp_size > 0 &&
            near_generation == buffer_generation && base >= near->begin &&
            base + sharp_size <= near->end && near->dirty_pages.empty()) {
            via_near = true;
            size = sharp_size;
            found = std::pair{vk::Buffer{Handle<VkBuffer_T>(near->buffer->NativeHandle())},
                              u64{base - near->begin}};
            ++stats.mirror_binds;
            ++stats.vertex_near;
        } else if (base != 0 && sharp_size > 0 && r.memory->MappedAreaEnd(base, base + 1) != 0) {
            size = r.memory->ClampRangeSize(base, sharp_size);
            found = GuestBuffer(base, size, false, MirrorKind('v')); // FIX-177
            if (sampled) {
                in_place.Note(BoundKind(0), base, size); // NAT-142
            }
            near = found ? last_mirror : nullptr;
            near_generation = buffer_generation;
        }
        v.buffers.push_back(found ? found->first : vk::Buffer{});
        v.offsets.push_back(found ? found->second : 0);
        v.sizes.push_back(found ? size : 0);
        v.strides.push_back(guest.GetStride());
        {
            // DIAG-211 (diagnosis, RYGT_DIAG_REGION_COMPARE with SHADGT_WATCH_SHADERS): a watched
            // shader's vertex buffers, how they are bound, every 10th frame.
            static const bool on = [] {
                const char* env = Common::DiagEnv("RYGT_DIAG_REGION_COMPARE");
                return env != nullptr && env[0] != '4' &&
                       Common::DiagEnv("SHADGT_WATCH_SHADERS") != nullptr;
            }();
            if (on && frame % 10 == 0) {
                bool watched = false;
                for (const auto* stage : pipeline.GetStages()) {
                    watched |= stage && IsWatchedShader(stage->pgm_hash);
                }
                if (watched) {
                    std::array<u32, 4> head{};
                    Core::RytyMemory::GuardedCopy(head.data(), reinterpret_cast<const void*>(base),
                                                  std::min<u64>(sharp_size, sizeof(head)));
                    LOG_WARNING(Render_Vulkan, "DIAG-211: frame {} vertex binding {} {:#x}+{:#x} "
                                "stride {} bound {} (offset {:#x}); memory {:08x} {:08x} {:08x} "
                                "{:08x}", frame, index, base, sharp_size, guest.GetStride(),
                                !found                        ? "nowhere"
                                : via_near                    ? "near mirror"
                                : last_bound == Bound::Region ? "region"
                                : last_bound == Bound::Mirror ? "mirror"
                                                              : "in place",
                                found ? found->second : 0, head[0], head[1], head[2], head[3]);
                    if (found && size <= 512_KB) {
                        CompareRegion(base, size, std::max<u32>(guest.GetStride(), 4u),
                                      found->first, found->second, 0, false); // GPU vs memory
                    }
                }
            }
        }
        if (probing && found && probe_vertices.size() < 2) {
            std::array<u8, 512> bytes{};
            const u64 length = std::min<u64>(size, 512);
            if (Core::RytyMemory::GuardedCopy(bytes.data(), reinterpret_cast<const void*>(base),
                                              length)) {
                std::array<u32, 4> head{};
                std::memcpy(head.data(), bytes.data(), std::min<u64>(length, 16));
                probe_vertices.push_back({base, length, found->first, found->second,
                                          XXH3_64bits(bytes.data(), length),
                                          via_near ? 9u : u32(last_bound),
                                          head});
            }
        }
        Mirror* bound_mirror = found && near && base >= near->begin &&
                                       base + std::max<u64>(size, 1) <= near->end
                                   ? near
                                   : nullptr;
        if (bound_mirror && found->first == vk::Buffer{Handle<VkBuffer_T>(
                                                bound_mirror->buffer->NativeHandle())}) {
            CheckMirrorBind(*bound_mirror, base, size); // FIX-179 (diagnosis)
        }
    }
}

void NativeRenderer::CheckVertexReach(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs,
                                      bool is_indexed, u32 index_offset) {
    // FIX-181 (diagnosis, RYGT_NATIVE_VERTEX_REACH_CHECK=<n>): every n-th draw, the highest
    // vertex it can read (its indices read from guest memory, or its vertex count) against each
    // vertex V#'s end and the mirror bound there: draws reading past a mirror's end got zeros
    // (the tunnel explosions went away with mirrors 256 KB longer).
    static const u32 every = [] {
        const char* env = std::getenv("RYGT_NATIVE_VERTEX_REACH_CHECK");
        return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    if (every == 0 || ++reach_check_counter % every != 0) {
        return;
    }
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    if (guest_buffers.empty()) {
        return;
    }
    const auto [vertex_offset, instance_offset] =
        DrawOffsets(regs, pipeline.GetStage(Shader::SwStage::Vertex), pipeline.GetFetchShader());
    u64 max_index = 0;
    const u32 count = regs.num_indices;
    if (is_indexed) {
        const bool is16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
        const u32 index_size = is16 ? 2 : 4;
        const VAddr address =
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
        const u64 bytes = std::min<u64>(u64(count) * index_size, 1_MB);
        static thread_local std::vector<u8> indices;
        indices.resize(bytes);
        if (bytes == 0 ||
            !Core::RytyMemory::GuardedCopy(indices.data(), reinterpret_cast<const void*>(address),
                                           bytes)) {
            return;
        }
        const bool restart = regs.enable_primitive_restart & 1;
        const u32 restart_index = regs.primitive_restart_index;
        for (u64 i = 0; i < bytes / index_size; ++i) {
            const u32 value = is16 ? u32(reinterpret_cast<const u16*>(indices.data())[i])
                                   : reinterpret_cast<const u32*>(indices.data())[i];
            if (restart && value == (is16 ? (restart_index & 0xffff) : restart_index)) {
                continue;
            }
            max_index = std::max<u64>(max_index, value);
        }
    } else {
        max_index = count == 0 ? 0 : count - 1;
    }
    ++stats.reach_checks;
    for (u32 i = 0; i < guest_buffers.size(); ++i) {
        const auto& guest = guest_buffers[i];
        const u64 stride = guest.GetStride();
        if (stride == 0) {
            continue;
        }
        const bool per_instance = bindings[i].inputRate == vk::VertexInputRate::eInstance;
        const u64 divisor = per_instance ? std::max<u32>(bindings[i].divisor, 1) : 1;
        const u64 instances = std::max<u32>(regs.num_instances.NumInstances(), 1);
        const u64 last = per_instance ? u64(instance_offset) + (instances - 1) / divisor
                                      : u64(vertex_offset) + max_index;
        const VAddr base = guest.base_address;
        const VAddr sharp_end = base + guest.GetSize();
        const VAddr reach = base + (last + 1) * stride;
        if (reach <= sharp_end) {
            continue;
        }
        ++stats.reach_past_sharp;
        const Mirror* mirror = MirrorHolding(base, 1);
        const bool past_mirror = mirror && reach > mirror->end;
        stats.reach_past_mirror += past_mirror ? 1 : 0;
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-181: draw (vs {:#x}) binding {} ({}) V# {:#x} stride {} records {} "
                        "reads to element {} (vertex offset {}, max index {}, instance offset {}, "
                        "{} instances): {:#x} past the V#'s end{}",
                        pipeline.GetStage(Shader::SwStage::Vertex).pgm_hash, i,
                        per_instance ? "per instance" : "per vertex", base, stride,
                        guest.num_records, last, vertex_offset, max_index, instance_offset,
                        instances, reach - sharp_end,
                        mirror ? fmt::format(", mirror {:#x}+{:#x}{}", mirror->begin,
                                             mirror->end - mirror->begin,
                                             past_mirror ? " (past its end too)" : "")
                               : std::string{});
        }
    }
}

void NativeRenderer::CheckMirrorBind(Mirror& mirror, VAddr base, u64 size) {
    // FIX-179 (diagnosis, RYGT_NATIVE_MIRROR_CHECK=<n>): every n-th vertex range bound from a
    // mirror, the mirror's bytes and guest memory's bytes there are copied out side by side right
    // before the draw (the GPU's view of both at that point) and compared once the GPU is done.
    static const u32 every = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRROR_CHECK");
        return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    if (every == 0) {
        return;
    }
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = mirror_checks.begin(); it != mirror_checks.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        const u8* a = it->buffer->Mapped();
        const u8* b = a + it->size;
        u64 first = it->size, last = 0, differ = 0;
        for (u64 i = 0; i < it->size; ++i) {
            if (a[i] != b[i]) {
                first = std::min(first, i);
                last = i;
                ++differ;
            }
        }
        ++stats.mirror_checks;
        if (differ != 0) {
            ++stats.mirror_check_differ;
            static std::atomic<u32> logged{};
            if (Sample(logged)) {
                LOG_WARNING(Render_Vulkan,
                            "FIX-179: vertex range {:#x}+{:#x} (frame {}): mirror {:#x}+{:#x} "
                            "differs from guest memory in {} bytes ({:#x}..{:#x}); mirror made or "
                            "copied in last at frame {}, {} dirty pages then",
                            it->base, it->size, it->frame, it->mirror_begin, it->mirror_size,
                            differ, first, last, it->mirror_used, it->dirty);
            }
        }
        it = mirror_checks.erase(it);
    }
    if (++mirror_check_counter % every != 0 || mirror_checks.size() >= 64) {
        return;
    }
    const u64 bytes = std::min<u64>(size, 16_KB) & ~u64{3};
    if (bytes == 0) {
        return;
    }
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct || !direct->EnsureImported(base, bytes)) {
        return;
    }
    auto buffer = device->CreateBuffer({.size = bytes * 2,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::Readback,
                                        .name = "mirror check"});
    if (!buffer || !buffer->Mapped()) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(buffer->NativeHandle())};
    s.cmd.copyBuffer(vk::Buffer{Handle<VkBuffer_T>(mirror.buffer->NativeHandle())}, dst,
                     vk::BufferCopy{.srcOffset = base - mirror.begin, .dstOffset = 0,
                                    .size = bytes});
    direct->ForEachPart(base, bytes,
                        [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                            u64 part) {
                            s.cmd.copyBuffer(view.Handle(), dst,
                                             vk::BufferCopy{.srcOffset = view_offset,
                                                            .dstOffset = bytes + offset,
                                                            .size = part});
                        });
    s.buffers_written = true;
    mirror_checks.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, bytes, base,
                             frame, mirror.begin, mirror.end - mirror.begin, mirror.used_frame,
                             mirror.dirty_pages.size()});
}

bool NativeRenderer::ResolveIndices(const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                                    bool quad_triangles, VertexIndexState& v) {
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    v.index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    v.index_count = regs.num_indices;
    if (quad_triangles) {
        const auto& corners =
            regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last
                ? QuadLastVertex
                : QuadFirstVertex;
        const u32 count = regs.num_indices;
        const u32 max_indices = count / 4 * 6;
        v.index_count = 0;
        if (max_indices == 0) {
            return true;
        }
        if (!is_indexed) {
            const auto [buffer, offset, data] = Allocate(u64(max_indices) * sizeof(u32), 4);
            auto* out = reinterpret_cast<u32*>(data);
            for (u32 quad = 0; quad + 4 <= count; quad += 4) {
                for (const u32 corner : corners) {
                    *out++ = quad + corner;
                }
            }
            v.index_buffer = buffer;
            v.index_offset = offset;
            v.index_type = vk::IndexType::eUint32;
            v.index_count = max_indices;
            return true;
        }
        const VAddr address =
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
        if (!Native::Readable(reinterpret_cast<const void*>(address), u64(count) * index_size)) {
            return false; // NAT-106: indices the CPU can't read
        }
        const bool restart = (regs.enable_primitive_restart & 1) != 0;
        const auto [buffer, offset, data] = Allocate(u64(max_indices) * index_size, 4);
        if (is_index16) {
            v.index_count = ExpandQuads<u16>(
                reinterpret_cast<const u16*>(address), count,
                restart ? std::optional<u16>{u16(regs.primitive_restart_index)} : std::nullopt,
                corners, reinterpret_cast<u16*>(data));
        } else {
            v.index_count = ExpandQuads<u32>(
                reinterpret_cast<const u32*>(address), count,
                restart ? std::optional<u32>{regs.primitive_restart_index} : std::nullopt,
                corners, reinterpret_cast<u32*>(data));
        }
        v.index_buffer = buffer;
        v.index_offset = offset;
        return true;
    }
    if (!is_indexed) {
        return true;
    }
    const VAddr address =
        regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
    const auto found =
        GuestBuffer(address, u64(regs.num_indices) * index_size, false, MirrorKind('i')); // FIX-177
    if (!found) {
        return false;
    }
    if (InPlaceSampled()) {
        in_place.Note(BoundKind(1), address, u64(regs.num_indices) * index_size); // NAT-142
    }
    v.index_buffer = found->first;
    v.index_offset = found->second;
    return true;
}

void NativeRenderer::RecordVertexIndex(const VertexIndexState& v) {
    Stream& s = stream;
    const CmdBuf cmdbuf{s.cmd};
    const bool dynamic_input = r.instance.IsVertexInputDynamicState();
    // OPT-005: what the list already has is not recorded again (vertex input state and vertex
    // and index bindings outlive pipeline binds; the list's other pipelines are compute).
    const bool known = s.vertex_state_known;
    s.vertex_state_known = true;
    if (dynamic_input) {
        if (!known || !std::ranges::equal(s.input_bindings, v.input_bindings) ||
            !std::ranges::equal(s.attributes, v.attributes)) {
            cmdbuf.setVertexInputEXT(v.input_bindings, v.attributes);
            s.input_bindings.assign(v.input_bindings.begin(), v.input_bindings.end());
            s.attributes.assign(v.attributes.begin(), v.attributes.end());
        } else {
            ++stats.vertex_skipped;
        }
    }
    if (const u32 count = static_cast<u32>(v.buffers.size()); count != 0) {
        const bool same = known && std::ranges::equal(s.vertex_buffers, v.buffers) &&
                          std::ranges::equal(s.vertex_offsets, v.offsets) &&
                          (dynamic_input || (std::ranges::equal(s.vertex_sizes, v.sizes) &&
                                             std::ranges::equal(s.vertex_strides, v.strides)));
        if (!same) {
            if (dynamic_input) {
                cmdbuf.bindVertexBuffers(0, count, v.buffers.data(), v.offsets.data());
            } else {
                cmdbuf.bindVertexBuffers2(0, count, v.buffers.data(), v.offsets.data(),
                                          v.sizes.data(), v.strides.data());
            }
            s.vertex_buffers.assign(v.buffers.begin(), v.buffers.end());
            s.vertex_offsets.assign(v.offsets.begin(), v.offsets.end());
            s.vertex_sizes.assign(v.sizes.begin(), v.sizes.end());
            s.vertex_strides.assign(v.strides.begin(), v.strides.end());
        }
    }
    if (v.index_buffer) {
        if (!known || s.index_buffer != v.index_buffer || s.index_offset != v.index_offset ||
            s.index_type != v.index_type) {
            cmdbuf.bindIndexBuffer(v.index_buffer, v.index_offset, v.index_type);
            s.index_buffer = v.index_buffer;
            s.index_offset = v.index_offset;
            s.index_type = v.index_type;
        }
    }
}

// PERF-GT7-043: dispatches in a row of a listed program run without a barrier between them when
// none of them writes buffers and none samples or stores a picture over memory another touched.
// GT7's probe filter 0x490b6362 runs ~8 such dispatches at a time, each of a few groups: without
// the barriers it takes 3.8 times less GPU time. A listed program must store into its images
// without loading from them, and must not store the same texels twice in a run: their order is no
// longer defined. RYGT_NATIVE_OVERLAP_PROGRAMS=<hex>[,...] sets the list (empty: none).
bool NativeRenderer::RunNeedsBarrier(const Stream& s, const DrawBindings& b, u64 program,
                                     bool recorded, std::optional<u64> texel) const {
    static const std::vector<u64> listed = [] {
        std::vector<u64> list;
        const char* env = std::getenv("RYGT_NATIVE_OVERLAP_PROGRAMS");
        for (const char* at = env ? env : "490b6362"; *at;) {
            char* end = nullptr;
            const u64 value = std::strtoull(at, &end, 16);
            if (end == at) {
                break;
            }
            list.push_back(value);
            at = *end == ',' ? end + 1 : end;
        }
        return list;
    }();
    if ((std::ranges::find(listed, program) == listed.end() && !texel) ||
        s.run_program != program || s.run_commands != s.commands || s.buffers_written.foreign ||
        recorded || !writebacks.empty() || b.writes_buffers || !s.attach_dirty.empty()) {
        return true;
    }
    const auto meets = [](const std::vector<const Native::ImageEntry*>& pictures,
                          const Native::ImageEntry* entry) {
        return std::ranges::any_of(pictures, [&](const Native::ImageEntry* picture) {
            return picture && entry->info.guest_address < picture->info.guest_address +
                                                              picture->info.guest_size &&
                   picture->info.guest_address < entry->info.guest_address + entry->info.guest_size;
        });
    };
    // PERF-GT7-057: a same-texel program (SameTexelKey) may read the picture it stores where the
    // run wrote it, if the run hasn't stored that mip and layer since its last barrier: each of
    // its dispatches reads only the texels it then stores. FIX-GT7-080: the key says no other of
    // its pictures meets the stored one, so the sampled picture let through below is the one the
    // code reads at the texel it stores. FIX-GT7-081: not after a dispatch of the run stored
    // without a key. FIX-GT7-087: nor onto a picture the run sampled elsewhere.
    const Native::ImageEntry* self = texel && b.stored.size() == 1 ? b.stored[0] : nullptr;
    const auto same_picture = [&](const Native::ImageEntry* entry) {
        return self && entry && entry->info.guest_address == self->info.guest_address &&
               entry->info.guest_size == self->info.guest_size;
    };
    const bool texel_new =
        self && s.run_texels.IsNew(self->info.guest_address, self->info.guest_size, *texel);
    // The pictures of a list that meet an entry, but for the picture this dispatch stores.
    const auto meets_other = [&](const std::vector<const Native::ImageEntry*>& pictures,
                                 const Native::ImageEntry* entry) {
        return std::ranges::any_of(pictures, [&](const Native::ImageEntry* picture) {
            return picture && !same_picture(picture) &&
                   entry->info.guest_address <
                       picture->info.guest_address + picture->info.guest_size &&
                   picture->info.guest_address < entry->info.guest_address + entry->info.guest_size;
        });
    };
    for (const auto* entry : b.sampled) {
        if (entry && meets(s.written, entry) &&
            !(texel_new && same_picture(entry) && !meets_other(s.written, entry))) {
            return true; // reads what the run wrote
        }
    }
    for (const auto* entry : b.stored) {
        if (entry && meets(s.read, entry) &&
            !(texel_new && same_picture(entry) && !meets_other(s.read, entry))) {
            return true; // writes what the run read
        }
    }
    return false;
}

// PERF-GT7-057 (RYGT_NATIVE_SAME_TEXEL=0 turns it off): programs that read a picture they store
// only at the texel they store, at the mip and layer words of their user data name. GT7's probe
// filter 0xa8d4b6cd took 30-40% of the GPU's time in races: one run per probe cube, a dispatch per
// mip [65] and face [66] (layer face + 6 * probe [67]), 42 or 48 of them, each reading cs_img3
// with texelFetch at the texel it then stores into cs_img4[mip] (its decompiled code, all four
// variants), and sampling cs_img0 to 2 with textureLod at other coordinates, pictures the run
// doesn't write. Each waited for the one before at a global barrier, with 1 to 256 groups of 8x8
// threads on the whole GPU.
// FIX-GT7-080: the key is the level and layer the two views reach in the picture (Native::
// SameTexelKey), from the pictures and views bound to the fetched and stored descriptors, which the
// list names; any other descriptor that meets the stored picture keeps the barrier.
namespace {
struct SameTexelProgram {
    u64 program;
    u32 mip_word;
    u32 face_word;
    u32 probe_word;
    u32 faces;
    u32 fetch_image; ///< the descriptor read with texelFetch at the texel stored
    u32 store_image;
};
constexpr SameTexelProgram SameTexelPrograms[] = {{0xa8d4b6cd, 65, 66, 67, 6, 3, 4}};

bool IsSameTexelProgram(u64 program) {
    return std::ranges::find(SameTexelPrograms, program, &SameTexelProgram::program) !=
           std::end(SameTexelPrograms);
}
} // namespace

static std::optional<u64> SameTexelKey(const Shader::Info& cs,
                                       std::span<const Native::ImageEntry* const> entries) {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_SAME_TEXEL");
        return !(env && env[0] == '0');
    }();
    if (!on) {
        return std::nullopt;
    }
    const auto it = std::ranges::find(SameTexelPrograms, cs.pgm_hash, &SameTexelProgram::program);
    if (it == std::end(SameTexelPrograms)) {
        return std::nullopt;
    }
    const auto& words = cs.flattened_ud_buf;
    if (std::max({it->mip_word, it->face_word, it->probe_word}) >= words.size() ||
        std::max(it->fetch_image, it->store_image) >= cs.images.size() ||
        cs.images[it->store_image].mip_fallback_mode ==
            Shader::MipStorageFallbackMode::ConstantIndex) {
        return std::nullopt;
    }
    boost::container::small_vector<Native::SameTexelImage, 8> images;
    for (size_t i = 0; i < cs.images.size(); ++i) {
        const auto& desc = cs.images[i];
        const Native::ImageEntry* entry = i < entries.size() ? entries[i] : nullptr;
        Native::SameTexelImage& image = images.emplace_back();
        image.written = desc.is_written;
        if (!entry) {
            continue;
        }
        image.address = entry->info.guest_address;
        image.size = std::max<u64>(entry->info.guest_size, 1);
        if (i != it->fetch_image && i != it->store_image) {
            continue;
        }
        // The view as BindTextures asks for it, and the picture View keeps it in.
        const VideoCore::ImageViewInfo view{desc.GetSharp(cs), desc};
        const auto& picture = entry->image->Desc();
        image.base_level = view.range.base.level;
        image.base_layer = view.range.base.layer;
        image.layers = view.range.extent.layers;
        image.image_mips = std::max<u32>(picture.mips, 1);
        image.image_layers =
            picture.dim == Native::Rhi::ImageDim::D3 ? 1u : std::max<u32>(picture.layers, 1);
        image.level_per_binding =
            desc.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex;
        image.levels = image.level_per_binding ? desc.NumBindings(cs) : view.range.extent.levels;
    }
    const u64 layer = u64(words[it->face_word]) + u64(it->faces) * words[it->probe_word];
    if (layer > ~u32{0}) {
        return std::nullopt;
    }
    return Native::SameTexelKey(images, it->fetch_image, it->store_image, words[it->mip_word],
                                u32(layer));
}

bool NativeRenderer::NeedsBarrierFor(const DrawBindings& b) const {
    const Stream& s = stream;
    if (s.buffers_written || writebacks_due) {
        return true;
    }
    for (const auto* entry : b.sampled) {
        if (std::ranges::find(s.written, entry) != s.written.end()) {
            return true;
        }
    }
    for (const auto* entry : b.stored) {
        if (std::ranges::find(s.written, entry) != s.written.end()) {
            return true;
        }
    }
    return false;
}

void NativeRenderer::Written(const Native::ImageEntry* entry, u64 at) {
    WrittenSubresources(entry, at, nullptr);
}

void NativeRenderer::WrittenRange(const Native::ImageEntry* entry, const DrawnRange& range) {
    WrittenSubresources(entry, 0, &range);
}

bool NativeRenderer::SubVersionsOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_SUB_VERSIONS"); // FIX-208 off: =0
        return !(env && env[0] == '0');
    }();
    return on;
}

u64 NativeRenderer::SubVersion(Native::ImageEntry& entry, u32 mip, u32 layer) {
    const u64 whole = entry.write_version.load(std::memory_order_relaxed);
    if (!SubVersionsOn()) {
        return whole;
    }
    std::scoped_lock lock{entry.sources_mutex};
    const u32 layers = std::max<u32>(entry.info.resources.layers, 1);
    const size_t at = size_t(mip) * layers + layer;
    if (entry.sub_versions.empty() || layer >= layers || at >= entry.sub_versions.size()) {
        return whole;
    }
    return entry.sub_versions[at];
}

void NativeRenderer::WrittenSubresources(const Native::ImageEntry* entry, u64 at,
                                         const DrawnRange* range) {
    if (!entry) {
        return;
    }
    auto* written = const_cast<Native::ImageEntry*>(entry);
    const u64 previous = written->write_version.load(std::memory_order_relaxed);
    const u64 version = at != 0 ? at : ++write_clock;
    written->write_version.store(version, std::memory_order_relaxed);
    // FIX-198: the frames it was drawn in.
    if (const u64 last = written->written_frame.load(std::memory_order_relaxed); last != frame) {
        written->written_frame_before.store(last, std::memory_order_relaxed);
        written->written_frame.store(frame, std::memory_order_relaxed);
    }
    if (Native::WatchedTexture(entry->info.guest_address, entry->info.guest_address + 1)) {
        // FIX-159 (diagnosis, RYGT_FRAME_DUMP_ON_STORE=<seconds>): from that long after start,
        // the first write of a watched picture starts a frame dump there (the frames a
        // time-triggered dump never lands on).
        static const double on_store = [] {
            const char* env = Common::DiagEnv("RYGT_FRAME_DUMP_ON_STORE");
            return env && *env ? std::atof(env) : -1.0;
        }();
        static const auto started = std::chrono::steady_clock::now();
        static bool dumped = false;
        if (on_store >= 0 && !dumped &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
                on_store) {
            dumped = true;
            Native::g_dump_request = true;
        }
        if (Native::WatchSample()) {
            LOG_WARNING(Render_Vulkan, "FIX-157 watch: {} {:#x} {}x{} layers {} written, version {} "
                        "(frame {})", entry->from_cpu ? "texture" : "target",
                        entry->info.guest_address, entry->info.size.width,
                        entry->info.size.height, entry->info.resources.layers,
                        written->write_version.load(), frame);
        }
    }
    // NAT-111: the textures made of it copy its picture in before their next read.
    boost::container::small_vector<Native::ImageEntry*, 4> dependents;
    {
        std::scoped_lock lock{written->sources_mutex};
        // FIX-208: which subresources hold this picture now. GT7 draws each reflection probe's
        // mips into 18-layer targets six layers (one probe's faces) at a time, and its compute
        // passes write other probes' layers of the cube array those targets are mips of: copied
        // in whole, a target's never-drawn layers (zeros) went over the compute output, and the
        // cockpit and the crowd lit by those probes were black two frames in three.
        const u32 mips = std::max<u32>(written->info.resources.levels, 1);
        const u32 layers = std::max<u32>(written->info.resources.layers, 1);
        auto& subs = written->sub_versions;
        const bool eligible = SubVersionsOn() && mips <= 16 && layers <= 64 &&
                              written->info.type != AmdGpu::ImageType::Color3D;
        const bool partial = range && range->mips != 0 &&
                             (range->mip != 0 || range->mips < mips || range->layer != 0 ||
                              range->layers < layers);
        if (eligible && partial) {
            if (subs.empty()) {
                subs.assign(size_t(mips) * layers, previous);
            }
            for (u32 m = range->mip; m < std::min<u32>(u32(range->mip) + range->mips, mips); ++m) {
                const u32 last = std::min<u32>(u32(range->layer) + std::max<u32>(range->layers, 1),
                                               layers);
                for (u32 l = range->layer; l < last; ++l) {
                    subs[size_t(m) * layers + l] = version;
                }
            }
        } else if (!subs.empty()) {
            std::fill(subs.begin(), subs.end(), version);
        }
        for (auto* texture : written->dependents) {
            texture->sources_dirty.store(true, std::memory_order_relaxed);
            dependents.push_back(texture);
        }
    }
    // FIX-207: and the textures made of those (see RefreshSources).
    if (WholeSourcesOn()) {
        for (auto* whole : dependents) {
            std::scoped_lock lock{whole->sources_mutex};
            for (auto* texture : whole->dependents) {
                texture->sources_dirty.store(true, std::memory_order_relaxed);
            }
        }
    }
}

bool NativeRenderer::WholeSourcesOn() {
    // Off unless RYGT_NATIVE_WHOLE_SOURCES=1: it did not change the black-cockpit rate it was
    // written for (64% of frames with and without), and a whole's version counted as its parts'
    // could pick its older mip 0 over a fresher plain target of the same memory.
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_WHOLE_SOURCES");
        return env && env[0] == '1';
    }();
    return on;
}

u64 NativeRenderer::WholeVersion(Native::ImageEntry& source) {
    // FIX-207: a target made of targets (its mips drawn as targets of their own) holds a picture
    // as new as the newest of them.
    u64 version = source.write_version.load(std::memory_order_relaxed);
    if (!WholeSourcesOn()) {
        return version;
    }
    std::scoped_lock lock{source.sources_mutex};
    for (const auto& link : source.sources) {
        if (link.source->image && !link.source->reused.load(std::memory_order_relaxed)) {
            version = std::max(version, link.source->write_version.load(std::memory_order_relaxed));
        }
    }
    return version;
}

void NativeRenderer::RefreshSources(Native::ImageEntry& texture) {
    // NAT-111: only when one of its targets was drawn since the last time.
    const bool traced = Rasterizer::Traced(texture.info.guest_address, texture.info.guest_size);
    if (!texture.sources_dirty.exchange(false, std::memory_order_relaxed)) {
        if (traced) {
            LOG_WARNING(Render_Vulkan, "DIAG-165: texture {:#x} {}x{} not dirty (frame {})",
                        texture.info.guest_address, texture.info.size.width,
                        texture.info.size.height, DebugState.GetFrameNum());
        }
        return;
    }
    // FIX-207: textures made of a target that is itself made of targets. GT7 draws each
    // reflection probe into a 128x128 cube target and its mips 1-7 as targets of their own
    // (FIX-197 copies those into the cube target), and the lighting samples the probe through a
    // texture of another layout over the same memory, made of the cube target only: just the
    // target's mip 0 was copied in, the texture's other mips stayed empty, and the cockpit and
    // the crowd lit by them were black in most frames. Such a whole is brought up to date first,
    // and all its mips are copied. RYGT_NATIVE_WHOLE_SOURCES=0 turns it off.
    static thread_local u32 refresh_depth = 0;
    if (WholeSourcesOn() && refresh_depth == 0) {
        boost::container::small_vector<Native::ImageEntry*, 8> links;
        {
            std::scoped_lock lock{texture.sources_mutex};
            for (const auto& link : texture.sources) {
                if (link.source != &texture) {
                    links.push_back(link.source);
                }
            }
        }
        ++refresh_depth;
        for (auto* source : links) {
            bool whole;
            {
                std::scoped_lock lock{source->sources_mutex};
                whole = !source->sources.empty();
            }
            if (whole) {
                RefreshSources(*source);
            }
        }
        --refresh_depth;
    }
    static thread_local std::vector<Native::SourceLink> sources;
    {
        std::scoped_lock lock{texture.sources_mutex};
        if (texture.sources.empty()) {
            return;
        }
        sources.assign(texture.sources.begin(), texture.sources.end());
    }
    // Per subresource of the texture, the newest picture drawn for it (several layouts take
    // turns in the same memory), when newer than what the texture holds.
    struct Pick {
        Native::ImageEntry* source;
        Native::SourcePlace place;
        u64 version;
        u32 run_layers = 0; ///< FIX-208: layers of a per-layer run (0: as many as fit)
    };
    boost::container::small_vector<Pick, 16> picks;
    auto& held = held_versions[&texture];
    // FIX-200: a texture's memory rewritten by a GPU buffer write counts as its own picture then.
    const u64 foreign =
        texture.from_cpu ? texture.foreign_clock.load(std::memory_order_relaxed) : u64{0};
    const u64 own = std::max(texture.write_version.load(std::memory_order_relaxed), foreign);
    const auto pick_whole = [&](Native::ImageEntry* source, const Native::SourcePlace& place) {
        const u64 version = WholeVersion(*source); // FIX-207
        const u32 key = (u32(place.mip) << 16) | u32(place.layer);
        if (version <= std::max(held[key], own)) {
            return;
        }
        const auto same = std::ranges::find_if(picks, [&](const Pick& pick) {
            return pick.run_layers == 0 && pick.place.mip == place.mip &&
                   pick.place.layer == place.layer;
        });
        if (same == picks.end()) {
            picks.push_back({source, place, version});
        } else if (version > same->version) {
            *same = {source, place, version};
        }
    };
    if (!SubVersionsOn()) {
        for (const auto& [source, place] : sources) {
            if (WholeVersion(*source) == 0 || !source->image ||
                source->reused.load(std::memory_order_relaxed)) {
                continue; // FIX-157: a reused one is stale
            }
            pick_whole(source, place);
        }
    } else {
        // FIX-208: per layer of the texture, the newest picture any source drew there, against
        // what the texture holds in that layer; then runs of layers one source drew at once.
        struct LayerPick {
            u32 key;
            Native::ImageEntry* source;
            Native::SourcePlace place;
            u32 offset;
            u64 version;
        };
        boost::container::small_vector<LayerPick, 64> best;
        std::vector<u64> source_subs;
        for (const auto& [source, place] : sources) {
            if (WholeVersion(*source) == 0 || !source->image ||
                source->reused.load(std::memory_order_relaxed)) {
                continue; // FIX-157: a reused one is stale
            }
            const u32 src_layers = std::max<u32>(source->image->Desc().layers, 1u);
            const u32 dst_layers = std::max<u32>(texture.image->Desc().layers, 1u);
            const u32 count = std::max<u32>(
                1u, std::min<u32>(src_layers - std::min<u32>(u32(place.src_layer), src_layers),
                                  dst_layers - std::min<u32>(u32(place.layer), dst_layers)));
            if (count > 64 || src_layers > 64 || dst_layers > 64) {
                pick_whole(source, place); // big arrays: one version for all, as before
                continue;
            }
            {
                std::scoped_lock lock{source->sources_mutex};
                source_subs = source->sub_versions;
            }
            const u64 source_version = source->write_version.load(std::memory_order_relaxed);
            for (u32 offset = 0; offset < count; ++offset) {
                const u32 src_layer = u32(place.src_layer) + offset;
                const size_t at = size_t(place.src_mip) * src_layers + src_layer;
                const u64 version = at < source_subs.size() ? source_subs[at] : source_version;
                if (version == 0) {
                    continue; // never drawn there
                }
                const u32 layer = u32(place.layer) + offset;
                const u32 key = (u32(place.mip) << 16) | layer;
                const u64 holds =
                    std::max({held[key], SubVersion(texture, u32(place.mip), layer), foreign});
                if (version <= holds) {
                    continue;
                }
                const auto same = std::ranges::find_if(
                    best, [&](const LayerPick& pick) { return pick.key == key; });
                if (same == best.end()) {
                    best.push_back({key, source, place, offset, version});
                } else if (version > same->version) {
                    *same = {key, source, place, offset, version};
                }
            }
        }
        std::ranges::sort(best, [](const LayerPick& a, const LayerPick& b) {
            return std::tie(a.source, a.place.mip, a.place.src_mip, a.place.layer, a.offset) <
                   std::tie(b.source, b.place.mip, b.place.src_mip, b.place.layer, b.offset);
        });
        for (size_t i = 0; i < best.size();) {
            size_t j = i + 1;
            while (j < best.size() && best[j].source == best[i].source &&
                   best[j].place.mip == best[i].place.mip &&
                   best[j].place.src_mip == best[i].place.src_mip &&
                   best[j].place.layer == best[i].place.layer &&
                   best[j].place.src_layer == best[i].place.src_layer &&
                   best[j].offset == best[j - 1].offset + 1 &&
                   best[j].version == best[i].version) {
                ++j;
            }
            Native::SourcePlace place = best[i].place;
            place.layer += s32(best[i].offset);
            place.src_layer += s32(best[i].offset);
            picks.push_back({best[i].source, place, best[i].version, u32(j - i)});
            i = j;
        }
    }
    if (traced) {
        // DIAG-165: each source's version against what the texture holds.
        std::string text;
        for (const auto& [source, place] : sources) {
            text += fmt::format(" {}x{}p{}:{}@{}{}", source->info.size.width,
                                source->info.size.height, source->info.pitch,
                                vk::to_string(source->info.pixel_format),
                                source->write_version.load(std::memory_order_relaxed),
                                source->reused.load(std::memory_order_relaxed) ? "(reused)" : "");
        }
        LOG_WARNING(Render_Vulkan, "DIAG-165: texture {:#x} {}x{}p{}:{} own {} held {} picks {}:{} "
                    "(frame {})", texture.info.guest_address, texture.info.size.width,
                    texture.info.size.height, texture.info.pitch,
                    vk::to_string(texture.info.pixel_format), own, held[0], picks.size(), text,
                    DebugState.GetFrameNum());
    }
    if (picks.empty()) {
        return;
    }
    Stream& s = Begin();
    for (const auto& pick : picks) {
        ApplyRewrite(*pick.source); // NAT-122
    }
    const Native::ImageEntry* const self = &texture;
    // The texture's own upload goes first (PERF-109: it too).
    PrepareImages(std::span<const Native::ImageEntry* const>{&self, 1});
    GlobalBarrier();  // the sources' pictures are written
    const vk::Image dst{Handle<VkImage_T>(texture.image->NativeHandle())};
    for (const auto& pick : picks) {
        const auto& place = pick.place;
        // FIX-GT7-126: into a block-compressed texture from a target of one texel per block, the
        // extent counts the source's texels (Vulkan's rule between compressed and uncompressed
        // formats): the level's blocks.
        const u32 block_dim = place.blocks ? 4u : 1u;
        const u32 width = std::min(
            {place.width,
             (std::max(texture.info.size.width >> place.mip, 1u) + block_dim - 1) / block_dim,
             std::max(pick.source->info.size.width >> place.src_mip, 1u)});
        const u32 height = std::min(
            {place.height,
             (std::max(texture.info.size.height >> place.mip, 1u) + block_dim - 1) / block_dim,
             std::max(pick.source->info.size.height >> place.src_mip, 1u)});
        // NAT-125: a layered target (a cube's faces drawn in one pass, the layer from the
        // vertex shader) gives all its layers; NAT-131: from the layer the texture starts at.
        const u32 layers =
            pick.run_layers != 0
                ? pick.run_layers // FIX-208: the layers this source drew last
                : std::max<u32>(
                      1u, std::min<u32>(std::max<u32>(pick.source->image->Desc().layers, 1u) -
                                            u32(place.src_layer),
                                        std::max<u32>(texture.image->Desc().layers, 1u) -
                                            u32(place.layer)));
        const vk::ImageCopy region{
            .srcSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = u32(place.src_mip),
                            .baseArrayLayer = u32(place.src_layer),
                            .layerCount = layers},
            .srcOffset = {0, 0, 0},
            .dstSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = u32(place.mip),
                            .baseArrayLayer = u32(place.layer),
                            .layerCount = layers},
            .dstOffset = {0, 0, 0},
            .extent = {width, height, 1},
        };
        CmdBuf{s.cmd}.copyImage(vk::Image{Handle<VkImage_T>(pick.source->image->NativeHandle())},
                                vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, region);
        // FIX-207: a whole made of targets gives its other mips too (brought up to date above).
        bool whole = false;
        if (WholeSourcesOn()) {
            std::scoped_lock lock{pick.source->sources_mutex};
            whole = !pick.source->sources.empty();
        }
        const u32 src_mips = std::max<u32>(pick.source->image->Desc().mips, 1u);
        const u32 dst_mips = std::max<u32>(texture.image->Desc().mips, 1u);
        // Not for FIX-GT7-126's blocks: each level has its own target.
        for (u32 m = 1; whole && !place.blocks && u32(place.src_mip) + m < src_mips &&
                        u32(place.mip) + m < dst_mips;
             ++m) {
            vk::ImageCopy mip = region;
            mip.srcSubresource.mipLevel = u32(place.src_mip) + m;
            mip.dstSubresource.mipLevel = u32(place.mip) + m;
            mip.extent.width = std::max(std::min(texture.info.size.width >> (place.mip + m),
                                                 pick.source->info.size.width >> (place.src_mip + m)),
                                        1u);
            mip.extent.height = std::max(std::min(texture.info.size.height >> (place.mip + m),
                                                  pick.source->info.size.height >> (place.src_mip + m)),
                                         1u);
            CmdBuf{s.cmd}.copyImage(
                vk::Image{Handle<VkImage_T>(pick.source->image->NativeHandle())},
                vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, mip);
        }
        if (Rasterizer::Traced(texture.info.guest_address, texture.info.guest_size)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-130: texture {:#x} {}x{} mips {} layers {} <- target {:#x} {}x{} "
                        "layers {} at mip {} layer {} ({} layers, version {})",
                        texture.info.guest_address, texture.info.size.width,
                        texture.info.size.height, texture.info.resources.levels,
                        texture.info.resources.layers, pick.source->info.guest_address,
                        pick.source->info.size.width, pick.source->info.size.height,
                        pick.source->info.resources.layers, place.mip, place.layer, layers,
                        pick.version);
        }
        if (SeqTraced(&texture) || SeqTraced(pick.source)) {
            LOG_WARNING(Render_Vulkan, "DIAG-208: frame {} copy {:#x} mip {} layer {} -> {:#x} "
                        "mip {} layer {} x{} ({}x{}) v{} (own {}, run {})", frame,
                        pick.source->info.guest_address, place.src_mip, place.src_layer,
                        texture.info.guest_address, place.mip, place.layer, layers, width,
                        height, pick.version, own, pick.run_layers); // DIAG-208 (diagnosis)
        }
        if (pick.run_layers != 0) {
            for (u32 i = 0; i < pick.run_layers; ++i) {
                held[(u32(place.mip) << 16) | (u32(place.layer) + i)] = pick.version; // FIX-208
            }
        } else {
            held[(u32(place.mip) << 16) | u32(place.layer)] = pick.version;
        }
        // DIAG-181 (diagnosis): copies into tiny textures.
        static const bool small_links = Common::DiagEnv("RYGT_DIAG_SMALL_LINKS") != nullptr;
        static u32 small_logged = 0;
        if (small_links && texture.info.size.width <= 8 && texture.info.size.height <= 8 &&
            small_logged++ < 300) {
            LOG_WARNING(Render_Vulkan, "DIAG-181: frame {} texture {:#x} {}x{} {} <- target {:#x} "
                        "{}x{} {} (version {}, own {}) at mip {} layer {}", frame,
                        texture.info.guest_address, texture.info.size.width,
                        texture.info.size.height, vk::to_string(texture.info.pixel_format),
                        pick.source->info.guest_address, pick.source->info.size.width,
                        pick.source->info.size.height,
                        vk::to_string(pick.source->info.pixel_format), pick.version, own,
                        place.mip, place.layer);
        }
    }
    s.written.push_back(&texture); // read after the next barrier
    if (traced) {
        std::string what;
        for (const auto& pick : picks) {
            what += fmt::format(" {:#x} {}x{} {} v{}", pick.source->info.guest_address,
                                pick.source->info.size.width, pick.source->info.size.height,
                                vk::to_string(pick.source->info.pixel_format), pick.version);
        }
        ProbeRefresh(texture, what); // DIAG-179 (diagnosis)
    }
    stats.source_copies += picks.size();
    ++s.commands;
}

void NativeRenderer::SyncDepthTwin(Native::ImageEntry& entry) {
    // FIX-212: on the PS4 a depth buffer and a colour picture of its layout at its address are
    // the same memory. GT7 double-buffers its 5120x5120 static shadow map (D16 at 0x2900000000
    // and 0x2920000000): each frame a compute pass reprojects the last map into the other
    // through an R16 storage image, then draws a few tiles into it as a depth buffer and samples
    // it the next frame. The stores went into an R16 image of their own; the depth buffer, made
    // from memory (which never holds what the GPU wrote), stayed zero, so one map in two put the
    // whole scene in shadow: the flickering dark band at Alsace. Vulkan copies no colour image
    // into a depth one, so the picture goes through a buffer.
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_DEPTH_TWINS");
        return !(env && env[0] == '0');
    }();
    const auto& info = entry.info;
    if (!on || entry.from_cpu || !entry.image || info.num_samples != 1 ||
        info.props.has_stencil ||
        (!info.props.is_depth && !LiverpoolToVK::IsFormatDepthCompatible(info.pixel_format))) {
        return;
    }
    Native::ImageEntry* twin =
        registry->FindExisting(info.guest_address, [&](const Native::ImageEntry& other) {
            const auto& o = other.info;
            return &other != &entry && !other.from_cpu && other.image &&
                   o.props.is_depth != info.props.is_depth && !o.props.has_stencil &&
                   o.num_samples == 1 && o.num_bits == info.num_bits &&
                   o.size.width == info.size.width && o.size.height == info.size.height &&
                   o.resources.levels == info.resources.levels &&
                   o.resources.layers == info.resources.layers && o.type == info.type &&
                   (o.props.is_depth || LiverpoolToVK::IsFormatDepthCompatible(o.pixel_format));
        });
    if (!twin) {
        return;
    }
    const u64 version = twin->write_version.load(std::memory_order_relaxed);
    if (version == 0 || version <= entry.write_version.load(std::memory_order_relaxed)) {
        return;
    }
    const u32 texel = std::max<u32>(info.num_bits / 8, 1);
    const u32 mips = std::max<u32>(info.resources.levels, 1);
    const u32 layers = std::max<u32>(info.resources.layers, 1);
    const auto aspect = [](const Native::ImageEntry& e) {
        return e.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                     : vk::ImageAspectFlagBits::eColor;
    };
    boost::container::small_vector<vk::BufferImageCopy, 16> from, into;
    u64 offset = 0;
    for (u32 m = 0; m < mips; ++m) {
        const u32 width = std::max(info.size.width >> m, 1u);
        const u32 height = std::max(info.size.height >> m, 1u);
        vk::BufferImageCopy copy{
            .bufferOffset = offset,
            .imageSubresource{.mipLevel = m, .baseArrayLayer = 0, .layerCount = layers},
            .imageExtent = {width, height, 1},
        };
        copy.imageSubresource.aspectMask = aspect(*twin);
        from.push_back(copy);
        copy.imageSubresource.aspectMask = aspect(entry);
        into.push_back(copy);
        offset += u64(width) * height * layers * texel;
    }
    auto staging = device->CreateBuffer({.size = offset,
                                         .usage = Native::Rhi::BufferUsageTransferSrc |
                                                  Native::Rhi::BufferUsageTransferDst,
                                         .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                         .name = "native depth twin copy"});
    if (!staging) {
        return;
    }
    Stream& s = Begin();
    ApplyRewrite(*twin);
    ApplyRewrite(entry);
    // Their own uploads first (a new target starts with what memory holds, NAT-127).
    const std::array<const Native::ImageEntry*, 2> both{&entry, twin};
    PrepareImages(std::span<const Native::ImageEntry* const>{both});
    GlobalBarrier(); // the twin's picture is written
    const vk::Buffer buffer{Handle<VkBuffer_T>(staging->NativeHandle())};
    s.cmd.copyImageToBuffer(vk::Image{Handle<VkImage_T>(twin->image->NativeHandle())},
                            vk::ImageLayout::eGeneral, buffer, from);
    GlobalBarrier();
    s.cmd.copyBufferToImage(buffer, vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())},
                            vk::ImageLayout::eGeneral, into);
    s.keep.push_back(std::move(staging)); // freed once the GPU ran this list
    s.written.push_back(&entry);          // read after the next barrier
    ++s.commands;
    WrittenSubresources(&entry, version, nullptr);
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 40) {
        LOG_INFO(Render_Vulkan, "FIX-212: {} {:#x} {}x{} {} <- {} {} (version {}, frame {})",
                 info.props.is_depth ? "depth" : "colour", info.guest_address, info.size.width,
                 info.size.height, vk::to_string(info.pixel_format),
                 twin->info.props.is_depth ? "depth" : "colour",
                 vk::to_string(twin->info.pixel_format), version, frame);
    }
}

bool NativeRenderer::PartSyncOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_PART_SYNC"); // FIX-210 off: =0
        return !(env && env[0] == '0');
    }();
    return on;
}

void NativeRenderer::SyncPartFromWholes(Native::ImageEntry& part, const DrawnRange& range) {
    // FIX-210: on the PS4 a target that is mips or layers of a larger one (FIX-197) is the same
    // memory, and what other work wrote into the larger one is in the part when the part is
    // drawn into. GT7 draws its reflection probes' mips 1-8 into 18-layer targets through a
    // 12-layer view, the six layers it draws picked by the vertex data; the other six belong to
    // a probe its compute filter writes in the cube array. Copied back as the view drawn
    // (FIX-208), those never-drawn layers blanked that probe's mips 1-8, and the cockpit and the
    // crowd it lights were black two frames in three. The subresources of the view that the
    // larger target holds newer pictures for are copied into the part first.
    const auto eligible = [](const Native::ImageEntry& e) {
        return e.image && std::max<u32>(e.info.resources.levels, 1) <= 16 &&
               std::max<u32>(e.info.resources.layers, 1) <= 64 &&
               e.info.type != AmdGpu::ImageType::Color3D;
    };
    if (!PartSyncOn() || !SubVersionsOn() || part.from_cpu || range.mips == 0 ||
        !eligible(part)) {
        return;
    }
    struct Whole {
        Native::ImageEntry* entry;
        Native::SourcePlace place;
    };
    boost::container::small_vector<Whole, 2> wholes;
    {
        boost::container::small_vector<Native::ImageEntry*, 4> dependents;
        {
            std::scoped_lock lock{part.sources_mutex};
            dependents.assign(part.dependents.begin(), part.dependents.end());
        }
        for (auto* dependent : dependents) {
            if (dependent->from_cpu || dependent->reused.load(std::memory_order_relaxed) ||
                !eligible(*dependent)) {
                continue;
            }
            std::scoped_lock lock{dependent->sources_mutex};
            for (const auto& link : dependent->sources) {
                if (link.source == &part) {
                    wholes.push_back({dependent, link.place});
                    break;
                }
            }
        }
    }
    for (const auto& [whole, place] : wholes) {
        if (range.mip < u32(place.src_mip)) {
            continue;
        }
        const u32 whole_mip = u32(place.mip) + (range.mip - u32(place.src_mip));
        if (whole_mip >= std::min<u32>(std::max<u32>(whole->info.resources.levels, 1),
                                       std::max<u32>(whole->image->Desc().mips, 1u))) {
            continue;
        }
        RefreshSources(*whole); // its other parts' newer pictures first
        auto& held = held_versions[whole];
        const u32 part_layers = std::min<u32>(std::max<u32>(part.info.resources.layers, 1),
                                              std::max<u32>(part.image->Desc().layers, 1u));
        const u32 whole_layers = std::min<u32>(std::max<u32>(whole->info.resources.layers, 1),
                                               std::max<u32>(whole->image->Desc().layers, 1u));
        struct Run {
            u32 part_layer;
            u32 whole_layer;
            u32 count;
        };
        boost::container::small_vector<Run, 4> runs;
        boost::container::small_vector<std::pair<u32, u64>, 32> taken; // part layer, version
        const u32 last = std::min<u32>(u32(range.layer) + std::max<u32>(range.layers, 1),
                                       part_layers);
        for (u32 layer = range.layer; layer < last; ++layer) {
            if (layer < u32(place.src_layer)) {
                continue;
            }
            const u32 whole_layer = u32(place.layer) + (layer - u32(place.src_layer));
            if (whole_layer >= whole_layers) {
                break;
            }
            // What the whole holds there: its own drawing, or the newest part copied in.
            const auto copied = held.find((whole_mip << 16) | whole_layer);
            const u64 version = std::max(SubVersion(*whole, whole_mip, whole_layer),
                                         copied != held.end() ? copied->second : u64{0});
            if (version <= SubVersion(part, range.mip, layer)) {
                continue;
            }
            if (!runs.empty() && runs.back().part_layer + runs.back().count == layer &&
                runs.back().whole_layer + runs.back().count == whole_layer) {
                ++runs.back().count;
            } else {
                runs.push_back({layer, whole_layer, 1});
            }
            taken.emplace_back(layer, version);
        }
        if (runs.empty()) {
            continue;
        }
        Stream& s = Begin();
        GlobalBarrier(); // the whole's pictures are written
        const u32 width = std::min(std::max(part.info.size.width >> range.mip, 1u),
                                   std::max(whole->info.size.width >> whole_mip, 1u));
        const u32 height = std::min(std::max(part.info.size.height >> range.mip, 1u),
                                    std::max(whole->info.size.height >> whole_mip, 1u));
        for (const auto& run : runs) {
            const vk::ImageCopy region{
                .srcSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                .mipLevel = whole_mip,
                                .baseArrayLayer = run.whole_layer,
                                .layerCount = run.count},
                .srcOffset = {0, 0, 0},
                .dstSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                .mipLevel = range.mip,
                                .baseArrayLayer = run.part_layer,
                                .layerCount = run.count},
                .dstOffset = {0, 0, 0},
                .extent = {width, height, 1},
            };
            CmdBuf{s.cmd}.copyImage(vk::Image{Handle<VkImage_T>(whole->image->NativeHandle())},
                                    vk::ImageLayout::eGeneral,
                                    vk::Image{Handle<VkImage_T>(part.image->NativeHandle())},
                                    vk::ImageLayout::eGeneral, region);
        }
        GlobalBarrier(); // before the draw writes the part
        {
            // The part holds those pictures now, as new as the whole's: copied back only once
            // drawn again.
            std::scoped_lock lock{part.sources_mutex};
            const u32 mips = std::max<u32>(part.info.resources.levels, 1);
            const u32 layers = std::max<u32>(part.info.resources.layers, 1);
            auto& subs = part.sub_versions;
            if (subs.empty()) {
                subs.assign(size_t(mips) * layers,
                            part.write_version.load(std::memory_order_relaxed));
            }
            for (const auto& [layer, version] : taken) {
                const size_t at = size_t(range.mip) * layers + layer;
                if (at < subs.size()) {
                    subs[at] = version;
                }
            }
        }
        stats.source_copies += runs.size();
        ++s.commands;
        static std::atomic<u32> logged{};
        if (logged++ < 8 || SeqTraced(&part)) {
            LOG_INFO(Render_Vulkan, "FIX-210: target {:#x} {}x{} mip {} layers {}+{} takes {} "
                     "layers of target {:#x} mip {} first (frame {})", part.info.guest_address,
                     part.info.size.width, part.info.size.height, range.mip, range.layer,
                     range.layers, taken.size(), whole->info.guest_address, whole_mip, frame);
        }
    }
}

VAddr NativeRenderer::DiagWritesAt() {
    static const VAddr at = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_WRITES_AT");
        return env && *env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    return at;
}

bool NativeRenderer::DiagFrame() const {
    // DIAG-178 (RYGT_DIAG_WRITES_AT=<hex address>, RYGT_DIAG_WRITES_FRAME=<frame>, 2500 if not
    // given): one frame's work touching the address, in order, up to 300 lines.
    static const u64 at_frame = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_WRITES_FRAME");
        return env && *env ? u64(std::strtoull(env, nullptr, 0)) : u64{2500};
    }();
    return DiagWritesAt() != 0 && frame >= at_frame && diag_lines < 300;
}

void NativeRenderer::DiagWrites(const Targets* targets, const DrawBindings& b) {
    if (!DiagFrame()) {
        return;
    }
    const VAddr at = DiagWritesAt();
    // RYGT_DIAG_PROGRAM=<hex hash>: that program's work too, wherever it draws.
    static const std::string program = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_PROGRAM");
        return env && *env ? std::string{env} : std::string{};
    }();
    const bool by_program = !program.empty() && diag_what.find(program) != std::string::npos;
    const auto covers = [&](const Native::ImageEntry* e) {
        return e && (by_program || (e->info.guest_address <= at &&
                                    at < e->info.guest_address +
                                             std::max<u64>(e->info.guest_size, 1)));
    };
    const auto describe = [](const Native::ImageEntry* e) {
        return fmt::format("{:#x} {}x{}p{} {} mips {} layers {}{}", e->info.guest_address,
                           e->info.size.width, e->info.size.height, e->info.pitch,
                           vk::to_string(e->info.pixel_format), e->info.resources.levels,
                           e->info.resources.layers, e->from_cpu ? " (texture)" : "");
    };
    std::string text;
    if (targets) {
        for (const auto* color : targets->colors) {
            if (covers(color)) {
                text += " draws into " + describe(color) +
                        fmt::format(" (version {})", color->write_version.load());
            }
        }
        if (covers(targets->depth)) {
            text += " depth " + describe(targets->depth);
        }
    }
    for (const auto* stored : b.stored) {
        if (covers(stored)) {
            text += " stores " + describe(stored);
        }
    }
    for (const auto& [address, size] : b.gpu_writes) {
        if (address <= at && at < address + size) {
            text += fmt::format(" writes buffer {:#x}+{:#x}", address, size);
        }
    }
    const bool writes = !text.empty();
    for (const auto* sampled : b.sampled) {
        if (covers(sampled)) {
            text += " samples " + describe(sampled);
        }
    }
    // Reads alone: the first 20 lines only.
    static u32 read_lines = 0;
    if (!writes && !text.empty() && ++read_lines > 20) {
        return;
    }
    if (!text.empty()) {
        ++diag_lines;
        if (targets) {
            // How it draws: primitive, count, blend of MRT0, viewport, scissor.
            const auto& regs = r.liverpool->DrawRegs();
            const auto& blend = regs.blend_control[0];
            const auto& vp = regs.viewports[0];
            text += fmt::format(" | prim {} count {} blend {} src {} func {} dst {} mask {:#x} "
                                "vp {}x{} at {},{} scissor {},{}-{},{}",
                                u32(regs.primitive_type), regs.num_indices, u32(blend.enable),
                                u32(blend.color_src_factor), u32(blend.color_func),
                                u32(blend.color_dst_factor), regs.color_target_mask.raw,
                                vp.xscale * 2, -vp.yscale * 2, vp.xoffset - vp.xscale,
                                vp.yoffset + vp.yscale, regs.screen_scissor.top_left_x,
                                regs.screen_scissor.top_left_y, regs.screen_scissor.bottom_right_x,
                                regs.screen_scissor.bottom_right_y);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-178: frame {} {} {}:{}", frame,
                    targets ? "draw" : "dispatch", diag_what, text);
    }
}

void NativeRenderer::NoteWrites(const Targets* targets, const DrawBindings& b) {
    DiagWrites(targets, b);               // DIAG-178 (diagnosis)
    NoteProgramTextureWrites(targets, b); // DIAG-185 (diagnosis)
    if (WriteHistoryOn()) {
        // DIAG-187 (diagnosis)
        if (targets) {
            for (const auto* color : targets->colors) {
                if (color) {
                    NoteHistory(color->info.guest_address, color->info.guest_size, 0,
                                history_program);
                }
            }
            if (targets->depth) {
                NoteHistory(targets->depth->info.guest_address, targets->depth->info.guest_size,
                            1, history_program);
            }
        }
        for (const auto* stored : b.stored) {
            if (stored) {
                NoteHistory(stored->info.guest_address, stored->info.guest_size, 2,
                            history_program);
            }
        }
        for (const auto& [address, size] : b.gpu_writes) {
            NoteHistory(address, size, 3, history_program);
        }
    }
    Stream& s = stream;
    const auto note = [&](const Native::ImageEntry* entry, const DrawnRange* range = nullptr) {
        if (range && range->mips != 0) {
            WrittenRange(entry, *range); // FIX-208: only the mips and layers drawn
        } else {
            Written(entry);
        }
        if (entry && r.FrameDumpCollecting() &&
            std::ranges::find(dump_entries, entry) == dump_entries.end()) {
            dump_entries.push_back(entry); // NAT-116
        }
        if (entry && std::ranges::find(s.written, entry) == s.written.end()) {
            s.written.push_back(entry);
            // FIX-GT7-036: its bytes are no longer what a fill before it says.
            registry->ForgetFills(entry->info.guest_address, entry->info.guest_size);
        }
    };
    for (const auto* entry : b.sampled) {
        if (entry && std::ranges::find(s.read, entry) == s.read.end()) {
            s.read.push_back(entry); // NAT-140
        }
    }
    if (targets) {
        const auto attached = [&](const Native::ImageEntry* entry) {
            if (entry && std::ranges::find(s.attach_dirty, entry) == s.attach_dirty.end()) {
                s.attach_dirty.push_back(entry); // NAT-140
            }
        };
        for (const auto* entry : targets->colors) {
            attached(entry);
        }
        attached(targets->depth);
        for (size_t cb = 0; cb < targets->colors.size(); ++cb) {
            const auto* entry = targets->colors[cb];
            note(entry, &targets->color_ranges[cb]);
            // NAT-112: small pictures go back to guest memory.
            constexpr u64 WritebackSize = 64_KB;
            if (entry && entry->info.num_samples == 1) {
                if (entry->info.guest_size <= WritebackSize) {
                    QueueWriteback(entry); // PERF-GT7-044
                } else {
                    WritebackForCpu(entry); // FIX-GT7-041
                }
            }
        }
        note(targets->depth);
    }
    for (const auto* entry : b.stored) {
        note(entry);
        constexpr u64 WritebackSize = 64_KB; // NAT-112, as for render targets
        if (entry->info.guest_size <= WritebackSize && entry->info.num_samples == 1 &&
            !entry->info.props.is_depth) {
            QueueWriteback(entry); // PERF-GT7-044
        }
    }
    s.buffers_written |= b.writes_buffers;
    {
        // DIAG-GT7-004: for dispatches (no targets), direct and indirect.
        PartClock part{targets ? nullptr : &stats.dispatch_note_ns[0]};
        for (const auto& [address, size] : b.gpu_writes) {
            const auto before = std::ranges::find_if(written_befores, [&](const WrittenBefore& w) {
                return w.address == address && w.size == size;
            });
            NoteRegionWrite(address, size, before != written_befores.end() ? &*before : nullptr,
                            true); // FIX-194, FIX-203
        }
    }
    DropWrittenBefores();
    // NAT-119: images the shader overwrote in memory are uploaded again (after this work, in
    // GPU order: the next PrepareImages records the uploads behind a barrier).
    PartClock part{targets ? nullptr : &stats.dispatch_note_ns[1]}; // DIAG-GT7-004
    for (const auto& [address, size] : b.gpu_writes) {
        if (b.fill_pattern && address == b.fill_address) {
            GpuWrote(address, std::min(size, b.fill_size), &*b.fill_pattern, true); // NAT-120
        } else {
            GpuWrote(address, size, nullptr, false); // NAT-129: what the shader may write
        }
    }
}

void NativeRenderer::Draw(bool is_indexed, u32 index_offset, const IndirectDraw* indirect) {
    std::scoped_lock lock{mutex};
    AmdGpu::RecorderParts::Scope recorder_part{AmdGpu::RecorderParts::Part::Draws}; // DIAG-GT7-005
    const auto start = std::chrono::steady_clock::now();
    struct Timed {
        std::chrono::steady_clock::time_point start;
        u64& total;
        ~Timed() {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    } timed{start, stats.draw_ns};
    t_timed = (++timing_tick & 15) == 0; // OPT-001
    stats.timed_draws += t_timed ? 1 : 0;
    r.scheduler.PopPendingOperations();
    // NAT-161: an indirect draw that returns before its draw command is counted as skipped.
    struct IndirectSkip {
        u64* skipped;
        ~IndirectSkip() {
            if (skipped) {
                ++*skipped;
            }
        }
    } indirect_skip{indirect ? &stats.indirect_skipped : nullptr};
    const auto& regs = r.liverpool->DrawRegs();
    if (WriteHistoryOn()) {
        // DIAG-189 (diagnosis): every draw's targets as it comes in, drawn or skipped later
        // (the filter in the program field: 0 draw, 1 fast clear eliminate, 2 FMASK decompress,
        // 3 resolve, 4 no primitive, 5 depth copy).
        const u64 filter = u64(Classify(regs));
        for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
            const auto& col_buf = regs.color_buffers[i];
            if (col_buf && col_buf.Address() != 0) {
                NoteHistory(col_buf.Address(), u64(col_buf.GetColorSliceSize()) * col_buf.NumSlices(),
                            13, filter | (u64(i) << 8));
            }
        }
        if (regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteAddress() != 0) {
            NoteHistory(regs.depth_buffer.DepthWriteAddress(),
                        regs.depth_buffer.GetDepthSliceSize(), 14, filter);
        }
    }
    switch (Classify(regs)) {
    case Filter::EliminateFastClear: {
        // Fast-cleared tiles get the clear colour now when nothing drew into the target since
        // its CMASK was cleared (otherwise the pass that drew started with the clear).
        const auto& col_buf = regs.color_buffers[0];
        if (col_buf && col_buf.info.fast_clear && cleared_meta.erase(col_buf.CmaskAddress())) {
            const VideoCore::TextureCache::ImageDesc desc{col_buf, r.liverpool->DrawCbExtent(0)};
            if (Native::ImageEntry* entry = registry->Find(desc.info, true, frame)) {
                Stream& s = Begin();
                ApplyRewrite(*entry); // NAT-122
                PrepareImages();
                GlobalBarrier();
                const vk::ImageSubresourceRange range{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = desc.view_info.range.base.layer,
                    .layerCount = desc.view_info.range.extent.layers,
                };
                CmdBuf{s.cmd}.clearColorImage(
                    vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())},
                    vk::ImageLayout::eGeneral, LiverpoolToVK::ColorBufferClearValue(col_buf).color,
                    range);
                Written(entry);
                s.buffers_written = true; // the next work waits for the clear
                ++s.commands;
            }
        }
        ++stats.skipped;
        return;
    }
    case Filter::FmaskDecompress:
    case Filter::NoPrimitive:
        ++stats.skipped;
        return;
    case Filter::Resolve:
        ResolveDraw(regs);
        return;
    case Filter::Copy: {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan, "NAT-102: depth/stencil copy draw skipped");
        }
        ++stats.skipped;
        return;
    }
    case Filter::Draw:
        break;
    }

    const bool quad_list = regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
    // NAT-161: an indirect draw's vertex count is in GPU memory, so its quads are drawn through
    // the tessellation helpers (as the old path's DrawIndirect does).
    bool quad_triangles = quad_list && !indirect && Common::PerfFeatureEnabled(28) &&
                          u64(regs.num_indices) / 4 * 6 * sizeof(u32) <= MaxQuadIndexBytes;
    // OPT-010: the draw a game job thread prepared, if any (job threads prepare direct draws).
    static const bool prepare = PrepareEnabled();
    PreparedDraw* prepared = nullptr;
    if (prepare && PreparedOn() && !indirect && !r.FrameDumpCollecting()) {
        prepared = TakePrepared(Rasterizer::t_draw_packet);
        stats.prepared_missing += prepared ? 0 : 1;
    }
    struct GiveBack {
        NativeRenderer& n;
        PreparedDraw* p;
        ~GiveBack() {
            if (p) {
                n.FreePrepared(p);
            }
        }
    } give_back{*this, prepared};
    // OPT-012: when the command thread didn't select (it saw the prepared draw), the job
    // thread's selection is installed and trusted.
    bool trusted = false;
    // NAT-161: as the old path's DrawIndirect: the vertex shader reads the base vertex and
    // first instance the CP would write into its user data SGPRs from the BaseVertex and
    // BaseInstance built-ins, i.e. from the draw's arguments.
    const DrawIndirectParams params =
        indirect ? DrawIndirectParams{
                       .vertex_sgpr_offset = indirect->vertex_sgpr_offset,
                       .instance_sgpr_offset = indirect->instance_sgpr_offset,
                       .tessellate_quads = true,
                       .is_indirect = true, // FIX-192
                   }
                 : DrawIndirectParams{
                       .vertex_sgpr_offset = 0,
                       .instance_sgpr_offset = 0,
                       .tessellate_quads = quad_list && !quad_triangles,
                   };
    const GraphicsPipeline* pipeline = [&]() -> const GraphicsPipeline* {
        PhaseClock clock{stats.phase_ns[0]};
        if (prepared && !quad_list && r.selected_pipeline == nullptr && prepared->pipeline) {
            if (const GraphicsPipeline* installed = InstallPrepared(*prepared)) {
                trusted = true;
                return installed;
            }
        }
        return r.AcquireGraphicsPipeline(params, !indirect && !quad_list);
    }();
    const bool must_wait = !pipeline && DrawMustNotSkip(regs);
    // FIX-GT7-121: a picture known to be sampled before any redraw waits whatever the budget.
    const bool harmful_wait = must_wait && OneOffKnownHarmful();
    if (must_wait && (harmful_wait || OneOffBudgetLeft())) {
        // FIX-198: drawn once its pipeline is built (see DrawMustNotSkip).
        PhaseClock clock{stats.phase_ns[0]};
        const auto wait_start = std::chrono::steady_clock::now();
        pipeline = r.AcquireGraphicsPipelineWaiting(
            params, harmful_wait ? std::chrono::steady_clock::time_point::max()
                                 : OneOffDeadline()); // FIX-GT7-109, FIX-GT7-121
        one_off_wait_ns_frame += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - wait_start)
                                         .count()); // FIX-GT7-102
        if (pipeline) {
            ++stats.one_off_waits;
            stats.one_off_harmful_waits += harmful_wait ? 1 : 0;
            WatchOneOffs(regs, false); // DIAG-GT7-008, PERF-GT7-023
        } else {
            ++stats.one_off_budget_skips; // FIX-GT7-109: not ready within the play budget
            WatchOneOffs(regs, true);
        }
    } else if (must_wait) {
        // FIX-GT7-102: this frame already waited its budget for one-off pipelines: skipped as
        // other draws are, and watched as a skip (it never makes its kind safe).
        ++stats.one_off_budget_skips;
        WatchOneOffs(regs, true);
    } else if (!pipeline && !one_off_hits.empty()) {
        // PERF-GT7-023: one-off pictures of kinds that no longer wait only: skipped as other
        // draws are, and still watched (one sampled before its redraw makes its kind wait again).
        ++stats.one_off_exempt;
        WatchOneOffs(regs, true);
    } else if (pipeline && OneOffDiagAll() && DrawMustNotSkip(regs)) {
        WatchOneOffs(regs, false); // DIAG-GT7-008: it would have waited
    }
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    // DIAG-152 (RYGT_NATIVE_SKIP_PS=<hex,...>): draws with these pixel shaders are skipped.
    static const std::vector<u64> skip_ps = [] {
        std::vector<u64> hashes;
        if (const char* env = std::getenv("RYGT_NATIVE_SKIP_PS"); env && *env) {
            const std::string list{env};
            for (size_t start = 0; start < list.size();) {
                const size_t end = std::min(list.find(',', start), list.size());
                hashes.push_back(std::stoull(list.substr(start, end - start), nullptr, 16));
                start = end + 1;
            }
        }
        return hashes;
    }();
    const u64 diagnostic_skip = DiagnosticPassOptions(frame).second;
    if (!skip_ps.empty() || diagnostic_skip) {
        const auto& stages = pipeline->GetStages();
        const Shader::Info* fs = stages[u32(Shader::SwStage::Fragment)];
        if (fs && (fs->pgm_hash == diagnostic_skip ||
                   std::ranges::find(skip_ps, fs->pgm_hash) != skip_ps.end())) {
            ++stats.skipped;
            return;
        }
    }
    if (indirect) {
        // DIAG-162: the first kinds of indirect draws, by shaders and argument layout.
        static std::mutex seen_mutex;
        static std::set<std::tuple<u64, u64, bool, u32, bool, u32>> seen;
        const auto& stages = pipeline->GetStages();
        const Shader::Info* vs = stages[u32(Shader::SwStage::Vertex)];
        const Shader::Info* fs = stages[u32(Shader::SwStage::Fragment)];
        const auto key = std::make_tuple(vs ? vs->pgm_hash : 0, fs ? fs->pgm_hash : 0, is_indexed,
                                         indirect->max_count, indirect->count_address != 0,
                                         indirect->stride);
        std::scoped_lock lock{seen_mutex};
        if (seen.size() < 64 && seen.insert(key).second) {
            LOG_WARNING(Render_Vulkan,
                        "DIAG-162: indirect draw vs {:#x} ps {:#x} {} max_count {} count {:#x} "
                        "stride {} args {:#x} sgprs {}/{}",
                        std::get<0>(key), std::get<1>(key), is_indexed ? "indexed" : "arrays",
                        indirect->max_count, indirect->count_address, indirect->stride,
                        indirect->address, indirect->vertex_sgpr_offset,
                        indirect->instance_sgpr_offset);
        }
    }
    Stream& s = Begin();
    // NAT-161: the arguments are looked up before the draw's other buffers, as DispatchIndirect
    // does: a region a later lookup makes takes theirs in and leaves it as it was, holding what
    // the GPU wrote there; made last, their region could take in one this draw binds for
    // writing, whose writes would then go to the buffer left behind.
    std::optional<IndirectBuffers> arguments;
    if (indirect) {
        arguments = ResolveIndirect(*indirect, is_indexed);
        if (!arguments) {
            ++stats.skipped;
            return;
        }
        // DIAG-211 (diagnosis): a watched shader's arguments as the GPU reads them.
        static const bool watching = Common::DiagEnv("SHADGT_WATCH_SHADERS") != nullptr;
        if (watching && frame % 10 == 0) {
            for (const auto* stage : pipeline->GetStages()) {
                if (stage && IsWatchedShader(stage->pgm_hash)) {
                    const u32 command = is_indexed ? 20u : 16u;
                    const u64 bytes =
                        u64(std::max(indirect->stride, command)) * (arguments->max_count - 1) +
                        command;
                    CompareRegion(indirect->address, bytes, std::max(indirect->stride, command),
                                  arguments->args, arguments->args_offset, stage->pgm_hash, true);
                    break;
                }
            }
        }
    }
    Targets targets;
    {
        PhaseClock clock{stats.phase_ns[1]};
        ResolveTargets(regs, *pipeline, targets);
    }
    // GT7-UW: the frame's first draw into a display buffer is the scene; later ones the HUD,
    // drawn into the display's black and white HUD pictures instead.
    std::optional<RenderState> hud_black_state;
    std::optional<RenderState> hud_white_state;
    HudLayer* hud_layer = nullptr;
    // (Multisampled, as a race's track and course maps are; a menu's small pictures drawn with
    // depth, like the World Map's 256x256 probe, are not.)
    if (HudSplitOn() && targets.colors[0] && targets.depth &&
        targets.colors[0]->info.num_samples > 1 &&
        targets.colors[0]->info.size.width <= 1024 && targets.colors[0]->info.size.height <= 600) {
        hud_depth_drawn[targets.colors[0]->info.guest_address] = frame;
    }
    if (HudSplitOn() && targets.colors[0] && targets.state.num_color_attachments == 1 &&
        IsDisplayAddress(targets.colors[0]->info.guest_address)) {
        HudLayer& layer = hud_layers[targets.colors[0]->info.guest_address];
        if (layer.composite_frame != frame) {
            layer.composite_frame = frame;
            layer.clear_pending = true;
            if (!layer.black) {
                layer.black = MakeHudPicture(*targets.colors[0]);
                layer.white = MakeHudPicture(*targets.colors[0]);
            }
        } else if (layer.black && layer.white) {
            const VideoCore::TextureCache::ImageDesc desc{regs.color_buffers[0],
                                                          r.liverpool->DrawCbExtent(0)};
            const vk::ImageView black = View(*layer.black, desc.view_info, ViewUse::Attachment);
            const vk::ImageView white = View(*layer.white, desc.view_info, ViewUse::Attachment);
            if (black && white) {
                hud_black_state = targets.state;
                hud_white_state = targets.state;
                auto& b0 = hud_black_state->color_attachments[0];
                auto& w0 = hud_white_state->color_attachments[0];
                b0.image_view = black;
                w0.image_view = white;
                b0.is_clear = w0.is_clear = layer.clear_pending ? 1u : 0u;
                b0.clear_value = {};
                const u32 one = std::bit_cast<u32>(1.0f);
                w0.clear_value = layer.clear_pending ? std::array<u32, 4>{one, one, one, one}
                                                     : std::array<u32, 4>{};
                layer.clear_pending = false;
                layer.hud_frame = frame;
                hud_layer = &layer;
            }
        }
    }
    if (r.FrameDumpActive() || Rasterizer::ExposureLogOn()) {
        if (r.FrameDumpCollecting()) {
            SnapPass(targets); // DIAG-226 (diagnosis)
        }
        r.FrameDumpPass(regs, targets.state, pipeline); // NAT-117
    }
    // OPT-010: the bindings the job thread worked out ahead, when they hold.
    bool from_prepared = false;
    if (prepared) {
        PhaseClock clock{stats.phase_ns[2]};
        from_prepared = CommitPrepared(*prepared, *pipeline, regs, trusted);
    }
    DrawBindings& b = from_prepared ? prepared->bindings : bindings;
    // GT7-UW: a HUD draw showing a small picture drawn with depth this frame (the race's maps).
    const auto note_race_hud = [&] {
        if (!hud_layer || hud_layer->race_frame == frame) {
            return;
        }
        for (const auto* sampled : b.sampled) {
            const auto it = hud_depth_drawn.find(sampled->info.guest_address);
            if (it != hud_depth_drawn.end() && it->second == frame) {
                hud_layer->race_frame = frame;
                return;
            }
        }
    };
    if (!from_prepared) {
        b.Clear();
        b.push = UserData(regs);
        BindStagesStable(pipeline->GetStages(), regs, false, b); // FIX-GT7-090
        WritebacksBefore(b); // PERF-GT7-044
    }
    if (b.failed) {
        ++stats.skipped;
        return;
    }
    static thread_local VertexIndexState vertex_index;
    const VertexIndexState* vi = &vertex_index;
    {
        PhaseClock clock{stats.phase_ns[5]};
        if (from_prepared && prepared->has_vertices &&
            CommitPreparedVertices(*prepared, *pipeline, regs, is_indexed, index_offset,
                                   quad_triangles, trusted)) {
            vi = prepared->vertex_index.get(); // OPT-011
        } else {
            vertex_index = {};
            ResolveVertexBuffers(*pipeline, regs, vertex_index);
            // NAT-161: an indexed indirect draw's indices start at the index base; num_indices
            // is then the index buffer's size (INDEX_BUFFER_SIZE), as the old path binds it.
            if (!ResolveIndices(regs, is_indexed, index_offset, quad_triangles, vertex_index)) {
                if (indirect && is_indexed) {
                    static const bool logged = [&] {
                        LOG_WARNING(Render_Vulkan,
                                    "NAT-161: indexed indirect draw skipped: its {} indices at "
                                    "{:#x} can't be bound",
                                    regs.num_indices, regs.index_base_address.Address());
                        return true;
                    }();
                }
                ++stats.skipped;
                return;
            }
            if (indirect && is_indexed && regs.index_offset != 0) {
                // The GPU adds VGT_INDX_OFFSET to every index it fetches; the arguments have
                // their own base vertex, and nothing adds this one (as in the old path).
                static const bool logged = [&] {
                    LOG_WARNING(Render_Vulkan,
                                "NAT-161: indexed indirect draw with VGT_INDX_OFFSET {} drawn "
                                "without it",
                                regs.index_offset);
                    return true;
                }();
            }
            if (!indirect) { // NAT-161: an indirect draw's counts are in GPU memory
                CheckVertexReach(*pipeline, regs, is_indexed, index_offset); // FIX-181 (diagnosis)
            }
        }
    }
    {
        PhaseClock clock{stats.phase_ns[6]};
        PrepareImages(b); // PERF-109
    }
    SnapProgramMips(pipeline->GetStages(), b); // DIAG-234 (diagnosis)
    PhaseClock record_clock{stats.phase_ns[7]};
    {
        PhaseClock part{stats.record_ns[0]}; // OPT-011
        // NAT-140: a full barrier only when the draw reads what was written since the last
        // one, or draws into what was read since; a pass that draws into pictures earlier
        // passes drew waits for their attachment writes only; other passes don't wait.
        static const bool pass_barriers = [] {
            const char* env = std::getenv("RYGT_NATIVE_PASS_BARRIERS");
            return !(env && env[0] == '0');
        }();
        const bool pass_change =
            !s.rendering || !(s.render == (hud_black_state ? *hud_black_state : targets.state));
        // FIX-183 (diagnosis, RYGT_NATIVE_DRAW_BARRIER=1): a full barrier before every draw.
        static const bool draw_barrier = [] {
            const char* env = std::getenv("RYGT_NATIVE_DRAW_BARRIER");
            return env && env[0] == '1';
        }();
        if (draw_barrier) {
            GlobalBarrier();
        }
        const auto targets_in = [&](const std::vector<const Native::ImageEntry*>& list) {
            if (list.empty()) {
                return false;
            }
            for (const auto* entry : targets.colors) {
                if (entry && std::ranges::find(list, entry) != list.end()) {
                    return true;
                }
            }
            return targets.depth && std::ranges::find(list, targets.depth) != list.end();
        };
        // NAT-161: an indirect draw reads its arguments (and count) at the draw-indirect stage.
        // Buffers written since the last global barrier (by the dispatch that wrote them) get
        // one now, whose second scope (all commands, all memory reads) takes in indirect command
        // reads; NeedsBarrierFor asks for it too, but the arguments keep it whatever that checks.
        const bool reads_arguments = indirect && s.buffers_written;
        if (!pass_barriers) {
            if (pass_change || NeedsBarrierFor(b) || reads_arguments) {
                GlobalBarrier();
            }
        } else if (NeedsBarrierFor(b) || reads_arguments || (pass_change && targets_in(s.read))) {
            GlobalBarrier();
        } else if (pass_change && targets_in(s.attach_dirty)) {
            PassBarrier();
        } else if (pass_change) {
            ++stats.passes_unblocked;
        }
    }
    // FIX-210: a target's pending pictures (from the targets it is made of, and from the larger
    // target it is a part of) go in before the draw, which counts as drawing its whole view.
    if (PartSyncOn()) {
        const u64 copies_before = stats.source_copies;
        for (size_t cb = 0; cb < targets.colors.size(); ++cb) {
            if (auto* color = const_cast<Native::ImageEntry*>(targets.colors[cb])) {
                RefreshSources(*color);
                SyncPartFromWholes(*color, targets.color_ranges[cb]);
            }
        }
        if (stats.source_copies != copies_before) {
            GlobalBarrier(); // the copies are done before the draw writes the targets
        }
    }
    {
        // DIAG-214 (diagnosis, RYGT_DIAG_RECORD_DRAWS=1): every 30th frame, each draw reading a
        // 144-byte-stride buffer (GT7's crowd records): its programs, records, depth test, targets.
        static const bool record_draws = Common::DiagEnv("RYGT_DIAG_RECORD_DRAWS") != nullptr;
        if (record_draws && b.diag_records != 0 && frame % 30 == 0) {
            std::string programs;
            for (const auto* stage : pipeline->GetStages()) {
                if (stage) {
                    programs += fmt::format(" {:#x}", stage->pgm_hash);
                }
            }
            const auto& dc = regs.depth_control;
            const auto& dsa = targets.state.depth_stencil_attachment;
            LOG_WARNING(Render_Vulkan, "DIAG-214: frame {} draw{} records {:#x} | depth {:#x} test {} "
                        "write {} func {} | entry {} {} {}x{} v{} clear {} view {} | c0 {:#x} mask "
                        "{:#x} | indirect {} n {} inst {}", frame,
                        programs, b.diag_records,
                        targets.depth ? targets.depth->info.guest_address : 0, u32(dc.depth_enable),
                        u32(dc.depth_write_enable), u32(dc.depth_func),
                        static_cast<const void*>(targets.depth),
                        targets.depth ? vk::to_string(targets.depth->info.pixel_format) : "-",
                        targets.depth ? targets.depth->info.size.width : 0,
                        targets.depth ? targets.depth->info.size.height : 0,
                        targets.depth ? targets.depth->write_version.load() : 0, dsa.depth_clear,
                        static_cast<const void*>(static_cast<VkImageView>(dsa.image_view)),
                        targets.colors[0] ? targets.colors[0]->info.guest_address : 0,
                        regs.color_target_mask.raw, indirect != nullptr, regs.num_indices,
                        regs.num_instances.NumInstances());
        }
    }
    {
        // DIAG-212 (diagnosis, RYGT_DIAG_CAPTURE=<hex program>,<every n frames>): every n-th
        // frame, the color target the program's first draw writes, right before that draw, and
        // (in FindDisplay) the display picture the frame ends with.
        // Several programs: <hex>+<hex>+...,<n>; each one's first draw in the frame is captured.
        struct CaptureConfig {
            std::vector<u64> programs;
            u64 every = 30;
        };
        static const CaptureConfig capture = [] {
            CaptureConfig c;
            if (const char* env = Common::DiagEnv("RYGT_DIAG_CAPTURE"); env && *env) {
                const char* at = env;
                char* end = nullptr;
                while (true) {
                    c.programs.push_back(std::strtoull(at, &end, 16));
                    if (!end || *end != '+') {
                        break;
                    }
                    at = end + 1;
                }
                c.every = end && *end == ',' ? std::max<u64>(std::strtoull(end + 1, nullptr, 10), 1)
                                             : 30;
            }
            return c;
        }();
        static u64 captured_frame = ~u64{0};
        static std::vector<u64> captured;
        if (!capture.programs.empty() && frame % capture.every == 0 && targets.colors[0]) {
            if (captured_frame != frame) {
                captured_frame = frame;
                captured.clear();
            }
            for (const auto* stage : pipeline->GetStages()) {
                if (stage && std::ranges::find(capture.programs, stage->pgm_hash) !=
                                 capture.programs.end() &&
                    std::ranges::find(captured, stage->pgm_hash) == captured.end()) {
                    captured.push_back(stage->pgm_hash);
                    capture_frame = frame;
                    CaptureImage(*targets.colors[0], frame,
                                 fmt::format("before_{:x}", stage->pgm_hash));
                    break;
                }
            }
        }
    }
    // DIAG-208 (diagnosis): the vertex data of a logged draw as the GPU reads it (GT7's probe
    // passes take each triangle's layer, LOD and scale from it).
    if (std::ranges::any_of(targets.colors, [&](const auto* color) { return SeqTraced(color); })) {
        for (size_t i = 0; i < vi->buffers.size() && i < 2 && refresh_probes.size() < 64; ++i) {
            const u64 bytes = std::min<u64>(vi->sizes[i], 2048);
            if (bytes == 0 || !vi->buffers[i]) {
                continue;
            }
            auto buffer = device->CreateBuffer({.size = bytes,
                                                .usage = Native::Rhi::BufferUsageStorage |
                                                         Native::Rhi::BufferUsageTransferDst,
                                                .memory = Native::Rhi::MemoryKind::Readback,
                                                .name = "vertex probe"});
            if (!buffer || !buffer->Mapped()) {
                continue;
            }
            GlobalBarrier();
            stream.cmd.copyBuffer(vi->buffers[i],
                                  vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
                                  vk::BufferCopy{.srcOffset = vi->offsets[i], .dstOffset = 0,
                                                 .size = bytes});
            GlobalBarrier();
            const auto* color = *std::ranges::find_if(
                targets.colors, [&](const auto* c) { return SeqTraced(c); });
            refresh_probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, frame,
                                      fmt::format("vertex floats of binding {} (stride {}) of a "
                                                  "draw into {:#x}",
                                                  i, vi->strides[i], color->info.guest_address)});
        }
    }
    // FIX-227: the scratch of a draw whose hull shader reads other patches' LS outputs (GCN LDS).
    const u32 lds_mode = pipeline->LdsEmulation();
    u64 lds_address = 0, lds_bytes = 0, lds_patches = 0;
    u32 lds_num_cp = 0, lds_vertex_bytes = 0;
    if (lds_mode != 0 && !indirect && b.push.quad_sample_mask[0] == ~0u &&
        b.push.quad_sample_mask[1] == ~0u) {
        lds_num_cp = std::max<u32>(regs.ls_hs_config.hs_input_control_points, 1u);
        lds_patches = u64(regs.num_indices / lds_num_cp) * regs.num_instances.NumInstances();
        Shader::TessellationDataConstantBuffer tess_constants{};
        pipeline->GetStage(Shader::SwStage::TessellationControl)
            .ReadTessConstantBuffer(tess_constants);
        // At least 64 bytes a vertex: the hull shader's stride is the one it was translated with.
        lds_vertex_bytes = std::max<u32>((tess_constants.ls_stride + 15) / 16 * 16, 64u);
        lds_bytes = std::max<u64>(lds_patches, 1) * lds_num_cp * lds_vertex_bytes;
        lds_address = LdsScratch(lds_bytes);
    }
    // PERF-GT7-060: a run of such draws in one pass, with nothing else recorded between them,
    // records each one's writer pass, then one barrier, then the reader passes in their order
    // (FlushLdsReaders), instead of a pass break and a full barrier inside each draw. A draw that
    // doesn't join (any other draw, or one of these that runs alone) records after the run.
    std::optional<LdsReader> lds_reader;
    if (lds_address != 0 && LdsBatchOn() && !hud_black_state && !hud_white_state &&
        !b.writes_buffers && b.stored.empty() && b.gpu_writes.empty() &&
        !r.FrameDumpCollecting()) {
        lds_reader.emplace();
        if (!CaptureLdsReader(*lds_reader, *pipeline, b, *vi)) {
            lds_reader.reset();
        }
    }
    if (!lds_reader) {
        FlushLdsReaders(0);
    } else if (s.lds_bytes + lds_bytes > (32ull << 20) || s.lds_readers.size() >= 1024) {
        // The run's slices stay apart in the 128 MB ring; a reader takes ~7 KB. (The demo
        // race's longest runs are ~300 draws, Daytona's ~145 a frame in all.)
        FlushLdsReaders(2);
    }
    {
        PhaseClock part{stats.record_ns[1]};
        BeginRendering(hud_black_state ? *hud_black_state : targets.state);
    }

    // The renderer's dynamic state code, filling this list's state.
    const auto bind_state = [&] {
    {
        PhaseClock part{stats.record_ns[2]};
        DynamicState* const saved = Rasterizer::t_dynamic_state;
        Rasterizer::t_dynamic_state = &s.dynamic;
        r.ctx->attachment_feedback_loop = {};
        for (const auto* color : targets.colors) {
            if (color && std::ranges::find(b.sampled, color) != b.sampled.end()) {
                r.ctx->attachment_feedback_loop |= vk::ImageAspectFlagBits::eColor;
            }
        }
        if (targets.depth &&
            std::ranges::find(b.sampled, targets.depth) != b.sampled.end()) {
            r.ctx->attachment_feedback_loop |= vk::ImageAspectFlagBits::eDepth;
        }
        r.UpdateViewportScissorState();
        r.UpdateDepthStencilState();
        r.UpdatePrimitiveState(is_indexed, quad_triangles);
        r.UpdateRasterizationState();
        r.UpdateColorBlendingState(pipeline);
        s.dynamic.Commit(r.instance, CmdBuf{s.cmd});
        Rasterizer::t_dynamic_state = saved;
    }
    {
        PhaseClock part{stats.record_ns[3]};
        RecordVertexIndex(*vi);
    }
    {
        PhaseClock part{stats.record_ns[4]};
        BindGraphics(s, *pipeline); // SO-001/SO-010/SO-018
    }
    {
        PhaseClock part{stats.record_ns[5]};
        PushBindings(*pipeline, vk::PipelineBindPoint::eGraphics, b);
    }
    };
    bind_state();
    const u32 index_count = vi->index_count;
    const auto record_draw = [&] {
        const CmdBuf cmdbuf{stream.cmd}; // the HUD list's when it is swapped in
        if (indirect) {
            RecordIndirect(*indirect, *arguments, is_indexed); // NAT-161
        } else {
            const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
            const auto [vertex_offset, instance_offset] =
                DrawOffsets(regs, vs_info, pipeline->GetFetchShader());
            const u32 instances = regs.num_instances.NumInstances();
            if (quad_triangles) {
                if (index_count != 0) {
                    cmdbuf.drawIndexed(index_count, instances, 0, s32(vertex_offset),
                                       instance_offset);
                }
            } else if (is_indexed) {
                cmdbuf.drawIndexed(regs.num_indices, instances, 0, s32(vertex_offset),
                                   instance_offset);
            } else {
                cmdbuf.draw(regs.num_indices, instances, vertex_offset, instance_offset);
            }
        }
    };
    note_race_hud();
    // FIX-227: hull shaders that read other patches' LS outputs (GCN LDS) get them from a
    // scratch buffer: the draw first runs as a writer pass (each patch stores its inputs, nothing
    // is drawn), then, after a barrier, as itself.
    static u32 lds_seen = 0;
    if (pipeline->GetStages()[u32(Shader::SwStage::TessellationControl)] && lds_seen < 5) {
        ++lds_seen;
        LOG_WARNING(Render_Vulkan, "FIX-227: tessellated draw, LDS emulation {} (cp {} patches {})",
                    pipeline->LdsEmulation(), u32(regs.ls_hs_config.hs_input_control_points),
                    u32(regs.ls_hs_config.num_patches));
    }
    bool lds_pushed = false;
    bool lds_deferred = false; // PERF-GT7-060: its reader pass waits for its run's barrier
    bool lds_alone = false;    // PERF-GT7-060: it runs alone (for the runs it would have made)
    {
        const u64 address = lds_address;
        const u32 num_cp = lds_num_cp;
        const u64 patches = lds_patches;
        const u32 vertex_bytes = lds_vertex_bytes;
        const auto push_scratch = [&](u64 value) {
            const std::array<u32, 2> words{u32(value), u32(value >> 32)};
            stream.cmd.pushConstants(pipeline->GetLayout(), AllGraphicsStageBits,
                                     u32(offsetof(Shader::PushData, quad_sample_mask)), 8u,
                                     words.data());
        };
        if (address != 0 && lds_reader) {
            // PERF-GT7-060: the writer pass now, the reader pass after the run's barrier.
            push_scratch(address | 1);
            record_draw();
            LdsReader& reader = *lds_reader;
            reader.dynamic = s.dynamic;
            reader.address = address;
            const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
            const auto [vertex_offset, instance_offset] =
                DrawOffsets(regs, vs_info, pipeline->GetFetchShader());
            reader.instances = regs.num_instances.NumInstances();
            reader.indexed = quad_triangles || is_indexed;
            reader.count = quad_triangles ? index_count : regs.num_indices;
            reader.vertex_offset = u32(vertex_offset);
            reader.first_instance = instance_offset;
            const Shader::Info* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
            reader.ps = ps ? ps->pgm_hash : 0;
            reader.prims = u64(regs.num_indices) * std::max<u32>(reader.instances, 1);
            s.lds_readers.push_back(std::move(reader));
            s.lds_bytes += lds_bytes;
            // The quad sample mask's words hold the writer's scratch: pushed whole by the next
            // draw's bindings.
            s.push_known = false;
            lds_deferred = true;
            ++stats.lds_batched;
            if (++stats.lds_emulated_draws <= 5) {
                LOG_WARNING(Render_Vulkan, "FIX-227: draw of {} patches ({} instances) with hull "
                            "shader LDS emulation (mode {}, {} bytes a vertex) at {:#x}, its reader "
                            "pass after its run's barrier (PERF-GT7-060)",
                            patches, regs.num_instances.NumInstances(), lds_mode, vertex_bytes,
                            address);
            }
        } else if (address != 0) {
            lds_alone = true;
            // PERF-GT7-060: the run it continues, if nothing was recorded since the last one.
            const std::array<u64, 4> now{stats.barriers, stats.passes, stats.draws,
                                         stats.dispatches};
            if (stats.lds_alone_run == 0 || now != stats.lds_alone_marker) {
                if (stats.lds_alone_run != 0) {
                    ++stats.lds_alone_runs;
                    ++stats.lds_alone_lengths[std::min<size_t>(
                        std::bit_width(stats.lds_alone_run - 1), 6)];
                    stats.lds_alone_run_max =
                        std::max(stats.lds_alone_run_max, stats.lds_alone_run);
                }
                stats.lds_alone_run = 0;
            }
            ++stats.lds_alone_run;
            ++stats.lds_alone;
            push_scratch(address | 1);
            record_draw();
            const RenderState resumed = [&] {
                RenderState state = stream.render;
                for (u32 i = 0; i < state.num_color_attachments; ++i) {
                    state.color_attachments[i].is_clear = false;
                }
                state.depth_stencil_attachment.depth_clear = false;
                state.depth_stencil_attachment.stencil_clear = false;
                return state;
            }();
            const RenderState current = stream.render;
            EndRendering();
            const vk::MemoryBarrier2 written{
                .srcStageMask = vk::PipelineStageFlagBits2::eTessellationControlShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTessellationControlShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            };
            // A full barrier: one from and to the tessellation control stage alone left the
            // reader pass reading stale scratch on NVIDIA.
            (void)written;
            GlobalBarrier();
            BeginRendering(resumed);
            stream.render = current; // its clears are done; the same pass to later draws
            push_scratch(address);
            lds_pushed = true;
            lds_snap_address = address; // DIAG-241
            lds_snap_bytes = std::min<u64>(patches * num_cp * vertex_bytes, 1ull << 20);
            if (++stats.lds_emulated_draws <= 5) {
                LOG_WARNING(Render_Vulkan, "FIX-227: draw of {} patches ({} instances) with hull "
                            "shader LDS emulation (mode {}, {} bytes a vertex) at {:#x}",
                            patches, regs.num_instances.NumInstances(), lds_mode, vertex_bytes,
                            address);
            }
        }
    }
    {
        PhaseClock part{stats.record_ns[6]};
        // DIAG-243: the draws recorded last, for the report of a lost device.
        static const bool note_draws = [] {
            const char* env = std::getenv("RYGT_DEVICE_FAULT");
            return env && env[0] == '1';
        }();
        if (note_draws) {
            const auto& key = pipeline->GetGraphicsKey();
            NoteDrawForFault(XXH3_64bits(&key, sizeof(key)),
                             (pipeline->LdsEmulation() & 0xff) |
                                 (u32(regs.stage_enable.hs_en != 0) << 8));
        }
        if (!lds_deferred) {
            record_draw();
        }
        // DIAG-241 (diagnosis): in a dumped frame, the LDS scratch of the first such draw.
        if (lds_snap_address != 0 && r.FrameDumpCollecting() &&
            std::ranges::none_of(pass_snaps, [](const PassSnap& snap) {
                return snap.name.starts_with("snap_lds_");
            })) {
            auto buffer = device->CreateBuffer({.size = lds_snap_bytes,
                                                .usage = Native::Rhi::BufferUsageStorage |
                                                         Native::Rhi::BufferUsageTransferDst,
                                                .memory = Native::Rhi::MemoryKind::Readback,
                                                .name = "lds scratch snapshot"});
            if (buffer && buffer->Mapped()) {
                EndRendering();
                GlobalBarrier();
                stream.cmd.copyBuffer(
                    vk::Buffer{Handle<VkBuffer_T>(lds_scratch->NativeHandle())},
                    vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
                    vk::BufferCopy{.srcOffset = lds_snap_address - lds_scratch->GpuAddress(),
                                   .dstOffset = 0,
                                   .size = lds_snap_bytes});
                GlobalBarrier();
                pass_snaps.push_back({std::move(buffer),
                                      fmt::format("snap_lds_cp{}_{}.bin",
                                                  u32(regs.ls_hs_config.hs_input_control_points),
                                                  lds_snap_bytes)});
            }
        }
        if (lds_pushed) {
            // FIX-227: the quad sample mask's words back for the draws after.
            stream.cmd.pushConstants(pipeline->GetLayout(), AllGraphicsStageBits,
                                     u32(offsetof(Shader::PushData, quad_sample_mask)), 8u,
                                     b.push.quad_sample_mask.data());
        }
        lds_snap_address = 0;
        if (hud_white_state && HudStreamOn() && !stream.wrapped_list) {
            // GT7-UW: the same draw over white, in the HUD list (its state recorded there too).
            std::swap(stream, hud_stream);
            BeginHudStream();
            BeginRendering(*hud_white_state);
            bind_state();
            record_draw();
            std::swap(stream, hud_stream);
        } else if (hud_white_state) {
            // GT7-UW: the same draw again over white (the pipeline, bindings and dynamic state
            // stay bound across the change of attachment).
            BeginRendering(*hud_white_state);
            record_draw();
        }
    }
    {
        PhaseClock part{stats.record_ns[7]};
        NoteWrites(&targets, b);
    }
    for (size_t cb = 0; cb < targets.colors.size(); ++cb) {
        const auto* color = targets.colors[cb];
        if (!SeqTraced(color)) {
            continue;
        }
        // DIAG-208 (diagnosis): the draw, the subresources it was noted as drawing, and whether
        // a stage picks the layer itself.
        u64 vs = 0, ps = 0;
        bool layer_out = false;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            layer_out |= stage->stores.GetAny(Shader::IR::Attribute::RenderTargetIndex);
            if (stage->sw_stage == Shader::SwStage::Fragment) {
                ps = stage->pgm_hash;
            } else if (vs == 0) {
                vs = stage->pgm_hash;
            }
        }
        const auto& range = targets.color_ranges[cb];
        const auto& blend = regs.blend_control[cb];
        LOG_WARNING(Render_Vulkan, "DIAG-208: frame {} draw into {:#x} {}x{} mips {} layers {}: "
                    "mip {} layers {}+{}{} -> v{} @{} | vs {:#x} ps {:#x} layer out {} prim {} "
                    "n {} inst {} blend {} mask {:#x} pass {}x{}x{}", frame,
                    color->info.guest_address, color->info.size.width, color->info.size.height,
                    color->info.resources.levels, color->info.resources.layers, range.mip,
                    range.layer, range.layers,
                    targets.state.color_attachments[cb].is_clear ? " CLEAR" : "",
                    color->write_version.load(), static_cast<const void*>(color), vs, ps,
                    layer_out, u32(regs.primitive_type), regs.num_indices,
                    regs.num_instances.NumInstances(), u32(blend.enable),
                    regs.color_target_mask.raw, targets.state.width, targets.state.height,
                    targets.state.num_layers);
    }
    ProbeDraw(targets, regs); // DIAG-170 (diagnosis)
    StripDraw(targets);       // DIAG-173 (diagnosis)
    // DIAG-GT7-001: a draw whose reader pass waits counts in the segment the reader draws in.
    if (GpuSegment* seg = lds_deferred ? nullptr : GpuSeg()) {
        ++seg->draws;
        const u64 prims = u64(regs.num_indices) * std::max<u32>(regs.num_instances.NumInstances(), 1);
        if (prims > seg->max_prims) {
            const Shader::Info* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
            seg->max_prims = prims;
            seg->max_prims_ps = ps ? ps->pgm_hash : 0;
        }
    }
    ++stats.draws;
    if (lds_alone) {
        // PERF-GT7-060: the next one continues its run if none of these moved meanwhile.
        stats.lds_alone_marker = {stats.barriers, stats.passes, stats.draws, stats.dispatches};
    }
    if (lds_address != 0) {
        ReportLdsRuns();
    }
    if (indirect) {
        indirect_skip.skipped = nullptr;
        ++stats.indirect_draws;
    }
    ++s.commands;
    ChunkFlushCheck();
    Report();
}

std::optional<NativeRenderer::IndirectBuffers> NativeRenderer::ResolveIndirect(
    const IndirectDraw& indirect, bool is_indexed) {
    // What Vulkan can't draw as the GPU would is skipped, rather than drawn wrong.
    const u32 command_size =
        is_indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
    if (indirect.max_count == 0) {
        return std::nullopt; // no draws (none on the GPU either)
    }
    if (!r.instance.IsDrawIndirectFirstInstanceSupported()) {
        // The shaders read the first instance from the arguments (BaseInstance), which Vulkan
        // leaves undefined there without the feature unless it is 0.
        static const bool logged = [] {
            LOG_WARNING(Render_Vulkan, "NAT-161: indirect draws skipped: the device can't start "
                                       "them at an instance other than 0 "
                                       "(drawIndirectFirstInstance)");
            return true;
        }();
        return std::nullopt;
    }
    if (indirect.count_address != 0 && (!r.instance.IsDrawIndirectCountSupported() ||
                                        indirect.max_count > r.instance.MaxDrawIndirectCount())) {
        static const bool logged = [&] {
            LOG_WARNING(
                Render_Vulkan,
                "NAT-161: indirect draw of up to {} draws counted at {:#x} skipped: the "
                "device can't take that many draws from a count in memory "
                "(drawIndirectCount, at most {})",
                indirect.max_count, indirect.count_address,
                r.instance.IsDrawIndirectCountSupported() ? r.instance.MaxDrawIndirectCount() : 0u);
            return true;
        }();
        return std::nullopt;
    }
    // Several draws step through the arguments by the stride, which Vulkan wants 4-byte aligned
    // and no shorter than one draw's arguments (one draw's stride is ignored).
    if ((indirect.max_count > 1 || indirect.count_address != 0) &&
        (indirect.stride % 4 != 0 || indirect.stride < command_size)) {
        static const bool logged = [&] {
            LOG_WARNING(Render_Vulkan,
                        "NAT-161: indirect draw of up to {} draws with arguments {} bytes apart "
                        "skipped (Vulkan needs a multiple of 4, at least {})",
                        indirect.max_count, indirect.stride, command_size);
            return true;
        }();
        return std::nullopt;
    }
    // With a count in memory, max_count is only an upper bound (the GPU reads min(count,
    // max_count) draws), so the arguments looked up end where the mapping does, and at most
    // MaxCountArgs in: a huge bound would otherwise make GuestBuffer merge and drop regions for
    // a range that cannot exist.
    constexpr u64 MaxCountArgs = 16_MB;
    u32 max_count = indirect.max_count;
    u64 size = u64(indirect.stride) * (max_count - 1) + command_size;
    if (indirect.count_address != 0) {
        const u64 reach = std::min(r.memory->ClampRangeSize(indirect.address, size), MaxCountArgs);
        if (reach < command_size) {
            static const bool logged = [&] {
                LOG_WARNING(Render_Vulkan,
                            "NAT-161: indirect draw skipped: its arguments at {:#x} are not mapped",
                            indirect.address);
                return true;
            }();
            return std::nullopt;
        }
        if (reach < size) {
            max_count = static_cast<u32>((reach - command_size) / indirect.stride + 1);
            size = u64(indirect.stride) * (max_count - 1) + command_size;
        }
    }
    // The arguments where the GPU has them: in their region when shaders wrote them (NAT-114;
    // GT7's come from compute shader 850c97f9), else guest memory read in place.
    const auto args = GuestBuffer(indirect.address, size, false);
    if (!args || args->second % 4 != 0) {
        static const bool logged = [&] {
            LOG_WARNING(Render_Vulkan,
                        "NAT-161: indirect draw skipped: its arguments at {:#x}+{:#x} are {}",
                        indirect.address, size,
                        args ? "not 4-byte aligned" : "not readable by the GPU");
            return true;
        }();
        return std::nullopt;
    }
    IndirectBuffers buffers{.args = args->first, .args_offset = args->second, .max_count = max_count};
    if (indirect.count_address != 0) {
        const auto count = GuestBuffer(indirect.count_address, sizeof(u32), false);
        if (!count || count->second % 4 != 0) {
            static const bool logged = [&] {
                LOG_WARNING(Render_Vulkan,
                            "NAT-161: indirect draw skipped: its draw count at {:#x} is {}",
                            indirect.count_address,
                            count ? "not 4-byte aligned" : "not readable by the GPU");
                return true;
            }();
            return std::nullopt;
        }
        buffers.count = count->first;
        buffers.count_offset = count->second;
    }
    return buffers;
}

void NativeRenderer::RecordIndirect(const IndirectDraw& indirect, const IndirectBuffers& buffers,
                                    bool is_indexed) {
    // The guest's argument structures are Vulkan's (pm4_cmds.h): index or vertex count,
    // instance count, first index or vertex, (base vertex,) first instance.
    const CmdBuf cmdbuf{stream.cmd};
    if (buffers.count) {
        if (is_indexed) {
            cmdbuf.drawIndexedIndirectCount(buffers.args, buffers.args_offset, buffers.count,
                                            buffers.count_offset, buffers.max_count,
                                            indirect.stride);
        } else {
            cmdbuf.drawIndirectCount(buffers.args, buffers.args_offset, buffers.count,
                                     buffers.count_offset, buffers.max_count, indirect.stride);
        }
        return;
    }
    // A command makes at most maxDrawIndirectCount draws (1 without multiDrawIndirect).
    const u32 per_command = std::max(r.instance.MaxDrawIndirectCount(), 1u);
    for (u32 first = 0; first < buffers.max_count;) {
        const u32 count = std::min(per_command, buffers.max_count - first);
        const u64 offset = buffers.args_offset + u64(first) * indirect.stride;
        if (is_indexed) {
            cmdbuf.drawIndexedIndirect(buffers.args, offset, count, indirect.stride);
        } else {
            cmdbuf.drawIndirect(buffers.args, offset, count, indirect.stride);
        }
        first += count;
    }
}

bool NativeRenderer::PrepareEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_PREPARE");
        if (!(env && env[0] == '1')) {
            return false;
        }
        // FIX-229: prepared draws (GT7-PREP) lost the GPU device with FIX-227's hull-shader LDS
        // emulation on (two of three Alsace runs with eight workers); ignored unless forced.
        const char* lds = std::getenv("RYGT_TESS_LDS_EMULATION");
        const char* force = std::getenv("RYGT_NATIVE_PREPARE_FORCE");
        if (!(lds && lds[0] == '0') && !(force && force[0] == '1')) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-229: RYGT_NATIVE_PREPARE=1 is ignored: prepared draws are not safe "
                        "with the race-start car fix (FIX-227); RYGT_NATIVE_PREPARE_FORCE=1 uses "
                        "them anyway");
            return false;
        }
        return true;
    }();
    return enabled;
}

void NativeRenderer::PreparedDraw::Reset() {
    packet = nullptr;
    epoch = 0;
    pipeline = nullptr;
    stage_hashes = {};
    bindings.Clear();
    regions.clear();
    image_ranges.clear();
    buffer_ranges.clear();
    upload_bytes.clear();
    uploads.clear();
    null_images = 0;
    has_vertices = false;
    vertex_hash = 0;
    indexed = false;
    index_offset = 0;
    index_address = 0;
    mirrors.clear();
    in_place.clear();
    for (auto& copy : stage_copies) {
        copy.present = false;
    }
}

void NativeRenderer::ChangeLog::Note(VAddr begin, VAddr end) {
    const u64 next = seq.load(std::memory_order_relaxed) + 1;
    entries[next % entries.size()] = {next, begin, end};
    seq.store(next, std::memory_order_release);
}

bool NativeRenderer::ChangeLog::Untouched(u64 since,
                                          std::span<const std::pair<VAddr, VAddr>> ranges) const {
    const u64 now = seq.load(std::memory_order_acquire);
    if (now == since) {
        return true;
    }
    if (now - since > entries.size()) {
        return false;
    }
    for (u64 at = since + 1; at <= now; ++at) {
        const Entry& entry = entries[at % entries.size()];
        for (const auto& [begin, end] : ranges) {
            if (entry.begin < end && entry.end > begin) {
                return false;
            }
        }
    }
    return true;
}

NativeRenderer::PreparedDraw* NativeRenderer::NewPrepared() {
    std::scoped_lock lock{prepared_pool_mutex};
    if (!prepared_free.empty()) {
        PreparedDraw* draw = prepared_free.back();
        prepared_free.pop_back();
        return draw;
    }
    prepared_all.push_back(std::make_unique<PreparedDraw>());
    return prepared_all.back().get();
}

void NativeRenderer::FreePrepared(PreparedDraw* draw) {
    draw->Reset();
    std::scoped_lock lock{prepared_pool_mutex};
    prepared_free.push_back(draw);
}

bool NativeRenderer::HasPrepared(const void* packet) {
    if (!packet) {
        return false;
    }
    PreparedShard& shard = ShardFor(packet);
    std::scoped_lock lock{shard.mutex};
    return shard.draws.contains(packet);
}

const GraphicsPipeline* NativeRenderer::InstallPrepared(PreparedDraw& p) {
    // As AcquireGraphicsPipeline with the command thread's selection: the stages' user data
    // and flattened user data, the flattened buffers moved in (and the old ones kept here).
    std::array<Rasterizer::SelectedStage, Shader::MaxStageTypes> selected{};
    for (u32 i = 0; i < selected.size(); ++i) {
        const auto& copy = p.stage_copies[i];
        auto& stage = selected[i];
        stage.present = copy.present;
        if (!copy.present) {
            continue;
        }
        stage.pgm_base = copy.pgm_base;
        stage.num_user_data = copy.num_user_data;
        std::copy_n(copy.user_data.begin(), copy.num_user_data, stage.user_data.begin());
        stage.flattened = std::move(p.stage_copies[i].flattened);
    }
    const bool installed = r.InstallDrawStages(p.pipeline, selected, true);
    for (u32 i = 0; i < selected.size(); ++i) {
        p.stage_copies[i].flattened = std::move(selected[i].flattened);
    }
    if (!installed) {
        return nullptr;
    }
    ++stats.prepared_installed;
    return p.pipeline;
}

NativeRenderer::PreparedDraw* NativeRenderer::TakePrepared(const void* packet) {
    if (!packet) {
        return nullptr;
    }
    PreparedShard& shard = ShardFor(packet);
    std::scoped_lock lock{shard.mutex};
    const auto it = shard.draws.find(packet);
    if (it == shard.draws.end()) {
        return nullptr;
    }
    PreparedDraw* draw = it->second;
    shard.draws.erase(it);
    return draw;
}

u64 NativeRenderer::StageHash(const Shader::Info& info) {
    // What the bindings were worked out from: the stage's sharps (the user data registers the
    // program doesn't read hold whatever earlier work left, which a job and the command
    // processor need not agree on).
    static thread_local std::vector<u32> words;
    words.clear();
    const auto add = [](const auto& sharp) {
        const auto* data = reinterpret_cast<const u32*>(&sharp);
        words.insert(words.end(), data, data + sizeof(sharp) / sizeof(u32));
    };
    for (const auto& desc : info.buffers) {
        if (!desc.IsSpecial()) {
            add(desc.GetSharp(info));
        }
    }
    for (const auto& desc : info.images) {
        add(desc.GetSharp(info));
    }
    for (const auto& desc : info.samplers) {
        add(desc.GetSharp(info));
    }
    return XXH3_64bits_withSeed(words.data(), words.size() * sizeof(u32), info.pgm_base);
}

void NativeRenderer::PrepareJobDraw(const void* packet, u32 epoch, const AmdGpu::Regs& regs,
                                    const GraphicsPipeline& pipeline,
                                    std::span<const Shader::Info* const> stages, bool is_indexed,
                                    u32 index_offset) {
    const auto start = std::chrono::steady_clock::now();
    PreparedDraw* p = NewPrepared();
    p->packet = packet;
    p->epoch = epoch;
    p->pipeline = &pipeline;
    // What the lookups depend on, read before them.
    p->image_generation = registry->Generation();
    p->buffer_seq = buffer_log.Seq();
    p->unmap_gen = unmap_generation.load(std::memory_order_acquire);
    p->bindings.push = UserData(regs);
    const u32 count = std::min<u32>(static_cast<u32>(stages.size()), Shader::MaxStageTypes - 1);
    std::array<StageResourceBindings, Shader::MaxStageTypes> plans{};
    bool ok = true;
    size_t total = 0;
    for (u32 i = 0; i < count; ++i) {
        if (const auto* stage = stages[i]) {
            total += stage->buffers.size() + stage->images.size() + stage->samplers.size();
        }
    }
    p->bindings.writes.reserve(total);
    for (u32 i = 0; i < count; ++i) {
        // OPT-012: what the recorder installs (as the command thread's selection would copy).
        auto& copy = p->stage_copies[i];
        const auto* stage = stages[i];
        copy.present = stage != nullptr;
        if (stage) {
            copy.pgm_base = stage->pgm_base;
            copy.num_user_data =
                static_cast<u32>(std::min<size_t>(stage->user_data.size(), copy.user_data.size()));
            std::copy_n(stage->user_data.begin(), copy.num_user_data, copy.user_data.begin());
            copy.flattened.assign(stage->flattened_ud_buf.begin(), stage->flattened_ud_buf.end());
        }
    }
    for (u32 i = 0; ok && i < count; ++i) {
        const auto* stage = stages[i];
        p->stage_hashes[i] = stage ? StageHash(*stage) : 0;
        if (!stage) {
            continue;
        }
        plans[i] = PlanStageResourceBindings(*stage,
                                             Shader::Backend::StageStart(u32(stage->sw_stage)));
        auto buffer_binding = plans[i].buffers;
        ok = PrepareBuffers(*stage, regs, buffer_binding, *p);
    }
    for (u32 i = 0; ok && i < count; ++i) {
        if (const auto* stage = stages[i]) {
            auto texture_binding = plans[i].textures;
            ok = PrepareTextures(*stage, regs, texture_binding, *p);
        }
    }
    if (!ok) {
        ++prepare_stats.failed;
        FreePrepared(p);
        return;
    }
    // OPT-011: the vertex and index buffers too, when they need nothing ordered.
    constexpr u32 VertexStage = static_cast<u32>(Shader::SwStage::Vertex);
    if (VertexStage < count && stages[VertexStage]) {
        p->has_vertices =
            PrepareVertices(*p, regs, pipeline, *stages[VertexStage], is_indexed, index_offset);
    }
    std::vector<PreparedDraw*> dropped;
    {
        PreparedShard& shard = ShardFor(packet);
        std::scoped_lock lock{shard.mutex};
        if (shard.draws.size() > 1024) {
            // Draws the recorder never took (not submitted, or filtered): from a few
            // submissions ago, they go.
            for (auto it = shard.draws.begin(); it != shard.draws.end();) {
                if (epoch - it->second->epoch > 4) {
                    dropped.push_back(it->second);
                    it = shard.draws.erase(it);
                } else {
                    ++it;
                }
            }
        }
        const auto [it, inserted] = shard.draws.try_emplace(packet, p);
        if (!inserted) {
            dropped.push_back(it->second);
            it->second = p;
            ++prepare_stats.replaced;
        }
    }
    prepare_stats.stale += dropped.size();
    for (PreparedDraw* draw : dropped) {
        FreePrepared(draw);
    }
    ++prepare_stats.made;
    prepare_stats.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::JobGuestBuffer(VAddr address, u64 size,
                                                                         bool write,
                                                                         PreparedDraw& p) {
    {
        std::shared_lock map_lock{buffer_map_mutex};
        if (Region* region = RegionOver(address, size)) {
            if (address < region->begin || address + size > region->end) {
                return std::nullopt; // the region grows over it first
            }
            p.regions.push_back(region);
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
    }
    if (write) {
        return std::nullopt; // a region is made for it (NAT-144), or mirrors dropped
    }
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (direct && direct->EnsureImported(address, size)) {
        if (const auto found = direct->FindBuffer(address, size)) {
            return std::pair{found->first->Handle(), found->second};
        }
    }
    return std::nullopt; // gathered or copied, in GPU order
}

bool NativeRenderer::PrepareBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs,
                                    Shader::Backend::Bindings& binding, PreparedDraw& p) {
    // As BindBuffers, without its side effects.
    DrawBindings& b = p.bindings;
    const u64 alignment = r.instance.StorageMinAlignment();
    auto* direct = stage.uses_direct ? r.buffer_cache.GetDirectMemory() : nullptr;
    static thread_local std::vector<std::array<u32, 4>> direct_entries;
    direct_entries.assign(stage.buffers.size(), {});
    constexpr u64 GpuAddressLimit = 1ULL << 40;
    const auto upload = [&](const void* data, u32 size) {
        const u32 offset = static_cast<u32>(p.upload_bytes.size());
        const auto* bytes = static_cast<const u8*>(data);
        p.upload_bytes.insert(p.upload_bytes.end(), bytes, bytes + size);
        p.uploads.push_back({static_cast<u32>(b.buffers.size()), offset, size});
        return vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, size};
    };
    u32 index = 0;
    for (const auto& desc : stage.buffers) {
        const u32 buffer_index = index++;
        vk::DescriptorBufferInfo info{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        if (desc.IsSpecial()) {
            switch (desc.buffer_type) {
            case Shader::BufferType::GdsBuffer:
                info = {Handle<VkBuffer_T>(gds->NativeHandle()), 0, gds->Size()};
                break;
            case Shader::BufferType::Flatbuf: {
                // The recorder's own copy of the stage's user data (the registers the program
                // doesn't read may differ here), uploaded at commit.
                const u32 size = static_cast<u32>(stage.flattened_ud_buf.size() * sizeof(u32));
                if (size != 0) {
                    p.uploads.push_back({static_cast<u32>(b.buffers.size()), 0, size,
                                         static_cast<s32>(stage.sw_stage)});
                    info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, size};
                }
                break;
            }
            case Shader::BufferType::ClipPlanes:
                if (regs.clipper_control.user_clip_plane_enable != 0) {
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    info = upload(planes.data(), sizeof(planes));
                }
                break;
            case Shader::BufferType::DirectTable: {
                const u32 words = static_cast<u32>(4 + stage.buffers.size() * 4);
                static thread_local std::vector<u32> table;
                table.assign(words, 0);
                if (auto* memory = r.buffer_cache.GetDirectMemory()) {
                    const u64 l1 = memory->L1Address();
                    const u64 zero = memory->ZeroPageAddress();
                    table[0] = u32(l1);
                    table[1] = u32(l1 >> 32);
                    table[2] = u32(zero);
                    table[3] = u32(zero >> 32);
                    for (size_t i = 0; i < stage.buffers.size(); ++i) {
                        std::memcpy(&table[4 + i * 4], direct_entries[i].data(), 16);
                    }
                }
                info = upload(table.data(), words * sizeof(u32));
                break;
            }
            default:
                return false; // compute-only kinds and the unsupported ones: the recorder
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            const u64 base = vsharp.base_address;
            const u64 sharp_size = vsharp.GetSize();
            const bool impossible =
                vsharp.num_records != UINT32_MAX &&
                (sharp_size >= GpuAddressLimit || base + sharp_size > GpuAddressLimit ||
                 (sharp_size >= 1_GB && base != 0 && !r.memory->IsValidMapping(base)));
            const bool unmapped =
                !impossible && base != 0 && sharp_size != 0 && !r.IsMappedStart(base);
            if (base != 0 && sharp_size != 0 && !impossible && !unmapped) {
                const u64 size = r.memory->ClampRangeSize(base, sharp_size);
                bool region_over;
                {
                    std::shared_lock map_lock{buffer_map_mutex};
                    region_over = RegionOver(base, size) != nullptr;
                }
                p.buffer_ranges.emplace_back(base, base + size);
                if (direct && desc.IsDirectEligible() && size < (u64{1} << 32) && !region_over &&
                    direct->EnsureImported(base, size)) {
                    direct_entries[buffer_index] = {u32(base), u32(base >> 32), u32(size), 1u};
                } else if (const auto found = JobGuestBuffer(base, size, desc.is_written, p)) {
                    const u64 aligned = Common::AlignDown(found->second, alignment);
                    const u64 adjust = found->second - aligned;
                    b.push.AddOffset(binding.buffer, static_cast<u32>(adjust));
                    info = {found->first, aligned, size + adjust};
                    b.writes_buffers |= desc.is_written;
                    if (desc.is_written) {
                        b.gpu_writes.emplace_back(base, size); // NAT-119
                    }
                } else {
                    return false;
                }
            }
        }
        b.buffers.push_back(info);
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
        b.writes.push_back(vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &b.buffers.back(),
        });
        ++binding.buffer;
    }
    return true;
}

bool NativeRenderer::PrepareTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                                     Shader::Backend::Bindings& binding, PreparedDraw& p) {
    // As BindTextures, without its side effects: pictures that exist only, no rewrites or
    // copies of sources (the recorder does them).
    DrawBindings& b = p.bindings;
    const u64 now_frame = shared_frame.load(std::memory_order_relaxed);
    static thread_local std::array<JobTextureMemo, 4096> memos{};
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        const bool cube_sampling =
            image_desc.post_op == Shader::SharpFetchPostOp::NativeCubeSampling;
        const u32 num_bindings = image_desc.NumBindings(stage);
        const u32 first = static_cast<u32>(b.images.size());
        const bool storage = image_desc.is_written;
        const auto add_write = [&] {
            b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
            b.writes.push_back(vk::WriteDescriptorSet{
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = binding.unified,
                .dstArrayElement = 0,
                .descriptorCount = num_bindings,
                .descriptorType =
                    storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
                .pImageInfo = &b.images[first],
            });
            binding.unified += num_bindings;
        };
        const auto bind_null = [&] {
            for (u32 i = 0; i < num_bindings; ++i) {
                b.images.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral});
            }
            ++p.null_images;
            add_write();
        };
        std::array<u64, 4> raw_sharp{};
        static_assert(sizeof(tsharp) == sizeof(raw_sharp));
        std::memcpy(raw_sharp.data(), &tsharp, sizeof(raw_sharp));
        const u32 memo_flags = u32(storage) | (u32(image_desc.is_array) << 1) |
                               (u32(image_desc.is_depth) << 2) |
                               (u32(image_desc.mip_fallback_mode) << 3) |
                               ((image_desc.constant_mip_index & 0xff) << 6) |
                               (num_bindings << 16) | (u32(cube_sampling) << 31);
        if (num_bindings > 4 || stage.pgm_hash == 0xaa3822a3) {
            return false; // the recorder (FIX-081's grass shader, long arrays)
        }
        const u64 memo_generation = p.image_generation + p.unmap_gen;
        JobTextureMemo& memo = memos[XXH3_64bits_withSeed(raw_sharp.data(), sizeof(raw_sharp),
                                                       memo_flags) &
                                  (memos.size() - 1)];
        const VAddr address = tsharp.Address();
        if (memo.entry && memo.flags == memo_flags && memo.sharp == raw_sharp &&
            memo.generation == memo_generation) {
            Native::ImageEntry* entry = memo.entry;
            entry->last_use.store(now_frame, std::memory_order_relaxed);
            for (u32 i = 0; i < num_bindings; ++i) {
                b.images.push_back({VK_NULL_HANDLE, memo.views[i], vk::ImageLayout::eGeneral});
            }
            (storage ? b.stored : b.sampled).push_back(entry);
            p.image_ranges.emplace_back(address, address + std::max<u64>(entry->info.guest_size, 1));
            add_write();
            continue;
        }
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (address == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt) ||
            !r.memory->IsValidGpuMapping(address, 0) ||
            LiverpoolToVK::TrySurfaceFormat(data_fmt, num_fmt) == vk::Format::eUndefined ||
            VideoCore::CheckImageDescriptorGeometry(tsharp, r.instance.GetImageLimits()) !=
                VideoCore::ImageDescriptorGeometryError::None) {
            bind_null();
            continue;
        }
        const VideoCore::TextureCache::ImageDesc desc{tsharp, image_desc};
        if (!r.IsMappedStart(desc.info.guest_address)) {
            bind_null();
            continue;
        }
        Native::ImageEntry* entry = registry->Find(desc.info, storage, now_frame, false);
        if (!entry) {
            return false; // made by the recorder, in GPU order
        }
        const auto fallback = image_desc.mip_fallback_mode;
        for (u32 i = 0; i < num_bindings; ++i) {
            auto view_info = desc.view_info;
            if (fallback == Shader::MipStorageFallbackMode::ConstantIndex) {
                view_info.range.base.level += image_desc.constant_mip_index;
                view_info.range.extent.levels = 1;
            } else if (fallback == Shader::MipStorageFallbackMode::DynamicIndex) {
                view_info.range.base.level += i;
                view_info.range.extent.levels = 1;
            }
            const vk::ImageView view =
                View(*entry, view_info, storage        ? ViewUse::Storage
                                        : cube_sampling ? ViewUse::CubeSampled
                                                        : ViewUse::Sampled);
            b.images.push_back({VK_NULL_HANDLE, view, vk::ImageLayout::eGeneral});
        }
        memo.sharp = raw_sharp;
        memo.flags = memo_flags;
        memo.entry = entry;
        for (u32 i = 0; i < num_bindings; ++i) {
            memo.views[i] = b.images[first + i].imageView;
        }
        memo.generation = memo_generation;
        (storage ? b.stored : b.sampled).push_back(entry);
        p.image_ranges.emplace_back(desc.info.guest_address,
                                    desc.info.guest_address +
                                        std::max<u64>(desc.info.guest_size, 1));
        add_write();
    }
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                regs.ta_bc_base.Address() == 0)) {
            ssharp = AmdGpu::Sampler{};
        }
        b.images.push_back({SamplerFor(ssharp, regs.ta_bc_base, sampler.is_depth), VK_NULL_HANDLE,
                            vk::ImageLayout::eGeneral});
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
        b.writes.push_back(vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &b.images.back(),
        });
    }
    return true;
}

u64 NativeRenderer::VertexHash(const GraphicsPipeline& pipeline, const Shader::Info& vs,
                                u32 step_rate_0, u32 step_rate_1) {
    // The V#s the fetch shader reads (what the vertex buffers were worked out from).
    const auto& fetch = pipeline.GetFetchShader();
    std::array<u32, 2 + 4 * MaxVertexBufferCount> words{step_rate_0, step_rate_1};
    u32 count = 2;
    if (!fetch.Empty()) {
        for (const auto& attrib : fetch.attributes) {
            if (count + 4 > words.size()) {
                break;
            }
            const auto sharp = attrib.GetSharp(vs);
            static_assert(sizeof(sharp) == 4 * sizeof(u32));
            std::memcpy(&words[count], &sharp, sizeof(sharp));
            count += 4;
        }
    }
    return XXH3_64bits(words.data(), count * sizeof(u32));
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::JobVertexBuffer(VAddr address, u64 size,
                                                                          PreparedDraw& p,
                                                                          Mirror** mirror_out) {
    // As GuestBufferLookup for vertices and indices (a region, else a mirror, else in place).
    // The caller holds buffer_map_mutex shared.
    *mirror_out = nullptr;
    {
        if (Region* region = RegionOver(address, size)) {
            if (address < region->begin || address + size > region->end) {
                return std::nullopt;
            }
            if (p.regions.empty() || p.regions.back() != region) {
                p.regions.push_back(region);
            }
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
        static const bool mirrors_on = [] {
            const char* env = std::getenv("RYGT_NATIVE_MIRRORS");
            return !(env && env[0] == '0');
        }();
        if (mirrors_on && size <= 64_MB) {
            if (Mirror* mirror = MirrorHolding(address, size)) {
                if (std::ranges::find(p.mirrors, mirror) == p.mirrors.end()) {
                    p.mirrors.push_back(mirror);
                }
                *mirror_out = mirror;
                return std::pair{vk::Buffer{Handle<VkBuffer_T>(mirror->buffer->NativeHandle())},
                                 u64{address - mirror->begin}};
            }
        }
    }
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (direct && direct->EnsureImported(address, size)) {
        if (const auto found = direct->FindBuffer(address, size)) {
            p.in_place.emplace_back(address, size);
            return std::pair{found->first->Handle(), found->second};
        }
    }
    return std::nullopt;
}

bool NativeRenderer::PrepareVertices(PreparedDraw& p, const AmdGpu::Regs& regs,
                                     const GraphicsPipeline& pipeline, const Shader::Info& vs,
                                     bool is_indexed, u32 index_offset) {
    if (regs.primitive_type == AmdGpu::PrimitiveType::QuadList) {
        return false; // the recorder makes their triangles
    }
    if (!p.vertex_index) {
        p.vertex_index = std::make_unique<VertexIndexState>();
    }
    VertexIndexState& v = *p.vertex_index;
    v = {};
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(v.attributes, v.input_bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1, vs);
    p.vertex_hash = VertexHash(pipeline, vs, regs.vgt_instance_step_rate_0,
                               regs.vgt_instance_step_rate_1);
    // Regions and mirrors stay put while this thread reads them.
    std::shared_lock map_lock{buffer_map_mutex};
    Mirror* near = nullptr;
    for (const auto& guest : guest_buffers) {
        std::optional<std::pair<vk::Buffer, u64>> found;
        u64 size = 0;
        const VAddr base = guest.base_address;
        const u64 sharp_size = guest.GetSize();
        if (near && base != 0 && sharp_size > 0 && base >= near->begin &&
            base + sharp_size <= near->end) {
            size = sharp_size; // OPT-004 on this thread
            found = std::pair{vk::Buffer{Handle<VkBuffer_T>(near->buffer->NativeHandle())},
                              u64{base - near->begin}};
        } else if (base != 0 && sharp_size > 0 && r.memory->MappedAreaEnd(base, base + 1) != 0) {
            size = r.memory->ClampRangeSize(base, sharp_size);
            found = JobVertexBuffer(base, size, p, &near);
            if (!found) {
                return false;
            }
        }
        if (size != 0) {
            p.buffer_ranges.emplace_back(base, base + size);
        }
        v.buffers.push_back(found ? found->first : vk::Buffer{});
        v.offsets.push_back(found ? found->second : 0);
        v.sizes.push_back(found ? size : 0);
        v.strides.push_back(guest.GetStride());
    }
    p.indexed = is_indexed;
    p.index_offset = index_offset;
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    v.index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    v.index_count = regs.num_indices;
    if (is_indexed) {
        const VAddr address =
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
        const u64 bytes = u64(regs.num_indices) * index_size;
        if (bytes == 0) {
            return false;
        }
        Mirror* unused = nullptr;
        const auto found = JobVertexBuffer(address, bytes, p, &unused);
        if (!found) {
            return false;
        }
        p.buffer_ranges.emplace_back(address, address + bytes);
        p.index_address = address;
        v.index_buffer = found->first;
        v.index_offset = found->second;
    }
    return true;
}

bool NativeRenderer::CommitPreparedVertices(PreparedDraw& p, const GraphicsPipeline& pipeline,
                                            const AmdGpu::Regs& regs, bool is_indexed,
                                            u32 index_offset, bool quad_triangles,
                                            bool trusted) {
    // (The ranges were checked with the bindings'.)
    const auto reject = [&] {
        ++stats.prepared_vertices_rejected;
        return false;
    };
    if (quad_triangles || p.indexed != is_indexed || p.index_offset != index_offset) {
        return reject();
    }
    if (!trusted &&
        VertexHash(pipeline, pipeline.GetStage(Shader::SwStage::Vertex),
                   regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1) != p.vertex_hash) {
        return reject();
    }
    const auto& v = *p.vertex_index;
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    if (v.index_count != regs.num_indices ||
        v.index_type != (is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32)) {
        return reject();
    }
    if (is_indexed) {
        const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
        if (p.index_address !=
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size) {
            return reject();
        }
    }
    // The ordered part: pages the CPU wrote into the mirrors, and the ranges read in place
    // (noted as mirror candidates).
    for (Mirror* mirror : p.mirrors) {
        RefreshMirror(*mirror);
        mirror->used_frame = frame;
        ++stats.mirror_binds;
    }
    for (const auto& [address, size] : p.in_place) {
        MirrorBuffer(address, size);
    }
    ++stats.prepared_vertices;
    return true;
}

bool NativeRenderer::CommitPrepared(PreparedDraw& p, const GraphicsPipeline& pipeline,
                                    const AmdGpu::Regs& regs, bool trusted) {
    const auto reject = [&](u32 why) {
        ++stats.prepared_rejected[why];
        return false;
    };
    if (p.pipeline != &pipeline) {
        return reject(0);
    }
    const auto stages = pipeline.GetStages();
    // OPT-012: stages installed from this prepared draw are what it was worked out from.
    for (u32 i = 0; !trusted && i < stages.size() && i < p.stage_hashes.size(); ++i) {
        if ((stages[i] ? StageHash(*stages[i]) : 0) != p.stage_hashes[i]) {
            return reject(1);
        }
    }
    for (const auto& upload : p.uploads) {
        if (upload.flat_stage >= 0) {
            const Shader::Info* stage = stages[upload.flat_stage];
            if (!stage || stage->flattened_ud_buf.size() * sizeof(u32) != upload.size) {
                return reject(1);
            }
        }
    }
    if (p.unmap_gen != unmap_generation.load(std::memory_order_acquire)) {
        return reject(2);
    }
    if (!registry->Untouched(p.image_generation, p.image_ranges)) {
        return reject(3);
    }
    SortRegionWrites(); // CPU writes since (may drop mirrors)
    if (!buffer_log.Untouched(p.buffer_seq, p.buffer_ranges)) {
        return reject(4);
    }
    // The ordered part, as BindBuffers and BindTextures do it.
    for (Region* region : p.regions) {
        RefreshRegion(*region);
        ++stats.region_binds;
    }
    const u64 alignment = r.instance.StorageMinAlignment();
    for (const auto& upload : p.uploads) {
        const u8* source = p.upload_bytes.data() + upload.offset;
        u32 size = upload.size;
        if (upload.flat_stage >= 0) {
            source = reinterpret_cast<const u8*>(stages[upload.flat_stage]->flattened_ud_buf.data());
        }
        const auto [buffer, offset, data] = Allocate(size, alignment);
        std::memcpy(data, source, size);
        p.bindings.buffers[upload.info] = vk::DescriptorBufferInfo{buffer, offset, size};
    }
    for (const auto* stored : p.bindings.stored) {
        auto& entry = *const_cast<Native::ImageEntry*>(stored);
        entry.last_use.store(frame, std::memory_order_relaxed);
        ApplyRewrite(entry); // NAT-120
        SyncDepthTwin(entry); // FIX-212
    }
    for (const auto* sampled : p.bindings.sampled) {
        auto& entry = *const_cast<Native::ImageEntry*>(sampled);
        entry.last_use.store(frame, std::memory_order_relaxed);
        ApplyRewrite(entry); // NAT-120, NAT-133
        SyncDepthTwin(entry); // FIX-212
        if (entry.from_cpu && registry->RewrittenSmall(entry)) {
            ++stats.small_rewrites; // FIX-202
        }
        RefreshSources(entry); // NAT-104 (FIX-197: targets too)
    }
    // The viewport transform the shaders get, from the registers as recorded.
    const Shader::PushData fresh = UserData(regs);
    p.bindings.push.xoffset = fresh.xoffset;
    p.bindings.push.xscale = fresh.xscale;
    p.bindings.push.yoffset = fresh.yoffset;
    p.bindings.push.yscale = fresh.yscale;
    p.bindings.push.quad_sample_mask = fresh.quad_sample_mask;
    stats.null_images += p.null_images;
    ++stats.prepared_used;
    return true;
}

void NativeRenderer::GpuWrote(VAddr address, u64 size, const std::array<u32, 2>* pattern,
                              bool exact) {
    const u64 clock = ++write_clock; // NAT-148: when the memory was rewritten
    registry->GpuWrote(address, size, pattern, exact, frame, clock);
    // A fill of a stencil plane: depth images whose stencil is there clear it at next use.
    const auto it = stencil_owners.find(address);
    if (it == stencil_owners.end() || !pattern) {
        return;
    }
    const u32 lo = (*pattern)[0];
    const bool uniform = (*pattern)[0] == (*pattern)[1] && lo == (lo & 0xffu) * 0x01010101u;
    if (!uniform) {
        return;
    }
    for (auto* entry : it->second) {
        entry->stencil_clear_pending = true;
        entry->stencil_clear_value = static_cast<u8>(lo);
        entry->rewrite_clock = std::max(entry->rewrite_clock, clock);
    }
}

const Native::Rhi::Buffer* NativeRenderer::PatternBuffer(u32 lo, u32 hi, u64 bytes) {
    // NAT-134: a device buffer holding lo, hi, lo, hi... for fills, made once per pattern and
    // grown when a bigger picture needs it (only ever read afterwards).
    const u64 key = u64(lo) | (u64(hi) << 32);
    auto& slot = fill_patterns[key];
    if (slot && slot->Size() >= bytes) {
        return slot.get();
    }
    const u64 size = std::max<u64>(Common::AlignUp(bytes, 64_KB), 1_MB);
    auto buffer = device->CreateBuffer({.size = size,
                                        .usage = Native::Rhi::BufferUsageTransferDst |
                                                 Native::Rhi::BufferUsageTransferSrc,
                                        .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                        .name = "native fill pattern"});
    if (!buffer) {
        return slot.get();
    }
    Stream& s = Begin();
    EndRendering();
    const vk::Buffer handle{Handle<VkBuffer_T>(buffer->NativeHandle())};
    const auto transfer_barrier = [&] {
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
        };
        s.cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    };
    if (lo == hi) {
        s.cmd.fillBuffer(handle, 0, VK_WHOLE_SIZE, lo);
    } else {
        // 8 KB of the pattern written in, then doubled by copies until the buffer is full.
        std::array<u32, 2048> words;
        for (size_t i = 0; i < words.size(); i += 2) {
            words[i] = lo;
            words[i + 1] = hi;
        }
        u64 filled = sizeof(words);
        s.cmd.updateBuffer(handle, 0, filled, words.data());
        while (filled < size) {
            transfer_barrier();
            const u64 chunk = std::min(filled, size - filled);
            s.cmd.copyBuffer(handle, handle, vk::BufferCopy{.srcOffset = 0, .dstOffset = filled,
                                                            .size = chunk});
            filled += chunk;
        }
    }
    transfer_barrier();
    ++s.commands;
    if (slot) {
        retired_patterns.push_back(std::move(slot)); // still read by recorded copies
    }
    slot = std::move(buffer);
    return slot.get();
}

bool NativeRenderer::FillImage(Native::ImageEntry& entry, u32 lo, u32 hi, u32 rows,
                               u32 first_layers) {
    // NAT-132/134: a picture filled with the pattern (two dwords, repeated), as memory holds it
    // after a fill: copied in from a buffer holding it (any format: every texel is the pattern's
    // bytes). `rows` != 0: only those rows of the first mip and layer (NAT-124), or of its first
    // layers with `first_layers` (FIX-GT7-031).
    const auto& info = entry.info;
    const u32 texel = info.props.is_depth
                          ? (info.pixel_format == vk::Format::eD16Unorm ||
                                     info.pixel_format == vk::Format::eD16UnormS8Uint
                                 ? 2u
                                 : 4u)
                          : info.num_bits / 8;
    if (texel == 0 || info.props.is_block || info.num_samples != 1 || !entry.image ||
        (lo != hi && texel < 8 && texel != 4)) {
        return false;
    }
    const auto& desc = entry.image->Desc();
    const u32 layers = desc.dim == Native::Rhi::ImageDim::D3 ? 1u : std::max(desc.layers, 1u);
    const u32 depth = desc.dim == Native::Rhi::ImageDim::D3 ? std::max(desc.depth, 1u) : 1u;
    const u32 mips = rows ? 1u : std::max(desc.mips, 1u);
    const u32 height0 = rows ? std::min(rows, desc.height) : desc.height;
    const u32 row_layers = std::clamp(first_layers, 1u, layers);
    const u64 bytes = u64(desc.width) * height0 * texel * (rows ? row_layers : layers * depth);
    if (info.props.is_bgr_packed_float) {
        const auto repack = [](u32 value) {
            return (value >> 21) | (((value >> 10) & 0x7ffu) << 11) |
                   ((value & 0x3ffu) << 22);
        };
        lo = repack(lo);
        hi = repack(hi);
    }
    const Native::Rhi::Buffer* pattern = PatternBuffer(lo, hi, bytes);
    if (!pattern || pattern->Size() < bytes) {
        return false;
    }
    Stream& s = Begin();
    GlobalBarrier(); // the work before is done with the picture
    const vk::Buffer buffer{Handle<VkBuffer_T>(pattern->NativeHandle())};
    boost::container::small_vector<vk::BufferImageCopy, 16> copies;
    for (u32 m = 0; m < mips; ++m) {
        const u32 w = std::max(desc.width >> m, 1u);
        const u32 h = rows ? height0 : std::max(desc.height >> m, 1u);
        const u32 d = std::max(depth >> m, 1u);
        copies.push_back(vk::BufferImageCopy{
            .bufferOffset = 0,
            .bufferRowLength = w,
            .bufferImageHeight = h,
            .imageSubresource{.aspectMask = info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                                : vk::ImageAspectFlagBits::eColor,
                              .mipLevel = m,
                              .baseArrayLayer = 0,
                              .layerCount = rows ? row_layers : layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {w, h, d},
        });
    }
    s.cmd.copyBufferToImage(buffer, vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())},
                            vk::ImageLayout::eGeneral,
                            vk::ArrayProxy<const vk::BufferImageCopy>(
                                static_cast<u32>(copies.size()), copies.data()));
    Written(&entry, entry.rewrite_clock); // NAT-148
    s.buffers_written = true; // the next work waits for the fill
    ++s.commands;
    return true;
}

void NativeRenderer::ApplyRewrite(Native::ImageEntry& entry) {
    using Rewrite = Native::ImageEntry::Rewrite;
    if (entry.rewrite == Rewrite::None && !entry.stencil_clear_pending) {
        return;
    }
    const auto& info = entry.info;
    if (Native::WatchedTexture(info.guest_address, info.guest_address + 1) &&
        Native::WatchSample()) {
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: apply {} to {} {:#x} ({:#x} bytes of {:#x}, "
                    "pattern {:#x} {:#x})",
                    entry.rewrite == Rewrite::Clear    ? "clear"
                    : entry.rewrite == Rewrite::Upload ? "upload"
                                                       : "stencil clear",
                    entry.from_cpu ? "texture" : "target", info.guest_address, entry.rewrite_bytes,
                    info.guest_size, entry.rewrite_pattern[0], entry.rewrite_pattern[1]);
    }
    if (SeqTraced(&entry)) {
        LOG_WARNING(Render_Vulkan, "DIAG-208: frame {} rewrite {} {:#x} ({:#x} bytes of {:#x}, "
                    "pattern {:#x} {:#x}, clock {}) @{}", frame,
                    entry.rewrite == Rewrite::Clear    ? "clear"
                    : entry.rewrite == Rewrite::Upload ? "upload"
                                                       : "stencil clear",
                    info.guest_address, entry.rewrite_bytes, info.guest_size,
                    entry.rewrite_pattern[0], entry.rewrite_pattern[1], entry.rewrite_clock,
                    static_cast<const void*>(&entry)); // DIAG-208 (diagnosis)
    }
    const Rewrite rewrite = std::exchange(entry.rewrite, Rewrite::None);
    const bool stencil = std::exchange(entry.stencil_clear_pending, false) &&
                         info.props.has_stencil && entry.image;
    const u32 lo = entry.rewrite_pattern[0];
    const u32 hi = entry.rewrite_pattern[1];
    const bool partial = rewrite == Rewrite::Clear && entry.rewrite_bytes < info.guest_size;
    // What a clear command can do: a depth plane filled with one value (32-bit float, or 16-bit
    // unorm), a color picture filled with zeros.
    std::optional<float> depth_value;
    bool color_zero = false;
    if (rewrite == Rewrite::Clear && !partial) {
        if (info.props.is_depth) {
            const auto format = info.pixel_format;
            const bool d16 = format == vk::Format::eD16Unorm || format == vk::Format::eD16UnormS8Uint;
            if (!d16 && lo == hi) {
                depth_value = std::bit_cast<float>(lo);
            } else if (d16 && lo == hi && (lo >> 16) == (lo & 0xffffu)) {
                depth_value = float(lo & 0xffffu) / 65535.0f;
            }
        } else if (!info.props.is_block && lo == 0 && hi == 0) {
            color_zero = true;
        }
    }
    if (rewrite == Rewrite::Clear && partial) {
        // NAT-124: a fill of the first bytes only: those rows of the first mip and layer.
        const u32 texel = info.props.is_depth ? 4u : std::max(info.num_bits / 8, 1u);
        const u64 row_bytes = u64(std::max(info.pitch, info.size.width)) * texel;
        const u32 rows = u32(std::min<u64>((entry.rewrite_bytes + row_bytes - 1) / row_bytes,
                                           info.size.height));
        // FIX-GT7-031: a fill of whole layers clears those layers. GT7 clears the layers of its
        // headlight shadow maps that this frame uses, fewer than the picture has.
        const u64 layer_bytes = info.guest_size / std::max<u32>(info.resources.layers, 1);
        const u32 whole_layers = layer_bytes != 0 ? u32(entry.rewrite_bytes / layer_bytes) : 0;
        if (rows != 0 && FillImage(entry, lo, hi, rows, whole_layers)) {
            ++stats.rewrite_partial;
        }
    } else if (rewrite == Rewrite::Clear && !depth_value && !color_zero) {
        // NAT-132: other patterns (GT Sport fills pictures in its scratch pool with non-zero
        // values) are filled in on the GPU; uploading them from memory cost 1-3 GB a second in
        // the menus.
        if (FillImage(entry, lo, hi, 0)) {
            ++stats.rewrite_fills;
        } else if (info.num_samples == 1) {
            registry->RequestUpload(entry); // in GPU order, before the next draw or dispatch
            ++stats.rewrite_uploads;
        }
    } else if (rewrite == Rewrite::Upload && info.num_samples == 1) {
        registry->RequestUpload(entry);
        ++stats.rewrite_uploads;
    }
    if (!depth_value && !color_zero && !stencil) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier(); // the work before is done with the picture
    const vk::Image image{Handle<VkImage_T>(entry.image->NativeHandle())};
    const auto range = [](vk::ImageAspectFlags aspect) {
        return vk::ImageSubresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };
    };
    if (depth_value) {
        s.cmd.clearDepthStencilImage(image, vk::ImageLayout::eGeneral,
                                     vk::ClearDepthStencilValue{.depth = *depth_value},
                                     range(vk::ImageAspectFlagBits::eDepth));
    }
    if (stencil) {
        s.cmd.clearDepthStencilImage(
            image, vk::ImageLayout::eGeneral,
            vk::ClearDepthStencilValue{.stencil = entry.stencil_clear_value},
            range(vk::ImageAspectFlagBits::eStencil));
    }
    if (color_zero) {
        s.cmd.clearColorImage(image, vk::ImageLayout::eGeneral, vk::ClearColorValue{},
                              range(vk::ImageAspectFlagBits::eColor));
    }
    Written(&entry, entry.rewrite_clock); // NAT-148
    s.buffers_written = true; // the next work waits for the clear
    ++s.commands;
    ++stats.rewrite_clears;
}

void NativeRenderer::ReadFillShader(const Shader::Info& cs, DrawBindings& b) {
    // GT Sport's fill shader reads [lo, hi, count] from its first buffer and writes lo, hi,
    // lo, hi... (count dwords) to its second: a memset of depth buffers, stencil planes and
    // pictures.
    constexpr u64 FillShader = 0x8aa6e00c;
    // FIX-GT7-031: GT7's is the same shader. Without it, its fills of targets were lost: the
    // headlight shadow maps were never cleared and kept the nearest depth of every frame since
    // the race began, so the lights lit only what no object ever covered (the middle of the road).
    // RYGT_NATIVE_GT7_FILLS=0: as before.
    constexpr u64 FillShaderGt7 = 0xabe484108a360b66;
    static const bool gt7_fills = [] {
        const char* env = std::getenv("RYGT_NATIVE_GT7_FILLS");
        return !(env && env[0] == '0');
    }();
    if ((cs.pgm_hash != FillShader && !(gt7_fills && cs.pgm_hash == FillShaderGt7)) ||
        cs.buffers.size() < 2) {
        return;
    }
    const Shader::BufferResource* args = nullptr;
    const Shader::BufferResource* out = nullptr;
    for (const auto& desc : cs.buffers) {
        if (desc.IsSpecial()) {
            continue;
        }
        if (desc.is_written) {
            out = out ? out : &desc;
        } else {
            args = args ? args : &desc;
        }
    }
    if (!args || !out) {
        return;
    }
    const auto args_sharp = args->GetSharp(cs);
    const auto out_sharp = out->GetSharp(cs);
    const void* arguments = reinterpret_cast<const void*>(args_sharp.base_address);
    if (args_sharp.base_address == 0 || !Native::Readable(arguments, 12)) {
        return;
    }
    std::array<u32, 3> words{};
    std::memcpy(words.data(), arguments, sizeof(words));
    b.fill_pattern = std::array<u32, 2>{words[0], words[1]};
    b.fill_address = out_sharp.base_address;
    b.fill_size = std::min<u64>(u64(words[2]) * 4, out_sharp.GetSize());
    ++stats.fill_shaders;
}

bool NativeRenderer::MetaClearDispatch(const Shader::Info& cs) {
    // As IsComputeMetaClear: a shader writing a known CMASK/HTILE without tile address math
    // (no XOR) and reading none clears it. The dispatch still runs (FIX-112: GT Sport reuses
    // such memory for pictures).
    for (const auto& desc : cs.buffers) {
        if (!desc.IsSpecial() && !desc.is_written &&
            known_meta.contains(desc.GetSharp(cs).base_address)) {
            return false;
        }
    }
    if (cs.has_bitwise_xor) {
        return false;
    }
    bool cleared = false;
    for (const auto& desc : cs.buffers) {
        const VAddr address = desc.GetSharp(cs).base_address;
        if (!desc.IsSpecial() && desc.is_written && known_meta.contains(address)) {
            cleared_meta.insert(address);
            cleared = true;
            if (r.FrameDumpActive()) {
                r.FrameDumpEvent(fmt::format("dispatch: metadata clear {:#x}", address));
            }
        }
    }
    stats.meta_clears += cleared;
    return cleared;
}

void NativeRenderer::DispatchDirect() {
    std::scoped_lock lock{mutex};
    AmdGpu::RecorderParts::Scope recorder_part{AmdGpu::RecorderParts::Part::Dispatches};
    const auto start = std::chrono::steady_clock::now();
    struct Timed {
        std::chrono::steady_clock::time_point start;
        u64& total;
        ~Timed() {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    } timed{start, stats.dispatch_ns};
    t_timed = false; // OPT-001: the phase timers are the draws'
    // DIAG-188: the time by part.
    auto lap = start;
    const auto split = [&](u32 part) {
        const auto now = std::chrono::steady_clock::now();
        stats.dispatch_part_ns[part] +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - lap).count();
        lap = now;
    };
    r.scheduler.PopPendingOperations();
    const auto& cs_program = r.liverpool->DrawCsRegs();
    const ComputePipeline* pipeline = r.AcquireComputePipeline();
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    const auto& regs = r.liverpool->DrawRegs();
    MetaClearDispatch(pipeline->GetStage(Shader::SwStage::Compute));
    split(0);
    if (r.FrameDumpActive()) {
        r.FrameDumpDispatch(pipeline->GetStage(Shader::SwStage::Compute), cs_program,
                            false); // NAT-118
    }
    const u64 refresh_before = stats.region_refresh_ns; // DIAG-GT7-004
    // PERF-GT7-043: what is recorded for this dispatch needs its barrier.
    const u64 uploads_before = registry->GetStats().uploads.load(std::memory_order_relaxed);
    const u64 barriers_before = stats.barriers;
    const u64 copies_before = stats.source_copies;
    Stream& s = [&]() -> Stream& {
        PartClock part{&stats.dispatch_bind_ns[0]};
        return Begin();
    }();
    DrawBindings& b = bindings;
    b.Clear();
    b.push = UserData(regs);
    BindStagesStable(pipeline->GetStages(), regs, true, b); // FIX-GT7-090
    WritebacksBefore(b); // PERF-GT7-044
    if (b.failed) {
        ++stats.skipped;
        return;
    }
    {
        PartClock part{&stats.dispatch_bind_ns[4]};
        ReadFillShader(pipeline->GetStage(Shader::SwStage::Compute), b); // NAT-120
    }
    stats.dispatch_refresh_ns += stats.region_refresh_ns - refresh_before;
    split(1);
    const bool image_init = device->PendingImageInit(); // PERF-GT7-043
    PrepareImages(b); // PERF-109
    split(2);
    const u64 program = pipeline->GetStage(Shader::SwStage::Compute).pgm_hash;
    const bool recorded = image_init || stats.barriers != barriers_before ||
                          stats.source_copies != copies_before ||
                          stats.region_refresh_ns != refresh_before ||
                          registry->GetStats().uploads.load(std::memory_order_relaxed) !=
                              uploads_before;
    // PERF-GT7-057, FIX-GT7-080
    const std::optional<u64> texel =
        b.stored.size() == 1
            ? SameTexelKey(pipeline->GetStage(Shader::SwStage::Compute), b.image_entries)
            : std::nullopt;
    if (RunNeedsBarrier(s, b, program, recorded, texel)) { // PERF-GT7-043, 057
        GlobalBarrier();
    } else {
        EndRendering();
    }
    // PERF-GT7-057; FIX-GT7-081: a dispatch that stores without a key ends the run's exemptions.
    s.run_texels.Note(texel ? b.stored[0]->info.guest_address : 0, texel, !b.stored.empty());
    // FIX-GT7-087: what a same-texel program's dispatch sampled, but for its destination read at
    // the texel it stores (with a key, no other input meets the destination).
    if (IsSameTexelProgram(program)) {
        for (const auto* entry : b.sampled) {
            if (entry && !(texel && entry->info.guest_address == b.stored[0]->info.guest_address &&
                           entry->info.guest_size == b.stored[0]->info.guest_size)) {
                s.run_texels.NoteRead(entry->info.guest_address, entry->info.guest_size);
            }
        }
    }
    split(3);
    const CmdBuf cmdbuf{s.cmd};
    // FIX-188: Intel Arc (#65) binds the pipeline every time (tile work runs between).
    if (r.instance.IsIntel() || s.pipeline != pipeline->Handle()) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
        s.pipeline = pipeline->Handle();
        s.so_known = false; // SO-010
    }
    PushBindings(*pipeline, vk::PipelineBindPoint::eCompute, b);
    ProbeDispatch(pipeline->GetStage(Shader::SwStage::Compute), b, false); // DIAG-175
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    if (GpuSegment* seg = GpuSeg()) { // DIAG-GT7-001
        ++seg->dispatches;
        const u64 groups = u64(cs_program.dim_x) * cs_program.dim_y * cs_program.dim_z;
        if (groups > seg->max_groups) {
            seg->max_groups = groups;
            seg->max_groups_cs = pipeline->GetStage(Shader::SwStage::Compute).pgm_hash;
        }
    }
    ProbeDispatch(pipeline->GetStage(Shader::SwStage::Compute), b, true); // DIAG-175
    split(4);
    NoteWrites(nullptr, b);
    for (const auto* stored : b.stored) {
        if (SeqTraced(stored)) {
            // DIAG-208 (diagnosis): with the flattened user data the shader reads its cube and
            // mip from (GT7's probe filter: dwords 8, 194 and 195).
            const auto& flat = pipeline->GetStage(Shader::SwStage::Compute).flattened_ud_buf;
            std::string words;
            for (const size_t at : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 192, 193, 194, 195, 196, 197}) {
                words += at < flat.size() ? fmt::format(" [{}]{:08x}", at, flat[at])
                                          : std::string{};
            }
            LOG_WARNING(Render_Vulkan, "DIAG-208: frame {} dispatch cs {:#x} {}x{}x{} stored {:#x} "
                        "-> v{} @{} | flat{}", frame,
                        pipeline->GetStage(Shader::SwStage::Compute).pgm_hash, cs_program.dim_x,
                        cs_program.dim_y, cs_program.dim_z, stored->info.guest_address,
                        stored->write_version.load(), static_cast<const void*>(stored), words);
        }
    }
    split(5);
    StripDispatch(b); // DIAG-173 (diagnosis)
    if (Native::g_dump_request.exchange(false)) {
        r.StartFrameDumpNow(); // FIX-156 (diagnosis): a watched fill
    }
    s.buffers_written = true; // what the dispatch wrote is seen after the next barrier
    s.buffers_written.foreign = false; // PERF-GT7-043: a barrier came first, or nothing else wrote
    ++stats.dispatches;
    ++s.commands;
    s.run_program = program; // PERF-GT7-043
    s.run_commands = s.commands;
    ChunkFlushCheck();
    Report();
}

void NativeRenderer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    std::scoped_lock lock{mutex};
    AmdGpu::RecorderParts::Scope recorder_part{AmdGpu::RecorderParts::Part::Dispatches};
    t_timed = false; // OPT-001
    r.scheduler.PopPendingOperations();
    const ComputePipeline* pipeline = r.AcquireComputePipeline();
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    const auto& regs = r.liverpool->DrawRegs();
    if (r.FrameDumpActive()) {
        r.FrameDumpDispatch(pipeline->GetStage(Shader::SwStage::Compute),
                            r.liverpool->DrawCsRegs(), true); // NAT-118
    }
    Stream& s = Begin();
    const auto args = GuestBuffer(address + offset, std::max<u32>(size, 12), false);
    if (!args) {
        ++stats.skipped;
        return;
    }
    DrawBindings& b = bindings;
    b.Clear();
    b.push = UserData(regs);
    BindStagesStable(pipeline->GetStages(), regs, true, b); // FIX-GT7-090
    WritebacksBefore(b); // PERF-GT7-044
    PrepareImages(b); // PERF-109
    GlobalBarrier();
    const CmdBuf cmdbuf{s.cmd};
    // FIX-188: Intel Arc (#65) binds the pipeline every time (tile work runs between).
    if (r.instance.IsIntel() || s.pipeline != pipeline->Handle()) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
        s.pipeline = pipeline->Handle();
        s.so_known = false; // SO-010
    }
    PushBindings(*pipeline, vk::PipelineBindPoint::eCompute, b);
    cmdbuf.dispatchIndirect(args->first, args->second);
    if (GpuSegment* seg = GpuSeg()) { // DIAG-GT7-001 (groups unknown here)
        ++seg->dispatches;
    }
    NoteWrites(nullptr, b);
    StripDispatch(b); // DIAG-173 (diagnosis)
    s.buffers_written = true;
    ++stats.dispatches;
    ++s.commands;
    ChunkFlushCheck();
}

void NativeRenderer::ResolveDraw(const AmdGpu::Regs& regs) {
    const VideoCore::TextureCache::ImageDesc src_desc{regs.color_buffers[0],
                                                      r.liverpool->DrawCbExtent(0)};
    const VideoCore::TextureCache::ImageDesc dst_desc{regs.color_buffers[1],
                                                      r.liverpool->DrawCbExtent(1)};
    Native::ImageEntry* src = registry->Find(src_desc.info, true, frame);
    Native::ImageEntry* dst = registry->Find(dst_desc.info, true, frame);
    if (!src || !dst) {
        return;
    }
    // GT7-UW: a resolve of a picture drawn with depth this frame (a race's minimaps are drawn
    // with MSAA and resolved before the HUD shows them) passes that on to its result.
    if (HudSplitOn()) {
        const auto it = hud_depth_drawn.find(src->info.guest_address);
        if (it != hud_depth_drawn.end() && it->second == frame) {
            hud_depth_drawn[dst->info.guest_address] = frame;
        }
    }
    Stream& s = Begin();
    // NAT-122: a fill before the resolve is applied before it (left pending, it cleared the
    // resolved picture at its first read: GT Sport's minimaps, resolved into the scratch pool).
    ApplyRewrite(*src);
    ApplyRewrite(*dst);
    registry->ForgetFills(dst->info.guest_address, dst->info.guest_size); // FIX-GT7-036
    if (r.FrameDumpActive()) {
        r.FrameDumpEvent(fmt::format("resolve {:#x} {}x{} x{} -> {:#x} {}x{} x{}",
                                     src->info.guest_address, src->info.size.width,
                                     src->info.size.height, src->info.num_samples,
                                     dst->info.guest_address, dst->info.size.width,
                                     dst->info.size.height, dst->info.num_samples));
        if (r.FrameDumpCollecting() &&
            std::ranges::find(dump_entries, dst) == dump_entries.end()) {
            dump_entries.push_back(dst);
        }
    }
    PrepareImages();
    GlobalBarrier();
    const auto layers = std::min(src_desc.view_info.range.extent.layers,
                                 dst_desc.view_info.range.extent.layers);
    const vk::ImageResolve region{
        .srcSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = src_desc.view_info.range.base.layer,
                        .layerCount = layers},
        .dstSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = dst_desc.view_info.range.base.layer,
                        .layerCount = layers},
        // FIX-154: the targets the resolve names (they may share larger pictures).
        .extent{std::min({src->info.size.width, dst->info.size.width, src_desc.info.size.width,
                          dst_desc.info.size.width}),
                std::min({src->info.size.height, dst->info.size.height,
                          src_desc.info.size.height, dst_desc.info.size.height}),
                1},
    };
    const vk::Image src_image{Handle<VkImage_T>(src->image->NativeHandle())};
    const vk::Image dst_image{Handle<VkImage_T>(dst->image->NativeHandle())};
    // FIX-GT7-044: the scratch pool's targets keep the format they were made with, whatever is drawn
    // into them later (NAT-113). vkCmdResolveImage wants one format on both sides: GT7's menu course
    // map, drawn in R8G8B8A8, came out converted to 10-bit fields once a 512x512 A2B10G10R10 target
    // was left at 0x100a0b0000 (the rewards screen draws one), and the map's texture read those bits
    // as R8G8B8A8: an orange line and a rainbow shadow instead of white and grey. A view of the
    // source's format writes the bits a guest resolve would. RYGT_NATIVE_RESOLVE_VIEW=0: as before.
    static const bool resolve_view = [] {
        const char* env = std::getenv("RYGT_NATIVE_RESOLVE_VIEW");
        return !(env && env[0] == '0');
    }();
    const auto src_format = src->image->Desc().format;
    if (resolve_view && layers == 1 && src->image->Desc().samples > 1 &&
        dst->image->Desc().samples == 1 && src_format != dst->image->Desc().format &&
        src->info.num_bits == dst->info.num_bits) {
        static std::atomic<u32> logged{};
        if (const u32 n = logged++; n < 8) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-GT7-044: resolve {:#x} {} into {:#x}, a {} target, the guest's {}",
                        src->info.guest_address, vk::to_string(src->info.pixel_format),
                        dst->info.guest_address, vk::to_string(dst->info.pixel_format),
                        vk::to_string(dst_desc.info.pixel_format));
        }
        // Views made for attachments: ImageEntry::View's are for sampling or storage only. The mode
        // follows the views' format, which can differ from the one the guest names.
        const vk::Format view_format = static_cast<vk::Format>(src_format);
        const auto attachment_view = [&](const Native::ImageEntry& entry, u32 layer) {
            VideoCore::ImageViewInfo view_info{};
            view_info.format = view_format;
            view_info.range.base.layer = u16(layer);
            return View(entry, view_info, ViewUse::Attachment);
        };
        const vk::ImageView src_view = attachment_view(*src, region.srcSubresource.baseArrayLayer);
        const vk::ImageView dst_view = attachment_view(*dst, region.dstSubresource.baseArrayLayer);
        if (src_view && dst_view) {
            const std::string_view numeric = vk::componentNumericFormat(view_format, 0);
            const vk::RenderingAttachmentInfo attachment{
                .imageView = src_view,
                .imageLayout = vk::ImageLayout::eGeneral,
                .resolveMode = numeric == "UINT" || numeric == "SINT"
                                   ? vk::ResolveModeFlagBits::eSampleZero
                                   : vk::ResolveModeFlagBits::eAverage,
                .resolveImageView = dst_view,
                .resolveImageLayout = vk::ImageLayout::eGeneral,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            };
            s.cmd.beginRendering(vk::RenderingInfo{
                .renderArea{.extent = {region.extent.width, region.extent.height}},
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &attachment,
            });
            s.cmd.endRendering();
            Written(dst);
            s.buffers_written = true;
            ++s.commands;
            return;
        }
    }
    if (src->image->Desc().samples > 1 && dst->image->Desc().samples == 1) {
        CmdBuf{s.cmd}.resolveImage(src_image, vk::ImageLayout::eGeneral, dst_image,
                                   vk::ImageLayout::eGeneral, region);
    } else {
        const vk::ImageCopy copy{
            .srcSubresource = region.srcSubresource,
            .dstSubresource = region.dstSubresource,
            .extent = region.extent,
        };
        CmdBuf{s.cmd}.copyImage(src_image, vk::ImageLayout::eGeneral, dst_image,
                                vk::ImageLayout::eGeneral, copy);
    }
    Written(dst);
    s.buffers_written = true;
    ++s.commands;
}

void NativeRenderer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    if (!is_gds) {
        NoteProgramTextureWrite(address, num_bytes, "fill"); // DIAG-185 (diagnosis)
        NoteHistory(address, num_bytes, 4, value);           // DIAG-187 (diagnosis)
    }
    if (DiagFrame() && !is_gds && address <= DiagWritesAt() &&
        DiagWritesAt() < address + num_bytes) {
        ++diag_lines;
        LOG_WARNING(Render_Vulkan, "DIAG-178: frame {} fill {:#x}+{:#x} with {:#x}", frame,
                    address, num_bytes, value);
    }
    if (num_bytes == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Stream& s = Begin();
    if (is_gds) {
        GlobalBarrier();
        CmdBuf{s.cmd}.fillBuffer(vk::Buffer{Handle<VkBuffer_T>(gds->NativeHandle())}, address,
                                 std::min<u64>(num_bytes, gds->Size() - address), value);
        s.buffers_written = true;
        ++s.commands;
        return;
    }
    const bool meta = known_meta.contains(address);
    if (meta) {
        cleared_meta.insert(address); // a CMASK/HTILE cleared by the CP
        ++stats.meta_clears;
    }
    if (r.FrameDumpActive()) {
        r.FrameDumpEvent(fmt::format("fill {:#x}+{:#x} = {:#x}{}", address, num_bytes, value,
                                     meta ? " (metadata clear)" : "")); // NAT-117
    }
    ++stats.fills;
    WritebacksBefore(address, num_bytes); // PERF-GT7-044
    SendWritebacksDue();                  // FIX-GT7-040
    if (const auto found = GuestBuffer(address, num_bytes, true)) {
        GlobalBarrier();
        CmdBuf{s.cmd}.fillBuffer(found->first, found->second, num_bytes, value);
        s.buffers_written = true;
        ++s.commands;
        NoteRegionWrite(address, num_bytes); // FIX-194
    } else {
        // Not GPU-visible as one range: written now (before the GPU runs what precedes it).
        if (r.memory->IsValidMapping(address, num_bytes)) {
            std::fill_n(reinterpret_cast<u32*>(address), num_bytes / 4, value);
        }
    }
    const std::array<u32, 2> pattern{value, value};
    GpuWrote(address, num_bytes, &pattern, true); // NAT-119/120
}

void NativeRenderer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds) {
        NoteProgramTextureWrite(dst, num_bytes, "copy"); // DIAG-185 (diagnosis)
        NoteHistory(dst, num_bytes, 5, src_gds ? 0 : src); // DIAG-187 (diagnosis)
    }
    if (DiagFrame() && !dst_gds && dst <= DiagWritesAt() && DiagWritesAt() < dst + num_bytes) {
        ++diag_lines;
        LOG_WARNING(Render_Vulkan, "DIAG-178: frame {} copy {:#x}+{:#x} from {:#x}", frame, dst,
                    num_bytes, src);
    }
    if (num_bytes == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Stream& s = Begin();
    ++stats.copies;
    if (r.FrameDumpActive()) {
        r.FrameDumpEvent(fmt::format("copy {:#x}{} <- {:#x}{} ({:#x} bytes)", dst,
                                     dst_gds ? " (GDS)" : "", src, src_gds ? " (GDS)" : "",
                                     num_bytes)); // NAT-117
    }
    // PERF-GT7-044: what it reads as well as what it overwrites.
    if (!src_gds) {
        WritebacksBefore(src, num_bytes);
    }
    if (!dst_gds) {
        WritebacksBefore(dst, num_bytes);
    }
    SendWritebacksDue(); // FIX-GT7-040
    const vk::Buffer gds_buffer{Handle<VkBuffer_T>(gds->NativeHandle())};
    const auto source = src_gds ? std::optional{std::pair{gds_buffer, u64(src)}}
                                : GuestBuffer(src, num_bytes, false);
    const auto dest = dst_gds ? std::optional{std::pair{gds_buffer, u64(dst)}}
                              : GuestBuffer(dst, num_bytes, true);
    if (source && dest) {
        GlobalBarrier();
        CmdBuf{s.cmd}.copyBuffer(source->first, dest->first,
                                 vk::BufferCopy{.srcOffset = source->second,
                                                .dstOffset = dest->second,
                                                .size = num_bytes});
        s.buffers_written = true;
        ++s.commands;
        if (!dst_gds) {
            NoteRegionWrite(dst, num_bytes); // FIX-194
        }
    } else if (!dst_gds && !src_gds) {
        if (r.memory->IsValidMapping(dst, num_bytes) &&
            Native::Readable(reinterpret_cast<const void*>(src), num_bytes)) {
            std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
                        num_bytes);
        }
    } else {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan, "NAT-101: GDS copy {:#x} <- {:#x} ({} bytes) dropped", dst,
                        src, num_bytes);
        }
    }
    if (!dst_gds) {
        GpuWrote(dst, num_bytes, nullptr, true); // NAT-119
    }
}

bool NativeRenderer::InlineWrite(VAddr address, std::span<const u8> data) {
    if (data.empty() || data.size() % 4 != 0 || data.size() > 65536 || address % 4 != 0) {
        return false;
    }
    NoteHistory(address, data.size(), 6, 0); // DIAG-187 (diagnosis)
    std::scoped_lock lock{mutex};
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct || !direct->EnsureImported(address, data.size())) {
        return false;
    }
    const auto found = direct->FindBuffer(address, data.size());
    if (!found) {
        return false;
    }
    Stream& s = Begin();
    WritebacksBefore(address, data.size()); // PERF-GT7-044
    GlobalBarrier();
    CmdBuf{s.cmd}.updateBuffer(found->first->Handle(), found->second, data.size(), data.data());
    registry->ForgetFills(address, data.size()); // FIX-GT7-036: the bytes are no longer the fill
    // NAT-146: a region over the range holds what the GPU reads there: it gets the bytes too.
    for (auto& [base, region] : regions) {
        const VAddr begin = std::max<VAddr>(address, region.begin);
        const VAddr end = std::min<VAddr>(address + data.size(), region.end);
        if (begin < end) {
            CmdBuf{s.cmd}.updateBuffer(vk::Buffer{Handle<VkBuffer_T>(region.buffer->NativeHandle())},
                                       begin - region.begin, end - begin,
                                       data.data() + (begin - address));
        }
    }
    // NAT-150: and mirrors.
    for (auto it = FirstMirror(address);
         it != mirrors.end() && it->second.begin < address + data.size(); ++it) {
        const Mirror& mirror = it->second;
        const VAddr begin = std::max<VAddr>(address, mirror.begin);
        const VAddr end = std::min<VAddr>(address + data.size(), mirror.end);
        CmdBuf{s.cmd}.updateBuffer(vk::Buffer{Handle<VkBuffer_T>(mirror.buffer->NativeHandle())},
                                   begin - mirror.begin, end - begin,
                                   data.data() + (begin - address));
        for (u64 g = (begin - mirror.begin) / MirrorGranule;
             g <= (end - 1 - mirror.begin) / MirrorGranule && g < mirror.hashes.size(); ++g) {
            const_cast<Mirror&>(mirror).hashes[g] = 0; // FIX-176
        }
    }
    s.buffers_written = true;
    ++s.commands;
    ++stats.inline_writes;
    return true;
}

void NativeRenderer::OnUnmap(VAddr address, u64 size) {
    if (Native::WatchedTexture(address, address + size)) {
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: unmap {:#x}+{:#x}", address, size);
    }
    Native::ForgetReadable(address, size); // NAT-147
    // FIX-153: the pages stop being watched (Ryty tells before it unmaps them). Memory mapped
    // there again comes writable; a watch left in place counted as one in the page manager but
    // protected nothing, and was never set again (the page was "watched"). GT Sport maps a car's
    // models and textures into the same memory each time: the showroom's 32x32 lighting texture
    // kept the zeros of the fresh mapping, and the car screens came out blown out.
    if (size != 0) {
        UnwatchPages(VideoCore::PageManager::GetPageAddr(address),
                     VideoCore::PageManager::GetNextPageAddr(address + size - 1));
    }
    registry->Invalidate(address, size, Native::ImageRegistry::Stale::Unmap);
    std::scoped_lock lock{mutex};
    DropRegionsOver(address, size);
    ++buffer_generation;        // OPT-003
    ++buffer_hard_generation;   // PERF-GT7-047
    ++unmap_generation;         // OPT-002
}

u32 NativeRenderer::DropRegionsOver(VAddr address, u64 size) {
    // NAT-146: regions over the range go (the device frees their buffers once the GPU is done
    // with them). With `mutex` held.
    u32 dropped = 0;
    {
        std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
        for (auto it = regions.begin(); it != regions.end();) {
            if (it->second.begin < address + size && it->second.end > address) {
                buffer_log.Note(it->second.begin, it->second.end);
                NoteHistory(it->second.begin, it->second.end - it->second.begin, 11,
                            0); // DIAG-187 (diagnosis)
                it = regions.erase(it);
                ++dropped;
            } else {
                ++it;
            }
        }
    }
    const size_t mirrors_before = mirrors.size();
    DropMirrors(address, size); // NAT-150
    dropped += u32(mirrors_before - mirrors.size());
    {
        std::scoped_lock pending_lock{region_dirty_mutex}; // FIX-173
        std::erase_if(page_snapshots, [&](const auto& snapshot) {
            return snapshot.first >= address && snapshot.first < address + size;
        });
    }
    return dropped;
}

void NativeRenderer::OnNewMapping(VAddr address, u64 size) {
    // FIX-GT7-077 (see Rasterizer::OnNewMapping).
    // FIX-GT7-086: regions and mirrors over the new mapping were made after Ryty's unmap report
    // (the report dropped those made before), from the old pages; a region whose memory failed
    // to import was filled from the CPU's bytes once and never refreshes, and a texture's upload
    // reads a region over it first. They go, as at an unmap, and only then are the textures
    // marked: an upload that takes the mark reads what is mapped now.
    u32 dropped = 0;
    {
        std::scoped_lock lock{mutex};
        dropped = DropRegionsOver(address, size);
        if (dropped != 0) {
            ++buffer_generation;      // OPT-003
            ++buffer_hard_generation; // PERF-GT7-047
        }
    }
    const u32 textures = registry->InvalidateNewMapping(address, size);
    if (textures != 0 || dropped != 0) {
        static std::atomic<u32> mappings{};
        if (const u32 n = ++mappings; n <= 40 || n % 100 == 0) {
            LOG_INFO(Render_Vulkan,
                     "FIX-GT7-077: {} textures over the new mapping {:#x}+{:#x} uploaded again, "
                     "{} regions or mirrors dropped ({} mappings so far)",
                     textures, address, size, dropped, n);
        }
    }
}

void NativeRenderer::WatchBind(const Native::ImageEntry& entry, const AmdGpu::Image& tsharp,
                               bool storage, bool memo, u64 program) {
    if (!Native::WatchSample()) {
        return;
    }
    u32 sources = 0;
    {
        std::scoped_lock lock{const_cast<Native::ImageEntry&>(entry).sources_mutex};
        sources = static_cast<u32>(entry.sources.size());
    }
    LOG_WARNING(Render_Vulkan, "FIX-157 watch: {} {:#x} {}x{} layers {}-{} type {} fmt {}/{}{} "
                "(program {:#x}) -> {} {}x{} layers {} version {}, {} sources (frame {})",
                storage ? "store" : "sample", tsharp.Address(), tsharp.width + 1,
                tsharp.height + 1, u32(tsharp.base_array), u32(tsharp.last_array),
                u32(tsharp.GetType()), u32(tsharp.GetDataFmt()), u32(tsharp.GetNumberFmt()),
                memo ? " (memo)" : "", program, entry.from_cpu ? "texture" : "target",
                entry.info.size.width, entry.info.size.height, entry.info.resources.layers,
                entry.write_version.load(), sources, frame);
    if (entry.from_cpu) {
        // DIAG-223 (diagnosis): whether the texture's memory still holds what was uploaded, and
        // which of its layers differ (CPU writes the page watch missed show here).
        const bool unchanged = registry->MemoryUnchanged(const_cast<Native::ImageEntry&>(entry));
        std::string layers;
        const u64 layer_size = entry.info.resources.layers > 1
                                   ? entry.info.guest_size / entry.info.resources.layers
                                   : 0;
        if (!unchanged && layer_size >= 4096) {
            static std::unordered_map<VAddr, std::vector<u64>> seen;
            auto& hashes = seen[entry.info.guest_address];
            hashes.resize(entry.info.resources.layers);
            u32 changed = 0;
            for (u32 l = 0; l < entry.info.resources.layers; ++l) {
                const u64 h = Core::HashGuestBytes(entry.info.guest_address + l * layer_size,
                                                   layer_size);
                if (hashes[l] != 0 && hashes[l] != h && changed++ < 12) {
                    layers += fmt::format(" {}", l);
                }
                hashes[l] = h;
            }
            layers += fmt::format(" ({} layers changed since the last bind)", changed);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-223: texture {:#x} memory {} the upload (uploads {}){}",
                    entry.info.guest_address, unchanged ? "matches" : "DIFFERS FROM",
                    u32(entry.total_uploads), layers);
    }
}

bool NativeRenderer::SeqTraced(const Native::ImageEntry* entry) const {
    // DIAG-208 (diagnosis, RYGT_DIAG_SEQ=<hex lo>-<hex hi>[,<every>[,<count>]]): in <count>
    // frames of every <every> (4 of 600 if not given), each bind, draw, dispatch, rewrite and
    // copy-in of the pictures starting in that range, with the mips and layers it covers.
    struct Config {
        VAddr lo = 0;
        VAddr hi = 0;
        u64 every = 600;
        u64 count = 4;
    };
    static const Config config = [] {
        Config c;
        const char* env = Common::DiagEnv("RYGT_DIAG_SEQ");
        if (!env || !*env) {
            return c;
        }
        char* end = nullptr;
        c.lo = std::strtoull(env, &end, 16);
        c.hi = end && *end == '-' ? std::strtoull(end + 1, &end, 16) : c.lo + 1;
        if (end && *end == ',') {
            c.every = std::max<u64>(std::strtoull(end + 1, &end, 10), 1);
        }
        if (end && *end == ',') {
            c.count = std::strtoull(end + 1, &end, 10);
        }
        return c;
    }();
    return entry && config.lo < config.hi && entry->info.guest_address >= config.lo &&
           entry->info.guest_address < config.hi && frame % config.every < config.count;
}

void NativeRenderer::SeqBind(const Native::ImageEntry& entry, const AmdGpu::Image& tsharp,
                             bool storage, u64 program) {
    if (!storage && !entry.from_cpu && OneOffTrackOn()) {
        // DIAG-GT7-008, PERF-GT7-023: a picture the GPU drew, sampled.
        const_cast<Native::ImageEntry&>(entry).sampled_frame.store(frame,
                                                                   std::memory_order_relaxed);
    }
    if (!SeqTraced(&entry)) {
        return;
    }
    LOG_WARNING(Render_Vulkan, "DIAG-208: frame {} {} T# {:#x} {}x{} mips {}-{} layers {}-{} "
                "(program {:#x}) -> {} {:#x} {}x{} mips {} layers {} v{} @{}", frame,
                storage ? "store" : "sample", tsharp.Address(), tsharp.width + 1,
                tsharp.height + 1, u32(tsharp.base_level), u32(tsharp.last_level),
                u32(tsharp.base_array), u32(tsharp.last_array), program,
                entry.from_cpu ? "texture" : "target", entry.info.guest_address,
                entry.info.size.width, entry.info.size.height, entry.info.resources.levels,
                entry.info.resources.layers, entry.write_version.load(),
                static_cast<const void*>(&entry));
}

void NativeRenderer::ProbeWatchedPictures() {
    // FIX-156 (diagnosis, RYGT_NATIVE_WATCH_TEXTURE): what the pictures at the watched address
    // hold, every half second (read back from the GPU), to see when one loses its content.
    const auto& watched_list = Native::WatchedTextures();
    if (watched_list.empty()) {
        return;
    }
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = probes.begin(); it != probes.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        it->buffer->Invalidate(0, it->size);
        const u8* bytes = it->buffer->Mapped();
        const u64 count = it->size / 4;
        double sum = 0;
        float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
        u64 zeros = 0;
        // FIX-171: B10G11R11 pictures (GT Sport's light cubes) as their mean colour channel.
        const auto small_float = [](u32 bits, u32 mantissa_bits) {
            const u32 e = bits >> mantissa_bits;
            const float m = float(bits & ((1u << mantissa_bits) - 1)) / float(1u << mantissa_bits);
            return e == 0 ? std::ldexp(m, -14) : e == 31 ? 0.0f : std::ldexp(1.0f + m, int(e) - 15);
        };
        for (u64 i = 0; i < count; ++i) {
            float v;
            std::memcpy(&v, bytes + i * 4, 4);
            if (it->packed_r11g11b10) {
                u32 word;
                std::memcpy(&word, bytes + i * 4, 4);
                v = (small_float(word & 0x7ff, 6) + small_float((word >> 11) & 0x7ff, 6) +
                     small_float(word >> 22, 5)) / 3.0f;
            }
            if (!std::isfinite(v)) {
                continue;
            }
            sum += v;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            zeros += v == 0.0f ? 1 : 0;
        }
        if (it->packed_r11g11b10 && it->layers != 0) {
            // FIX-171: per mip, the mean of each layer (untiled copy, mip after mip).
            std::string text;
            u64 at = 0;
            for (u32 mip = 0; mip < it->mips; ++mip) {
                const u64 w = std::max(it->width >> mip, 1u), h = std::max(it->height >> mip, 1u);
                text += fmt::format(" m{}:", mip);
                u64 non_finite = 0; // DIAG-177: texels with an Inf/NaN channel (exponent 31)
                float largest = 0.0f;
                for (u32 layer = 0; layer < it->layers; ++layer) {
                    double layer_sum = 0;
                    for (u64 i = 0; i < w * h && (at + i + 1) * 4 <= it->size; ++i) {
                        u32 word;
                        std::memcpy(&word, bytes + (at + i) * 4, 4);
                        if (((word >> 6) & 0x1f) == 31 || ((word >> 17) & 0x1f) == 31 ||
                            ((word >> 27) & 0x1f) == 31) {
                            ++non_finite;
                        }
                        const float value = (small_float(word & 0x7ff, 6) +
                                             small_float((word >> 11) & 0x7ff, 6) +
                                             small_float(word >> 22, 5)) / 3.0f;
                        largest = std::max(largest, value);
                        layer_sum += value;
                    }
                    at += w * h;
                    text += fmt::format(" {:.3f}", layer_sum / double(w * h));
                }
                text += fmt::format(" (max {:.1f}, {} Inf/NaN)", largest, non_finite);
            }
            if (it->layers == 1 && it->height >= 3) {
                // A plain picture (the frame's main view): its top, middle and bottom thirds.
                const u64 w = it->width, third = it->height / 3;
                std::array<double, 3> part{};
                for (u32 band = 0; band < 3; ++band) {
                    for (u64 y = band * third; y < (band + 1) * third; ++y) {
                        for (u64 x = 0; x < w; ++x) {
                            u32 word;
                            std::memcpy(&word, bytes + (y * w + x) * 4, 4);
                            part[band] += (small_float(word & 0x7ff, 6) +
                                           small_float((word >> 11) & 0x7ff, 6) +
                                           small_float(word >> 22, 5)) / 3.0f;
                        }
                    }
                    part[band] /= double(third * w);
                }
                text += fmt::format(" thirds: {:.4f} {:.4f} {:.4f}", part[0], part[1], part[2]);
            }
            LOG_WARNING(Render_Vulkan, "FIX-171 watch: frame {}: {} {:#x} {}x{} version {}:{}",
                        it->frame, it->target ? "target" : "texture", it->address, it->width,
                        it->height, it->version, text);
            it = probes.erase(it);
            continue;
        }
        LOG_WARNING(Render_Vulkan, "FIX-156 watch: frame {}: {} {:#x} {}x{} version {} holds "
                    "mean {:.4f} min {:.4f} max {:.4f}, {} of {} words zero (as 32-bit floats)",
                    it->frame, it->target ? "target" : "texture", it->address, it->width,
                    it->height,
                    it->version, count ? sum / count : 0.0, lo, hi, zeros, count);
        it = probes.erase(it);
    }
    // FIX-171 (diagnosis, RYGT_NATIVE_WATCH_EVERY=<frames>): how often (every 30 frames).
    static const u64 every = [] {
        const char* env = Common::DiagEnv("RYGT_NATIVE_WATCH_EVERY");
        return env && *env ? std::max<u64>(std::strtoull(env, nullptr, 0), 1) : u64{30};
    }();
    if (frame % every != 0 || probes.size() >= (every >= 30 ? 1u : 16u)) {
        return;
    }
    Begin();
    for (const VAddr watched : watched_list) {
        registry->ForEachOverlapping(watched, watched + 1, [&](Native::ImageEntry& entry) {
            if (entry.info.guest_address != watched || !entry.image ||
                entry.info.num_samples != 1) {
                return;
            }
            const bool packed = entry.info.pixel_format == vk::Format::eB10G11R11UfloatPack32;
            const auto& desc = entry.image->Desc();
            // FIX-171 (RYGT_NATIVE_WATCH_RECT=x,y,w,h): of a plain picture only that rectangle.
            static const std::array<u32, 4> rect = [] {
                std::array<u32, 4> r{};
                if (const char* env = Common::DiagEnv("RYGT_NATIVE_WATCH_RECT"); env && *env) {
                    std::sscanf(env, "%u,%u,%u,%u", &r[0], &r[1], &r[2], &r[3]);
                }
                return r;
            }();
            const bool in_rect = packed && rect[2] != 0 && rect[3] != 0 && desc.mips == 1 &&
                                 desc.layers <= 1 && rect[0] + rect[2] <= desc.width &&
                                 rect[1] + rect[3] <= desc.height;
            u64 untiled = 0;
            for (u32 mip = 0; mip < desc.mips; ++mip) {
                untiled += u64(std::max(desc.width >> mip, 1u)) *
                           std::max(desc.height >> mip, 1u) * std::max(desc.layers, 1u) * 4;
            }
            if (in_rect) {
                untiled = u64(rect[2]) * rect[3] * 4;
            }
            const u64 size = packed ? untiled : entry.info.guest_size;
            auto buffer = device->CreateBuffer({.size = size,
                                                .usage = Native::Rhi::BufferUsageStorage |
                                                         Native::Rhi::BufferUsageTransferDst,
                                                .memory = Native::Rhi::MemoryKind::Readback,
                                                .name = "watch probe"});
            if (!buffer || !buffer->Mapped()) {
                return;
            }
            GlobalBarrier();
            if (in_rect) {
                const vk::BufferImageCopy region{
                    .bufferOffset = 0,
                    .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                      .mipLevel = 0,
                                      .baseArrayLayer = 0,
                                      .layerCount = 1},
                    .imageOffset = {s32(rect[0]), s32(rect[1]), 0},
                    .imageExtent = {rect[2], rect[3], 1},
                };
                stream.cmd.copyImageToBuffer(
                    vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())},
                    vk::ImageLayout::eGeneral,
                    vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())}, region);
                stream.list->Barrier(Native::Rhi::ScopeTransfer, Native::Rhi::ScopeHost);
                probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, size, frame,
                                  entry.write_version.load(), rect[2], rect[3], !entry.from_cpu,
                                  watched, true, 1, 1});
            } else if (packed) {
                // FIX-171: each mip with all its layers, untiled.
                std::vector<vk::BufferImageCopy> regions;
                u64 offset = 0;
                for (u32 mip = 0; mip < desc.mips; ++mip) {
                    const u32 w = std::max(desc.width >> mip, 1u);
                    const u32 h = std::max(desc.height >> mip, 1u);
                    regions.push_back(vk::BufferImageCopy{
                        .bufferOffset = offset,
                        .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                          .mipLevel = mip,
                                          .baseArrayLayer = 0,
                                          .layerCount = std::max(desc.layers, 1u)},
                        .imageExtent = {w, h, 1},
                    });
                    offset += u64(w) * h * std::max(desc.layers, 1u) * 4;
                }
                stream.cmd.copyImageToBuffer(
                    vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())},
                    vk::ImageLayout::eGeneral,
                    vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())}, regions);
                stream.list->Barrier(Native::Rhi::ScopeTransfer, Native::Rhi::ScopeHost);
                probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, size, frame,
                                  entry.write_version.load(), desc.width, desc.height,
                                  !entry.from_cpu, watched, true, desc.mips,
                                  std::max(desc.layers, 1u)});
            } else {
                auto copies = Native::MipCopies(entry.info);
                stream.list->CopyImageToBuffer(entry.image.get(), buffer.get(), copies);
                stream.list->Barrier(Native::Rhi::ScopeTransfer, Native::Rhi::ScopeHost);
                stream.pipeline = vk::Pipeline{};
                stream.so_known = false; // SO-010
                ForgetBindings(stream); // OPT-014
                probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, size, frame,
                                  entry.write_version.load(), entry.info.size.width,
                                  entry.info.size.height, !entry.from_cpu, watched, false});
            }
        });
    }
}

void NativeRenderer::EvictImages() {
    ProbeWatchedPictures(); // FIX-156 (diagnosis)
    {
        // PERF-GT7-052: each frame, the per-frame textures uploaded since are looked at again:
        // stale, the next use uploads them, and NAT-109 skips the copy when their bytes are
        // unchanged.
        std::vector<std::pair<VAddr, u64>> polled;
        {
            std::scoped_lock lock{polled_mutex};
            polled.swap(polled_textures);
        }
        for (const auto& [address, size] : polled) {
            registry->Invalidate(address, size, Native::ImageRegistry::Stale::CpuWrite);
        }
    }
    // NAT-108: images not used for two seconds go (GT Sport's scratch pool holds render
    // targets of ever-changing sizes: the UI's blur panels, effects).
    constexpr u64 KeepFrames = 120;
    const u64 tick = r.scheduler.CurrentTick();
    {
        const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
        while (!retired_views.empty() && retired_views.front().first <= done) {
            retired_views.pop_front();
        }
    }
    if (frame <= KeepFrames || frame % 30 != 0) {
        return;
    }
    // FIX-157: a picture the GPU wrote and used for a second or more stays past its idle time
    // while nothing else writes its memory: its pixels exist nowhere else (the GPU's writes stay
    // in the picture, not in guest memory). GT Sport bakes its light probes (two cube arrays of
    // 322 32x32 cubes, at 0x107aee1400 and 0x108626d400) with compute shaders while a race
    // loads; nothing read them for about three seconds around the start, they were dropped, the
    // race's textures there uploaded the zeros of guest memory, and the scene's ambient light
    // went: a dark, blue cockpit and world, with the odd warm frame. It is kept when guest memory
    // still holds what it held when the picture was made (nothing but the GPU wrote there:
    // keeping pictures whose memory the CPU had reused drew stale pictures, and writing them
    // back overwrote shader code the game had loaded there), no newer picture overlaps it, and
    // GPU work wrote no part of its memory since; from then on its pages are watched, and a CPU
    // write, an unmap or GPU work there lets it go. Pictures in use for less than a second (the
    // scratch pool's) go as before; kept ones take at most 256 MB.
    constexpr u64 KeptBudget = 256ull << 20;
    u32 newly_kept = 0;
    const auto keep = [&](Native::ImageEntry& entry) {
        if (entry.from_cpu || !entry.image || entry.reused.load(std::memory_order_relaxed)) {
            return false;
        }
        const u64 version = entry.write_version.load(std::memory_order_relaxed);
        if (version == 0 || entry.foreign_clock.load(std::memory_order_relaxed) > version) {
            return false;
        }
        const VAddr address = entry.info.guest_address;
        const u64 size = std::max<u64>(entry.info.guest_size, 1);
        bool superseded = false;
        registry->ForEachOverlapping(address, address + size, [&](Native::ImageEntry& other) {
            superseded |= &other != &entry && !other.from_cpu &&
                          !other.reused.load(std::memory_order_relaxed) &&
                          other.write_version.load(std::memory_order_relaxed) > version;
        });
        if (superseded) {
            return false;
        }
        if (entry.kept.load(std::memory_order_relaxed)) {
            return true;
        }
        if (entry.last_use.load(std::memory_order_relaxed) < entry.made_frame + 60 ||
            kept_bytes + size > KeptBudget || !registry->MemoryUnchanged(entry)) {
            return false;
        }
        entry.kept.store(true, std::memory_order_relaxed);
        kept_bytes += size;
        WatchPages(address, size);
        ++newly_kept;
        if (Native::WatchedTexture(address, address + 1)) {
            LOG_WARNING(Render_Vulkan, "FIX-157 watch: {:#x} kept (version {}, frame {})",
                        address, version, frame);
        }
        return true;
    };
    // PERF-GT7-005: textures the game loaded stay past their idle time, the most recently used
    // first, up to a budget (RYGT_NATIVE_KEEP_TEXTURES_MB, 2048; 0 as before). Their pages are
    // watched, so a CPU write or an unmap still uploads them again at their next use. GT7's
    // license tests left the track's textures unused for over 4 s while a retry reloaded, and
    // at ~4 s into the next attempt 50-120 MB of them were made and uploaded again.
    static const u64 texture_budget = [] {
        const char* env = std::getenv("RYGT_NATIVE_KEEP_TEXTURES_MB");
        if (env && *env) {
            return u64{std::strtoull(env, nullptr, 10)} << 20;
        }
        // FIX-GT7-030 (NAT-VRAM-001): what the video memory setting can free here: idle textures
        // take at most half of it (not below the budget's 1 GB floor).
        const u64 vram = Instance::VideoMemorySettingBytes();
        return vram != 0 ? std::min(u64{2048} << 20, std::max<u64>(vram, u64{1} << 30) / 2)
                         : u64{2048} << 20;
    }();
    // PERF-GT7-011: only textures the game loaded and left alone: uploaded at most twice, never
    // rewritten by GPU work nor made of targets, and at most 4 at one address. Keeping every idle
    // texture kept those GT7 makes in its scratch pool (0x100a0b0000, a new layout every few
    // frames, rewritten by compute passes): 2,600 pictures at that one address after 19 minutes,
    // walked by every lookup, CPU write and GPU write there (ImageRegistry::Find took 5% of the
    // recorder; 2.5 million textures made stale by CPU writes every 10 s).
    const auto loaded_texture = [](Native::ImageEntry& entry) {
        if (entry.total_uploads.load(std::memory_order_relaxed) > 2 ||
            entry.last_stale == u8(Native::ImageRegistry::Stale::GpuWrite) ||
            entry.force_upload.load(std::memory_order_relaxed) ||
            entry.gpu_rewritten.load(std::memory_order_relaxed) != 0 ||
            entry.foreign_clock.load(std::memory_order_relaxed) != 0) {
            return false;
        }
        std::scoped_lock lock{entry.sources_mutex};
        return entry.sources.empty();
    };
    std::unordered_set<const Native::ImageEntry*> kept_textures;
    if (texture_budget != 0) {
        std::vector<Native::ImageEntry*> idle;
        registry->ForEachOverlapping(0, ~VAddr{0}, [&](Native::ImageEntry& entry) {
            if (entry.from_cpu && entry.image &&
                entry.last_use.load(std::memory_order_relaxed) < frame - KeepFrames) {
                idle.push_back(&entry);
            }
        });
        std::ranges::sort(idle, std::greater{}, [](const Native::ImageEntry* entry) {
            return entry->last_use.load(std::memory_order_relaxed);
        });
        std::unordered_map<VAddr, u32> kept_at;
        u64 bytes = 0;
        for (Native::ImageEntry* entry : idle) {
            if (!loaded_texture(*entry) || kept_at[entry->info.guest_address] >= 4) {
                continue;
            }
            bytes += std::max<u64>(entry->info.guest_size, 1);
            if (bytes > texture_budget) {
                break;
            }
            ++kept_at[entry->info.guest_address];
            kept_textures.insert(entry);
        }
    }
    registry->Evict(frame - KeepFrames, [&](Native::ImageEntry& entry) {
        if (entry.kept.load(std::memory_order_relaxed)) {
            kept_bytes -= std::min<u64>(kept_bytes, std::max<u64>(entry.info.guest_size, 1));
        }
        if (Native::WatchedTexture(entry.info.guest_address, entry.info.guest_address + 1)) {
            LOG_WARNING(Render_Vulkan, "FIX-157 watch: {} {:#x} {}x{} layers {} evicted (last "
                        "use frame {}, version {}, frame {})",
                        entry.from_cpu ? "texture" : "target", entry.info.guest_address,
                        entry.info.size.width, entry.info.size.height,
                        entry.info.resources.layers, entry.last_use.load(),
                        entry.write_version.load(), frame);
        }
        {
            std::scoped_lock lock{views_mutex};
            if (const auto it = views.find(&entry); it != views.end()) {
                for (auto& record : it->second) {
                    retired_views.emplace_back(tick, std::move(record.view));
                }
                views.erase(it);
            }
        }
        held_versions.erase(&entry);
        std::erase(stream.written, &entry);
        std::erase(writebacks, &entry);
        std::erase(frame_writebacks, &entry); // PERF-GT7-044
        std::erase(dump_entries, &entry);
        for (auto it = stencil_owners.begin(); it != stencil_owners.end();) {
            auto& owners = it->second;
            owners.erase(std::remove(owners.begin(), owners.end(), &entry), owners.end());
            it = owners.empty() ? stencil_owners.erase(it) : std::next(it);
        }
    }, [&](Native::ImageEntry& entry) {
        return kept_textures.contains(&entry) || keep(entry); // PERF-GT7-005, FIX-157
    });
    if (newly_kept != 0) {
        static std::atomic<u32> logged{};
        if (const u32 n = logged++; n < 20 || n % 100 == 0) {
            LOG_WARNING(Render_Vulkan, "FIX-157: {} pictures the GPU wrote are kept past their idle "
                        "time ({:.1f} MB kept in all)", newly_kept, kept_bytes / 1048576.0);
        }
    }
}

void NativeRenderer::WatchPages(VAddr address, u64 size) {
    const VAddr begin = VideoCore::PageManager::GetPageAddr(address);
    const VAddr end = VideoCore::PageManager::GetNextPageAddr(address + size - 1);
    if (Native::WatchedTexture(begin, end)) {
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: watch {:#x}-{:#x}", begin, end);
    }
    // PERF-105: the page manager keeps which pages are watched, under each page's lock. With a
    // set of its own under one mutex, held across the protections, a write fault waited ~7 us
    // for it (mostly behind the recorder protecting whole regions): half a game thread's time
    // went to ~50,000 faults a second in races.
    r.page_manager.UpdateNativeWatch(begin, end - begin, true);
}

void NativeRenderer::UnwatchPages(VAddr begin, VAddr end) {
    r.page_manager.UpdateNativeWatch(begin, end - begin, false); // PERF-105
}

// DIAG-194: where a CPU write fault's time goes (snapshot, unwatch, registry, the rest).
static std::array<std::atomic<u64>, 5> g_fault_part_ns{};
static std::atomic<u64> g_fault_count{};

void NativeRenderer::OnCpuWrite(VAddr address, u64 size, u64 exact) {
    // DIAG-194 (diagnosis, RYGT_DIAG_FAULT_COST=1)
    static const bool timed = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_FAULT_COST");
        return env && env[0] == '1';
    }();
    auto lap = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto split = [&](u32 part) {
        if (!timed) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        g_fault_part_ns[part].fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - lap).count(),
            std::memory_order_relaxed);
        lap = now;
    };
    if (timed) {
        g_fault_count.fetch_add(1, std::memory_order_relaxed);
    }
    // The written pages stop being watched (so they open up and the write goes on), and the
    // textures over them are uploaded again when next used (which watches them again).
    const VAddr begin = VideoCore::PageManager::GetPageAddr(address);
    const VAddr end = VideoCore::PageManager::GetNextPageAddr(address + std::max<u64>(size, 1) - 1);
    if (Native::WatchedTexture(begin, end)) {
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: CPU write fault {:#x}+{:#x} (exact {:#x}); "
                    "memory {}", address, size, exact, Native::WatchedBytes(address));
    }
    NoteHistory(begin, end - begin, 7, exact != 0 ? address + exact : 0); // DIAG-187 (diagnosis)
    if (RyDiag::Enabled()) {
        NoteFaultPages(begin, end); // DIAG-GT7-010
    }
    // FIX-173: a region page holds what the GPU wrote there, which guest memory never got; a
    // CPU write to the page copies in only the bytes the CPU changed (RefreshRegion compares
    // with this copy of the page from before the write), not the whole page from guest memory.
    // Copying whole pages put stale guest bytes over the GPU's results: GT Sport computes the
    // cockpit's light from its interior capture into 64 bytes at 0x1000e317c0 every third
    // frame, and the CPU writes other data in that page every frame, so the cockpit was lit in
    // one frame of three and dark in the other two.
    // PERF-107: each page's copy is taken before any lock is held (the page can't change: it is
    // still protected, and other threads writing it fault and wait in here too), and kept under
    // the lock before the page opens: once it is open, another thread's copy of it may hold a
    // write already, and the copy kept must predate them all. (Kept after the page opened, a
    // fault racing on the same page lost the other thread's write: stale data in regions, and
    // the GPU faulted - device lost - in most races.) Taken under the region lock, each fault
    // waited ~2.5 us for the recorder.
    split(0);
    bool region_fault = false; // PERF-GT7-030
    if (end - begin <= 64_KB) {
        bool in_region = false;
        {
            std::shared_lock map_lock{buffer_map_mutex};
            in_region = RegionOver(begin, end - begin) != nullptr;
        }
        region_fault = in_region;
        if (in_region) {
            boost::container::small_vector<
                std::pair<VAddr, std::unique_ptr<std::array<u8, 4096>>>, 2>
                copies;
            for (VAddr page = begin; page < end; page += 4096) {
                auto copy = std::make_unique<std::array<u8, 4096>>();
                if (Core::RytyMemory::GuardedCopy(copy->data(),
                                                  reinterpret_cast<const void*>(page), 4096)) {
                    copies.emplace_back(page, std::move(copy));
                }
            }
            std::scoped_lock pending_lock{region_dirty_mutex};
            for (auto& [page, copy] : copies) {
                auto& slot = page_snapshots[page];
                if (!slot) {
                    slot = std::move(copy); // the first write since the page was copied in counts
                }
            }
        }
    }
    if (WritebackRaceDiag()) {
        // DIAG-176: a CPU write into guest memory a FIX-194 write-back will still copy over (the
        // GPU hasn't run it yet): what the CPU writes there now may be lost.
        const u64 known = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
        std::scoped_lock race_lock{writeback_race_mutex};
        while (!pending_writebacks.empty() && pending_writebacks.front().tick <= known) {
            pending_writebacks.pop_front();
        }
        ++writeback_race_faults;
        for (const auto& pending : pending_writebacks) {
            if (pending.begin < end && pending.end > begin && pending.tick > known) {
                const u64 hits = ++writeback_race_hits;
                if (hits <= 40 || hits % 1000 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-176: CPU write {:#x}+{:#x} (frame {}) into a write-back "
                                "{:#x}..{:#x} of frame {} the GPU hasn't run (tick {}, GPU at {}); "
                                "{} such writes so far",
                                address, size, frame, pending.begin, pending.end, pending.frame,
                                pending.tick, known, hits);
                }
                break;
            }
        }
    }
    split(1);
    UnwatchPages(begin, end);
    split(2);
    registry->CpuWrite(begin, end - begin, exact != 0 ? (address - begin) + exact : 0);
    split(3);
    ++stats.cpu_writes;
    // NAT-114: pages of regions written by the CPU (a fault on any thread) are copied in before
    // the region's next use; the recorder sorts them to their regions.
    {
        std::scoped_lock pending_lock{region_dirty_mutex};
        pending_region_writes.emplace_back(begin, end);
    }
    has_pending_region_writes.store(true, std::memory_order_release);
    if (region_fault) {
        FaultAhead(begin, end); // PERF-GT7-030
    }
    split(4);
}

void NativeRenderer::FaultAhead(VAddr begin, VAddr end) {
    // PERF-GT7-030: in GT7's pause menu, ~2,700 write faults a frame (25-28 ms of the game's
    // threads, the game at ~15 fps) on region pages: whole 64 KB chunks written again every few
    // frames, each of their pages faulting, copied (FIX-173), opened, then watched again at the
    // region's next use. The chunk's other pages are opened with the first one, as if written:
    // each copied first while still protected and kept under the lock before it opens, as the
    // faulting page is (PERF-107), so a kept copy never holds a CPU write the region lacks; a
    // page left unwritten copies nothing in at the refresh.
    // FIX-GT7-021: off unless RYGT_NATIVE_FAULT_AHEAD=1, as with FIX-GT7-017. GT7 races still
    // showed long black triangles with FIX-GT7-018 and FIX-GT7-020 (which found 11 pages in a
    // session). In a race it saves ~1,000 faults a second (~1% of a game thread); the pause menu
    // is where it counted.
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_FAULT_AHEAD");
        return env && env[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    const VAddr chunk = begin & ~VAddr{0xFFFF};
    if (end > chunk + 64_KB) {
        return;
    }
    const u64 now = shared_frame.load(std::memory_order_relaxed);
    {
        std::scoped_lock lock{region_dirty_mutex};
        if (fault_chunks.size() >= 65536) {
            fault_chunks.clear();
        }
        auto [it, first] = fault_chunks.try_emplace(chunk, now);
        const bool lately = !first && now - it->second <= 8;
        it->second = now;
        if (!lately) {
            return;
        }
    }
    // The chunk's other pages of the same region that are still watched.
    boost::container::small_vector<VAddr, 16> pages;
    {
        std::shared_lock map_lock{buffer_map_mutex};
        const Region* region = RegionOver(begin, 4096);
        if (!region) {
            return;
        }
        const VAddr from = std::max(chunk, region->begin);
        const VAddr to = std::min<VAddr>(chunk + 64_KB, region->end);
        for (VAddr page = from; page + 4096 <= to; page += 4096) {
            if ((page < begin || page >= end) && r.page_manager.IsNativeWatched(page)) {
                pages.push_back(page);
            }
        }
    }
    {
        // FIX-GT7-006: only pages the CPU wrote lately (FIX-194: for those guest memory gets what
        // the GPU writes there too, so the refresh can't put stale bytes over the GPU's).
        std::scoped_lock lock{region_dirty_mutex};
        pages.erase(std::remove_if(pages.begin(), pages.end(),
                                   [&](VAddr page) {
                                       const auto it = cpu_recent_pages.find(page);
                                       return it == cpu_recent_pages.end() ||
                                              now - it->second > RegionRecentFrames;
                                   }),
                    pages.end());
    }
    // FIX-GT7-018: not a page under a picture the registry tracks. Opened, the page was handed
    // to registry->CpuWrite as written by the CPU: a picture the GPU drew there and kept past
    // its idle time (FIX-157: GT7's sky and reflection pictures) was then taken as reused, no
    // lookup found it any more, and its draws read a new one made from guest memory, which never
    // got what the GPU drew: long black triangles, the whole sky black for a frame. A real CPU
    // write into the page still goes through OnCpuWrite.
    if (!pages.empty()) {
        const size_t before = pages.size();
        pages.erase(std::remove_if(pages.begin(), pages.end(),
                                   [&](VAddr page) {
                                       bool picture = false;
                                       registry->ForEachOverlapping(
                                           page, page + 4096,
                                           [&](auto&) { picture = true; });
                                       return picture;
                                   }),
                    pages.end());
        fault_ahead_pictures.fetch_add(before - pages.size(), std::memory_order_relaxed);
    }
    if (pages.empty()) {
        return;
    }
    boost::container::small_vector<std::pair<VAddr, std::unique_ptr<std::array<u8, 4096>>>, 16>
        copies;
    for (const VAddr page : pages) {
        auto copy = std::make_unique<std::array<u8, 4096>>();
        if (Core::RytyMemory::GuardedCopy(copy->data(), reinterpret_cast<const void*>(page),
                                          4096)) {
            copies.emplace_back(page, std::move(copy));
        }
    }
    if (copies.empty()) {
        return;
    }
    {
        std::scoped_lock pending_lock{region_dirty_mutex};
        for (auto& [page, copy] : copies) {
            auto& slot = page_snapshots[page];
            if (!slot) {
                slot = std::move(copy); // the first copy since the page was copied in counts
            }
            ahead_pages.insert(page); // FIX-GT7-006
        }
    }
    // Opened, then noted as written, in runs (one protect each), as OnCpuWrite does.
    for (size_t i = 0; i < copies.size();) {
        size_t j = i + 1;
        while (j < copies.size() && copies[j].first == copies[j - 1].first + 4096) {
            ++j;
        }
        const VAddr run_begin = copies[i].first;
        const VAddr run_end = copies[j - 1].first + 4096;
        UnwatchPages(run_begin, run_end);
        registry->CpuWrite(run_begin, run_end - run_begin, 0);
        {
            std::scoped_lock pending_lock{region_dirty_mutex};
            pending_region_writes.emplace_back(run_begin, run_end);
        }
        i = j;
    }
    has_pending_region_writes.store(true, std::memory_order_release);
    fault_ahead_faults.fetch_add(1, std::memory_order_relaxed);
    fault_ahead_pages.fetch_add(copies.size(), std::memory_order_relaxed);
}

void NativeRenderer::NoteFaultPages(VAddr begin, VAddr end) {
    const u64 now = shared_frame.load(std::memory_order_relaxed);
    std::scoped_lock lock{fault_pages_mutex};
    for (VAddr page = begin; page < end; page += 4096) {
        if (fault_pages.size() >= 65536 && !fault_pages.contains(page)) {
            return;
        }
        FaultPage& p = fault_pages[page];
        const u64 gap = now - p.last_frame;
        if (p.faults == 0 || gap >= 32) {
            p.frames = 1;
        } else if (gap != 0) {
            p.frames = (p.frames << gap) | 1u;
        }
        p.last_frame = now;
        ++p.faults;
    }
}

std::string NativeRenderer::FaultPagesText() {
    std::unordered_map<VAddr, FaultPage> pages;
    {
        std::scoped_lock lock{fault_pages_mutex};
        pages.swap(fault_pages);
    }
    if (pages.empty()) {
        return {};
    }
    const u64 now = shared_frame.load(std::memory_order_relaxed);
    u64 faults = 0;
    u32 hot = 0;
    std::array<u32, 4> by_class{}; // region, texture up to 4 MB (hashed), larger texture, neither
    std::map<VAddr, std::pair<u32, std::string>> chunks; // 64 KB: hot pages, what one holds
    for (const auto& [page, p] : pages) {
        faults += p.faults;
        // Hot: written in 6 or more of the 8 frames up to its last write, and lately.
        if (now - p.last_frame > 8 || std::popcount(p.frames & 0xffu) < 6) {
            continue;
        }
        ++hot;
        bool region = false;
        {
            std::shared_lock map_lock{buffer_map_mutex};
            region = RegionOver(page, 4096) != nullptr;
        }
        u64 biggest = 0;
        std::string what;
        registry->ForEachOverlapping(page, page + 4096, [&](Native::ImageEntry& entry) {
            if (entry.from_cpu && entry.info.guest_size > biggest) {
                biggest = entry.info.guest_size;
                what = fmt::format("texture {:#x} {}x{} {} ({} KB)", entry.info.guest_address,
                                   entry.info.size.width, entry.info.size.height,
                                   vk::to_string(entry.info.pixel_format),
                                   entry.info.guest_size >> 10);
            }
        });
        const size_t cls = region ? 0 : biggest == 0 ? 3 : biggest <= 4_MB ? 1 : 2;
        ++by_class[cls];
        auto& chunk = chunks[page & ~VAddr{0xFFFF}];
        ++chunk.first;
        if (chunk.second.empty()) {
            chunk.second = region ? std::string{"region"} : biggest ? what : std::string{"neither"};
        }
    }
    std::vector<std::pair<u32, VAddr>> busiest;
    for (const auto& [chunk, info] : chunks) {
        busiest.emplace_back(info.first, chunk);
    }
    std::ranges::sort(busiest, std::greater{});
    std::string top;
    for (size_t i = 0; i < std::min<size_t>(busiest.size(), 6); ++i) {
        top += fmt::format("; {:#x} {} hot pages: {}", busiest[i].second, busiest[i].first,
                           chunks[busiest[i].second].second);
    }
    return fmt::format("{} CPU write faults on {} pages; {} pages written in 6+ of 8 frames: {} in "
                       "regions, {} in textures up to 4 MB, {} in larger textures, {} neither{}",
                       faults, pages.size(), hot, by_class[0], by_class[1], by_class[2],
                       by_class[3], top);
}

u32 NativeRenderer::ReadGds(u32 gds_offset) {
    u32 value = 0;
    if (gds_offset + sizeof(u32) <= gds->Size()) {
        std::memcpy(&value, gds->Mapped() + gds_offset, sizeof(value));
    }
    return value;
}

void NativeRenderer::Splice() {
    std::scoped_lock lock{mutex};
    AmdGpu::RecorderParts::Scope recorder_part{AmdGpu::RecorderParts::Part::Splices};
    Stream& s = stream;
    if (!s.list) {
        return;
    }
    EndRendering();
    if (!writebacks.empty()) {
        GlobalBarrier(); // NAT-112: written back in this submission
    }
    const bool wrapped = s.wrapped_list != nullptr; // FIX-188: Intel Arc (#65)
    GpuTimeEnd(s, r.scheduler.CurrentTick()); // DIAG-GT7-001
    if (!wrapped) {
        s.list->End();
    }
    r.scheduler.EndRendering();
    if (!wrapped) {
        r.scheduler.CommandBuffer().executeCommands(s.cmd); // PERF-GT7-018: its slot's
    }
    const u64 tick = r.scheduler.CurrentTick();
    GpuTimeCollect(); // DIAG-GT7-001: lists the GPU finished
    SpliceHudStream(tick); // GT7-UW
    if (!wrapped) {
        device->MarkExecuted(s.list, {timeline.get(), tick});
    }
    VideoCore::DirectMemory::ReleaseHeldAfter(tick); // FIX-205
    {
        std::scoped_lock chunk_lock{chunk_mutex};
        for (auto& chunk : s.chunks) {
            chunk.retire_tick = tick;
            retiring_chunks.push_back(std::move(chunk));
        }
        for (const auto pool : s.pools) {
            retiring_pools.emplace_back(tick, pool);
        }
    }
    s.chunks.clear();
    s.chunk_used = 0;
    s.pools.clear();
    RetireSnapshots(s, tick); // PERF-GT7-007
    s.keep.clear(); // freed by the device once the GPU ran this list (NAT-107)
    device->Collect(); // objects the GPU finished with (staging buffers of uploads)
    s.wrapped_list.reset();
    s.list = nullptr;
    s.cmd = CmdBuf{};
    s.rendering = false;
    ++s.serial;        // OPT-015: the list's upload memory retires with it
    ForgetBindings(s); // OPT-014
    ++stats.splices;
}

std::optional<NativeRenderer::Display> NativeRenderer::FindDisplay(const VideoCore::ImageInfo& info,
                                                                   vk::Format view_format) {
    std::scoped_lock lock{mutex};
    if (InPlaceSampled()) {
        for (u32 i = 0; i < in_place.pages.size(); ++i) {
            in_place.total[i] += in_place.pages[i].size();
            in_place.pages[i].clear();
        }
        ++in_place.frames;
    }
    // PERF-GT7-044: this frame's pictures go back after the next global barrier.
    for (auto* entry : frame_writebacks) {
        if (std::ranges::find(writebacks, entry) == writebacks.end()) {
            writebacks.push_back(entry);
        }
    }
    frame_writebacks.clear();
    ++frame;
    TrimChunks();
    shared_frame.store(frame, std::memory_order_relaxed); // OPT-010
    if (RyDiag::Enabled() && !AbOn()) {
        // DIAG-GT7-011: the recorder's stack samples of heavy frames (over 3,300 draws, as GT7's
        // pause menu) go to their own tag in recorder_profile.txt (2; the others 0).
        const u64 drawn = stats.draws >= draws_at_frame ? stats.draws - draws_at_frame : stats.draws;
        draws_at_frame = stats.draws;
        RyDiag::SetProfileTag(drawn > 3300 ? 2 : 0);
    }
    CheckOneOffs(); // DIAG-GT7-008
    // The game's render target there (its layout from the CB registers may differ in details
    // from the video out description), else an image of the display layout.
    Native::ImageEntry* entry =
        registry->FindExisting(info.guest_address, [&](const Native::ImageEntry& candidate) {
            return !candidate.from_cpu && !candidate.info.props.is_depth &&
                   candidate.info.size.width == info.size.width &&
                   candidate.info.size.height == info.size.height &&
                   candidate.info.num_bits == info.num_bits;
        });
    if (!entry) {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-103: nothing drew the display buffer {:#x} ({}x{}, {} bits) yet",
                        info.guest_address, info.size.width, info.size.height, info.num_bits);
        }
        entry = registry->Find(info, true, frame);
    }
    if (!entry || !entry->image) {
        return std::nullopt;
    }
    VideoCore::ImageViewInfo view_info{};
    view_info.format = view_format;
    view_info.mapping.a = AmdGpu::CompSwizzle::One; // as the presenter: no alpha blending
    const vk::ImageView view = View(*entry, view_info, ViewUse::Sampled);
    if (!view) {
        return std::nullopt;
    }
    // A display buffer nothing drew into yet starts in the general layout too.
    if (device->PendingImageInit()) {
        Begin();
        PrepareImages();
    }
    ProbeDisplay(*entry); // FIX-175 (diagnosis)
    if (capture_frame != ~u64{0} && capture_frame + 1 == frame) {
        CaptureImage(*entry, capture_frame, "display"); // DIAG-212 (diagnosis)
    }
    WriteCaptures();      // DIAG-212 (diagnosis)
    StripFrameEnd();      // DIAG-173 (diagnosis)
    ReadDispatchProbes(); // DIAG-175 (diagnosis)
    ReadRefreshProbes();  // DIAG-179 (diagnosis)
    ReadRegionCompares(); // DIAG-211 (diagnosis)
    CheckStaleFlats();    // DIAG-237 (diagnosis)
    ProbeVolume();        // DIAG-180 (diagnosis)
    CheckSmallTextures(); // DIAG-182 (diagnosis)
    ReportProgramTextures(); // DIAG-184 (diagnosis)
    WatchHistoryRanges();    // DIAG-187 (diagnosis)
    // DIAG-174 (diagnosis, RYGT_NATIVE_REUPLOAD=1): every texture the last frames used, not
    // made of targets, is uploaded from guest memory again before its next use (=2: only those
    // from 0x200000000 to 0x208000000, GT7's per-frame memory).
    static const int reupload = [] {
        const char* env = std::getenv("RYGT_NATIVE_REUPLOAD");
        return env && *env ? std::atoi(env) : 0;
    }();
    if (reupload) {
        const VAddr lo = reupload == 2 ? 0x200000000 : 0;
        const VAddr hi = reupload == 2 ? 0x208000000 : ~VAddr{0};
        registry->ForEachOverlapping(lo, hi, [&](Native::ImageEntry& texture) {
            if (!texture.from_cpu || texture.info.guest_address < lo ||
                texture.last_use.load(std::memory_order_relaxed) + 2 < frame) {
                return;
            }
            {
                std::scoped_lock lock{texture.sources_mutex};
                if (!texture.sources.empty()) {
                    return;
                }
            }
            registry->RequestUpload(texture);
        });
    }
    if (!draw_probes.empty() || probe_frame != 0) {
        LOG_WARNING(Render_Vulkan, "DIAG-170: frame {} shows {:#x}", frame - 1,
                    info.guest_address);
        ReadDrawProbes();
    }
    Splice();
    Display display{vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())}, view,
                    entry->info.size.width, entry->info.size.height};
    // GT7-UW: the HUD drawn into this display in the frame just finished (frame counted it).
    if (HudSplitOn()) {
        const auto it = hud_layers.find(info.guest_address);
        if (it != hud_layers.end() && it->second.black && it->second.white &&
            it->second.hud_frame + 1 == frame) {
            VideoCore::ImageViewInfo hud_view{};
            hud_view.format = view_format;
            display.hud_black = View(*it->second.black, hud_view, ViewUse::Sampled);
            display.hud_white = View(*it->second.white, hud_view, ViewUse::Sampled);
            display.hud_race = it->second.race_frame + 1 == frame;
        }
    }
    VerifyMirrors(); // FIX-176 (diagnosis)
    EvictImages(); // (the display picture was just used: it stays)
    return display;
}

u64 NativeRenderer::MirrorVerifyBudget() {
    static const u64 budget = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRROR_VERIFY");
        return env && *env ? std::strtoull(env, nullptr, 10) * 1_MB : u64{0};
    }();
    return budget;
}

void NativeRenderer::VerifyMirrors() {
    // FIX-176 (diagnosis, RYGT_NATIVE_MIRROR_VERIFY=<MB a frame>): guest memory under the mirrors,
    // a few MB a frame, hashed per 64 KB granule and compared with the hash taken the last time;
    // a change no CPU write fault or copy-in accounts for means the mirror went stale unseen (the
    // tunnel explosions went away with RYGT_NATIVE_MIRRORS=0). Such granules are logged and
    // copied in again.
    const u64 budget = MirrorVerifyBudget();
    if (budget == 0 || mirrors.empty()) {
        return;
    }
    SortRegionWrites();
    static thread_local std::vector<u8> scratch(MirrorGranule);
    u64 hashed = 0;
    u32 changed = 0;
    auto it = mirrors.lower_bound(mirror_verify_cursor);
    for (u32 visited = 0; hashed < budget && visited < 2 * mirrors.size() + 2; ++visited) {
        if (it == mirrors.end()) {
            it = mirrors.begin();
            mirror_verify_cursor = 0;
            if (it == mirrors.end()) {
                break;
            }
        }
        Mirror& mirror = it->second;
        const u64 granules = (mirror.end - mirror.begin + MirrorGranule - 1) / MirrorGranule;
        if (mirror.hashes.size() != granules) {
            mirror.hashes.assign(granules, 0);
        }
        u64 g = mirror_verify_cursor > mirror.begin && mirror_verify_cursor < mirror.end
                    ? (mirror_verify_cursor - mirror.begin) / MirrorGranule
                    : 0;
        for (; g < granules && hashed < budget; ++g) {
            const VAddr from = mirror.begin + g * MirrorGranule;
            const u64 size = std::min<u64>(MirrorGranule, mirror.end - from);
            const bool dirty = std::ranges::any_of(mirror.dirty_pages, [&](VAddr page) {
                return page >= from && page < from + size;
            });
            if (dirty) {
                mirror.hashes[g] = 0; // copied in at the next use, hashed again after
                continue;
            }
            if (!Core::RytyMemory::GuardedCopy(scratch.data(), reinterpret_cast<const void*>(from),
                                               size)) {
                continue;
            }
            const u64 hash = XXH3_64bits(scratch.data(), size) | 1;
            hashed += size;
            if (mirror.hashes[g] != 0 && mirror.hashes[g] != hash) {
                // A CPU write that faulted just now is no news: it waits in the pending writes
                // (not sorted here: sorting can drop this very mirror).
                bool pending = false;
                {
                    std::scoped_lock lock{region_dirty_mutex};
                    pending = std::ranges::any_of(pending_region_writes, [&](const auto& w) {
                        return w.first < from + size && w.second > from;
                    });
                }
                if (pending) {
                    mirror.hashes[g] = 0;
                    continue;
                }
                ++changed;
                ++stats.mirror_untrapped;
                static std::atomic<u32> logged{};
                if (Sample(logged)) {
                    LOG_WARNING(Render_Vulkan,
                                "FIX-176: mirror {:#x}+{:#x} granule {:#x}+{:#x} changed with no "
                                "CPU write seen (frame {}, mirror last used frame {})",
                                mirror.begin, mirror.end - mirror.begin, from, size, frame,
                                mirror.used_frame);
                }
                for (VAddr page = from; page < from + size; page += 4096) {
                    mirror.dirty_pages.push_back(page); // copied in again at the next use
                }
                mirror.hashes[g] = 0;
                continue;
            }
            mirror.hashes[g] = hash;
        }
        if (g >= granules) {
            ++it;
            mirror_verify_cursor = it == mirrors.end() ? 0 : it->first;
        } else {
            mirror_verify_cursor = mirror.begin + g * MirrorGranule;
        }
    }
    if (changed != 0) {
        ++buffer_generation; // OPT-003: memos see the dirty pages
        ++buffer_hard_generation; // PERF-GT7-047
    }
}

void NativeRenderer::ProbeDisplay(const Native::ImageEntry& entry) {
    // FIX-175 (diagnosis, RYGT_NATIVE_WATCH_DISPLAY=x,y,w,h): a rectangle of each picture the
    // presenter shows, read back from the GPU: the mean brightness of its top, middle and bottom
    // thirds, one line per presented frame (single-frame glitches stand out against their
    // neighbours; the game's working targets are no good for that, the next frame's work may
    // already have cleared them when a probe runs).
    static const std::array<u32, 4> rect = [] {
        std::array<u32, 4> r{};
        if (const char* env = Common::DiagEnv("RYGT_NATIVE_WATCH_DISPLAY"); env && *env) {
            std::sscanf(env, "%u,%u,%u,%u", &r[0], &r[1], &r[2], &r[3]);
        }
        return r;
    }();
    if (rect[2] == 0 || rect[3] < 3 || !entry.image) {
        return;
    }
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = display_probes.begin(); it != display_probes.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        const u8* bytes = it->buffer->Mapped();
        const u32 w = rect[2], third = rect[3] / 3;
        std::array<double, 3> part{};
        for (u32 band = 0; band < 3; ++band) {
            for (u64 y = u64(band) * third; y < u64(band + 1) * third; ++y) {
                for (u64 x = 0; x < w; ++x) {
                    u32 word;
                    std::memcpy(&word, bytes + (y * w + x) * 4, 4);
                    part[band] += double((word & 1023) + ((word >> 10) & 1023) +
                                         ((word >> 20) & 1023)) / (3.0 * 1023.0);
                }
            }
            part[band] /= double(third) * w;
        }
        LOG_WARNING(Render_Vulkan, "FIX-175 display: frame {} {:#x} thirds: {:.4f} {:.4f} {:.4f}",
                    it->frame, it->address, part[0], part[1], part[2]);
        // FIX-182 (diagnosis): a frame much darker than the one before in two bands or more is
        // saved with that one (raw A2B10G10R10, user\log\glitch_<frame>_<now|before>.raw).
        u32 darker = 0;
        for (u32 band = 0; band < 3; ++band) {
            darker += display_last_part[band] > 0.03 && part[band] < 0.45 * display_last_part[band];
        }
        if (darker >= 2 && display_glitch_dumps < 12 && !display_last_bytes.empty()) {
            ++display_glitch_dumps;
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
            const auto save = [&](const char* name, const u8* data) {
                const auto path = dir / fmt::format("glitch_{}_{}_{}x{}.raw", it->frame, name, w,
                                                    rect[3]);
                if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
                    std::fwrite(data, 1, u64(w) * rect[3] * 4, file);
                    std::fclose(file);
                }
            };
            save("now", bytes);
            save("before", display_last_bytes.data());
            LOG_WARNING(Render_Vulkan, "FIX-182: frame {} saved (darker than the one before)",
                        it->frame);
        }
        display_last_part = part;
        display_last_bytes.assign(bytes, bytes + u64(w) * rect[3] * 4);
        it = display_probes.erase(it);
    }
    const auto& desc = entry.image->Desc();
    if (display_probes.size() >= 16 || rect[0] + rect[2] > desc.width ||
        rect[1] + rect[3] > desc.height || entry.info.num_bits != 32) {
        return;
    }
    auto buffer = device->CreateBuffer({.size = u64(rect[2]) * rect[3] * 4,
                                        .usage = Native::Rhi::BufferUsageStorage |
                                                 Native::Rhi::BufferUsageTransferDst,
                                        .memory = Native::Rhi::MemoryKind::Readback,
                                        .name = "display probe"});
    if (!buffer || !buffer->Mapped()) {
        return;
    }
    Begin();
    GlobalBarrier();
    stream.cmd.copyImageToBuffer(
        vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())}, vk::ImageLayout::eGeneral,
        vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
        vk::BufferImageCopy{.bufferOffset = 0,
                            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                              .mipLevel = 0,
                                              .baseArrayLayer = 0,
                                              .layerCount = 1},
                            .imageOffset = {s32(rect[0]), s32(rect[1]), 0},
                            .imageExtent = {rect[2], rect[3], 1}});
    display_probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1,
                              u64(rect[2]) * rect[3] * 4, frame, 0, rect[2], rect[3], true,
                              entry.info.guest_address});
}

void NativeRenderer::SnapPass(const Targets& targets) {
    // DIAG-226 (diagnosis, RYGT_DUMP_PASS_SNAPS=<hex address>[,<hex address>...]): in a dumped frame,
    // each time a pass starts that draws into one of those colour targets, a copy of the targets as
    // they are then (what the passes before left), written with the dump as
    // snap_<n>_<address>_<w>x<h>_<format>.bin.
    static const std::vector<VAddr> watched = [] {
        std::vector<VAddr> list;
        const char* env = Common::DiagEnv("RYGT_DUMP_PASS_SNAPS");
        for (const char* at = env; at && *at;) {
            char* end = nullptr;
            const VAddr address = std::strtoull(at, &end, 16);
            if (end == at) {
                break;
            }
            list.push_back(address);
            at = *end == ',' ? end + 1 : end;
        }
        return list;
    }();
    if (watched.empty() || pass_snaps.size() >= 160) {
        return;
    }
    const std::array<const Native::ImageEntry*, 3> key{targets.colors[0], targets.colors[1],
                                                       targets.depth};
    if (key == last_snap_key) {
        return;
    }
    last_snap_key = key;
    const u32 index = pass_snap_index++;
    for (u32 cb = 0; cb < 2; ++cb) {
        const auto* entry = targets.colors[cb];
        if (!entry || !entry->image ||
            std::find(watched.begin(), watched.end(), entry->info.guest_address) ==
                watched.end()) {
            continue;
        }
        const auto& desc = entry->image->Desc();
        const u32 texel = std::max(entry->info.num_bits / 8, 1u);
        const u64 size = u64(desc.width) * desc.height * texel;
        auto buffer = device->CreateBuffer({.size = size,
                                            .usage = Native::Rhi::BufferUsageStorage |
                                                     Native::Rhi::BufferUsageTransferDst,
                                            .memory = Native::Rhi::MemoryKind::Readback,
                                            .name = "pass snapshot"});
        if (!buffer || !buffer->Mapped()) {
            continue;
        }
        Stream& s = Begin();
        if (s.rendering) {
            EndRendering();
        }
        GlobalBarrier();
        s.cmd.copyImageToBuffer(
            vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())}, vk::ImageLayout::eGeneral,
            vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
            vk::BufferImageCopy{.bufferOffset = 0,
                                .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                  .mipLevel = 0,
                                                  .baseArrayLayer = 0,
                                                  .layerCount = 1},
                                .imageOffset = {0, 0, 0},
                                .imageExtent = {desc.width, desc.height, 1}});
        GlobalBarrier();
        pass_snaps.push_back({std::move(buffer),
                              fmt::format("snap_{:03}_{:x}_{}x{}_{}.bin", index,
                                          entry->info.guest_address, desc.width, desc.height,
                                          u32(desc.format))});
    }
}

void NativeRenderer::SnapProgramMips(std::span<const Shader::Info* const> stages,
                                     const DrawBindings& b) {
    // DIAG-234 (diagnosis): in a dumped frame, every mip of the mipmapped textures in GT7's
    // per-frame memory that draws with RYGT_DIAG_STAGE_TEXTURES's program sample, as uploaded
    // for them (snap_pyr_<address>_m<mip>_<w>x<h>.bin), once each.
    static const u64 program = [] {
        const char* env = Common::DiagEnv("RYGT_DIAG_STAGE_TEXTURES");
        return env && *env ? u64(std::strtoull(env, nullptr, 16)) : u64{0};
    }();
    if (program == 0 || !r.FrameDumpCollecting() || pass_snaps.size() >= 160 ||
        !std::ranges::any_of(stages, [](const Shader::Info* stage) {
            return stage && stage->pgm_hash == program;
        })) {
        return;
    }
    for (const auto* entry : b.sampled) {
        if (!entry || !entry->image || entry->info.resources.levels < 2 ||
            entry->info.guest_address < 0x200000000 || entry->info.guest_address >= 0x208000000 ||
            std::ranges::any_of(pass_snaps, [&](const PassSnap& snap) {
                return snap.name.starts_with(
                    fmt::format("snap_pyr_{:x}_", entry->info.guest_address));
            })) {
            continue;
        }
        const auto& desc = entry->image->Desc();
        const u32 texel = std::max(entry->info.num_bits / 8, 1u);
        Stream& s = Begin();
        if (s.rendering) {
            EndRendering();
        }
        GlobalBarrier();
        for (u32 mip = 0; mip < entry->info.resources.levels; ++mip) {
            const u32 w = std::max(desc.width >> mip, 1u);
            const u32 h = std::max(desc.height >> mip, 1u);
            auto buffer = device->CreateBuffer({.size = u64(w) * h * texel,
                                                .usage = Native::Rhi::BufferUsageStorage |
                                                         Native::Rhi::BufferUsageTransferDst,
                                                .memory = Native::Rhi::MemoryKind::Readback,
                                                .name = "pyramid snapshot"});
            if (!buffer || !buffer->Mapped()) {
                continue;
            }
            s.cmd.copyImageToBuffer(
                vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())},
                vk::ImageLayout::eGeneral, vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())},
                vk::BufferImageCopy{.bufferOffset = 0,
                                    .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .mipLevel = mip,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1},
                                    .imageOffset = {0, 0, 0},
                                    .imageExtent = {w, h, 1}});
            pass_snaps.push_back({std::move(buffer),
                                  fmt::format("snap_pyr_{:x}_m{}_{}x{}.bin",
                                              entry->info.guest_address, mip, w, h)});
        }
        GlobalBarrier();
    }
}

u64 NativeRenderer::LdsScratch(u64 bytes) {
    // FIX-227: a ring; a part is written and read by one draw right away, and is written again
    // only after the ring went round (frames later, long done on the GPU).
    constexpr u64 RingSize = 128ull << 20;
    if (bytes > RingSize) {
        return 0;
    }
    if (!lds_scratch) {
        lds_scratch = device->CreateBuffer(
            {.size = RingSize,
             .usage = Native::Rhi::BufferUsageStorage | Native::Rhi::BufferUsageDeviceAddress,
             .memory = Native::Rhi::MemoryKind::DeviceLocal,
             .name = "hull shader LDS scratch"});
        if (!lds_scratch) {
            return 0;
        }
    }
    if (Common::AlignUp(lds_scratch_used, 256) + bytes > RingSize) {
        lds_scratch_used = 0;
    }
    const u64 offset = Common::AlignUp(lds_scratch_used, 256);
    lds_scratch_used = offset + bytes;
    return lds_scratch->GpuAddress() + offset;
}

bool NativeRenderer::LdsBatchOn() {
    // On by default since the Daytona A/B (2820 MHz): race 50.7 fps on against 44.3 off, no car
    // defect seen. RYGT_NATIVE_LDS_BATCH=0: each draw alone, as before.
    static const bool on = [] {
        const char* env = std::getenv("RYGT_NATIVE_LDS_BATCH");
        return !(env && env[0] == '0');
    }();
    return on;
}

bool NativeRenderer::CaptureLdsReader(LdsReader& reader, const GraphicsPipeline& pipeline,
                                      const DrawBindings& b, const VertexIndexState& vertices) {
    reader.pipeline = &pipeline;
    reader.vertices = vertices;
    DrawBindings& c = reader.bindings;
    c.buffers.assign(b.buffers.begin(), b.buffers.end());
    c.images.assign(b.images.begin(), b.images.end());
    c.writes = b.writes;
    c.write_sets = b.write_sets;
    c.push = b.push;
    // Where each write's infos are in the draw's own (FlushLdsReaders points the writes there).
    const auto index_of = [](const auto* info, u32 count, const auto& from) {
        if (!info) {
            return u32{~0u};
        }
        const auto at = reinterpret_cast<uintptr_t>(info);
        const auto begin = reinterpret_cast<uintptr_t>(from.data());
        using Info = std::remove_cvref_t<decltype(*info)>;
        if (at < begin || (at - begin) % sizeof(Info) != 0 ||
            (at - begin) / sizeof(Info) + count > from.size()) {
            return u32{~0u - 1};
        }
        return u32((at - begin) / sizeof(Info));
    };
    reader.infos.clear();
    for (auto& write : c.writes) {
        if (write.pNext || write.pTexelBufferView) {
            return false;
        }
        const u32 buffer = index_of(write.pBufferInfo, write.descriptorCount, b.buffers);
        const u32 image = index_of(write.pImageInfo, write.descriptorCount, b.images);
        if (buffer == ~0u - 1 || image == ~0u - 1) {
            return false; // infos outside the draw's own
        }
        reader.infos.emplace_back(buffer, image);
        write.pBufferInfo = nullptr;
        write.pImageInfo = nullptr;
    }
    return true;
}

void NativeRenderer::BindGraphics(Stream& s, const GraphicsPipeline& pipeline) {
    if (pipeline.UsesShaderObjects()) {
        // SO-001: no pipeline; the shader objects and the state a pipeline fixes.
        // SO-010: only what changed since the stream's last shader object draw.
        // RYGT_SO_DELTA=0 or so_delta.off in the game folder: every state each draw.
        static const bool delta = [] {
            const char* env = std::getenv("RYGT_SO_DELTA");
            return !(env && env[0] == '0') && !std::filesystem::exists("so_delta.off");
        }();
        const bool known = s.so_known && delta;
        const auto previous = s.so_state;
        const auto previous_shaders = s.so_shaders;
        const GraphicsPipeline* const bound = &pipeline;
        s.cmd.Run([bound, known, previous, previous_shaders](vk::CommandBuffer cmd) {
            bound->BindShaderObjects(cmd, known ? &previous : nullptr,
                                     known ? &previous_shaders : nullptr);
        });
        s.pipeline = vk::Pipeline{};
        s.so_known = true;
        s.so_state = pipeline.GetShaderObjectState();
        s.so_shaders = pipeline.ShaderObjects();
    } else if (s.pipeline != pipeline.Handle()) {
        s.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.Handle());
        s.pipeline = pipeline.Handle();
        s.so_known = false; // SO-010
    }
}

void NativeRenderer::FlushLdsReaders(u32 reason) {
    Stream& s = stream;
    if (s.lds_readers.empty()) {
        return;
    }
    // Out of the stream first: what records them below ends and begins the pass again.
    std::vector<LdsReader> readers = std::move(s.lds_readers);
    s.lds_readers.clear();
    s.lds_bytes = 0;
    const u64 length = readers.size();
    ++stats.lds_runs;
    ++stats.lds_run_lengths[std::min<size_t>(std::bit_width(length - 1), 6)];
    stats.lds_run_max = std::max(stats.lds_run_max, length);
    ++stats.lds_run_ends[std::min<u32>(reason, 2)];
    // The run's pass, its clears done: the reader passes draw into it after the barrier.
    const RenderState current = s.render;
    RenderState resumed = current;
    for (u32 i = 0; i < resumed.num_color_attachments; ++i) {
        resumed.color_attachments[i].is_clear = false;
    }
    resumed.depth_stencil_attachment.depth_clear = false;
    resumed.depth_stencil_attachment.stencil_clear = false;
    if (s.rendering) {
        s.cmd.endRendering();
        s.rendering = false;
    }
    // A full barrier, as FIX-227's: one from and to the tessellation control stage alone left
    // the reader pass reading stale scratch on NVIDIA. Unlike GlobalBarrier it keeps what was
    // noted written and read: the reader passes come after it.
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    GpuTimeMark(s); // DIAG-GT7-001: the reader passes' segment
    ++stats.barriers;
    BeginRendering(resumed);
    s.render = current; // the same pass to later draws
    // What the stream had bound is the last writer's: everything again for the first reader.
    // SO-018 (diagnosis): what each reader drew with before SO-018, when only pipelines were
    // bound here: a shader object reader (no pipeline handle) drew with the shaders bound last
    // (the run's last writer's), or with no pipeline after a pipeline reader bound a null one.
    int so018_kind = s.so_known ? 0 : 1; // 0 shader objects, 1 a pipeline, 2 a null pipeline
    const auto so018_writer_shaders = s.so_shaders;
    vk::Pipeline so018_pipeline{};
    u32 so018_readers = 0, so018_other = 0, so018_null = 0;
    s.pipeline = vk::Pipeline{};
    s.so_known = false; // SO-018: the first shader object reader sets all its state
    s.vertex_state_known = false;
    ForgetBindings(s);
    const CmdBuf cmdbuf{s.cmd};
    const LdsReader* previous = nullptr;
    const auto same_dynamic = [](const DynamicState& a, const DynamicState& b) {
        return std::ranges::equal(a.viewports, b.viewports) &&
               std::ranges::equal(a.scissors, b.scissors) &&
               a.depth_test_enabled == b.depth_test_enabled &&
               a.depth_write_enabled == b.depth_write_enabled &&
               a.depth_compare_op == b.depth_compare_op &&
               a.depth_bounds_test_enabled == b.depth_bounds_test_enabled &&
               a.depth_bounds_min == b.depth_bounds_min &&
               a.depth_bounds_max == b.depth_bounds_max &&
               a.depth_bias_enabled == b.depth_bias_enabled &&
               a.depth_bias_constant == b.depth_bias_constant &&
               a.depth_bias_clamp == b.depth_bias_clamp &&
               a.depth_bias_slope == b.depth_bias_slope &&
               a.stencil_test_enabled == b.stencil_test_enabled &&
               a.stencil_front_ops == b.stencil_front_ops &&
               a.stencil_front_reference == b.stencil_front_reference &&
               a.stencil_front_write_mask == b.stencil_front_write_mask &&
               a.stencil_front_compare_mask == b.stencil_front_compare_mask &&
               a.stencil_back_ops == b.stencil_back_ops &&
               a.stencil_back_reference == b.stencil_back_reference &&
               a.stencil_back_write_mask == b.stencil_back_write_mask &&
               a.stencil_back_compare_mask == b.stencil_back_compare_mask &&
               a.primitive_restart_enable == b.primitive_restart_enable &&
               a.rasterizer_discard_enable == b.rasterizer_discard_enable &&
               a.cull_mode == b.cull_mode && a.front_face == b.front_face &&
               a.blend_constants == b.blend_constants &&
               a.color_write_masks == b.color_write_masks && a.line_width == b.line_width &&
               a.feedback_loop_enabled == b.feedback_loop_enabled;
    };
    for (LdsReader& reader : readers) {
        const GraphicsPipeline& pipeline = *reader.pipeline;
        if (pipeline.UsesShaderObjects()) { // SO-018 (diagnosis)
            ++so018_readers;
            if (so018_pipeline) {
                so018_pipeline = vk::Pipeline{};
                so018_kind = 2;
                ++so018_null;
            }
            if (so018_kind != 0 || so018_writer_shaders != pipeline.ShaderObjects()) {
                ++so018_other;
            }
        } else if (so018_pipeline != pipeline.Handle()) {
            so018_pipeline = pipeline.Handle();
            so018_kind = 1;
        }
        // SO-018: a reader drawn with shader objects binds them and the state a pipeline fixes,
        // as its writer pass did (its Handle() is null: binding only pipelines here left it the
        // shaders bound last, or bound a null pipeline after a pipeline reader).
        BindGraphics(s, pipeline);
        if (!previous || !same_dynamic(previous->dynamic, reader.dynamic)) {
            s.dynamic = reader.dynamic;
            s.dynamic.Invalidate();
            s.dynamic.Commit(r.instance, cmdbuf);
        }
        previous = &reader;
        RecordVertexIndex(reader.vertices);
        DrawBindings& bound = reader.bindings;
        for (size_t i = 0; i < bound.writes.size() && i < reader.infos.size(); ++i) {
            const auto [buffer, image] = reader.infos[i];
            bound.writes[i].pBufferInfo = buffer != ~0u ? bound.buffers.data() + buffer : nullptr;
            bound.writes[i].pImageInfo = image != ~0u ? bound.images.data() + image : nullptr;
        }
        PushBindings(pipeline, vk::PipelineBindPoint::eGraphics, bound);
        const std::array<u32, 2> words{u32(reader.address), u32(reader.address >> 32)};
        cmdbuf.pushConstants(pipeline.GetLayout(), AllGraphicsStageBits,
                             u32(offsetof(Shader::PushData, quad_sample_mask)), 8u, words.data());
        if (reader.indexed) {
            if (reader.count != 0) {
                cmdbuf.drawIndexed(reader.count, reader.instances, 0, s32(reader.vertex_offset),
                                   reader.first_instance);
            }
        } else {
            cmdbuf.draw(reader.count, reader.instances, reader.vertex_offset,
                        reader.first_instance);
        }
        if (GpuSegment* seg = GpuSeg()) { // DIAG-GT7-001
            ++seg->draws;
            if (reader.prims > seg->max_prims) {
                seg->max_prims = reader.prims;
                seg->max_prims_ps = reader.ps;
            }
        }
    }
    s.push_known = false; // the quad sample mask's words hold the last reader's scratch
    if (so018_readers != 0) {
        static std::atomic<u64> total_readers{}, total_other{}, total_null{};
        static std::atomic<u32> logged{};
        const u64 readers_so_far = total_readers += so018_readers;
        const u64 other_so_far = total_other += so018_other;
        const u64 null_so_far = total_null += so018_null;
        if (so018_other != 0) {
            if (const u32 n = ++logged; n <= 5 || n % 500 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "SO-018: a run of {} reader passes, {} with shader objects: {} of "
                            "them drew with other shaders before SO-018 ({} after a null "
                            "pipeline bind); so far {} shader object readers, {} with other "
                            "shaders, {} after a null bind ({} such runs)",
                            length, so018_readers, so018_other, so018_null, readers_so_far,
                            other_so_far, null_so_far, n);
            }
        }
    }
    readers.clear();
    if (s.lds_readers.empty()) {
        s.lds_readers = std::move(readers); // its storage for the next run
    }
}

void NativeRenderer::ReportLdsRuns() {
    static auto reported = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - reported < std::chrono::seconds{10}) {
        return;
    }
    const auto lengths = [](const std::array<u64, 7>& a) {
        return fmt::format("{}/{}/{}/{}/{}/{}/{}", a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    };
    LOG_WARNING(Render_Vulkan,
                "PERF-GT7-060 ({:.1f} s): {} FIX-227 draws in {} runs sharing a barrier ({:.2f} a "
                "run, longest {}; by length 1/2/3-4/5-8/9-16/17-32/33+ {}; ended by another draw/"
                "the pass's end/the limit {}/{}/{}); {} drawn alone, the runs they made {} ({:.2f} "
                "a run, longest {}; by length {})",
                std::chrono::duration<double>(now - reported).count(), stats.lds_batched,
                stats.lds_runs, double(stats.lds_batched) / double(std::max<u64>(stats.lds_runs, 1)),
                stats.lds_run_max, lengths(stats.lds_run_lengths), stats.lds_run_ends[0],
                stats.lds_run_ends[1], stats.lds_run_ends[2], stats.lds_alone,
                stats.lds_alone_runs,
                double(stats.lds_alone) / double(std::max<u64>(stats.lds_alone_runs, 1)),
                stats.lds_alone_run_max, lengths(stats.lds_alone_lengths));
    stats.lds_batched = stats.lds_runs = stats.lds_run_max = 0;
    stats.lds_run_lengths = {};
    stats.lds_run_ends = {};
    stats.lds_alone = stats.lds_alone_runs = stats.lds_alone_run_max = 0;
    stats.lds_alone_lengths = {};
    reported = now;
}

void NativeRenderer::DumpFrame(const std::filesystem::path& dir) {
    std::scoped_lock lock{mutex};
    struct Readback {
        const Native::ImageEntry* entry;
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 size;
        u32 index;
    };
    std::vector<Readback> readbacks;
    std::vector<std::unique_ptr<Native::Rhi::Image>> resolved_images;
    std::vector<std::unique_ptr<Native::Rhi::ImageView>> resolved_views;
    Stream& s = Begin();
    GlobalBarrier();
    // NAT-123: the sampled GPU-written textures follow the targets (named tex_<key>).
    std::vector<std::pair<const Native::ImageEntry*, std::string>> jobs;
    {
        u32 order = 0;
        for (const auto* entry : dump_entries) {
            jobs.emplace_back(entry, fmt::format("{:03}_{:x}_{}x{}_{}{}.bin", order++,
                                                 entry->info.guest_address,
                                                 entry->image ? entry->image->Desc().width : 0,
                                                 entry->image ? entry->image->Desc().height : 0,
                                                 entry->image ? u32(entry->image->Desc().format)
                                                              : 0,
                                                 entry->info.props.is_depth ? "_depth" : ""));
        }
        std::string list;
        for (const auto& [key, entry] : dump_textures) {
            u32 sources;
            {
                std::scoped_lock lock{const_cast<Native::ImageEntry*>(entry)->sources_mutex};
                sources = u32(entry->sources.size());
            }
            const auto& info = entry->info;
            // FIX-153 (diagnosis): what guest memory holds there, and whether the GPU writes it
            // in a region.
            std::array<u32, 4> head{};
            const bool readable = Core::RytyMemory::GuardedCopy(
                head.data(), reinterpret_cast<const void*>(info.guest_address), sizeof(head));
            const Region* region =
                RegionOver(info.guest_address, std::max<u64>(info.guest_size, 1));
            // The bytes there now against the bytes last uploaded (NAT-109's hash).
            u64 hash_now = 0;
            const u64 size = info.guest_size;
            if (size != 0 && size <= (4ull << 20) && r.memory->IsValidMapping(info.guest_address) &&
                r.memory->ClampRangeSize(info.guest_address, size) == size &&
                Native::ReadablePrefix(reinterpret_cast<const void*>(info.guest_address), size) ==
                    size) {
                hash_now = Core::HashGuestBytes(info.guest_address, size) | 1;
            }
            list += fmt::format("{} -> {} {:#x} {}x{} mips {} layers {} {}, {} sources, "
                                "version {}{}; memory {}; {}; uploads {}, stale {}; "
                                "total uploads {} kind {}, tiled {} mode {} bits {} size {:#x}; "
                                "hash now {:016x} uploaded {:016x}{}{}\n",
                                key, entry->from_cpu ? "texture" : "target", info.guest_address,
                                info.size.width, info.size.height, info.resources.levels,
                                info.resources.layers, vk::to_string(info.pixel_format), sources,
                                entry->write_version.load(),
                                info.num_samples > 1 ? " (multisampled, not dumped)" : "",
                                readable ? fmt::format("{:08x} {:08x} {:08x} {:08x}", head[0],
                                                       head[1], head[2], head[3])
                                         : std::string{"unreadable"},
                                region ? fmt::format("in region {:#x}+{:#x}", region->begin,
                                                     region->end - region->begin)
                                       : std::string{"no region"},
                                entry->upload_count.load(), entry->needs_upload.load(),
                                entry->total_uploads.load(), entry->upload_kind,
                                info.props.is_tiled, u32(info.tile_mode), info.num_bits, size,
                                hash_now, entry->content_hash,
                                hash_now && entry->content_hash && hash_now != entry->content_hash
                                    ? " CHANGED"
                                    : "",
                                [&] { // FIX-GT7-027
                                    const int differing =
                                        registry->LayersDifferingFromUpload(*entry);
                                    return differing < 0
                                               ? std::string{}
                                               : fmt::format("; layers differing from their "
                                                             "upload {} of {}",
                                                             differing, info.resources.layers);
                                }());
            jobs.emplace_back(entry, fmt::format("tex_{}{}.bin", key,
                                                 info.props.is_depth ? "_depth" : ""));
        }
        if (std::FILE* file = std::fopen((dir / "textures.txt").string().c_str(), "w")) {
            std::fwrite(list.data(), 1, list.size(), file);
            std::fclose(file);
        }
    }
    u32 index = 0;
    for (const auto& [entry, name] : jobs) {
        const u32 order = index++;
        if (!entry->image) {
            continue;
        }
        const auto& desc = entry->image->Desc();
        const u32 texel = std::max(entry->info.num_bits / 8, 1u);
        // NAT-125: every layer of the first mip (cube faces, layered targets), one after another.
        const u32 layers = desc.dim == Native::Rhi::ImageDim::D3 ? 1u : std::max(desc.layers, 1u);
        const u32 depth = desc.dim == Native::Rhi::ImageDim::D3 ? std::max(desc.depth, 1u) : 1u;
        // FIX-152 (diagnosis): block-compressed images give their blocks (4x4 texels each).
        const bool block = entry->info.props.is_block;
        const u32 row = block ? Common::AlignUp(desc.width, 4u) : desc.width;
        const u32 rows = block ? Common::AlignUp(desc.height, 4u) : desc.height;
        const u64 size = block ? u64(row / 4) * (rows / 4) * texel * layers * depth
                               : u64(desc.width) * desc.height * texel * layers * depth;
        auto buffer = device->CreateBuffer({.size = size,
                                            .usage = Native::Rhi::BufferUsageTransferDst,
                                            .memory = Native::Rhi::MemoryKind::Readback,
                                            .name = "native dump"});
        if (!buffer || !buffer->Mapped()) {
            continue;
        }
        vk::Image source{Handle<VkImage_T>(entry->image->NativeHandle())};
        if (desc.samples != 1) {
            if (!entry->info.props.is_depth || layers != 1) continue;
            auto resolved_desc = desc;
            resolved_desc.samples = 1;
            auto resolved = device->CreateImage(resolved_desc);
            if (!resolved) continue;
            const vk::Image destination{Handle<VkImage_T>(resolved->NativeHandle())};
            const vk::ImageMemoryBarrier2 transition{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = destination,
                .subresourceRange{.aspectMask = vk::ImageAspectFlagBits::eDepth |
                    (entry->info.props.has_stencil ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlags{}),
                    .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
            };
            s.cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                    .pImageMemoryBarriers = &transition});
            Native::Rhi::ImageViewDesc view_desc{.aspect = Native::Rhi::AspectDepth};
            auto source_view = device->CreateImageView(entry->image.get(), view_desc);
            auto resolve_view = device->CreateImageView(resolved.get(), view_desc);
            const vk::RenderingAttachmentInfo attachment{
                .imageView = vk::ImageView{Handle<VkImageView_T>(source_view->NativeHandle())},
                .imageLayout = vk::ImageLayout::eGeneral,
                .resolveMode = vk::ResolveModeFlagBits::eSampleZero,
                .resolveImageView = vk::ImageView{Handle<VkImageView_T>(resolve_view->NativeHandle())},
                .resolveImageLayout = vk::ImageLayout::eGeneral,
                .loadOp = vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
            };
            s.cmd.beginRendering(vk::RenderingInfo{
                .renderArea{.extent = {desc.width, desc.height}}, .layerCount = 1,
                .pDepthAttachment = &attachment,
            });
            s.cmd.endRendering();
            GlobalBarrier();
            source = destination;
            resolved_images.push_back(std::move(resolved));
            resolved_views.push_back(std::move(source_view));
            resolved_views.push_back(std::move(resolve_view));
        }
        const vk::BufferImageCopy copy{
            .bufferOffset = 0,
            .bufferRowLength = row,
            .bufferImageHeight = rows,
            .imageSubresource{.aspectMask = entry->info.props.is_depth
                                               ? vk::ImageAspectFlagBits::eDepth
                                               : vk::ImageAspectFlagBits::eColor,
                              .mipLevel = 0,
                              .baseArrayLayer = 0,
                              .layerCount = layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {desc.width, desc.height, depth},
        };
        s.cmd.copyImageToBuffer(source,
                                vk::ImageLayout::eGeneral,
                                vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())}, copy);
        readbacks.push_back({entry, std::move(buffer), size, order});
    }
    Splice();
    r.scheduler.Finish();
    for (const auto& readback : readbacks) {
        const auto& info = readback.entry->info;
        const auto& desc = readback.entry->image->Desc();
        const auto path = dir / jobs[readback.index].second;
        if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
            std::fwrite(readback.buffer->Mapped(), 1, readback.size, file);
            std::fclose(file);
        }
    }
    for (const auto& snap : pass_snaps) { // DIAG-226
        if (std::FILE* file = std::fopen((dir / snap.name).string().c_str(), "wb")) {
            std::fwrite(snap.buffer->Mapped(), 1, snap.buffer->Size(), file);
            std::fclose(file);
        }
    }
    if (!pass_snaps.empty()) {
        LOG_WARNING(Render_Vulkan, "DIAG-226: {} pass snapshots dumped", pass_snaps.size());
    }
    pass_snaps.clear();
    last_snap_key = {};
    pass_snap_index = 0;
    LOG_WARNING(Render_Vulkan, "NAT-116: {} render targets and textures of the frame dumped to {}",
                readbacks.size(), dir.string());
    dump_entries.clear();
    dump_textures.clear();
}

void NativeRenderer::Report() {
    ApplyRecorderPriority(); // PERF-GT7-010 (on the recorder: Report runs at its draws' end)
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    last_report = now;
    // DIAG-GT7-005: the recorder's time in the window by part, in ms a second.
    const std::string recorder_parts = [&] {
        if (!AmdGpu::RecorderParts::On()) {
            return std::string{};
        }
        const auto ns = AmdGpu::RecorderParts::Take();
        u64 busy = 0;
        std::string text;
        for (u32 i = 0; i < ns.size(); ++i) {
            busy += ns[i];
            text += fmt::format("{}{:.1f}", i == 0 ? "" : "/", ns[i] / 1e6 / seconds);
        }
        return fmt::format("jobs/draw jobs/draws/dispatches/labels/DMAs/splices/submits/uploads/"
                           "write-backs/GPU waits/lists ahead {}, busy {:.1f}",
                           text, busy / 1e6 / seconds);
    }();
    if (!recorder_parts.empty()) {
        LOG_INFO(Render_Vulkan, "DIAG-GT7-005 (10 s): recorder ms a second in {}", recorder_parts);
    }
    const auto& registry_stats = registry->GetStats();
    const u64 uploads = registry_stats.uploads.load();
    const u64 upload_bytes = registry_stats.upload_bytes.load();
    const u64 evicted = registry_stats.evicted.load();
    std::array<u64, 3> stale{};
    for (u32 i = 0; i < stale.size(); ++i) {
        stale[i] = registry_stats.stale[i].load();
    }
    // The textures uploaded most in the window.
    std::vector<std::tuple<u32, VAddr, u32, u32, u64, bool>> busy;
    registry->ForEachOverlapping(0, ~VAddr{0}, [&](Native::ImageEntry& entry) {
        if (const u32 count = entry.upload_count.exchange(0); count >= 4) {
            busy.emplace_back(count, entry.info.guest_address, entry.info.size.width,
                              entry.info.size.height, entry.info.guest_size, entry.from_cpu);
        }
    });
    std::ranges::sort(busy, std::greater{});
    std::string busiest;
    for (size_t i = 0; i < std::min<size_t>(busy.size(), 5); ++i) {
        const auto& [count, address, width, height, size, texture] = busy[i];
        busiest += fmt::format(" {} {:#x} {}x{} ({} KB) x{};", texture ? "texture" : "target",
                               address, width, height, size / 1024, count);
    }
    std::string crowded;
    for (const auto& [address, count] : registry->Crowded(4)) {
        crowded += fmt::format(" {:#x} x{};", address, count);
        if (crowded.size() < 40) {
            // PERF-GT7-011: what the busiest address holds (textures, of them idle and made of
            // targets).
            u32 textures = 0, idle = 0, made_of_targets = 0;
            registry->ForEachOverlapping(address, address + 1, [&](Native::ImageEntry& entry) {
                if (entry.info.guest_address != address || !entry.from_cpu) {
                    return;
                }
                ++textures;
                idle += entry.last_use.load(std::memory_order_relaxed) + 120 < frame ? 1 : 0;
                std::scoped_lock lock{entry.sources_mutex};
                made_of_targets += entry.sources.empty() ? 0 : 1;
            });
            crowded.pop_back();
            crowded += fmt::format(" ({} textures, {} idle, {} made of targets);", textures, idle,
                                   made_of_targets);
        }
    }
    LOG_INFO(Render_Vulkan,
             "NAT-108 images (10 s): {} uploads ({:.1f} MB), {} skipped as unchanged (NAT-109), "
             "textures made stale by CPU writes {}, unmaps {}, GPU writes {} ({} targets so far, "
             "NAT-119); {} targets parts of others so far (FIX-197); {} evicted, {} live, most at "
             "one address:{}; uploaded most:{}",
             uploads - report_base.uploads, (upload_bytes - report_base.upload_bytes) / 1048576.0,
             registry->GetUploadStats().unchanged.exchange(0),
             stale[0] - report_base.stale[0], stale[1] - report_base.stale[1],
             stale[2] - report_base.stale[2], registry_stats.targets_rewritten.load(),
             registry_stats.target_parts.load(), evicted - report_base.evicted, registry->Count(),
             crowded,
             busiest.empty() ? std::string{" none"} : busiest);
    report_base = {uploads, upload_bytes, evicted, stale};
    LOG_INFO(Render_Vulkan,
             "NAT-100 native (10 s): {} draws, {} dispatches, {} skipped, {} without pipeline, "
             "{} indirect draws ({} not drawn, NAT-161); "
             "{} passes, {} barriers ({} pass barriers, {} passes unblocked, NAT-140), {} lists "
             "run; {} reads copied ({:.1f} MB), {} ranges bound "
             "empty, {} images bound empty; {} fills, {} copies, {} meta clears, {} inline "
             "writes, {} pictures copied from targets, {} written back, {} CPU writes to textures, "
             "{} ranges gathered, region binds/made/refreshes {}/{}/{} (NAT-114; {} regions, "
             "{:.1f} MB, {} pages refreshed in {:.1f} ms; {} guest buffers in {:.1f} ms), "
             "mirrors {} ({:.1f} MB; NAT-150: {} binds, {} made ({:.1f} MB), {} dropped as "
             "CPU-written, {} evicted, {} over budget, {} pages refreshed, {} granules read in "
             "place), "
             "memo hits textures {}/{} buffers {} (OPT-002/003), vertex ranges in the "
             "previous one's mirror {} (OPT-004), vertex records skipped {} (OPT-005), "
             "prepared draws (OPT-010): {} used, {} without one, rejected (pipeline/user "
             "data/unmap/images/buffers) {}, made on job threads {} ({:.2f} us each, {} could "
             "not, {} replaced, {} stale), vertices from them {} ({} not), pipelines from "
             "them {} (OPT-012), record parts "
             "barrier/begin/dynamic/vertex/pipeline/bindings/draw/notes {}, "
             "fill shaders {}, targets rewritten by clears/uploads/partial clears/fills "
             "{}/{}/{}/{} (NAT-120/124/132), "
             "on the GPU ({:.1f} MB), uploads from guest memory on the GPU/CPU {}/{}; images so far: "
             "{} lookups, {} textures and {} targets made, {} uploads ({:.1f} MB); {:.0f} draws/s, "
             "{:.2f} us a draw, {:.2f} us a dispatch on the recorder; images bound empty by "
             "reason (none/unmapped/format/geometry/outside/FIX-081/no image) {}; us per draw in "
             "pipeline/targets/buffers/textures(sources)/vertices/prepare/record {}",
             stats.draws, stats.dispatches, stats.skipped, stats.no_pipeline,
             stats.indirect_draws, stats.indirect_skipped, stats.passes,
             stats.barriers, stats.pass_barriers, stats.passes_unblocked, stats.splices,
             stats.copied_reads, stats.copied_bytes / 1048576.0,
             stats.unbound, stats.null_images, stats.fills, stats.copies, stats.meta_clears,
             stats.inline_writes, stats.source_copies, stats.writebacks,
             stats.cpu_writes.exchange(0),
             stats.gathers, stats.region_binds, stats.regions_made, stats.region_refreshes,
             regions.size(),
             [&] {
                 u64 bytes = 0;
                 for (const auto& [base, region] : regions) {
                     bytes += region.end - region.begin;
                 }
                 return bytes / 1048576.0;
             }(),
             stats.region_refresh_pages, stats.region_refresh_ns / 1e6, stats.guest_buffer_calls,
             stats.guest_buffer_ns * 16 / 1e6, // OPT-001: timed on one draw in 16
             mirrors.size(), mirror_bytes / 1048576.0,
             stats.mirror_binds, stats.mirrors_made, stats.mirror_made_bytes / 1048576.0,
             stats.mirrors_dropped, stats.mirrors_evicted, stats.mirror_over_budget,
             stats.mirror_refresh_pages, dynamic_granules.size(), stats.texture_memo_hits,
             stats.texture_lookups, stats.buffer_memo_hits, stats.vertex_near,
             stats.vertex_skipped, stats.prepared_used, stats.prepared_missing,
             fmt::join(stats.prepared_rejected, "/"), prepare_stats.made.load(),
             prepare_stats.made.load() ? prepare_stats.ns.load() / 1000.0 / prepare_stats.made.load()
                                       : 0.0,
             prepare_stats.failed.load(), prepare_stats.replaced.load(),
             prepare_stats.stale.load(), stats.prepared_vertices,
             stats.prepared_vertices_rejected, stats.prepared_installed,
             [&] {
                 std::string text;
                 for (u32 i = 0; i < stats.record_ns.size(); ++i) {
                     text += fmt::format("{}{:.2f}", i == 0 ? "" : "/",
                                         stats.timed_draws
                                             ? stats.record_ns[i] / 1000.0 / stats.timed_draws
                                             : 0.0);
                 }
                 return text;
             }(),
             stats.fill_shaders, stats.rewrite_clears, stats.rewrite_uploads,
             stats.rewrite_partial, stats.rewrite_fills,
             stats.gathered_bytes / 1048576.0,
             registry->GetUploadStats().gpu.load(), registry->GetUploadStats().cpu.load(),
             registry_stats.lookups.load(),
             registry_stats.created_textures.load(), registry_stats.created_targets.load(),
             registry_stats.uploads.load(), registry_stats.upload_bytes.load() / 1048576.0,
             stats.draws / seconds, stats.draws ? stats.draw_ns / 1000.0 / stats.draws : 0.0,
             stats.dispatches ? stats.dispatch_ns / 1000.0 / stats.dispatches : 0.0,
             fmt::join(stats.null_reasons, "/"), [&] {
                 std::string text;
                 for (u32 i = 0; i < stats.phase_ns.size(); ++i) {
                     text += fmt::format(
                         "{}{:.2f}", i == 0 ? "" : (i == 4 ? "(" : (i == 5 ? ")/" : "/")),
                         stats.timed_draws ? stats.phase_ns[i] / 1000.0 / stats.timed_draws
                                           : 0.0);
                 }
                 return text;
             }());
    LOG_INFO(Render_Vulkan,
             "OPT-013..015 (10 s): targets reused {} of {} draws; sets bound again {} of {}, "
             "pushed sets kept {}, push constants kept {}; uploads reused {}",
             stats.targets_reused, stats.draws, stats.sets_reused,
             stats.sets_reused + stats.sets_written, stats.pushes_skipped,
             stats.push_constants_skipped, stats.uploads_reused);
    stats.targets_reused = stats.sets_reused = stats.sets_written = stats.pushes_skipped = 0;
    stats.push_constants_skipped = stats.uploads_reused = 0;
    if (AbOn()) {
        // DIAG-GT7-003 (RYGT_NATIVE_AB=1): the recorder's cost in the window that ends (A the
        // defaults without PERF-GT7-018, B with it), then the next window in the next mode.
        const auto per_draw = [&](u64 ns) {
            return stats.timed_draws ? ns / 1000.0 / stats.timed_draws : 0.0;
        };
        std::string phases;
        for (u32 i = 0; i < stats.phase_ns.size(); ++i) {
            phases += fmt::format("{}{:.2f}", i == 0 ? "" : "/", per_draw(stats.phase_ns[i]));
        }
        std::string parts;
        for (u32 i = 0; i < stats.record_ns.size(); ++i) {
            parts += fmt::format("{}{:.2f}", i == 0 ? "" : "/", per_draw(stats.record_ns[i]));
        }
        LOG_WARNING(Render_Vulkan,
                    "DIAG-GT7-003 A/B window {} mode {}: {:.1f} s, {} draws, {:.3f} us a draw, {} "
                    "dispatches, {:.2f} us a dispatch, {} frames; phases pipeline/targets/buffers/"
                    "textures/sources/vertices/prepare/record {}; record parts barrier/begin/"
                    "dynamic/vertex/pipeline/bindings/draw/notes {}; dispatches: {}; recorder ms a "
                    "second in {}; lists begun ahead {}",
                    ab_window, char('A' + AbMode()), seconds, stats.draws,
                    stats.draws ? stats.draw_ns / 1000.0 / stats.draws : 0.0, stats.dispatches,
                    stats.dispatches ? stats.dispatch_ns / 1000.0 / stats.dispatches : 0.0,
                    frame - ab_frame, phases, parts, DispatchPartsText(), recorder_parts,
                    stats.lists_ahead);
        stats.lists_ahead = 0;
        ab_frame = frame;
        ++ab_window;
        const u32 mode = AbMode();
        g_ring_on.store(mode == 1, std::memory_order_relaxed); // PERF-GT7-018 (B; new lists)
        // DIAG-GT7-007: where the window begins, for the frames of each mode.
        RyDiag::Timeline('W', RyDiag::Seconds(std::chrono::steady_clock::now()), 0.0, 0.0, 0.0,
                         (ab_window << 8) | mode);
        targets_memo.valid = false;
        ForgetBindings(stream);
        ForgetBindings(hud_stream);
        ++stream.serial;
        ++hud_stream.serial;
        // DIAG-GT7-002: the mode, and the window for the raw samples.
        RyDiag::SetProfileTag(u32(ab_window << 8) | (mode + 1));
    }
    if (in_place.frames != 0) {
        const auto mb = [&](u32 i) { return in_place.total[i] * 64.0 / 1024.0 / in_place.frames; };
        LOG_INFO(Render_Vulkan,
                 "NAT-142: guest memory used in place per frame (64 KB pages, MB): vertices "
                 "{:.1f}, indices {:.1f}, read-only storage {:.1f}, storage read {:.1f}, storage "
                 "written {:.1f}; from video memory instead: mirrors {:.1f}, regions {:.1f} "
                 "(NAT-150; {} frames sampled)",
                 mb(0), mb(1), mb(2), mb(3), mb(4), mb(5), mb(6), in_place.frames);
        in_place.total = {};
        in_place.frames = 0;
    }
    const u64 cpu_writes = stats.cpu_writes.load();
    registry->ReportUploadTally(); // DIAG-197 (diagnosis)
    if (const u64 faults = g_fault_count.exchange(0); faults != 0) {
        const auto per = [&](u32 part) {
            return g_fault_part_ns[part].exchange(0) / 1000.0 / double(faults);
        };
        LOG_INFO(Render_Vulkan, "DIAG-194: {} CPU write faults, us each in start/snapshot/unwatch/"
                 "registry/rest {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}", faults, per(0), per(1), per(2),
                 per(3), per(4));
    }
    if (stats.dispatches != 0) {
        const auto per = [&](u64 ns) { return ns / 1000.0 / double(stats.dispatches); };
        LOG_INFO(Render_Vulkan, "DIAG-188: us a dispatch in pipeline/bindings/images/barrier/"
                 "record/notes {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}; write-backs {:.1f} ms",
                 per(stats.dispatch_part_ns[0]), per(stats.dispatch_part_ns[1]),
                 per(stats.dispatch_part_ns[2]), per(stats.dispatch_part_ns[3]),
                 per(stats.dispatch_part_ns[4]), per(stats.dispatch_part_ns[5]),
                 stats.writeback_ns / 1e6);
        LOG_INFO(Render_Vulkan, "DIAG-GT7-004 (10 s): {} dispatches: {}", stats.dispatches,
                 DispatchPartsText());
    }
    stats.dispatch_part_ns = {};
    stats.writeback_ns = 0;
    stats.dispatch_bind_ns = {};
    stats.dispatch_note_ns = {};
    stats.dispatch_refresh_ns = 0;
    stats.snapshot_bytes = stats.snapshots_made = stats.snapshots_reused = 0;
    stats.compute_texture_lookups = stats.compute_texture_hits = 0;
    stats.compute_texture_bindings = 0;
    if (stats.writeback_befores != 0 || stats.precise_writebacks != 0) {
        LOG_INFO(Render_Vulkan, "FIX-203: {} written ranges copied before the work, {} page runs "
                 "written back by compare", std::exchange(stats.writeback_befores, 0),
                 std::exchange(stats.precise_writebacks, 0));
    }
    if (stats.small_rewrites != 0) {
        LOG_INFO(Render_Vulkan, "FIX-202: {} small textures uploaded again after larger storage "
                 "writes over them", std::exchange(stats.small_rewrites, 0));
    }
    if (stats.one_off_waits != 0) {
        LOG_INFO(Render_Vulkan, "FIX-198: {} draws into pictures not drawn lately waited for their "
                 "new pipeline", std::exchange(stats.one_off_waits, 0));
    }
    if (const u64 opened = fault_ahead_pages.exchange(0); opened != 0) {
        LOG_INFO(Render_Vulkan, "PERF-GT7-030: {} write faults in region chunks written lately "
                 "opened {} more pages with them; FIX-GT7-018: {} left shut under a picture",
                 fault_ahead_faults.exchange(0), opened, fault_ahead_pictures.exchange(0));
    }
    if (downloaded_with_copy != 0) {
        LOG_INFO(Render_Vulkan, "FIX-GT7-020: {} region pages a picture's download wrote had a "
                 "copy from before ({} opened by PERF-GT7-030): copied in whole",
                 std::exchange(downloaded_with_copy, 0), std::exchange(downloaded_ahead, 0));
    }
    if (RyDiag::Enabled()) {
        if (std::string text = FaultPagesText(); !text.empty()) {
            LOG_WARNING(Render_Vulkan, "DIAG-GT7-010 (10 s): {}", text);
        }
    }
    if (stats.one_off_exempt != 0) {
        LOG_INFO(Render_Vulkan, "PERF-GT7-023: {} draws into one-off pictures of kinds always "
                 "redrawn skipped instead of waiting", std::exchange(stats.one_off_exempt, 0));
    }
    if (stats.one_off_budget_skips != 0) {
        LOG_INFO(Render_Vulkan, "FIX-GT7-102: {} draws into pictures not drawn lately skipped "
                 "once their frame had waited its budget for one-off pipelines",
                 std::exchange(stats.one_off_budget_skips, 0));
    }
    if (stats.one_off_harmful_waits != 0) {
        LOG_INFO(Render_Vulkan, "FIX-GT7-121: {} draws into pictures of kinds seen sampled before "
                 "their redraw waited for their pipeline whatever the budget",
                 std::exchange(stats.one_off_harmful_waits, 0));
    }
    SaveOneOffKinds(false); // PERF-GT7-023
    if (OneOffDiagOn()) {
        if (std::string text = OneOffText(); !text.empty()) {
            LOG_WARNING(Render_Vulkan, "DIAG-GT7-008 (10 s): {}", text);
        }
    }
    if (WritebackRaceDiag()) {
        LOG_INFO(Render_Vulkan, "DIAG-176: {} CPU write faults, {} into pending write-backs so far",
                 writeback_race_faults.exchange(0), writeback_race_hits.load());
    }
    if (stats.region_whole_pages || stats.region_writebacks || stats.region_pages_forgotten) {
        LOG_INFO(Render_Vulkan,
                 "FIX-194: region pages copied in whole {}, write-backs {} ({:.1f} MB), pages "
                 "left to the next CPU write {}",
                 stats.region_whole_pages, stats.region_writebacks,
                 stats.region_writeback_bytes / 1048576.0, stats.region_pages_forgotten);
    }
    stats.region_whole_pages = stats.region_writebacks = 0;
    stats.region_writeback_bytes = stats.region_pages_forgotten = 0;
    stats.draws = stats.dispatches = stats.skipped = stats.no_pipeline = stats.passes = 0;
    stats.indirect_draws = stats.indirect_skipped = 0;
    stats.barriers = stats.splices = stats.copied_reads = stats.copied_bytes = stats.unbound = 0;
    stats.null_images = stats.fills = stats.copies = stats.meta_clears = stats.inline_writes = 0;
    stats.source_copies = stats.draw_ns = stats.dispatch_ns = 0;
    stats.timed_draws = 0;
    stats.texture_memo_hits = stats.texture_lookups = stats.buffer_memo_hits = 0;
    stats.vertex_near = stats.vertex_skipped = 0;
    stats.prepared_used = stats.prepared_missing = 0;
    stats.prepared_rejected = {};
    stats.prepared_vertices = stats.prepared_vertices_rejected = 0;
    stats.prepared_installed = 0;
    stats.record_ns = {};
    prepare_stats.made = 0;
    prepare_stats.failed = 0;
    prepare_stats.replaced = 0;
    prepare_stats.stale = 0;
    prepare_stats.ns = 0;
    stats.gathers = stats.gathered_bytes = stats.writebacks = 0;
    stats.region_binds = stats.regions_made = stats.region_refreshes = 0;
    stats.fill_shaders = stats.rewrite_clears = stats.rewrite_uploads = 0;
    stats.rewrite_partial = stats.rewrite_fills = 0;
    stats.pass_barriers = stats.passes_unblocked = 0;
    stats.region_refresh_pages = stats.region_refresh_ns = 0;
    stats.guest_buffer_calls = stats.guest_buffer_ns = 0;
    stats.mirror_binds = stats.mirrors_made = stats.mirror_made_bytes = stats.mirrors_dropped = 0;
    stats.mirrors_evicted = stats.mirror_over_budget = stats.mirror_refresh_pages = 0;
    stats.null_reasons = {};
    stats.phase_ns = {};
    (void)cpu_writes;
}

} // namespace Vulkan
