// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-100: native mode (RYGT_NATIVE=1, ryGT builds); see vk_rasterizer_native.cpp.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/diagnostics.h"
#include "common/types.h"
#include "ryty_gpu/native/same_texel.h"
#include "ryty_gpu/ryty_upscale.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_depth.h"
#include "video_core/amdgpu/regs_texture.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h" // SO-010

namespace AmdGpu {
union Regs;
}

namespace Shader {
struct Info;
}

namespace VideoCore {
class Sampler;
}

namespace Native {
class ImageEntry;
class ImageRegistry;
struct MadeImage;
namespace Rhi {
class Buffer;
class Image;     // FIX-GT7-133
class ImageView; // FIX-GT7-133
class CommandList;
class Device;
class Pipeline;
class PipelineLayout;
class Fence;
} // namespace Rhi
} // namespace Native

namespace Vulkan {

class Rasterizer;
class GraphicsPipeline;
class ComputePipeline;

/// NAT-100: whether native mode was asked for (RYGT_NATIVE=1).
bool NativeModeRequested();

class NativeRenderer {
public:
    explicit NativeRenderer(Rasterizer& rasterizer);
    ~NativeRenderer();

    NativeRenderer(const NativeRenderer&) = delete;
    NativeRenderer& operator=(const NativeRenderer&) = delete;

    /// NAT-161: an indirect draw's arguments in memory (DRAW_INDIRECT, DRAW_INDEX_INDIRECT and
    /// their MULTI and COUNT_MULTI forms), as Rasterizer::DrawIndirect gets them.
    struct IndirectDraw {
        VAddr address = 0;            ///< the first draw's arguments
        u32 stride = 0;               ///< from one draw's arguments to the next
        u32 max_count = 0;            ///< draws (at most, with a count in memory)
        VAddr count_address = 0;      ///< the number of draws in memory, or 0
        u16 vertex_sgpr_offset = 0;   ///< where the CP writes the base vertex (0: nowhere)
        u16 instance_sgpr_offset = 0; ///< where it writes the first instance (0: nowhere)
    };
    /// `indirect` (NAT-161): the draw's counts and offsets are its arguments in memory.
    void Draw(bool is_indexed, u32 index_offset, const IndirectDraw* indirect = nullptr);
    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    /// WRITE_DATA in command order: later work on the GPU reads the new value. False when the
    /// address can't be written by the GPU.
    bool InlineWrite(VAddr address, std::span<const u8> data);
    /// Guest memory is unmapped: textures over it are uploaded again when next used.
    void OnUnmap(VAddr address, u64 size);
    /// FIX-GT7-077: guest memory was mapped where no view of older memory is left: textures over
    /// it are uploaded again when next used.
    void OnNewMapping(VAddr address, u64 size);
    /// NAT-105: the CPU wrote watched memory (a page fault on any thread): the textures there
    /// are uploaded again before their next use, and their pages are no longer watched.
    void OnCpuWrite(VAddr address, u64 size, u64 exact);
    u32 ReadGds(u32 gds_offset);

    /// Ends the command list recorded so far and runs it in the scheduler's current command
    /// buffer (called before anything else is recorded there or it is submitted).
    void Splice();

    /// The picture at a display buffer, for the presenter (none when nothing drew there).
    struct Display {
        vk::Image image;
        vk::ImageView view;
        u32 width;
        u32 height;
        /// GT7-UW: the frame's HUD drawn over black and over white (both null without one).
        vk::ImageView hud_black{};
        vk::ImageView hud_white{};
        /// GT7-UW: the frame's HUD is a race's (its moved pieces apply; not to menus).
        bool hud_race = false;
        /// DLSS-GT7-001: the frame's depth, motion vectors and jitter, when the game's temporal
        /// anti-aliasing pass was seen in it (RYGT_DLSS_TEMPORAL=1).
        std::optional<RytyUpscale::TemporalInputs> temporal;
    };
    std::optional<Display> FindDisplay(const VideoCore::ImageInfo& info, vk::Format view_format);

    /// NAT-116: saves the render targets noted while the rasterizer collected a frame.
    void DumpFrame(const std::filesystem::path& dir);

    /// OPT-010 (RYGT_NATIVE_PREPARE=1): whether game job threads prepare draws.
    static bool PrepareEnabled();
    /// Whether the prepared draws are used (the recorder takes them, the command thread trusts them).
    static bool PreparedOn();
    /// PERF-GT7-010: whether the recorder and the command processor run above normal priority.
    static bool BoostOn();
    /// PERF-GT7-013..016 (RYGT_NATIVE_NEXT=1, A/B mode C): the draw path's next changes under test.
    static bool NextOn();
    /// PERF-GT7-018: whether new lists are recorded by the scheduler's command thread.
    static bool RingOn();
    /// PERF-GT7-008 (B): on the recorder when it runs out of work, the next list is begun.
    void BeginAhead();
    /// OPT-012: on the command thread, whether a job thread prepared the draw at the packet
    /// (with its stages): then the command thread doesn't select its pipeline.
    bool HasPrepared(const void* packet);
    /// OPT-010: on a game job thread whose walk reached a draw (`regs` its state there) and
    /// found its built pipeline, the stage infos holding the draw's user data: the draw's
    /// bindings worked out ahead for the recorder (see PreparedDraw).
    void PrepareJobDraw(const void* packet, u32 epoch, const AmdGpu::Regs& regs,
                        const GraphicsPipeline& pipeline,
                        std::span<const Shader::Info* const> stages, bool is_indexed,
                        u32 index_offset);

private:
    enum class ViewUse : u8 { Sampled, Storage, Attachment, CubeSampled };

    struct Chunk {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 retire_tick = 0;
    };

    /// FIX-208: the mips and layers a draw renders into (mips == 0: all of the picture).
    struct DrawnRange {
        u16 mip = 0;
        u16 mips = 0;
        u16 layer = 0;
        u16 layers = 0;
    };

    /// What a draw renders into (the scheduler's RenderState, in the general layout).
    struct Targets {
        RenderState state{};
        std::array<const Native::ImageEntry*, 8> colors{};
        std::array<DrawnRange, 8> color_ranges{}; ///< FIX-208
        const Native::ImageEntry* depth = nullptr;
    };

    /// The bindings of one draw or dispatch.
    struct DrawBindings {
        Pipeline::DescriptorWrites writes;
        Pipeline::WriteSets write_sets; ///< SHD-002: the set of each write
        boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffers;
        boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> images;
        Shader::PushData push{};
        std::vector<const Native::ImageEntry*> sampled;
        std::vector<const Native::ImageEntry*> stored;
        /// FIX-GT7-080: a compute stage's picture for each image descriptor, in the stage's order
        /// (null, or past the end, when bound empty).
        std::vector<const Native::ImageEntry*> image_entries;
        /// NAT-119: guest memory the draw or dispatch writes through storage buffers.
        boost::container::small_vector<std::pair<VAddr, u64>, 8> gpu_writes;
        /// NAT-120: the dispatch is GT Sport's fill shader: what it fills, with what.
        std::optional<std::array<u32, 2>> fill_pattern;
        VAddr fill_address = 0;
        u64 fill_size = 0;
        bool writes_buffers = false;
        bool failed = false;
        /// DIAG-214 (diagnosis): a 144-byte-stride buffer this work reads (GT7's crowd records).
        VAddr diag_records = 0;
        void Clear();
    };

    /// DIAG-GT7-001 (RYGT_DIAG_GPU_TIME=1): what the GPU ran between two timestamps of a list.
    struct GpuSegment {
        u32 draws = 0, dispatches = 0, uploads = 0;
        u64 upload_bytes = 0;
        u64 max_groups = 0, max_groups_cs = 0; ///< the largest direct dispatch and its program
        u64 max_prims = 0, max_prims_ps = 0;   ///< the largest draw (indices x instances), its PS
    };
    /// DIAG-GT7-001: a list's timestamps (at its start, after each global barrier, at its end).
    struct GpuTiming {
        vk::QueryPool pool{};
        u32 used = 0;
        u64 tick = 0;
        u64 frame = 0;
        double cpu_s = 0.0;     ///< when the recorder began it
        double cpu_end_s = 0.0; ///< when the recorder ended it
        std::vector<GpuSegment> segments;
        u64 uploads_at = 0, upload_bytes_at = 0; ///< registry counters at the segment's start
    };
    /// OPT-014: what a list bound in one set of its last graphics pipeline layout.
    struct SetMemo {
        bool valid = false;
        u64 generation = 0;
        std::vector<u64> signature;
    };
    /// OPT-015: the small upload memory a stage's set last filled in a list.
    struct UploadMemo {
        u64 list = ~u64{0};
        vk::Buffer buffer{};
        u64 offset = 0;
        std::vector<u8> bytes;
    };

    struct LdsReader; ///< PERF-GT7-060

    /// The command list being recorded and its state.
    struct Stream {
        std::unique_ptr<Native::Rhi::CommandList> wrapped_list;
        Native::Rhi::CommandList* list = nullptr;
        /// PERF-GT7-018: into the command thread's ring for a list it records (RingOn).
        CmdBuf cmd{};
        bool rendering = false;
        RenderState render{};
        DynamicState dynamic{};
        vk::Pipeline pipeline{};
        /// SO-010: the shader objects and state the last draw set on this stream (valid while no
        /// pipeline was bound since).
        bool so_known = false;
        GraphicsPipeline::ShaderObjectState so_state{};
        std::array<vk::ShaderEXT, 4> so_shaders{};
        /// OPT-005: the vertex input layout, vertex buffers and index buffer last recorded in
        /// this list (the same ones are not recorded again).
        bool vertex_state_known = false;
        boost::container::static_vector<vk::VertexInputBindingDescription2EXT, 32> input_bindings;
        boost::container::static_vector<vk::VertexInputAttributeDescription2EXT, 32> attributes;
        boost::container::static_vector<vk::Buffer, 32> vertex_buffers;
        boost::container::static_vector<vk::DeviceSize, 32> vertex_offsets;
        boost::container::static_vector<vk::DeviceSize, 32> vertex_sizes;
        boost::container::static_vector<vk::DeviceSize, 32> vertex_strides;
        vk::Buffer index_buffer{};
        u64 index_offset = 0;
        vk::IndexType index_type = vk::IndexType::eUint32;
        std::vector<const Native::ImageEntry*> written; ///< since the last barrier
        /// PERF-GT7-043: `foreign` says something else than a dispatch wrote them.
        struct WrittenFlag {
            bool value = false;
            bool foreign = false;
            WrittenFlag& operator=(bool written) {
                value = foreign = written;
                return *this;
            }
            WrittenFlag& operator|=(bool written) {
                value |= written;
                foreign |= written;
                return *this;
            }
            operator bool() const {
                return value;
            }
        };
        WrittenFlag buffers_written;                    ///< by shaders since the last barrier
        /// PERF-GT7-043: the program of the last dispatch, if it is the last command so far.
        u64 run_program = 0;
        u32 run_commands = 0;
        /// PERF-GT7-057: what the run's same-texel dispatches stored since the last barrier:
        /// (picture, mip << 32 | layer), what they sampled elsewhere (FIX-GT7-087), or that one
        /// stored without a key (FIX-GT7-081).
        Native::SameTexelRun run_texels;
        /// NAT-140: attachments drawn since the last pass or global barrier, and pictures
        /// sampled since the last global barrier: what a new pass has to wait for.
        std::vector<const Native::ImageEntry*> attach_dirty;
        std::vector<const Native::ImageEntry*> read;
        std::vector<Chunk> chunks;
        u64 chunk_used = 0;
        std::vector<vk::DescriptorPool> pools; ///< for pipelines without push descriptors
        std::vector<std::unique_ptr<Native::Rhi::Buffer>> keep; ///< alive until the list ran
        /// PERF-GT7-007: FIX-203 copies this list reads, back to the pool when it ran.
        std::vector<std::unique_ptr<Native::Rhi::Buffer>> pooled;
        u32 commands = 0;
        std::unique_ptr<GpuTiming> gpu_timing; ///< DIAG-GT7-001
        /// OPT-014: the sets and push constants this list bound for its last graphics pipeline
        /// layout (a draw binding the same ones binds nothing again).
        vk::PipelineLayout memo_layout{};
        bool push_known = false;
        Shader::PushData last_push{};
        std::array<SetMemo, 5> set_memos{};
        /// OPT-015: which list the upload memos belong to (moves on when a list begins or ends).
        u64 serial = 0;
        std::array<UploadMemo, 5 * 3> upload_memos{};
        /// PERF-GT7-060: FIX-227 draws of the current pass whose writer pass is recorded and
        /// whose reader pass waits for the run's one barrier (FlushLdsReaders), and their scratch.
        std::vector<LdsReader> lds_readers;
        u64 lds_bytes = 0;
    };

    struct ViewRecord {
        VideoCore::ImageViewInfo info;
        ViewUse use;
        vk::UniqueImageView view;
    };

    Native::MadeImage MakeImage(const VideoCore::ImageInfo& info, bool render_target);

    /// GT7-UW: with a wider-than-16:9 screen (GPU.aspect_ratio), the draws into a display
    /// buffer after the frame's first one (the picture of the 3D scene) are the HUD. They go
    /// to two pictures of the display's layout instead, one cleared black and one white, so the
    /// presenter can stretch the scene over the window and lay the HUD over it at 16:9.
    struct HudLayer {
        Native::ImageEntry* black = nullptr; ///< made once, kept for the run
        Native::ImageEntry* white = nullptr;
        u64 composite_frame = ~u64{0}; ///< the frame the scene was drawn in
        u64 hud_frame = ~u64{0};       ///< the frame HUD draws went to the layer in
        bool clear_pending = false;    ///< the next HUD draw clears both pictures first
        u64 race_frame = ~u64{0};      ///< the frame its HUD showed a race's maps
        u32 scale = 1;                 ///< DLSS-GT7-001: the pictures' size over the display's
    };
    /// GT7-UW: small pictures drawn with a depth buffer, and the frame (a race HUD's track
    /// and course maps are drawn so every frame; menus' HUDs take none).
    std::unordered_map<VAddr, u64> hud_depth_drawn; ///< by address
    std::unordered_map<VAddr, HudLayer> hud_layers;
    static bool HudSplitOn();

    /// DLSS-GT7-001 (temporal DLSS): what GT7's temporal anti-aliasing pass
    /// read in the frame being drawn, for the presenter's DLSS pass (ryty_upscale.h).
    struct TemporalCapture {
        const Native::ImageEntry* depth = nullptr;
        const Native::ImageEntry* motion = nullptr;
        u64 frame = ~u64{0}; ///< the frame the pass was last seen in
        bool reset = true;   ///< the frame before that one had none
        float jitter[2]{};
        float mvec_scale[2]{};
        /// The pass's result is replaced with this colour before the next draw into it or into
        /// a display buffer (after the game's bloom and exposure passes read it).
        const Native::ImageEntry* pending_color = nullptr;
        const Native::ImageEntry* pending_out = nullptr;
        u64 seen = 0;     ///< passes noted
        u64 replaced = 0; ///< results replaced with the un-resolved colour
        u64 refused = 0;  ///< passes whose inputs or constants did not fit
        /// The frame a draw into a display buffer last showed a race's HUD maps; the pass is
        /// only taken in a race (NoteTemporalDraw).
        u64 race_hud_frame = ~u64{0};
        /// The frame a draw last sampled a display picture (the interface reading it back).
        u64 display_read_frame = ~u64{0};
        /// Outside a race the game's pictures are left as it draws them, and the tone-mapped
        /// un-resolved scene DLSS needs is drawn beside them (NoteTemporalDraw).
        bool mirror = false;        ///< the frame being drawn is such a frame
        bool white_inline = false;  ///< else: the HUD's record over white runs in place
        u64 run_start = 0;          ///< the frame the pass came back in after frames without it
        u64 seen_frame = ~u64{0};   ///< the last frame the pass was seen in, taken or not
        bool left = false;          ///< the last pass seen was left to the game (logged)
        u32 left_changes = 0;       ///< logged
        const Native::ImageEntry* result = nullptr; ///< the pass's result picture
        u64 replaced_frame = ~u64{0}; ///< the frame the result was last replaced in place
        u64 through_frame = ~u64{0};  ///< the last menu frame whose interface goes through DLSS
        u64 scene_draw_frame = ~u64{0}; ///< the frame the first draw into a display was seen in
        bool through = false;         ///< logged
        u32 through_changes = 0;      ///< logged
        /// The display as it was before its interface, in such frames (two, by frame parity:
        /// the presenter may still read the last one while the next frame draws).
        std::array<Native::ImageEntry*, 2> ui_scene{};
        u64 ui_scene_frame = ~u64{0}; ///< the frame the last one was taken in
        u32 ui_scene_slot = 0;        ///< which
        VAddr ui_scene_address = 0;   ///< the display buffer it was taken from
        bool hud_batch_open = false;  ///< HUD records wait in the HUD lists
        bool display_draw_now = false; ///< the draw being recorded goes into a display buffer
        u64 batches = 0;              ///< batches closed
        u64 batched_draws = 0;        ///< HUD draws recorded so
        u64 mirror_frame = ~u64{0}; ///< the last such frame
        Native::ImageEntry* kept = nullptr;        ///< the pass's own result, while swapped
        Native::ImageEntry* raw_display = nullptr; ///< the display as DLSS is to see it
        /// DLSS-GT7-003: the un-resolved colour with the game's draws into the result since the
        /// pass (motion blur, glows), while the result itself stays resolved for the game.
        Native::ImageEntry* raw_result = nullptr;
        u64 raw_result_frame = ~u64{0}; ///< the frame it was last started in
        u32 beside_logged = 0;          ///< logged
        /// The result picture, while it holds the un-resolved colour.
        const Native::ImageEntry* swapped_out = nullptr;
        u64 raw_frame = ~u64{0}; ///< the frame raw_display was last drawn in
        VAddr raw_address = 0;   ///< the display buffer it stands for
        u64 mirrored = 0;        ///< frames drawn so
        std::vector<Native::ImageEntry*> retired; ///< pictures of another size, kept alive
        bool race = false;    ///< as of the last pass seen
        u32 race_changes = 0; ///< logged
        /// The pass's result was replaced but its size is not the display's (a picture cropped
        /// or scaled before it is shown): DLSS could not take the inputs, so the pass is left
        /// to the game for the rest of the run (FindDisplay).
        bool size_refused = false;
        u32 size_mismatches = 0; ///< frames in a row so, before it is refused
    };
    TemporalCapture temporal;
    /// Called at the end of a draw: notes the pass's inputs and replaces its result.
    void NoteTemporalDraw(const GraphicsPipeline& pipeline, const Targets& targets,
                          const DrawBindings& b);
    void ApplyTemporalReplace();
    /// DLSS-GT7-001, frames left as the game draws them: the pass's result is put aside and
    /// holds the un-resolved colour until the tone mapping has been drawn for DLSS; then back.
    bool TemporalSwapIn();
    void TemporalSwapOut();
    /// DLSS-GT7-003: the result stays resolved until the tone mapping (default; off with
    /// RYGT_DLSS_TEMPORAL_BLOOM=0 or 1).
    static bool TemporalKeepsResolved();
    void CopyPicture(const Native::ImageEntry& from, const Native::ImageEntry& to);
    /// A picture of the prototype's own like another, made again when the size changes.
    /// size: the picture's size, when not like's.
    Native::ImageEntry* TemporalPicture(Native::ImageEntry*& slot, const Native::ImageEntry& like,
                                        vk::Extent2D size = {});
    /// DLSS-GT7-001: the HUD split for this frame: a wide screen's, or a frame with the pass.
    bool HudSplitNow() const;
    /// DLSS-GT7-001: how many times the display's size the prototype's HUD pictures are.
    u32 HudScale() const;
    /// size: the picture's size before scale, when not like's.
    Native::ImageEntry* MakeHudPicture(const Native::ImageEntry& like, u32 scale = 1,
                                       vk::Extent2D size = {});
    vk::ImageView View(const Native::ImageEntry& entry, const VideoCore::ImageViewInfo& info,
                       ViewUse use);
    /// explicit_lod (FIX-GT7-182): the shader reads this S# at a level it chose, which must not
    /// get the DLSS texture bias (DLSS-GT7-002).
    vk::Sampler SamplerFor(const AmdGpu::Sampler& sharp, AmdGpu::BorderColorBuffer border,
                           bool is_depth, bool explicit_lod);
    /// The sampler for an S# as SamplerFor adjusted it (FIX-GT7-002's lookup and pool), with the
    /// DLSS texture bias SamplerFor chose for it.
    vk::Sampler MakeSampler(const AmdGpu::Sampler& sharp, AmdGpu::BorderColorBuffer border,
                            bool is_depth, float resolution_bias);

    Stream& Begin();
    /// PERF-GT7-018: a list for the stream, with its commands going where RingOn says.
    void AcquireList(Stream& s);
    /// Where commands for `list` (a stream's) are recorded.
    CmdBuf CmdOf(Native::Rhi::CommandList& list) const;
    void EndRendering();
    void BeginRendering(const RenderState& state);
    void GlobalBarrier();
    /// NAT-140: attachment writes before it finish before attachment access after it (a pass
    /// drawing into what an earlier pass drew), without waiting for anything else.
    void PassBarrier();
    /// PERF-109: `bound` are the pictures the work about to be recorded uses.
    void PrepareImages(std::span<const Native::ImageEntry* const> bound = {});
    void PrepareImages(const DrawBindings& b);
    void ChunkFlushCheck();

    /// Upload memory of the stream's list: (buffer, offset, CPU pointer).
    std::tuple<vk::Buffer, u64, u8*> Allocate(u64 size, u64 align);
    void TrimChunks();
    Chunk TakeChunk(u64 size);
    vk::DescriptorPool TakeDescriptorPool();

    /// NAT-114: a range shaders write that no single imported view holds lives in a buffer of
    /// its own on the GPU (a region), filled from guest memory when made; every later binding
    /// over it uses the region, and CPU writes to its pages are copied in before the next use.
    struct Region {
        VAddr begin = 0;
        VAddr end = 0;
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        std::vector<VAddr> dirty_pages; ///< written by the CPU since the last use
        /// FIX-194: pages the CPU wrote, with the frame of the last write, whose bytes in
        /// guest memory are what the region holds except for the CPU's new writes (what
        /// FIX-173's compare needs). Pages not here are copied in whole at the next write.
        std::unordered_map<VAddr, u64> cpu_pages;
        /// FIX-GT7-033: of those, the pages the GPU wrote since their last copy.
        std::unordered_set<VAddr> gpu_pages;
    };
    std::map<VAddr, Region> regions; ///< by begin; guarded by mutex
    std::mutex region_dirty_mutex;   ///< pending_region_writes (CPU write faults: any thread)
    std::vector<std::pair<VAddr, VAddr>> pending_region_writes;
    std::atomic<bool> has_pending_region_writes{false};
    /// FIX-173: region pages as guest memory held them before the CPU's first write since they
    /// were last copied in (taken at the write fault; guarded by region_dirty_mutex).
    std::unordered_map<VAddr, std::unique_ptr<std::array<u8, 4096>>> page_snapshots;
    std::atomic<u32> logged_partial{};
    void SortRegionWrites();
    /// PERF-GT7-052: small textures in per-frame memory, checked each frame (not watched).
    bool PolledFrameTexture(VAddr address, u64 size);
    std::mutex polled_mutex;
    std::vector<std::pair<VAddr, u64>> polled_textures;
    Region* RegionOver(VAddr address, u64 size);
    /// FIX-GT7-184: no committed page, no region and no picture of the GPU's in the texture's
    /// guest range.
    bool NothingToRead(const VideoCore::ImageInfo& info);
    Region* MakeRegion(VAddr address, u64 size, bool write);
    void RefreshRegion(Region& region);
    /// FIX-194: the GPU wrote [address, address + size) (in GPU order, just recorded): where a
    /// region holds it over pages the CPU writes too, guest memory gets the bytes as well.
    /// FIX-203: the region bytes a range a draw or dispatch writes through a storage buffer held
    /// before it (whole pages), for NoteRegionWrite to write back only what the work changed.
    struct WrittenBefore {
        VAddr address = 0; ///< the written range
        u64 size = 0;
        VAddr begin = 0; ///< the pages copied
        VAddr end = 0;
        std::unique_ptr<Native::Rhi::Buffer> copy;
        bool pooled = false; ///< PERF-GT7-007: goes back to the pool
    };
    std::vector<WrittenBefore> written_befores;
    std::unique_ptr<Native::Rhi::PipelineLayout> writeback_layout;
    std::unique_ptr<Native::Rhi::Pipeline> writeback_pipeline;
    void SnapshotBeforeWrite(VAddr address, u64 size);
    void DropWrittenBefores();
    void NoteRegionWrite(VAddr address, u64 size, const WrittenBefore* before = nullptr,
                         bool shader_write = false);
    /// FIX-203: writes back the words of a run that differ from before the work.
    bool WriteBackChangedWords(Stream& s, const Region& region, const WrittenBefore& before,
                               VAddr run_from, VAddr run_to);
    /// FIX-198: whether a draw whose pipeline is still building must wait for it: it draws
    /// into a picture no recent frame drew into.
    bool DrawMustNotSkip(const AmdGpu::Regs& regs);
    /// DIAG-GT7-008 (RYGT_DIAG=1; RYGT_DIAG_ONE_OFF_REDRAW=0 off): what became of the pictures
    /// FIX-198 made draws wait for: drawn again in a later frame (how soon), sampled in a later
    /// frame before that, or neither in 300 frames; by why they waited (never drawn, or not for
    /// 64 frames), with the draws that waited for each. Whether a new scene's first frame may skip
    /// them as later frames do.
    static bool OneOffDiagOn();
    static bool OneOffDiagAll(); ///< RYGT_DIAG_ONE_OFF_REDRAW=1: draws with their pipeline too
    /// After a FIX-198 wait (or a draw PERF-GT7-023 skipped instead): from one_off_hits.
    void WatchOneOffs(const AmdGpu::Regs& regs, bool skipped);
    void CheckOneOffs();     ///< each frame
    std::string OneOffText(); ///< the window's counts, then cleared
    /// What the draw that first waited for a picture says of it: is it a colour target (and
    /// which) or the depth buffer, its pitch, height, slices, samples, format and tiling.
    struct OneOffHit {
        VAddr address = 0;
        bool never = false;
        bool depth = false;
        u32 cb = 0;
        u32 pitch = 0, height = 0, slices = 0, samples = 0;
        u32 format = 0, number_format = 0, tile = 0;
    };
    struct OneOffWatch {
        OneOffHit hit;
        u64 frame = 0;     ///< the frame of the first wait
        u64 ps_hash = 0;   ///< that draw's pixel shader
        u32 draws = 0;     ///< draws into it in the frame of the first wait (with a wait)
        u32 waits = 0;
        bool skipped = false; ///< PERF-GT7-023: every draw into it so far skipped, none waited
    };
    std::vector<OneOffHit> one_off_hits; ///< of the last DrawMustNotSkip
    /// FIX-GT7-135: the last DrawMustNotSkip waits for a never-drawn level of a texture.
    bool one_off_texture_level = false;
    /// FIX-GT7-153: the last DrawMustNotSkip's depth target is a circuit's shadow map (see
    /// kept_shadow_map.h): its draw waits, without a deadline when `first` (the map not drawn
    /// within the last frames) or out of play.
    bool one_off_kept_shadow = false;
    bool one_off_kept_shadow_first = false;
    /// FIX-GT7-135: the texture (address, level >= 1) whose level `target` is; {0, -1}: none.
    std::pair<VAddr, s32> TextureLevelOf(const VideoCore::ImageInfo& target);
    /// FIX-GT7-102: time the current frame's draws waited for one-off pipelines; past the
    /// budget (RYGT_ONE_OFF_BUDGET_MS, 250 by default, 0 no limit) the frame's other one-off
    /// draws are skipped as other draws are, and still watched.
    bool OneOffBudgetLeft();
    /// FIX-GT7-109: the deadline of a one-off draw's wait (max: no bound).
    std::chrono::steady_clock::time_point OneOffDeadline();
    /// FIX-GT7-121: a picture of the last DrawMustNotSkip's hits is of a kind (or at a place)
    /// already seen sampled before its redraw (PERF-GT7-023): its draw waits whatever the budget.
    bool OneOffKnownHarmful() const;
    std::chrono::steady_clock::time_point one_off_rate_since{};
    u64 one_off_rate_frames = 0;
    bool one_off_interactive = false;
    u64 one_off_budget_frame = ~u64{0};
    u64 one_off_wait_ns_frame = 0;
    std::unordered_map<VAddr, OneOffWatch> one_off_watch;
    struct {
        /// [never drawn, not lately][redrawn after 1, 2-4, 5-16, 17-64, 65-300 frames; sampled
        /// in a later frame first; neither in 300 frames]: pictures, and the draws that waited.
        std::array<std::array<u64, 7>, 2> pictures{};
        std::array<std::array<u64, 7>, 2> waits{};
    } one_off_stats;
    /// PERF-GT7-023: FIX-198's waits learned by kind of picture (colour or depth, pitch, height,
    /// slices, samples, format, tiling). A kind whose pictures were redrawn after a wait, without
    /// being sampled before, at least 8 times, no longer waits: its draws are skipped while their
    /// pipeline builds, as other draws are. One picture of the kind sampled before its redraw makes
    /// it wait again. Kept from run to run (<cache folder>/<serial>.one_off_kinds.txt).
    /// RYGT_ONE_OFF_LEARN=0: every one-off picture waits, as before.
    struct OneOffKind {
        bool depth = false;
        u32 pitch = 0, height = 0, slices = 0, samples = 0;
        u32 format = 0, number_format = 0, tile = 0;
        bool operator==(const OneOffKind&) const = default;
    };
    struct OneOffKindHash {
        size_t operator()(const OneOffKind& kind) const noexcept;
    };
    struct OneOffKindStats {
        u32 safe = 0;    ///< pictures redrawn after a wait, not sampled before
        u32 harmful = 0; ///< pictures sampled in a later frame before any redraw
    };
    /// A kind of picture at one address (GT7 keeps its per-frame targets at fixed addresses): such
    /// a place no longer waits after 4 redraws, none sampled first, even when its kind was sampled
    /// first elsewhere (the 256x256 depth pool at 0x100a0e8000 against the probes of that size).
    /// A family, the pictures of one format at one address whatever their size (a place with no
    /// pitch, height, slices or samples), after 8 (0x100a0b0000 holds pictures of 21 sizes).
    struct OneOffPlace {
        VAddr address = 0;
        OneOffKind kind{};
        bool operator==(const OneOffPlace&) const = default;
    };
    struct OneOffPlaceHash {
        size_t operator()(const OneOffPlace& place) const noexcept;
    };
    static OneOffKind KindOf(const OneOffHit& hit);
    static bool OneOffLearnOn();
    static bool OneOffTrackOn(); ///< PERF-GT7-023 or DIAG-GT7-008
    bool OneOffExempt(const OneOffHit& hit) const;
    void LoadOneOffKinds();
    void SaveOneOffKinds(bool wait); ///< when changed; wait: on this thread
    std::unordered_map<OneOffKind, OneOffKindStats, OneOffKindHash> one_off_kinds;
    std::unordered_map<OneOffPlace, OneOffKindStats, OneOffPlaceHash> one_off_places;
    /// DIAG-GT7-010 (with RYGT_DIAG): the pages CPU write faults hit, by frame, to find the ones
    /// the game writes every frame (watched again and faulted again each frame: 4-7 ms a frame of
    /// faults on the game's threads in a heavy scene) and what they hold.
    struct FaultPage {
        u64 last_frame = 0;
        u32 frames = 0; ///< bit n: written n frames before last_frame
        u32 faults = 0;
    };
    std::mutex fault_pages_mutex;
    std::unordered_map<VAddr, FaultPage> fault_pages;
    void NoteFaultPages(VAddr begin, VAddr end);
    /// PERF-GT7-030: after a write fault in a region page (its copy kept, FIX-173), the other
    /// still watched pages of its 64 KB chunk and region are opened too, when the chunk faulted
    /// within the last 8 frames: GT7 writes such chunks whole, a page at a time, every ~4 frames.
    void FaultAhead(VAddr begin, VAddr end);
    std::unordered_map<VAddr, u64> fault_chunks; ///< 64 KB chunk: frame of its last fault
    /// FIX-GT7-006 (under region_dirty_mutex): the regions' cpu_pages as FaultAhead may read
    /// them from any thread (page: frame the CPU last wrote it), and the pages it opened that no
    /// refresh has copied in yet.
    std::unordered_map<VAddr, u64> cpu_recent_pages;
    std::unordered_set<VAddr> ahead_pages;
    std::atomic<u64> fault_ahead_faults{};
    std::atomic<u64> fault_ahead_pages{};
    std::atomic<u64> fault_ahead_pictures{}; ///< FIX-GT7-018: pages left shut for a picture
    /// FIX-GT7-020 (under region_dirty_mutex): region pages a picture's download writes, copied
    /// in whole at their next refresh; of them (recorder), those that had a copy from before,
    /// and those PERF-GT7-030 had opened.
    std::unordered_set<VAddr> downloaded_pages;
    u64 downloaded_with_copy = 0;
    u64 downloaded_ahead = 0;
    u64 draws_at_frame = 0; ///< DIAG-GT7-011: stats.draws at the last frame's end
    std::string FaultPagesText(); ///< the window's pages, then cleared
    bool one_off_kinds_changed = false;
    /// DIAG-176 (diagnosis, RYGT_DIAG_WRITEBACK_RACE=1): FIX-194's write-backs recorded and not
    /// yet run by the GPU (guest range, GPU tick), and CPU writes that met one.
    struct PendingWriteback {
        VAddr begin;
        VAddr end;
        u64 tick;
        u64 frame;
    };
    std::mutex writeback_race_mutex;
    std::deque<PendingWriteback> pending_writebacks;
    std::atomic<u64> writeback_race_hits{0};
    std::atomic<u64> writeback_race_faults{0};
    static bool WritebackRaceDiag();
    static bool RegionWritebackOn();
    /// FIX-194: how long (frames) a page the CPU wrote counts as one it writes.
    static constexpr u64 RegionRecentFrames = 600;
    /// NAT-150: copies in video memory of guest memory the GPU reads as vertices and indices
    /// and the CPU leaves alone (meshes), instead of reading it over PCIe at every draw. A range
    /// bound in a second frame is copied at the start of the next list; CPU writes to its pages
    /// are copied in before its next use; 64 KiB granules the CPU writes in two frames (rings of
    /// per-frame data) are read in place again for a while. Mirrors never overlap regions or
    /// each other. RYGT_NATIVE_MIRRORS=0 turns them off; RYGT_NATIVE_MIRROR_MB is the budget.
    struct Mirror {
        VAddr begin = 0;
        VAddr end = 0;
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        std::vector<VAddr> dirty_pages; ///< written by the CPU (or written back) since copied
        u64 used_frame = 0;
        std::vector<u64> hashes; ///< FIX-176: guest memory per 64 KB granule when last verified
    };
    static constexpr u64 MirrorGranule = u64{64} * 1024; ///< FIX-176
    VAddr mirror_verify_cursor = 0;            ///< FIX-176
    static u64 MirrorVerifyBudget();
    static bool MirrorKind(char kind); ///< FIX-177
    /// FIX-179 (diagnosis): a vertex range bound from a mirror, copied out with guest memory.
    struct MirrorCheck {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 size;
        VAddr base;
        u64 frame;
        VAddr mirror_begin;
        u64 mirror_size;
        u64 mirror_used;
        u64 dirty;
    };
    std::vector<MirrorCheck> mirror_checks;
    u64 mirror_check_counter = 0;
    void CheckMirrorBind(Mirror& mirror, VAddr base, u64 size);
    /// FIX-181 (diagnosis): how far a draw reads past its vertex V#s and their mirrors.
    void CheckVertexReach(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs,
                          bool is_indexed, u32 index_offset);
    u64 reach_check_counter = 0;
    void VerifyMirrors();
    struct MirrorCandidate {
        VAddr end = 0;
        u64 frame = 0; ///< when last considered
    };
    std::map<VAddr, Mirror> mirrors; ///< by begin; guarded by mutex
    u64 mirror_bytes = 0;
    std::unordered_map<VAddr, MirrorCandidate> mirror_candidates; ///< by page of the range start
    std::vector<std::pair<VAddr, VAddr>> pending_mirrors;          ///< made at a list's start
    std::unordered_map<u64, u64> mirror_writes;    ///< granule -> frame the CPU wrote a mirror there
    std::unordered_map<u64, u64> dynamic_granules; ///< granule -> frame it was found CPU-written
    /// The first mirror that may overlap memory from `begin` on.
    std::map<VAddr, Mirror>::iterator FirstMirror(VAddr begin);
    /// The mirror holding all of the range, if any.
    Mirror* MirrorHolding(VAddr address, u64 size);
    bool IsDynamic(VAddr begin, VAddr end);
    std::optional<std::pair<vk::Buffer, u64>> MirrorBuffer(VAddr address, u64 size);
    void RefreshMirror(Mirror& mirror);
    void MakePendingMirrors();
    void DropMirrors(VAddr address, u64 size);
    /// NAT-146, FIX-GT7-086: regions and mirrors over the range go (with `mutex` held); how many.
    u32 DropRegionsOver(VAddr address, u64 size);
    /// FIX-GT7-122: a range starting in GT7's per-frame memory and ending past its known end
    /// moves the end there (at most 64 MB past the listed end, through mapped memory).
    void NotePerFrameRange(VAddr address, u64 size);
    /// FIX-GT7-122: a region found inside GT7's per-frame memory: the regions there go before the
    /// next draw or dispatch binds anything (DropDuePerFrameRegions), not while one binds.
    void NotePerFrameRegion(const Region& region);
    /// FIX-GT7-122: drops the regions inside the per-frame memory when NotePerFrameRegion asked
    /// for it (with `mutex` held, before a draw's or dispatch's first binding).
    void DropDuePerFrameRegions();
    bool per_frame_drop_due = false; ///< FIX-GT7-122
    u64 per_frame_drop_frame = 0;    ///< FIX-GT7-122: the frame of the last drop
    /// NAT-150: what the last GuestBuffer bound (for the NAT-142 counters).
    enum class Bound : u8 { InPlace, Region, Mirror };
    Bound last_bound = Bound::InPlace;
    /// OPT-003: what a guest range resolved to the last time (a region, a mirror, or imported
    /// guest memory read in place), valid while no region or mirror was made or dropped and no
    /// memory unmapped since (buffer_generation); a region or mirror with pages to copy in takes
    /// the full lookup.
    struct BufferMemo {
        VAddr address = 0;
        u64 size = 0;
        u32 flags = 0;
        u64 generation = ~u64{0};
        u64 hard_generation = 0; ///< PERF-GT7-047
        u64 log_seq = 0;
        vk::Buffer buffer{};
        u64 offset = 0;
        Region* region = nullptr;
        Mirror* mirror = nullptr;
        Bound bound = Bound::InPlace;
    };
    std::array<BufferMemo, 16384> buffer_memos{}; ///< OPT-006: 2048 thrashed
    u64 buffer_generation = 0;
    u64 buffer_hard_generation = 0; ///< PERF-GT7-047: what the change log doesn't tell
    /// OPT-003: what the last lookup can be remembered as (set by GuestBufferLookup).
    enum class MemoKind : u8 { None, Region, Mirror, InPlace };
    MemoKind memo_kind = MemoKind::None;
    Region* memo_region = nullptr;
    Mirror* memo_mirror = nullptr;
    /// OPT-004: the mirror the last GuestBuffer bound from (none when it bound anything else).
    Mirror* last_mirror = nullptr;
    /// FIX-170 (diagnosis): RYGT_NATIVE_SNAPSHOT_BELOW, 0 when off.
    static u64 SnapshotBelow();
    /// A copy of the range as it is now (NAT-106 zeros where the CPU can't read).
    std::optional<std::pair<vk::Buffer, u64>> SnapshotRange(VAddr address, u64 size);
    std::optional<std::pair<vk::Buffer, u64>> GuestBufferLookup(VAddr address, u64 size, bool write,
                                                                bool mirror);
    /// OPT-002: what a T# resolved to the last time (picture and views), by the T# and how the
    /// shader reads it, valid while the registry made or dropped no picture and no memory was
    /// unmapped since.
    /// PERF-GT7-012: up to 4 views again. PERF-GT7-009's 16 (for storage images indexed by mip)
    /// found GT7's compute slots at 1.3 views and doubled the table (3 MB), whose first touch
    /// at each texture slot was then the recorder's costliest line of BindTextures.
    static constexpr u32 TextureMemoViews = 4;
    template <u32 Views>
    struct TextureMemoOf {
        std::array<u64, 4> sharp{};
        u32 flags = 0;
        u64 generation = ~u64{0};
        Native::ImageEntry* entry = nullptr;
        std::array<vk::ImageView, Views> views{};
        u64 registry_generation = 0; ///< PERF-GT7-047
        u64 unmap_generation = 0;
        VAddr range_begin = 0;
        VAddr range_end = 0;
    };
    using TextureMemo = TextureMemoOf<TextureMemoViews>;
    using JobTextureMemo = TextureMemoOf<4>; ///< OPT-010's, per job thread (thread-local)
    std::array<TextureMemo, 16384> texture_memos{}; ///< OPT-006
    size_t TextureMemoSlot(const std::array<u64, 4>& raw_sharp, u32 memo_flags) const;
    std::atomic<u64> unmap_generation{0}; ///< (OPT-010: read by job threads)
    /// NAT-107: a guest range gathered on the GPU from the imported views holding it, into a
    /// buffer of the list (none when part of it is not imported).
    std::optional<std::pair<vk::Buffer, u64>> GatherOnGpu(VAddr address, u64 size);
    /// A guest range as a buffer the GPU can bind: imported guest memory read and written in
    /// place, else (reads only) a copy in upload memory.
    /// `mirror`: read-only data worth a copy in video memory (NAT-150: vertices, indices).
    std::optional<std::pair<vk::Buffer, u64>> GuestBuffer(VAddr address, u64 size, bool write,
                                                          bool mirror = false);

    bool ResolveTargets(const AmdGpu::Regs& regs, const GraphicsPipeline& pipeline, Targets& out);
    /// DIAG-GT7-001: GPU timestamps of the main stream's lists.
    static bool GpuTimeOn();
    void GpuTimeBegin(Stream& s);
    void GpuTimeMark(Stream& s);
    void GpuTimeEnd(Stream& s, u64 tick);
    void GpuTimeCollect();
    GpuSegment* GpuSeg() {
        auto* t = stream.gpu_timing.get();
        return t && !t->segments.empty() ? &t->segments.back() : nullptr;
    }
    static constexpr u32 GpuTimeQueries = 1024;
    std::deque<std::unique_ptr<GpuTiming>> gpu_timing_pending;
    std::vector<vk::QueryPool> gpu_timing_pools;
    u64 gpu_timing_prev_end = 0;
    double gpu_timing_prev_cpu_end = 0.0;
    std::chrono::steady_clock::time_point gpu_timing_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point gpu_timing_report = std::chrono::steady_clock::now();
    struct GpuTimeStats {
        u64 lists = 0, slow = 0;
        double busy_ms = 0.0, max_list_ms = 0.0, max_gap_ms = 0.0;
    } gpu_time_stats;
    /// OPT-013..015 (RYGT_NATIVE_REUSE=0 turns them off): work the previous draws of the list
    /// already did is not done again.
    static bool ReuseOn();
    /// DIAG-GT7-003 (RYGT_NATIVE_AB=1): the change under test alternates with the defaults
    /// every 10 s report window (A the defaults without PERF-GT7-018, B with it), each window's
    /// cost logged.
    static bool AbOn();
    u32 AbMode() const;
    /// DIAG-GT7-004: the window's dispatch costs by part and piece.
    std::string DispatchPartsText() const;
    /// PERF-GT7-010: the calling thread (the recorder) at the priority BoostOn asks for.
    void ApplyRecorderPriority();
    bool recorder_boosted = false;
    bool command_boosted = false; ///< PERF-GT7-018: the command thread's priority
    u64 ab_window = 0;
    u64 ab_frame = 0;
    /// OPT-013: the targets the previous draw resolved, by the registers that name them, while
    /// the registry made or dropped no picture and no memory was unmapped. Clears, rewrites,
    /// depth twins and last uses are still done at every draw.
    struct TargetsMemo {
        bool valid = false;
        u64 generation = 0;
        u32 mrt_mask = 0;
        u32 color_mode = 0;
        u32 target_mask = 0;
        u32 num_colors = 0;
        std::array<AmdGpu::ColorBuffer, 8> color_buffers{};
        std::array<AmdGpu::CbDbExtent, 8> cb_extents{};
        bool depth_used = false;
        u32 depth_flags = 0; ///< depth and stencil enable, depth write enable
        AmdGpu::DepthBuffer depth_buffer{};
        AmdGpu::DepthView depth_view{};
        VAddr htile = 0;
        AmdGpu::CbDbExtent db_extent{};
        std::array<VAddr, 8> cmasks{};
        bool htile_meta = false; ///< the depth image has HTILE (FIX-156)
        Targets targets{};
    } targets_memo;
    bool TargetsFromMemo(const AmdGpu::Regs& regs, const GraphicsPipeline& pipeline, Targets& out);
    /// OPT-014: the list's binding memos are no longer what is bound.
    static void ForgetBindings(Stream& s);
    /// OPT-015: upload memory of the list holding `data` (an earlier draw's with the same bytes).
    std::pair<vk::Buffer, u64> UploadBytes(u32 slot, const void* data, u32 size, u64 align);
    /// Everything OPT-014 depends on: pictures made or dropped, memory unmapped, regions or
    /// mirrors made or dropped.
    u64 ReuseGeneration() const;
    void BindStages(std::span<const Shader::Info* const> stages, const AmdGpu::Regs& regs,
                    bool compute, DrawBindings& b);
    /// FIX-GT7-090: BindStages again when a region made while binding took in one an earlier
    /// binding of the same draw or dispatch was given (its writes would go to the old buffer).
    void BindStagesStable(std::span<const Shader::Info* const> stages, const AmdGpu::Regs& regs,
                          bool compute, DrawBindings& b);
    u64 regions_taken_in = 0; ///< FIX-GT7-090: regions made that took in others
    u64 stage_rebinds = 0;    ///< FIX-GT7-090: BindStages done again for that
    void BindBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs, bool compute,
                     Shader::Backend::Bindings& binding, DrawBindings& b);
    void BindTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                      Shader::Backend::Bindings& binding, DrawBindings& b);
    void PushBindings(const Pipeline& pipeline, vk::PipelineBindPoint bind_point,
                      DrawBindings& b);
    /// A draw's vertex and index buffers: looked up before its pass starts (a lookup may gather
    /// on the GPU, outside rendering) and bound inside it.
    struct VertexIndexState {
        boost::container::static_vector<vk::VertexInputAttributeDescription2EXT, 32> attributes;
        boost::container::static_vector<vk::VertexInputBindingDescription2EXT, 32> input_bindings;
        boost::container::static_vector<vk::Buffer, 32> buffers;
        boost::container::static_vector<vk::DeviceSize, 32> offsets;
        boost::container::static_vector<vk::DeviceSize, 32> sizes;
        boost::container::static_vector<vk::DeviceSize, 32> strides;
        vk::Buffer index_buffer{};
        u64 index_offset = 0;
        vk::IndexType index_type = vk::IndexType::eUint32;
        u32 index_count = 0;
    };
    /// PERF-GT7-060: a FIX-227 draw's reader pass, recorded after its run's barrier: what it
    /// binds and draws.
    struct LdsReader {
        const GraphicsPipeline* pipeline = nullptr;
        DynamicState dynamic{};
        VertexIndexState vertices{};
        DrawBindings bindings; ///< writes, write_sets, buffers, images and push only
        /// Each write's first buffer and image info in `bindings` (~0u: none). The infos are
        /// held in the reader, which moves: the writes point at them only when recorded.
        std::vector<std::pair<u32, u32>> infos;
        u64 address = 0;       ///< its scratch
        bool indexed = false;
        u32 count = 0, instances = 0, vertex_offset = 0, first_instance = 0;
        u64 ps = 0, prims = 0; ///< DIAG-GT7-001: its pixel shader and size, for the segment
    };
    /// PERF-GT7-060 (on; RYGT_NATIVE_LDS_BATCH=0 off): the FIX-227 draws of a
    /// run share one barrier.
    static bool LdsBatchOn();
    /// PERF-GT7-060: copies what the reader pass binds; false when a descriptor write points
    /// outside the draw's own infos (the draw then runs alone, as before).
    static bool CaptureLdsReader(LdsReader& reader, const GraphicsPipeline& pipeline,
                                 const DrawBindings& b, const VertexIndexState& vertices);
    /// PERF-GT7-060: the stream's waiting reader passes, after one full barrier, in their pass
    /// (reason: 0 another draw, 1 the pass ends, 2 the run's limit).
    void FlushLdsReaders(u32 reason);
    /// SO-018: binds a draw's pipeline, or its shader objects and the state a pipeline would fix
    /// (SO-001; SO-010: only what changed since the stream's last shader object draw).
    void BindGraphics(Stream& s, const GraphicsPipeline& pipeline);
    /// PERF-GT7-060: FIX-227 runs by length, every 10 s.
    void ReportLdsRuns();
    void ResolveVertexBuffers(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs,
                              VertexIndexState& v);
    bool ResolveIndices(const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                        bool quad_triangles, VertexIndexState& v);
    void RecordVertexIndex(const VertexIndexState& v);
    /// NAT-161: the buffers an indirect draw reads its arguments (and count) from.
    struct IndirectBuffers {
        vk::Buffer args{};
        u64 args_offset = 0;
        vk::Buffer count{}; ///< none: max_count draws
        u64 count_offset = 0;
        u32 max_count = 0; ///< the draw's max_count, less when its arguments end sooner
    };
    /// NAT-161: none when the device can't draw it as the GPU would, or its arguments are not
    /// where the GPU can read them (each case logged once).
    std::optional<IndirectBuffers> ResolveIndirect(const IndirectDraw& indirect, bool is_indexed);
    /// NAT-161: the draw commands, counts and offsets read from the arguments by the GPU.
    void RecordIndirect(const IndirectDraw& indirect, const IndirectBuffers& buffers,
                        bool is_indexed);
    bool NeedsBarrierFor(const DrawBindings& b) const;
    // PERF-GT7-043
    bool RunNeedsBarrier(const Stream& s, const DrawBindings& b, u64 program, bool recorded,
                         std::optional<u64> texel = std::nullopt) const; // PERF-GT7-057: texel
    void NoteWrites(const Targets* targets, const DrawBindings& b);
    /// NAT-105: CPU writes to the pages of a range fault (once) from now on.
    void WatchPages(VAddr address, u64 size);
    /// NAT-105, FIX-153: the watched pages of [begin, end) (page aligned) are watched no more.
    void UnwatchPages(VAddr begin, VAddr end);
    /// NAT-112: small render targets drawn into are written back to guest memory (GT Sport's
    /// CPU reads its brightness chain there for auto exposure).
    void FlushWritebacks();
    /// FIX-156 (diagnosis): reads back the pictures at RYGT_NATIVE_WATCH_TEXTURE now and then.
    void ProbeWatchedPictures();
    /// FIX-157 (diagnosis): a T# at a watched address and the picture it resolved to.
    void WatchBind(const Native::ImageEntry& entry, const AmdGpu::Image& tsharp, bool storage,
                   bool memo, u64 program);
    /// DIAG-208 (diagnosis): whether the work on this picture is logged in this frame.
    bool SeqTraced(const Native::ImageEntry* entry) const;
    void SeqBind(const Native::ImageEntry& entry, const AmdGpu::Image& tsharp, bool storage,
                 u64 program);
    struct Probe {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 size;
        u64 frame;
        u64 version;
        u32 width, height;
        bool target;
        VAddr address;
        bool packed_r11g11b10 = false; // FIX-171
        u32 mips = 0, layers = 0;      // FIX-171: an untiled copy of every mip and layer
    };
    std::vector<Probe> probes;
    /// FIX-175 (diagnosis): rectangles of presented pictures on their way back from the GPU.
    std::vector<Probe> display_probes;
    /// DIAG-226: copies of watched colour targets taken at each new pass of a dumped frame.
    struct PassSnap {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        std::string name;
    };
    std::vector<PassSnap> pass_snaps;
    std::array<const Native::ImageEntry*, 3> last_snap_key{};
    u32 pass_snap_index = 0;
    void SnapPass(const Targets& targets);
    /// FIX-227: device address of `bytes` of the hull shader LDS emulation scratch ring.
    u64 LdsScratch(u64 bytes);
    std::unique_ptr<Native::Rhi::Buffer> lds_scratch;
    u64 lds_scratch_used = 0;
    u64 lds_snap_address = 0; ///< DIAG-241
    u64 lds_snap_bytes = 0;
    void NoteStaleFlats(const Shader::Info& stage);
    void CheckStaleFlats();
    void SnapProgramMips(std::span<const Shader::Info* const> stages, const DrawBindings& b);
    /// DIAG-170 (diagnosis): blocks of a watched draw's target read back right after it.
    struct DrawProbe {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 frame;
        u32 draw;          ///< the watched draw's number in its frame
        VAddr sampled;     ///< what it sampled (DIAG-169's match)
        u32 sampled_width, sampled_height;
        VAddr target;
        u32 width, height, bits;
        std::array<u32, 2> y; ///< the rows probed (at x RYGT_PROBE_AT's)
        std::string bound;    ///< the draw's images (views, empty ones)
        std::string recorded; ///< hashes of its buffers' bytes when it was recorded
        u32 extra = 0;        ///< bytes after the blocks: its buffers, then the texture's
        std::array<u32, 2> lengths{}; ///< of its buffers' copies
        u32 texture_bytes = 0;
        std::array<u32, 2> vertex_lengths{}; ///< of its vertex buffers' copies
        bool depth = false; ///< 16x16 floats of its depth buffer at the end
    };
    struct ProbeVertex {
        VAddr base;
        u64 size;
        vk::Buffer buffer;
        u64 offset;
        u64 hash; ///< of guest memory's bytes when the draw was recorded
        u32 how;  ///< Bound (or 9: near a mirror, OPT-004)
        std::array<u32, 4> head; ///< its first bytes then
    };
    std::vector<ProbeVertex> probe_vertices; ///< DIAG-170: the draw's vertex buffers
    std::string probe_bound;
    std::vector<std::pair<VAddr, u64>> probe_buffers; ///< DIAG-170: the draw's buffers
    std::string probe_bound_hashes;
    std::vector<DrawProbe> draw_probes;
    const Native::ImageEntry* probe_sampled = nullptr; ///< DIAG-169's match for this draw
    u32 probe_draw = 0;
    u64 probe_frame = 0;
    void ProbeDraw(const Targets& targets, const AmdGpu::Regs& regs);
    /// DIAG-173 (diagnosis): one frame's pixel history along a strip of rows of one target.
    struct StripProbe {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u32 draw;
        std::string what; ///< the draw's shaders and images
    };
    std::vector<StripProbe> strip_probes;
    u64 strip_frame = ~u64{0}; ///< the frame being recorded with strips
    u64 strip_written = 0;     ///< strips of that frame written out
    bool strip_screenshot = false;
    std::string strip_what;    ///< BindStages: the current draw's shaders and images
    void StripDraw(const Targets& targets);
    /// DIAG-178 (diagnosis): the work of one frame that writes or samples a watched address.
    std::string diag_what;
    u32 diag_lines = 0;
    static VAddr DiagWritesAt();
    bool DiagFrame() const;
    void DiagWrites(const Targets* targets, const DrawBindings& b);
    void StripDispatch(const DrawBindings& b);
    void StripImage(const Native::ImageEntry& target, bool dispatch);
    void StripFrameEnd();
    /// DIAG-175 (diagnosis): one texel of every layer of a dispatch's stored image, before and
    /// after it.
    struct DispatchProbe {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u32 layers;
        u32 mips;
        VAddr address;
        std::string what;
    };
    std::vector<DispatchProbe> dispatch_probes;
    void ProbeDispatch(const Shader::Info& stage, const DrawBindings& b, bool after);
    /// DIAG-179 (diagnosis): a traced R32F texture read back right after its sources were
    /// copied in (RYGT_DIAG_REFRESH_READ=<width>x<height>).
    struct RefreshProbe {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 frame;
        std::string what;
    };
    std::vector<RefreshProbe> refresh_probes;
    void ProbeRefresh(const Native::ImageEntry& texture, const std::string& what);
    void ReadRefreshProbes();
    /// DIAG-211 (diagnosis): a watched shader's buffer bound from a region, as the GPU holds it
    /// at the draw, against guest memory when the draw was recorded.
    struct RegionCompare {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 frame;
        VAddr address;
        u64 size;
        u32 stride;
        std::vector<u8> cpu;
        std::string what;
        bool always = false; ///< indirect arguments: logged, the first draws' words too
    };
    std::vector<RegionCompare> region_compares;
    void CompareRegion(VAddr base, u64 size, u32 stride, vk::Buffer buffer, u64 offset,
                       u64 program, bool always = false);
    /// DIAG-211: a watched shader's flattened user data against what the regions it was read
    /// from hold at the draw (GPU-written constants the CPU walk could not see).
    void CompareFlatSources(const Shader::Info& stage);
    /// DIAG-212 (diagnosis): a target right before a given program first draws into it, and the
    /// frame's display picture, written out raw (where a glitch enters a frame).
    struct Capture {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 frame;
        std::string what;
        u32 width;
        u32 height;
        u32 format;
    };
    std::vector<Capture> captures;
    u64 capture_frame = ~u64{0};
    void CaptureImage(const Native::ImageEntry& entry, u64 at_frame, const std::string& what);
    void WriteCaptures();
    void ReadRegionCompares();
    /// DIAG-180 (diagnosis, RYGT_DIAG_VOLUME=<hex address>): an RGBA16F volume read back every
    /// 300 frames, summarized per slice.
    std::unique_ptr<Native::Rhi::Buffer> volume_probe;
    u64 volume_tick = 0;
    u32 volume_width = 0, volume_height = 0, volume_depth = 0;
    void ProbeVolume();
    /// DIAG-182 (diagnosis, RYGT_DIAG_STALE_SMALL=1): small textures in use whose guest bytes
    /// differ from what was last uploaded while nothing asks for an upload.
    void CheckSmallTextures();
    /// DIAG-184 (diagnosis, RYGT_DIAG_PROGRAM_TEXTURES=<hex hash>): the small textures that
    /// program samples, and every 120 frames what memory holds there against what was uploaded.
    std::vector<Native::ImageEntry*> diag_program_textures;
    void NoteProgramTextures(u64 program, const DrawBindings& b);
    void ReportProgramTextures();
    /// DIAG-185 (diagnosis): writes over the DIAG-184 textures' memory.
    void NoteProgramTextureWrites(const Targets* targets, const DrawBindings& b);
    void NoteProgramTextureWrite(VAddr address, u64 size, const char* kind);
    u32 diag185_lines = 0;
    /// DIAG-187 (diagnosis, RYGT_DIAG_WRITE_HISTORY=1): GPU writes, region events and CPU write
    /// faults in a ring; DIAG-184 prints those over each of its textures when it first sees it.
    struct WriteNote {
        VAddr begin = 0;
        u64 size = 0;
        u64 frame = 0;
        u64 program = 0;
        u32 kind = 0;
    };
    std::vector<WriteNote> write_history;
    std::vector<WriteNote> kept_history; ///< over RYGT_DIAG_HISTORY_RANGES, from the start
    u64 write_history_count = 0;
    std::mutex write_history_mutex;
    u64 history_program = 0;
    void NoteHistory(VAddr begin, u64 size, u32 kind, u64 program);
    void PrintHistory(VAddr begin, u64 size);
    static const std::vector<std::pair<VAddr, VAddr>>& HistoryRanges();
    void WatchHistoryRanges();
    void ReadDispatchProbes();
    void ReadDrawProbes();
    std::array<double, 3> display_last_part{}; ///< FIX-182
    std::vector<u8> display_last_bytes;        ///< FIX-182
    u32 display_glitch_dumps = 0;              ///< FIX-182
    void ProbeDisplay(const Native::ImageEntry& entry);
    /// FIX-157: bytes of pictures kept past their idle time.
    u64 kept_bytes = 0;
    /// FIX-GT7-153: bytes of circuit shadow maps kept past their idle time (a budget of their own).
    u64 kept_shadow_bytes = 0;
    std::vector<Native::ImageEntry*> writebacks;
    /// PERF-GT7-044: the small pictures drawn this frame, written back when it ends.
    std::vector<Native::ImageEntry*> frame_writebacks;
    void QueueWriteback(const Native::ImageEntry* entry);
    void WritebackForCpu(const Native::ImageEntry* entry); ///< FIX-GT7-041
    bool SendWritebacksDue(); ///< FIX-GT7-040
    /// PERF-GT7-044: pictures waiting for the frame's end go back before work over their memory.
    void WritebacksBefore(VAddr address, u64 size);
    void WritebacksBefore(const DrawBindings& b);
    bool writebacks_due = false; ///< some of them wait for the next global barrier
    std::vector<const Native::ImageEntry*> dump_entries; // NAT-116
    /// NAT-123: GPU-written images the dumped frame sampled, by texture key.
    std::vector<std::pair<std::string, const Native::ImageEntry*>> dump_textures;
    /// NAT-120: depth images by the address of their stencil plane (fills clear it there).
    std::unordered_map<VAddr, boost::container::small_vector<Native::ImageEntry*, 2>>
        stencil_owners;
    /// NAT-120: GPU work wrote guest memory (targets there get a pending rewrite).
    void GpuWrote(VAddr address, u64 size, const std::array<u32, 2>* pattern, bool exact);
    /// NAT-120: a target's pending rewrite, before it is used: a clear, or an upload.
    void ApplyRewrite(Native::ImageEntry& entry);
    /// NAT-132: fills the picture with a 32-bit pattern on the GPU (rows != 0: only those rows
    /// of the first mip and layer); false when it can't (compressed, multisampled).
    bool FillImage(Native::ImageEntry& entry, u32 lo, u32 hi, u32 rows, u32 first_layers = 0);
    /// NAT-134: a device buffer holding the pattern (lo, hi, lo, hi...), at least `bytes` long.
    const Native::Rhi::Buffer* PatternBuffer(u32 lo, u32 hi, u64 bytes);
    /// FIX-GT7-123: a pattern's buffer and when it was last asked for (pattern_clock).
    struct FillPattern {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 used = 0;
    };
    std::unordered_map<u64, FillPattern> fill_patterns;
    u64 pattern_clock = 0;
    u64 pattern_bytes = 0; ///< FIX-GT7-123: the kept buffers' bytes
    /// NAT-120: GT Sport's fill shader (cs 0x8aa6e00c): its pattern and range, from the
    /// arguments it reads from memory.
    void ReadFillShader(const Shader::Info& cs, DrawBindings& b);
    /// NAT-104: the picture was written now (in GPU order).
    /// `at` != 0: the version is that clock value (NAT-148: a rewrite applied late).
    void Written(const Native::ImageEntry* entry, u64 at = 0);
    /// FIX-208: as Written, for a draw into some mips and layers only (their versions).
    void WrittenRange(const Native::ImageEntry* entry, const DrawnRange& range);
    void WrittenSubresources(const Native::ImageEntry* entry, u64 at, const DrawnRange* range);
    /// FIX-208: the version of one subresource (the picture's when it keeps none of its own).
    static u64 SubVersion(Native::ImageEntry& entry, u32 mip, u32 layer);
    static bool SubVersionsOn();
    /// NAT-104: copies the newer pictures of the render targets a texture is made of into it.
    void RefreshSources(Native::ImageEntry& texture);
    /// FIX-212: a depth target and a colour target of its layout at its address (D16/R16,
    /// D32/R32) are one picture: the other's newer one is copied in before either is used
    /// (RYGT_NATIVE_DEPTH_TWINS=0 off).
    void SyncDepthTwin(Native::ImageEntry& entry);
    /// FIX-210: before a draw into a target that is mips or layers of a larger one (FIX-197),
    /// the subresources of the drawn view that the larger one holds newer pictures for are
    /// copied in (RYGT_NATIVE_PART_SYNC=0 off).
    void SyncPartFromWholes(Native::ImageEntry& part, const DrawnRange& range);
    static bool PartSyncOn();
    /// FIX-207: textures made of targets made of targets (RYGT_NATIVE_WHOLE_SOURCES=0 off), and
    /// the version of such a whole: that of its newest part.
    static bool WholeSourcesOn();
    static u64 WholeVersion(Native::ImageEntry& source);

    void ResolveDraw(const AmdGpu::Regs& regs);
    bool MetaClearDispatch(const Shader::Info& cs);
    void Report();

    /// OPT-010: a draw's bindings worked out on the game job thread that recorded it, before
    /// the command processor reaches it: BindStages' lookups (buffers bound from regions or
    /// guest memory, textures from existing pictures and their views, samplers) with nothing
    /// done that must happen in GPU order (no picture or region made, nothing copied, uploaded
    /// or rewritten; upload memory is filled later). The recorder takes it by the draw packet's
    /// address and uses it when it was made for the same pipeline and user data and no
    /// picture, region or mirror over what it binds was made or dropped since (and no memory
    /// unmapped); then it does the ordered part (pending rewrites, copies in, upload memory).
    struct PreparedDraw {
        const void* packet = nullptr;
        u32 epoch = 0;
        const GraphicsPipeline* pipeline = nullptr;
        std::array<u64, Shader::MaxStageTypes> stage_hashes{};
        u64 image_generation = 0;
        u64 buffer_seq = 0;
        u64 unmap_gen = 0;
        DrawBindings bindings;
        boost::container::small_vector<Region*, 4> regions;
        std::vector<std::pair<VAddr, VAddr>> image_ranges;
        std::vector<std::pair<VAddr, VAddr>> buffer_ranges;
        /// Upload memory (flattened user data, clip planes, D3 tables): bytes copied at commit
        /// into the list's upload memory, and the buffer info that binds them.
        struct Upload {
            u32 info = 0;
            u32 offset = 0;
            u32 size = 0;
            s32 flat_stage = -1; ///< >= 0: that stage's flattened user data, the recorder's
        };
        std::vector<u8> upload_bytes;
        boost::container::small_vector<Upload, 4> uploads;
        u32 null_images = 0;
        /// OPT-011: vertex inputs and buffers and the index buffer (not for quad lists), as
        /// ResolveVertexBuffers and ResolveIndices work them out; the V#s they came from (a
        /// hash), the mirrors bound (copied in at commit) and the ranges read in place (mirror
        /// candidates, noted at commit).
        bool has_vertices = false;
        std::unique_ptr<VertexIndexState> vertex_index;
        u64 vertex_hash = 0;
        bool indexed = false;
        u32 index_offset = 0;
        VAddr index_address = 0;
        boost::container::small_vector<Mirror*, 4> mirrors;
        boost::container::small_vector<std::pair<VAddr, u64>, 4> in_place;
        /// OPT-012: the stages as the job thread matched them (what the command thread's
        /// selection would copy): the recorder installs them when the command thread didn't
        /// select (HasPrepared).
        struct StageCopy {
            bool present = false;
            VAddr pgm_base = 0;
            u32 num_user_data = 0;
            std::array<u32, 16> user_data{};
            std::vector<u32> flattened;
        };
        std::array<StageCopy, Shader::MaxStageTypes> stage_copies;
        void Reset();
    };
    /// OPT-010: the guest ranges whose region or mirror was made or dropped, by sequence number
    /// (written and checked by the recorder; job threads read the sequence before looking up).
    class ChangeLog {
    public:
        void Note(VAddr begin, VAddr end);
        /// Whether none of the ranges changed after `since` (false when the log no longer
        /// reaches back that far).
        bool Untouched(u64 since, std::span<const std::pair<VAddr, VAddr>> ranges) const;
        u64 Seq() const {
            return seq.load(std::memory_order_acquire);
        }

    private:
        struct Entry {
            u64 seq = 0;
            VAddr begin = 0;
            VAddr end = 0;
        };
        std::array<Entry, 1024> entries{};
        std::atomic<u64> seq{0};
    };
    ChangeLog buffer_log;
    /// OPT-010: regions and mirrors are added and removed under it (exclusive); job threads
    /// look regions up under it (shared).
    std::shared_mutex buffer_map_mutex;
    struct PreparedShard {
        std::mutex mutex;
        std::unordered_map<const void*, PreparedDraw*> draws;
    };
    std::array<PreparedShard, 64> prepared_shards;
    PreparedShard& ShardFor(const void* packet) {
        return prepared_shards[(reinterpret_cast<u64>(packet) * 0x9E3779B97F4A7C15ULL) >> 58];
    }
    std::mutex prepared_pool_mutex;
    std::vector<std::unique_ptr<PreparedDraw>> prepared_all;
    std::vector<PreparedDraw*> prepared_free;
    PreparedDraw* NewPrepared();
    void FreePrepared(PreparedDraw* draw);
    /// The prepared draw for the packet, taken out of the table (none if there is none).
    PreparedDraw* TakePrepared(const void* packet);
    bool PrepareBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs,
                        Shader::Backend::Bindings& binding, PreparedDraw& p);
    bool PrepareTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                         Shader::Backend::Bindings& binding, PreparedDraw& p);
    bool PrepareVertices(PreparedDraw& p, const AmdGpu::Regs& regs,
                         const GraphicsPipeline& pipeline, const Shader::Info& vs,
                         bool is_indexed, u32 index_offset);
    /// A vertex or index range from a region, a mirror or guest memory read in place (none
    /// when it would need ordered work).
    std::optional<std::pair<vk::Buffer, u64>> JobVertexBuffer(VAddr address, u64 size,
                                                              PreparedDraw& p,
                                                              Mirror** mirror_out);
    bool CommitPreparedVertices(PreparedDraw& p, const GraphicsPipeline& pipeline,
                                const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                                bool quad_triangles, bool trusted);
    static u64 VertexHash(const GraphicsPipeline& pipeline, const Shader::Info& vs,
                          u32 step_rate_0, u32 step_rate_1);
    /// GuestBufferLookup's lookups without its side effects: none when it would make a region,
    /// gather or copy.
    std::optional<std::pair<vk::Buffer, u64>> JobGuestBuffer(VAddr address, u64 size, bool write,
                                                             PreparedDraw& p);
    /// Recorder: whether the prepared draw holds for the draw being recorded (its ordered part
    /// is then done).
    bool CommitPrepared(PreparedDraw& p, const GraphicsPipeline& pipeline,
                        const AmdGpu::Regs& regs, bool trusted);
    /// OPT-012: installs the prepared draw's stages for its pipeline (nullptr if they don't
    /// fit it).
    const GraphicsPipeline* InstallPrepared(PreparedDraw& p);
    static u64 StageHash(const Shader::Info& info);
    std::atomic<u64> shared_frame{0}; ///< OPT-010: `frame`, for job threads
    std::atomic<u32> recorder_epoch{0};
    struct PrepareStats {
        std::atomic<u64> made{0}, failed{0}, replaced{0}, stale{0}, ns{0};
    } prepare_stats;

    Rasterizer& r;
    std::unique_ptr<Native::Rhi::Device> device;
    std::unique_ptr<Native::Rhi::Fence> timeline;
    std::unique_ptr<Native::ImageRegistry> registry;
    std::unique_ptr<Native::Rhi::Buffer> gds;
    std::unique_ptr<Native::Rhi::Buffer> lds;

    std::recursive_mutex mutex; ///< v0: the recorder thread records; flips may come from others
    Stream stream;
    /// GT7-UW: the HUD draws over white, recorded in a list of their own that runs right after
    /// the main one (which draws them over black): neither list switches between the two
    /// pictures, so consecutive HUD draws stay in one pass without barriers between them.
    Stream hud_stream;
    /// DLSS-GT7-001: in a frame left as the game draws it, the HUD draws over black, in a list of
    /// their own too (the main list then draws them into the display as the game does).
    Stream hud_black_stream;
    static bool HudStreamOn();
    void BeginHudStream();
    void SpliceHudStream(u64 tick);
    void SpliceStream(Stream& s, u64 tick);
    /// DLSS-GT7-001: the open batch of HUD draws runs (main list first, then the HUD lists).
    void CloseHudBatch();
    static bool HudBatchOn();
    DrawBindings bindings;
    u64 frame = 0;
    u64 timing_tick = 0; ///< OPT-001
    /// NAT-142: guest memory the GPU reads or writes in place (64 KB pages, every 8th frame):
    /// vertex buffers, indices, read-only storage through the page table, other storage
    /// buffers read, storage buffers written; NAT-150: then what is bound from video memory
    /// instead (mirrors, regions).
    struct InPlace {
        std::array<std::unordered_set<u64>, 7> pages;
        std::array<u64, 7> total{};
        u64 frames = 0;
        void Note(u32 kind, VAddr address, u64 size) {
            if (size == 0) {
                return;
            }
            for (u64 page = address >> 16; page <= (address + size - 1) >> 16; ++page) {
                pages[kind].insert(page);
            }
        }
    } in_place;
    bool InPlaceSampled() const {
        // PERF-GT7-001: diagnosis only (RYGT_NATIVE_INPLACE_STATS=1). Sampled frames put every
        // 64 KB page of every binding in a set and looked each vertex attribute up in full
        // (no OPT-004 shortcut): in GT7's busy scenes (~3,000 draws) those frames missed the
        // 30 fps deadline.
        static const bool on = [] {
            const char* env = Common::DiagEnv("RYGT_NATIVE_INPLACE_STATS");
            return env && env[0] == '1';
        }();
        return on && (frame & 7) == 0;
    }
    /// NAT-150: the counter for a binding of that kind, by what GuestBuffer bound.
    u32 BoundKind(u32 kind) const {
        return last_bound == Bound::Mirror ? 5 : (last_bound == Bound::Region ? 6 : kind);
    }

    std::mutex chunk_mutex;
    std::vector<Chunk> free_chunks;
    size_t free_chunks_low = 0; // FIX-GT7-046: the fewest free since the last trim
    std::deque<Chunk> retiring_chunks;
    /// PERF-GT7-007 (B): the device buffers of FIX-203's copies before the work (~2,000 a
    /// second in GT7, each made and destroyed), kept by size class and used again once the GPU
    /// ran the list that read them.
    std::vector<Chunk> free_snapshots;
    std::deque<Chunk> retiring_snapshots;
    u64 snapshot_pool_bytes = 0;
    std::unique_ptr<Native::Rhi::Buffer> TakeSnapshotBuffer(u64 size);
    void RetireSnapshots(Stream& s, u64 tick);
    std::vector<vk::DescriptorPool> free_pools;
    std::deque<std::pair<u64, vk::DescriptorPool>> retiring_pools;

    std::mutex sampler_mutex;
    /// FIX-GT7-002: by the state the device sees; sampler_lookup by the raw S#.
    std::unordered_map<u64, std::unique_ptr<VideoCore::Sampler>> samplers;
    std::unordered_map<u64, vk::Sampler> sampler_lookup;
    vk::Sampler last_sampler{};

    std::mutex views_mutex;
    std::unordered_map<const Native::ImageEntry*, std::vector<ViewRecord>> views;
    /// NAT-108: views of evicted images, destroyed once the GPU passed the tick.
    std::deque<std::pair<u64, vk::UniqueImageView>> retired_views;
    void EvictImages();
    struct ReportBase {
        u64 uploads = 0, upload_bytes = 0, evicted = 0;
        std::array<u64, 3> stale{};
    } report_base;

    /// NAT-104: the write clock, and which version of its sources each subresource of a
    /// texture holds (by mip << 16 | layer).
    u64 write_clock = 0;
    std::unordered_map<const Native::ImageEntry*, std::unordered_map<u32, u64>> held_versions;

    /// CMASK / HTILE addresses the game cleared: the next pass into their target clears it.
    std::unordered_set<VAddr> cleared_meta;
    /// FIX-GT7-133: a 1x1 depth image at 1.0 that stands in for an empty T# sampled with a
    /// depth comparison (2D and 2D-array views), so that the sampler's border still applies.
    std::unique_ptr<Native::Rhi::Image> depth_stand_in;
    std::array<std::unique_ptr<Native::Rhi::ImageView>, 2> depth_stand_in_views;
    std::array<std::atomic<VkImageView>, 2> depth_stand_in_handles{};
    /// FIX-GT7-133: the stand-in's view (made, and the image cleared, on the recorder only when
    /// `may_create`); none when off or not made yet.
    vk::ImageView DepthStandIn(bool array, bool may_create);
    /// FIX-GT7-133: RYGT_NATIVE_DEPTH_STAND_IN is not 0. A prepared draw that needs the stand-in
    /// before the recorder made it goes to the recorder.
    static bool DepthStandInOn();
    /// Color targets by their CMASK address, depth targets by their HTILE address (so a clear
    /// of the metadata can be told from a clear of a picture).
    std::unordered_set<VAddr> known_meta;

    struct Stats {
        u64 rewrite_clears = 0, rewrite_uploads = 0, fill_shaders = 0; ///< NAT-120
        u64 rewrite_partial = 0; ///< NAT-124
        u64 pass_barriers = 0, passes_unblocked = 0; ///< NAT-140
        u64 region_refresh_pages = 0, region_refresh_ns = 0; ///< NAT-145
        u64 region_partial_pages = 0, region_partial_bytes = 0; ///< FIX-173
        u64 region_whole_pages = 0, region_writebacks = 0;       ///< FIX-194
        u64 region_writeback_bytes = 0, region_pages_forgotten = 0;
        u64 mirror_untrapped = 0;                               ///< FIX-176
        u64 mirror_checks = 0, mirror_check_differ = 0;         ///< FIX-179
        u64 reach_checks = 0, reach_past_sharp = 0, reach_past_mirror = 0; ///< FIX-181
        u64 guest_buffer_calls = 0, guest_buffer_ns = 0;     ///< NAT-145
        u64 rewrite_fills = 0;   ///< NAT-132
        u64 timed_draws = 0;     ///< OPT-001
        u64 vertex_near = 0;     ///< OPT-004
        u64 vertex_skipped = 0;  ///< OPT-005
        /// OPT-013..015: targets from the memo, sets bound again without a new one, sets
        /// written, pushed sets and push constants not pushed again, uploads reused.
        u64 targets_reused = 0, sets_reused = 0, sets_written = 0, pushes_skipped = 0,
            push_constants_skipped = 0, uploads_reused = 0;
        /// OPT-010: draws bound from a prepared draw, without one, and prepared draws not used:
        /// another pipeline, other user data, memory unmapped, pictures or buffers changed.
        u64 prepared_used = 0, prepared_missing = 0;
        std::array<u64, 5> prepared_rejected{};
        u64 prepared_vertices = 0, prepared_vertices_rejected = 0; ///< OPT-011
        u64 prepared_installed = 0; ///< OPT-012: pipelines from a job thread's selection
        /// OPT-011: the record phase's parts: barrier choice, pass begin, dynamic state, vertex
        /// and index binds, pipeline bind, descriptors and push constants, the draw, write notes.
        std::array<u64, 8> record_ns{};
        u64 texture_memo_hits = 0, texture_lookups = 0; ///< OPT-002
        u64 buffer_memo_hits = 0;                        ///< OPT-003
        u64 mirror_binds = 0, mirrors_made = 0, mirror_made_bytes = 0, mirrors_dropped = 0,
            mirrors_evicted = 0, mirror_over_budget = 0, mirror_refresh_pages = 0; ///< NAT-150
        u64 indirect_draws = 0, indirect_skipped = 0; ///< NAT-161: recorded, and not
        u64 one_off_waits = 0; ///< FIX-198: draws that waited for their new pipeline
        u64 one_off_exempt = 0; ///< PERF-GT7-023: draws into exempt pictures, skipped
        u64 one_off_budget_skips = 0; ///< FIX-GT7-102: skipped once the frame's budget was spent
        u64 one_off_harmful_waits = 0; ///< FIX-GT7-121: waited whatever the budget
        u64 kept_shadow_waits = 0, kept_shadow_wait_ns = 0, kept_shadow_wait_max_ns = 0,
            kept_shadow_skips = 0; ///< FIX-GT7-153
        u64 writeback_befores = 0, precise_writebacks = 0; ///< FIX-203
        /// DIAG-188: a direct dispatch's recorder time by part (pipeline, bindings, images,
        /// barrier with write-backs, recording, notes), and the write-backs' share.
        std::array<u64, 6> dispatch_part_ns{};
        u64 writeback_ns = 0;
        /// DIAG-GT7-004: a direct dispatch's bindings part by piece (list begin, buffers,
        /// textures, FIX-203 copies before the work, fill shader read; the rest is the
        /// difference), its notes' region write-backs and GPU-write notes, and the region
        /// refreshes inside its bindings.
        std::array<u64, 5> dispatch_bind_ns{};
        std::array<u64, 2> dispatch_note_ns{};
        u64 dispatch_refresh_ns = 0;
        /// PERF-GT7-007: FIX-203 copies by bytes, and their buffers made and reused.
        u64 snapshot_bytes = 0, snapshots_made = 0, snapshots_reused = 0;
        u64 lists_ahead = 0; ///< PERF-GT7-008: lists begun while the recorder waited
        /// PERF-GT7-009: compute stages' image slots looked up, from the memo, and bindings.
        u64 compute_texture_lookups = 0, compute_texture_hits = 0, compute_texture_bindings = 0;
        u64 small_rewrites = 0; ///< FIX-202: small textures uploaded again after big writes
        u64 lds_emulated_draws = 0; // FIX-227
        /// PERF-GT7-060: FIX-227 draws whose reader pass waited for their run's barrier, runs
        /// by length (1, 2, 3-4, 5-8, 9-16, 17-32, 33+) and the longest, runs ended by another
        /// draw / the pass's end / the run's limit; draws that ran alone (switch off,
        /// not eligible), and the runs these would have made (nothing between them).
        u64 lds_batched = 0, lds_runs = 0, lds_run_max = 0;
        std::array<u64, 7> lds_run_lengths{};
        std::array<u64, 3> lds_run_ends{};
        u64 lds_alone = 0, lds_alone_runs = 0, lds_alone_run_max = 0;
        std::array<u64, 7> lds_alone_lengths{};
        u64 lds_alone_run = 0;                       ///< the current one's length
        std::array<u64, 4> lds_alone_marker{};       ///< barriers, passes, draws, dispatches after it
        u64 draws = 0, dispatches = 0, skipped = 0, no_pipeline = 0, passes = 0, barriers = 0,
            splices = 0, copied_reads = 0, copied_bytes = 0, unbound = 0, null_images = 0,
            fills = 0, copies = 0, meta_clears = 0, inline_writes = 0, source_copies = 0,
            draw_ns = 0, dispatch_ns = 0, gathers = 0, gathered_bytes = 0, writebacks = 0,
            region_binds = 0, regions_made = 0, region_refreshes = 0;
        /// Images bound empty: no address or format, not mapped for the GPU, format the
        /// device lacks, bad geometry, outside guest memory, FIX-081, no image made, nothing
        /// to read (FIX-GT7-184).
        std::array<u64, 8> null_reasons{};
        /// NAT-110: time in the draw's phases: pipeline, targets, buffers, textures, sources,
        /// vertices, prepare (inits and uploads), record.
        std::array<u64, 8> phase_ns{};
        std::atomic<u64> cpu_writes{0}; // NAT-105 (faults come from any thread)
    } stats;
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};

} // namespace Vulkan
