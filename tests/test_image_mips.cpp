// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FIX-039: render targets for the micro-tiled mips of a macro-tiled image. The values are the
// ones GT Sport programs for the environment map of the car purchase thumbnail (decoded from a
// diagnostic bundle's PM4 capture, tools/mcp/pm4_draws.py): a 2048x1024 B10G11R11 texture at
// 0x1009bc0000 with 12 mips and tile index 14, whose mips are rendered as separate targets, mips
// 1-4 with tile index 14 and mips 5-11 with tile index 13.

#include <gtest/gtest.h>
#include "core/libraries/kernel/process.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/image_info.h"

// image_info.cpp also holds constructors from guest descriptors that this test does not use.
namespace Libraries::Kernel {
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return 0;
}
} // namespace Libraries::Kernel
namespace Common::Log {
void VLog(Class, Level, const char*, int, const char*, fmt::string_view, fmt::format_args) {}
} // namespace Common::Log
namespace Vulkan::LiverpoolToVK {
const std::array<vk::Format, surface_format_table_size> surface_format_table{};
vk::Format DepthFormat(AmdGpu::DepthBuffer::ZFormat, AmdGpu::DepthBuffer::StencilFormat) {
    return vk::Format::eUndefined;
}
} // namespace Vulkan::LiverpoolToVK
void assert_fail_impl() {
    std::abort();
}
[[noreturn]] void unreachable_impl() {
    std::abort();
}

namespace {
using VideoCore::ImageInfo;

ImageInfo Color2D(VAddr address, u32 width, u32 height, u32 levels, AmdGpu::TileMode tile_mode,
                  vk::Format format = vk::Format::eB10G11R11UfloatPack32, u32 bits = 32) {
    ImageInfo info{};
    info.guest_address = address;
    info.type = AmdGpu::ImageType::Color2D;
    info.resources = {static_cast<u16>(levels), 1};
    info.props.is_block = false;
    info.props.is_pow2 = false;
    info.props.is_tiled = true;
    info.pixel_format = format;
    info.num_bits = bits;
    info.num_samples = 1;
    info.size = {width, height, 1};
    info.pitch = std::max(width, 8u);
    info.tile_mode = tile_mode;
    info.array_mode = AmdGpu::GetArrayMode(tile_mode);
    info.UpdateSize();
    return info;
}

constexpr VAddr EnvMap = 0x1009bc0000;
constexpr auto Macro = AmdGpu::TileMode::Thin2DThin; // tile index 14
constexpr auto Micro = AmdGpu::TileMode::Thin1DThin; // tile index 13

// The render targets the game binds for mips 1-11, as (address, width, height, tile mode).
struct MipTarget {
    VAddr address;
    u32 width;
    u32 height;
    AmdGpu::TileMode tile_mode;
};
constexpr MipTarget Targets[] = {
    {0x100a3c0000, 1024, 512, Macro}, {0x100a5c0000, 512, 256, Macro},
    {0x100a640000, 256, 128, Macro},  {0x100a660000, 128, 64, Macro},
    {0x100a668000, 64, 32, Micro},    {0x100a66a000, 32, 16, Micro},
    {0x100a66a800, 16, 8, Micro},     {0x100a66aa00, 8, 4, Micro},
    {0x100a66ab00, 4, 2, Micro},      {0x100a66ac00, 2, 1, Micro},
    {0x100a66ad00, 1, 1, Micro},
};
} // namespace

TEST(ImageMips, LayoutMatchesTheGameMipAddresses) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    for (u32 mip = 1; mip < 12; ++mip) {
        EXPECT_EQ(texture.guest_address + texture.mips_layout[mip].offset, Targets[mip - 1].address)
            << "mip " << mip;
    }
    // Mips 5 and smaller are micro-tiled, matching the tile index the game uses for them.
    EXPECT_EQ(texture.micro_tiled_mips & 0x1E, 0u);
    EXPECT_EQ(texture.micro_tiled_mips & 0xFE0, 0xFE0u);
}

TEST(ImageMips, RenderTargetsResolveToTheTextureMips) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    for (u32 mip = 1; mip < 12; ++mip) {
        const auto& t = Targets[mip - 1];
        const auto target = Color2D(t.address, t.width, t.height, 1, t.tile_mode);
        EXPECT_EQ(target.MipOf(texture), static_cast<s32>(mip)) << "mip " << mip;
        EXPECT_EQ(target.SliceOf(texture, mip), 0) << "mip " << mip;
    }
}

TEST(ImageMips, MacroTiledLevelsStillNeedTheSameArrayMode) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    // Mip 1 is macro-tiled: a 1D target at its address is a different surface.
    const auto wrong = Color2D(0x100a3c0000, 1024, 512, 1, Micro);
    EXPECT_EQ(wrong.MipOf(texture), -1);
    // A depth (different micro tile mode) target at a micro-tiled mip is not that mip either.
    const auto depth = Color2D(0x100a668000, 64, 32, 1, AmdGpu::TileMode::Depth1DThin);
    EXPECT_EQ(depth.MipOf(texture), -1);
}

// FIX-040: in the bad car thumbnail capture (bundle 20261009-130850), the environment memory at
// 0x1009bc0000 was cached as a 1-level R8G8B8A8 image (uid 8693) when the game rendered and
// sampled it as B10G11R11; in the good one (20261009-130627) it was cached as B10G11R11 (uid
// 1985). Sampling it as the 12-level texture must expand the cached image in both cases.
TEST(ImageMips, FullChainExtendsAnImageCachedUnderAnotherFormat) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    const auto cached_rgba8 =
        Color2D(EnvMap, 2048, 1024, 1, Macro, vk::Format::eR8G8B8A8Unorm, 32);
    EXPECT_TRUE(texture.ExtendsAcrossFormat(cached_rgba8));

    // Same format: the ordinary expansion path handles it.
    const auto cached_same = Color2D(EnvMap, 2048, 1024, 1, Macro);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_same));
    // A single level of the other format stays a view (no more resources to add).
    const auto level0 = Color2D(EnvMap, 2048, 1024, 1, Macro);
    EXPECT_FALSE(level0.ExtendsAcrossFormat(cached_rgba8));
    // Different texel size, size or tiling: a different surface, never copied bit for bit.
    const auto cached_rgba16 =
        Color2D(EnvMap, 2048, 1024, 1, Macro, vk::Format::eR16G16B16A16Sfloat, 64);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_rgba16));
    const auto cached_small = Color2D(EnvMap, 512, 512, 1, Macro, vk::Format::eR8G8B8A8Unorm, 32);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_small));
    const auto cached_linear =
        Color2D(EnvMap, 2048, 1024, 1, Micro, vk::Format::eR8G8B8A8Unorm, 32);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_linear));
}

// FIX-GT7-126: GT7's Scapes compress their HDR photo to BC6H on the GPU. The values come from a
// Scapes session (run-20261108-083523, NAT-041 and DIAG-197 lines, frame dump T#s): the scene
// samples a 6024x4016 BC6H texture at 0x1036e00000 (5 levels, pitch 8192, tile index 13, 0x2aa0000
// bytes), and the encoder wrote its levels through R32G32B32A32Uint storage images: 2048x1024 at
// the texture's address, 1024x512 at 0x1038e00000.
namespace {
constexpr VAddr ScapesPhoto = 0x1036e00000;

ImageInfo BlockTexture(VAddr address, u32 width, u32 height, u32 pitch, u32 levels, u32 bits,
                       vk::Format format, bool pow2) {
    ImageInfo info{};
    info.guest_address = address;
    info.type = AmdGpu::ImageType::Color2D;
    info.resources = {static_cast<u16>(levels), 1};
    info.props.is_block = true;
    info.props.is_pow2 = pow2;
    info.props.is_tiled = true;
    info.pixel_format = format;
    info.num_bits = bits;
    info.num_samples = 1;
    info.size = {width, height, 1};
    info.pitch = pitch;
    info.tile_mode = Micro;
    info.array_mode = AmdGpu::GetArrayMode(Micro);
    info.UpdateSize();
    return info;
}

ImageInfo ScapesBc6h() {
    return BlockTexture(ScapesPhoto, 6024, 4016, 8192, 5, 128, vk::Format::eBc6HUfloatBlock, true);
}

ImageInfo BlockTarget(VAddr address, u32 width, u32 height, u32 pitch,
                      vk::Format format = vk::Format::eR32G32B32A32Uint, u32 bits = 128,
                      AmdGpu::TileMode tile_mode = Micro) {
    auto info = Color2D(address, width, height, 1, tile_mode, format, bits);
    info.pitch = pitch;
    info.UpdateSize();
    return info;
}
} // namespace

TEST(ImageMips, ScapesBc6hLayoutMatchesTheGame) {
    const auto texture = ScapesBc6h();
    EXPECT_EQ(texture.guest_size, 0x2aa0000u);
    EXPECT_EQ(texture.mips_layout[1].offset, 0x2000000u);
}

TEST(ImageMips, EncoderTargetsAreTheBlockTextureLevels) {
    const auto texture = ScapesBc6h();
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 2048, 1024, 2048)), 0);
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto + 0x2000000, 1024, 512, 1024)), 1);
    // Only the level's blocks (1506x1004), with the level's pitch: the same surface.
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 1506, 1004, 2048)), 0);
    // BC1 (64-bit blocks) through a 64-bit target.
    const auto bc1 = BlockTexture(0x1000000000, 256, 256, 256, 1, 64, vk::Format::eBc1RgbaUnormBlock,
                                  false);
    EXPECT_EQ(bc1.BlockLevelOf(BlockTarget(0x1000000000, 64, 64, 64, vk::Format::eR32G32Uint, 64)),
              0);
}

TEST(ImageMips, OtherTargetsOverABlockTextureAreNotItsLevels) {
    const auto texture = ScapesBc6h();
    // Not at a level's address.
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto + 0x1000000, 2048, 1024, 2048)), -1);
    // Another row pitch, fewer blocks than the level, another element size, another tiling.
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 1506, 1004, 1506)), -1);
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 1024, 1024, 2048)), -1);
    EXPECT_EQ(texture.BlockLevelOf(
                  BlockTarget(ScapesPhoto, 2048, 1024, 2048, vk::Format::eR32G32Uint, 64)),
              -1);
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 2048, 1024, 2048,
                                               vk::Format::eR32G32B32A32Uint, 128, Macro)),
              -1);
    // An uncompressed texture has no block levels.
    const auto plain = Color2D(ScapesPhoto, 2048, 1024, 1, Micro, vk::Format::eR32G32B32A32Uint,
                               128);
    EXPECT_EQ(plain.BlockLevelOf(BlockTarget(ScapesPhoto, 2048, 1024, 2048)), -1);
}
