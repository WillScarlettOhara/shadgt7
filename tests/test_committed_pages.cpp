// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "ryty_gpu/native/committed_pages.h"

namespace {

using Native::AnyCommittedPage;

constexpr u64 Block = 64 * 1024; // Windows hands out addresses in 64 KB steps
constexpr u64 Page = 4096;

struct Reservation {
    explicit Reservation(u64 size, void* at = nullptr)
        : base{reinterpret_cast<u64>(VirtualAlloc(at, size, MEM_RESERVE, PAGE_NOACCESS))} {}
    ~Reservation() {
        if (base) {
            VirtualFree(reinterpret_cast<void*>(base), 0, MEM_RELEASE);
        }
    }
    u64 base;
};

// FIX-GT7-184: a texture descriptor read from a slot the game never wrote (GT7's shader
// 0x2a265dff) named 0.7 to 2.7 GiB of address space nothing was mapped in.
TEST(CommittedPages, ReservedMemoryHoldsNothing) {
    const Reservation memory{256 * Block};
    ASSERT_TRUE(memory.base != 0);
    EXPECT_FALSE(AnyCommittedPage(memory.base, 256 * Block));
    EXPECT_FALSE(AnyCommittedPage(memory.base + 5, Page));
}

TEST(CommittedPages, EmptyRangeHoldsNothing) {
    const Reservation memory{Block};
    ASSERT_TRUE(memory.base != 0);
    ASSERT_TRUE(VirtualAlloc(reinterpret_cast<void*>(memory.base), Page, MEM_COMMIT,
                             PAGE_READWRITE) != nullptr);
    EXPECT_FALSE(AnyCommittedPage(memory.base, 0));
}

// A streamed texture has unmapped leading mips and resident lower ones (NAT-042): one page
// anywhere in the range is enough.
TEST(CommittedPages, OnePageAnywhereCounts) {
    const Reservation memory{256 * Block};
    ASSERT_TRUE(memory.base != 0);
    const u64 page = memory.base + 200 * Block + 3 * Page;
    ASSERT_TRUE(VirtualAlloc(reinterpret_cast<void*>(page), Page, MEM_COMMIT, PAGE_READWRITE) !=
                nullptr);
    EXPECT_TRUE(AnyCommittedPage(memory.base, 256 * Block));
    EXPECT_TRUE(AnyCommittedPage(page + Page - 1, 1));
    EXPECT_TRUE(AnyCommittedPage(page - 1, 2));
    EXPECT_FALSE(AnyCommittedPage(memory.base, page - memory.base));
    EXPECT_FALSE(AnyCommittedPage(page + Page, 55 * Block));
    ASSERT_TRUE(VirtualFree(reinterpret_cast<void*>(page), Page, MEM_DECOMMIT) != 0);
    EXPECT_FALSE(AnyCommittedPage(memory.base, 256 * Block));
}

// Pages under write tracking, or protected, are committed pages the CPU may not read: their bytes
// are still there (MemoryManager::CopySparseMemory reads them through an alias).
TEST(CommittedPages, PagesTheCpuMayNotReadCount) {
    const Reservation memory{4 * Block};
    ASSERT_TRUE(memory.base != 0);
    ASSERT_TRUE(VirtualAlloc(reinterpret_cast<void*>(memory.base + Block), Page, MEM_COMMIT,
                             PAGE_NOACCESS) != nullptr);
    EXPECT_TRUE(AnyCommittedPage(memory.base, 4 * Block));
}

// Guest memory is mapped as views of a section, not allocated.
TEST(CommittedPages, MappedViewCounts) {
    const HANDLE section =
        CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 4 * Block, nullptr);
    ASSERT_TRUE(section != nullptr);
    void* const view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, 4 * Block);
    ASSERT_TRUE(view != nullptr);
    EXPECT_TRUE(AnyCommittedPage(reinterpret_cast<u64>(view), 4 * Block));
    EXPECT_TRUE(AnyCommittedPage(reinterpret_cast<u64>(view) + 3 * Block, Page));
    UnmapViewOfFile(view);
    CloseHandle(section);
}

TEST(CommittedPages, PastTheAddressSpaceHoldsNothing) {
    EXPECT_FALSE(AnyCommittedPage(~u64{0} - 0x10000, 0x8000));
}

// A size that carries the end past the top of the address space still covers what is there.
TEST(CommittedPages, EndThatWrapsStillSeesTheStart) {
    const Reservation memory{Block};
    ASSERT_TRUE(memory.base != 0);
    ASSERT_TRUE(VirtualAlloc(reinterpret_cast<void*>(memory.base), Page, MEM_COMMIT,
                             PAGE_READWRITE) != nullptr);
    EXPECT_TRUE(AnyCommittedPage(memory.base, ~u64{0} - 16)); // the end wraps
}

// The walk is bounded: a range in more pieces than it may ask about counts as holding something,
// so the caller does as it did before.
TEST(CommittedPages, TooManyRegionsCountAsCommitted) {
    constexpr u32 Pieces = 16;
    u64 base = 0;
    {
        const Reservation probe{Pieces * Block};
        ASSERT_TRUE(probe.base != 0);
        base = probe.base;
    }
    // Each reservation is a region of its own, even side by side.
    std::vector<std::unique_ptr<Reservation>> pieces;
    for (u32 i = 0; i < Pieces; ++i) {
        pieces.push_back(
            std::make_unique<Reservation>(Block, reinterpret_cast<void*>(base + i * Block)));
        if (pieces.back()->base != base + i * Block) {
            GTEST_SKIP(); // another thread took the address: nothing to tell here
        }
    }
    EXPECT_FALSE(AnyCommittedPage(base, Pieces * Block, 64));
    EXPECT_FALSE(AnyCommittedPage(base, Pieces * Block, Pieces));
    EXPECT_TRUE(AnyCommittedPage(base, Pieces * Block, Pieces - 1));
    EXPECT_TRUE(AnyCommittedPage(base, Pieces * Block, 1));
}

} // namespace
