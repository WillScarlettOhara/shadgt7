// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <windows.h>

#include "common/types.h"

namespace Native {

/// FIX-GT7-184: whether any page of [address, address + size) is committed, that is whether a
/// copy of the range could bring anything but zeros (MemoryManager::CopySparseMemory gives zeros
/// for every page that is not). A committed page the CPU may not read (protected, or under write
/// tracking) counts: its bytes are there. At most max_regions regions are asked for; past them,
/// or on a walk that does not advance, the answer is true, so that a caller who bounds its time
/// does as it did before on a range it could not walk. Past the last address a process can map
/// there is nothing, as CopySparseMemory has it: false.
inline bool AnyCommittedPage(u64 address, u64 size, u32 max_regions = 256) {
    const u64 end = address + size < address ? ~u64{0} : address + size;
    u64 at = address;
    for (u32 asked = 0; at < end; ++asked) {
        if (asked == max_regions) {
            return true;
        }
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info))) {
            return false; // past the last address a process can map
        }
        if (info.State == MEM_COMMIT) {
            return true;
        }
        const u64 next = reinterpret_cast<u64>(info.BaseAddress) + info.RegionSize;
        if (next <= at) {
            return true;
        }
        at = next;
    }
    return false;
}

} // namespace Native
