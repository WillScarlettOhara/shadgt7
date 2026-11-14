// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// GTS-001: run-time patches of GT Sport's code (CUSA03220 v1.69), applied in DllMain before the
// game's code runs. Each patch checks the original bytes first and is skipped when they differ
// (another game version, another converter build). The game's files are not changed.
//
// RYGT_PATCH=<rva>:<original hex>:<new hex>[,...] applies arbitrary patches the same way, for
// experiments (RVA in GTSport.exe: eboot.bin address + 0x10000).

#include <cstdio>
#include <array>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <windows.h>
#include <intrin.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/diagnostics.h"
#include "common/ryty_config.h"
#include "common/perf_monitor.h"
#include "common/types.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "core/file_format/psf.h"
#include "input/trigger_feedback.h"
#include "ryty_gpu/gt7_preset_options.h"
#include "ryty_gpu/ryty_gt7_hdr_status.h"

namespace Vulkan {
void RytyWatchWrites(std::uint64_t address, std::uint32_t ms, std::string label); // DIAG-067
}

namespace {

struct Patch {
    u64 rva;
    std::vector<u8> original;
    std::vector<u8> replacement;
    std::string name;
};

std::vector<u8> ParseHex(const std::string& text) {
    std::vector<u8> bytes;
    for (size_t i = 0; i + 1 < text.size(); i += 2) {
        bytes.push_back(static_cast<u8>(std::strtoul(text.substr(i, 2).c_str(), nullptr, 16)));
    }
    return bytes;
}

bool Apply(u8* base, const Patch& patch) {
    u8* target = base + patch.rva;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(target, &info, sizeof(info)) || info.State != MEM_COMMIT) {
        std::fprintf(stderr, "[patch] %s: %#llx is not mapped\n", patch.name.c_str(),
                     static_cast<unsigned long long>(patch.rva));
        return false;
    }
    if (patch.original.size() != patch.replacement.size() ||
        std::memcmp(target, patch.original.data(), patch.original.size()) != 0) {
        std::fprintf(stderr, "[patch] %s: original bytes at %#llx differ, not applied\n",
                     patch.name.c_str(), static_cast<unsigned long long>(patch.rva));
        return false;
    }
    DWORD old_protect = 0;
    if (!VirtualProtect(target, patch.replacement.size(), PAGE_EXECUTE_READWRITE, &old_protect)) {
        return false;
    }
    std::memcpy(target, patch.replacement.data(), patch.replacement.size());
    DWORD ignored = 0;
    VirtualProtect(target, patch.replacement.size(), old_protect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), target, patch.replacement.size());
    std::fprintf(stderr, "[patch] %s applied at %#llx\n", patch.name.c_str(),
                 static_cast<unsigned long long>(patch.rva));
    return true;
}

// DIAG-097 (temporary): GT Sport's per-course view distances, GTRender::EnvironmentSetting::Misc
// (reflection field list at eboot 0x2be6ac0; defaults set by its constructor at 0x6c7370). The
// three calls of that constructor are redirected through a stub so every Misc instance is known;
// a thread logs their values when the game changes them (RYGT_ENV_DIAG=1) and can scale chosen
// fields (RYGT_ENV_SCALE=far=0.5,farPassFar=0.5,...) to test how draw counts react.
struct MiscField {
    const char* name;
    u32 offset;
};
constexpr MiscField kMiscFields[] = {
    {"far", 0x0},          {"baseNear", 0x4},        {"extra_far", 0xc},
    {"depthPassFar", 0x10}, {"envFar", 0x18},         {"localEnvFar", 0x1c},
    {"mirrorFar", 0x24},    {"reflectFar", 0x30},     {"farPassBaseNear", 0x34},
    {"farPassFar", 0x38},   {"occluderFar", 0x3c},    {"tree_far", 0xbc},
    {"tree_far_mirror", 0xc0}, {"tree_lod_off0", 0xc4}, {"tree_lod_off1", 0xc8},
    {"tree_lod_off2", 0xcc},
    // EnvironmentSetting::lod (LODSetting at 0x54c): discreteBase, smoothMeshLODBase,
    // nonOnboardLODScale. Only meaningful for instances built by the EnvironmentSetting
    // constructor (site 0).
    {"lod.discreteBase", 0x54c}, {"lod.smoothMeshLODBase", 0x550},
    {"lod.nonOnboardLODScale", 0x558},
};
constexpr u64 kMiscCtor = 0x6c7370;
constexpr u64 kMiscCtorCalls[] = {0x10de5b9, 0x6c712a, 0x6c8faa};
constexpr u64 kTextRvaShift = 0x10000;

using MiscCtorFn = void(PS4_SYSV_ABI*)(void*);
MiscCtorFn g_misc_ctor = nullptr;

struct MiscInstance {
    u8* base;
    int site;
    float seen[std::size(kMiscFields)];    // last value the game left there
    float written[std::size(kMiscFields)]; // last value written by the scale override
    std::string reject;
};
std::mutex g_misc_mutex;
std::vector<MiscInstance> g_misc;
float g_misc_scale[std::size(kMiscFields)];
bool g_misc_any_scale = false;

bool ReadableRange(const void* address, size_t size) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_COMMIT ||
        (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
        return false;
    }
    return static_cast<const u8*>(address) + size <=
           static_cast<const u8*>(info.BaseAddress) + info.RegionSize;
}

// Instances are often temporaries that the game frees again; one whose memory was reused must
// not be written. Only whole EnvironmentSettings whose near/far planes and four LODSettings all
// hold plausible values pass.
bool LooksLikeEnvironmentSetting(const u8* base, std::string* reject_log = nullptr) {
    if (!ReadableRange(base, 0x590)) {
        return false;
    }
    const auto f = [base](u32 offset) {
        float value;
        std::memcpy(&value, base + offset, 4);
        return value;
    };
    const auto in = [](float value, float lo, float hi) { return value >= lo && value <= hi; };
    if (!in(f(0x0), 10.0f, 1e6f) || !in(f(0x4), 0.001f, 10.0f) || !in(f(0x8), 0.01f, 10.0f) ||
        !in(f(0x14), 0.001f, 100.0f) || !in(f(0x18), 1.0f, 1e6f) || !in(f(0x1c), 1.0f, 1e7f) ||
        !in(f(0x24), 1.0f, 1e6f) || !in(f(0x28), 1.0f, 1e6f) || !in(f(0x2c), 0.001f, 100.0f) ||
        base[0x40] > 1) {
        return false;
    }
    for (const u32 lod : {0x54c, 0x55c, 0x56c, 0x57c}) {
        if (!in(f(lod), 0.0f, 100.0f) || !in(f(lod + 4), 0.0f, 100.0f) || base[lod + 8] > 1 ||
            !in(f(lod + 0xc), 0.001f, 10.0f)) {
            if (reject_log) {
                *reject_log = fmt::format("LODSetting at {:#x}: {} {} {} {}", lod, f(lod),
                                          f(lod + 4), base[lod + 8], f(lod + 0xc));
            }
            return false;
        }
    }
    return true;
}

void MiscMonitor() {
    // RYGT_ENV_AB=<seconds>: the scale is on and off for alternating periods of the same run,
    // with draws per frame and frame rate logged for each state (run-to-run differences of the
    // test race are larger than the effects measured).
    const char* ab_env = Common::DiagEnv("RYGT_ENV_AB");
    const u32 ab_ticks = ab_env ? static_cast<u32>(std::atof(ab_env) * 10.0) : 0;
    struct AbTotals {
        u64 draws, frames, ticks;
    } ab[2]{};
    u64 last_draws = Common::GetWorkCounters().draws.load();
    u64 last_frames = DebugState.GetFrameNum();
    for (u32 tick = 0;; ++tick) {
        Sleep(100);
        const bool scale_on = ab_ticks == 0 || (tick / ab_ticks) % 2 == 0;
        if (ab_ticks) {
            const u64 draws = Common::GetWorkCounters().draws.load();
            const u64 frames = DebugState.GetFrameNum();
            // The first tick after a switch still shows the previous state's frames.
            if (tick % ab_ticks != 0) {
                auto& totals = ab[scale_on ? 0 : 1];
                totals.draws += draws - last_draws;
                totals.frames += frames - last_frames;
                ++totals.ticks;
            }
            last_draws = draws;
            last_frames = frames;
            if (tick % 200 == 199) {
                const auto line = [](const AbTotals& t) {
                    return fmt::format("{:.0f} draws/frame, {:.1f} fps",
                                       double(t.draws) / std::max<u64>(1, t.frames),
                                       t.frames * 10.0 / std::max<u64>(1, t.ticks));
                };
                std::fprintf(stderr, "[env] DIAG-097 A/B: scaled %s; original %s\n",
                             line(ab[0]).c_str(), line(ab[1]).c_str());
                std::fflush(stderr);
                ab[0] = ab[1] = {};
            }
        }
        std::scoped_lock lock{g_misc_mutex};
        for (auto& instance : g_misc) {
            const size_t span = 0x590;
            std::string reject;
            if (instance.site != 0 || !LooksLikeEnvironmentSetting(instance.base, &reject)) {
                if (!reject.empty() && reject != instance.reject) {
                    std::fprintf(stderr, "[env] DIAG-097 misc %p rejected: %s\n", instance.base,
                                 reject.c_str());
                    instance.reject = reject;
                }
                continue;
            }
            // RYGT_ENV_SET=<hex offset>:<i|f|b><value>,...: fields of every EnvironmentSetting
            // set to a value (graphics-option experiments: shadow splitCount at 0x3d8/0x42c...).
            static const auto env_sets = [] {
                struct Set {
                    u32 offset;
                    char type;
                    double value;
                };
                std::vector<Set> sets;
                if (const char* env = std::getenv("RYGT_ENV_SET")) {
                    std::string text = env;
                    size_t at = 0;
                    while (at < text.size()) {
                        const size_t end = std::min(text.find(',', at), text.size());
                        const std::string item = text.substr(at, end - at);
                        const size_t colon = item.find(':');
                        if (colon != std::string::npos && colon + 2 <= item.size()) {
                            sets.push_back({static_cast<u32>(std::strtoul(item.substr(0, colon).c_str(), nullptr, 16)),
                                            item[colon + 1], std::atof(item.c_str() + colon + 2)});
                        }
                        at = end + 1;
                    }
                }
                return sets;
            }();
            for (const auto& set : env_sets) {
                if (!scale_on || set.offset + 4 > span) {
                    continue;
                }
                u8* field = instance.base + set.offset;
                if (set.type == 'i') {
                    s32 old;
                    std::memcpy(&old, field, 4);
                    const s32 value = static_cast<s32>(set.value);
                    if (old != value) {
                        std::fprintf(stderr, "[env] RYGT_ENV_SET %p+%#x: %d -> %d\n", instance.base, set.offset, old, value);
                        std::memcpy(field, &value, 4);
                    }
                } else if (set.type == 'f') {
                    float old;
                    std::memcpy(&old, field, 4);
                    const float value = static_cast<float>(set.value);
                    if (old != value) {
                        std::fprintf(stderr, "[env] RYGT_ENV_SET %p+%#x: %g -> %g\n", instance.base, set.offset, old, value);
                        std::memcpy(field, &value, 4);
                    }
                } else if (set.type == 'b') {
                    const u8 value = static_cast<u8>(set.value);
                    if (*field != value) {
                        std::fprintf(stderr, "[env] RYGT_ENV_SET %p+%#x: %u -> %u\n", instance.base, set.offset, *field, value);
                        *field = value;
                    }
                }
            }
            std::string changes;
            for (size_t i = 0; i < std::size(kMiscFields); ++i) {
                if (kMiscFields[i].offset >= span) {
                    continue;
                }
                float* field = reinterpret_cast<float*>(instance.base + kMiscFields[i].offset);
                const float value = *field;
                if (value == instance.written[i]) {
                    if (!scale_on) {
                        *field = instance.seen[i]; // A/B: back to the game's value
                        instance.written[i] = -1.0f;
                    }
                    continue; // our own override
                }
                if (value != instance.seen[i]) {
                    changes += fmt::format(" {}={}", kMiscFields[i].name, value);
                    instance.seen[i] = value;
                }
                // A copy of another instance's already scaled value is not scaled again.
                const bool copied_override = std::ranges::any_of(
                    g_misc, [&](const MiscInstance& other) { return other.written[i] == value; });
                if (scale_on && g_misc_scale[i] != 1.0f && value > 0.0f && !copied_override) {
                    const float scaled = value * g_misc_scale[i];
                    *field = scaled;
                    instance.written[i] = scaled;
                }
            }
            if (!changes.empty()) {
                std::fprintf(stderr, "[env] DIAG-097 misc %p (site %d):%s\n", instance.base,
                             instance.site, changes.c_str());
                std::fflush(stderr);
            }
            // RYGT_ENV_WATCH=<hex offset>: which code reads that field of the first instance that
            // gets a course's values (a hardware read/write watch, at once and again 40 s later).
            static const char* watch_env = Common::DiagEnv("RYGT_ENV_WATCH");
            static const u32 watch_offset =
                watch_env ? static_cast<u32>(std::strtoul(watch_env, nullptr, 16)) : 0;
            static u8* watched = nullptr;
            static u32 watch_tick = 0;
            float far_plane;
            std::memcpy(&far_plane, instance.base, 4);
            if (watch_env && !watched && far_plane != 1000.0f) {
                watched = instance.base;
                watch_tick = tick;
                Vulkan::RytyWatchWrites(reinterpret_cast<u64>(watched) + watch_offset, 3000,
                                        fmt::format("rw env+{:#x}", watch_offset));
            } else if (watch_env && watched == instance.base && tick == watch_tick + 400) {
                Vulkan::RytyWatchWrites(reinterpret_cast<u64>(watched) + watch_offset, 2000,
                                        fmt::format("rw env+{:#x} later", watch_offset));
            }
        }
    }
}

void RecordMisc(void* self, int site) {
    static std::once_flag started;
    std::call_once(started, [] {
        std::thread{MiscMonitor}.detach();
    });
    std::scoped_lock lock{g_misc_mutex};
    for (auto& instance : g_misc) {
        if (instance.base == self) {
            instance.site = site;
            std::fill(std::begin(instance.seen), std::end(instance.seen), -1.0f);
            return;
        }
    }
    MiscInstance instance{static_cast<u8*>(self), site, {}, {}, {}};
    std::fill(std::begin(instance.seen), std::end(instance.seen), -1.0f);
    std::fill(std::begin(instance.written), std::end(instance.written), -1.0f);
    g_misc.push_back(instance);
    // RYGT_ENV_WATCH_NEW=<hex offset>: reads and writes of that field of each new
    // EnvironmentSetting for its first 1.5 s (skipped while another watch runs).
    // (Not during boot: the watch logs through the renderer's log, which starts later.)
    static const auto first_record = std::chrono::steady_clock::now();
    if (const char* env = Common::DiagEnv("RYGT_ENV_WATCH_NEW");
        env && site == 0 && std::chrono::steady_clock::now() - first_record > std::chrono::seconds{30}) {
        const u32 offset = static_cast<u32>(std::strtoul(env, nullptr, 16));
        Vulkan::RytyWatchWrites(reinterpret_cast<u64>(self) + offset, 1500,
                                fmt::format("rw new env {} +{:#x}", self, offset));
    }
}

template <int Site>
void PS4_SYSV_ABI MiscCtorHook(void* self) {
    g_misc_ctor(self);
    RecordMisc(self, Site);
}

// Memory for jump stubs within +-2 GB of the game's code, so its rel32 calls can reach them.
u8* AllocNear(u8* base) {
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    const u64 granularity = system.dwAllocationGranularity;
    for (u64 distance = granularity; distance < (1ULL << 30); distance += granularity) {
        for (const s64 sign : {-1, 1}) {
            const u64 address =
                (reinterpret_cast<u64>(base) + sign * static_cast<s64>(distance)) & ~(granularity - 1);
            if (void* memory = VirtualAlloc(reinterpret_cast<void*>(address), 0x1000,
                                            MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)) {
                return static_cast<u8*>(memory);
            }
        }
    }
    return nullptr;
}

// GTS-002: the Lewis Hamilton Time Trial Challenge. v1.69 is Polyphony's end-of-service build: its
// main menu (gtmode.ad, setupDisplay) shows the mode only when the player's inventory says the free
// add-on was downloaded, i.e. when the KeyValue ints unlock_dlc_gm_LewisHamilton and
// dlc_gm_validLewisHamilton are 1. The servers that set them are gone, but all ten events, Lewis's
// ghosts and voice-overs are in the game's files. The KeyValue lookup that MKeyValue::getInt calls
// answers 1 for those two keys; every other key is untouched. RYGT_LEWIS=0 turns this off.
constexpr u64 kKeyValueGetIntCall = 0x212204; // call of the lookup inside MKeyValue::getInt
constexpr u64 kKeyValueLookup = 0xbae080;     // bool lookup(store, const char* key, s64* value)

using KeyValueLookupFn = bool(PS4_SYSV_ABI*)(void*, const char*, s64*);
KeyValueLookupFn g_key_value_lookup = nullptr;

// GTS-004: every special paint (matte, chrome, colour-shift and maker colours, bought with mileage
// points from the shut-down servers). The inventory (inventory.ad) asks KeyValue for
// unlock_sp_paint_<id>; those answer 1. Off by default (Lance: the paints don't show on the car);
// RYGT_PAINTS=1 turns it on.
bool g_unlock_lewis = false;
bool g_unlock_paints = false;

bool PS4_SYSV_ABI KeyValueLookupHook(void* store, const char* key, s64* value) {
    const bool found = g_key_value_lookup(store, key, value);
    if (key && g_unlock_lewis && (std::strcmp(key, "unlock_dlc_gm_LewisHamilton") == 0 ||
                                  std::strcmp(key, "dlc_gm_validLewisHamilton") == 0)) {
        *value = 1;
        return true;
    }
    if (key && g_unlock_paints && std::strncmp(key, "unlock_sp_paint", 15) == 0) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            std::fprintf(stderr, "[patch] GTS-004 special paints unlocked (first key %s)\n", key);
        }
        *value = 1;
        return true;
    }
    return found;
}

// Points the rel32 call at `call_rva` (which must call `target_rva`) at `hook` through a stub.
bool RedirectCall(u8* base, u8* stub, u64 call_rva, u64 target_rva, void* hook, const char* name) {
    u8* call = base + call_rva + kTextRvaShift;
    s32 rel = 0;
    std::memcpy(&rel, call + 1, 4);
    if (call[0] != 0xe8 || call + 5 + rel != base + target_rva + kTextRvaShift) {
        std::fprintf(stderr, "[patch] %s: call at %#llx differs, not applied\n", name,
                     static_cast<unsigned long long>(call_rva));
        return false;
    }
    const u8 jump[6] = {0xff, 0x25, 0, 0, 0, 0}; // jmp qword ptr [rip]
    std::memcpy(stub, jump, 6);
    std::memcpy(stub + 6, &hook, 8);
    FlushInstructionCache(GetCurrentProcess(), stub, 16);
    const s32 new_rel = static_cast<s32>(stub - (call + 5));
    DWORD old_protect = 0;
    if (!VirtualProtect(call, 5, PAGE_EXECUTE_READWRITE, &old_protect)) {
        return false;
    }
    std::memcpy(call + 1, &new_rel, 4);
    DWORD ignored = 0;
    VirtualProtect(call, 5, old_protect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), call, 5);
    std::fprintf(stderr, "[patch] %s applied at %#llx\n", name,
                 static_cast<unsigned long long>(call_rva));
    return true;
}

#include "ryty_gpu/ryty_gt7_hdr_status.inl"

void InstallLewisChallenge(u8* base) {
    const char* env = std::getenv("RYGT_LEWIS");
    const char* paints = std::getenv("RYGT_PAINTS");
    g_unlock_lewis = !(env && env[0] == '0');
    g_unlock_paints = paints && paints[0] == '1'; // off: they unlock but don't show on the car
    if (!g_unlock_lewis && !g_unlock_paints) {
        return;
    }
    u8* stub = AllocNear(base);
    if (!stub) {
        std::fprintf(stderr, "[patch] GTS-002: no memory near the game's code for the hook\n");
        return;
    }
    g_key_value_lookup = reinterpret_cast<KeyValueLookupFn>(base + kKeyValueLookup + kTextRvaShift);
    RedirectCall(base, stub, kKeyValueGetIntCall, kKeyValueLookup,
                 reinterpret_cast<void*>(&KeyValueLookupHook), "GTS-002 Lewis Hamilton Challenge");
}

// GTS-003: the Livery Editor offline. The main menu (gtmode.ad, setupDisplay) disables the Livery
// button whenever the game is offline (`disable: [!enable_menu || DisableLivery]`, enable_menu =
// online), although the editor itself has an offline mode that saves liveries locally. The menu
// script is patched while the game reads it: that one condition uses completed_playgo (true once the
// game is installed) instead of enable_menu, like the Garage button. The bytes are only changed
// where the whole original sequence (78 bytes of v1.69's GTMODE.ADC) is found in one 4 KB read.
// RYGT_LIVERY=0 turns this off.
struct ScriptPatch {
    const char* name;
    std::vector<u8> original;
    size_t offset; // of `replacement` inside `original`
    std::vector<u8> replacement;
};
std::vector<ScriptPatch> g_script_patches;

struct GuestIoVec {
    u8* base;
    u64 length;
};
// The game's file read (eboot 0x1be7930); its buffered streams refill 4 KB at a time from these
// three calls, which is how scripts are read.
constexpr u64 kFileRead = 0x1be7930;
constexpr u64 kStreamRefillCalls[] = {0x17626ef, 0x170f88e, 0x1762a38};
using FileReadFn = s64(PS4_SYSV_ABI*)(void*, GuestIoVec*);
FileReadFn g_file_read = nullptr;

// GTS-005: whole-script replacements. A file in <exe dir>\patches\scripts replaces the game script
// whose symbol table starts the same way (its own path names, which a patched script keeps). It is served chunk by chunk as the game reads the original, so it may not be larger
// than the original (adcpatch pads it with zeros to the original size).
struct ScriptReplacement {
    std::string name;
    std::vector<u8> data;
};
std::vector<ScriptReplacement> g_script_replacements;
std::mutex g_replacement_mutex;
std::vector<std::pair<void*, std::pair<const ScriptReplacement*, size_t>>> g_active_replacements;

// The bytes after the "ADCH012" header and the symbol count (which grows when a patch adds names):
// the start of the symbol table, i.e. the script's own path names.
const u8* ScriptKey(const u8* data, size_t size) {
    if (size < 8 + 4 + 120 || std::memcmp(data, "ADCH012", 8) != 0) {
        return nullptr;
    }
    size_t extra = 0;
    for (u8 b = data[8]; b & 0x80; b <<= 1) {
        ++extra;
    }
    return data + 8 + 1 + extra;
}

void ServeReplacement(void* file, u8* begin, s64 read) {
    std::scoped_lock lock{g_replacement_mutex};
    auto it = std::find_if(g_active_replacements.begin(), g_active_replacements.end(),
                           [file](const auto& entry) { return entry.first == file; });
    // FIX-105: the game stops reading a script once it has parsed it, so a replacement smaller than
    // the original (zero padding at the end) can stay active. The next script read through the same
    // stream object then got the padding (zeros) instead of its own bytes: GRIM.ADC with the
    // livery ban checks replaced left NP.ADC empty ("stream is not adhoc script code") and the game
    // crashed at boot. A read that starts with a script header is always a new script.
    if (it != g_active_replacements.end() && read >= 8 && std::memcmp(begin, "ADCH012", 8) == 0) {
        g_active_replacements.erase(it);
        it = g_active_replacements.end();
    }
    if (it == g_active_replacements.end()) {
        const u8* key = ScriptKey(begin, static_cast<size_t>(read));
        for (const auto& replacement : g_script_replacements) {
            const u8* own = ScriptKey(replacement.data.data(), replacement.data.size());
            if (key && own && std::memcmp(key, own, 120) == 0) {
                g_active_replacements.push_back({file, {&replacement, 0}});
                it = std::prev(g_active_replacements.end());
                std::fprintf(stderr, "[patch] GTS-005 serving %s\n", replacement.name.c_str());
                break;
            }
        }
        if (it == g_active_replacements.end()) {
            return;
        }
    }
    auto& [replacement, offset] = it->second;
    const size_t available = offset < replacement->data.size() ? replacement->data.size() - offset : 0;
    const size_t count = std::min<size_t>(available, static_cast<size_t>(read));
    std::memcpy(begin, replacement->data.data() + offset, count);
    std::memset(begin + count, 0, static_cast<size_t>(read) - count);
    offset += static_cast<size_t>(read);
    if (offset >= replacement->data.size()) {
        g_active_replacements.erase(it);
    }
}

void LoadScriptReplacements() {
    // FIX-230: Unicode and long-path safe.
    const std::wstring dir = Common::RytyExeDirW() + L"patches\\scripts\\";
    WIN32_FIND_DATAW found{};
    HANDLE find = FindFirstFileW((dir + L"*.ADC").c_str(), &found);
    if (find == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        FILE* f = _wfopen((dir + found.cFileName).c_str(), L"rb");
        if (!f) {
            continue;
        }
        std::string name;
        for (const wchar_t* c = found.cFileName; *c; ++c) {
            name += *c < 0x80 ? static_cast<char>(*c) : '?';
        }
        ScriptReplacement replacement{name, {}};
        std::fseek(f, 0, SEEK_END);
        replacement.data.resize(static_cast<size_t>(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        const size_t got = std::fread(replacement.data.data(), 1, replacement.data.size(), f);
        std::fclose(f);
        if (got == replacement.data.size() && got >= 128) {
            std::fprintf(stderr, "[patch] GTS-005 script replacement %s (%zu bytes)\n",
                         replacement.name.c_str(), got);
            g_script_replacements.push_back(std::move(replacement));
        }
    } while (FindNextFileW(find, &found));
    FindClose(find);
}

// FIX-110: whether a race (a race_* project) or the menus (gtmode, arcade and the other menu
// projects) run, from the project script the game loads when it switches. Overlays a race loads
// (dialogs, the meter, the pause menu) don't change it. The renderer skips draws for new
// pipelines only in races; in the menus some scenes draw once into a texture they keep showing.
std::atomic<bool> g_in_race{false};

void NoteProjectScript(const u8* data, s64 read) {
    if (read < 16 || std::memcmp(data, "ADCH012", 8) != 0) {
        return;
    }
    const std::string_view head{reinterpret_cast<const char*>(data),
                                static_cast<size_t>(std::min<s64>(read, 200))};
    const auto at = head.find("projects/gt7sp/");
    if (at == std::string_view::npos) {
        return;
    }
    std::string_view name = head.substr(at + 15);
    if (std::getenv("RYGT_LOG_PROJECTS")) {
        // DIAG-135: every project script the game loads, with the time.
        SYSTEMTIME now;
        GetLocalTime(&now);
        const std::string_view path = name.substr(0, name.find_first_of(".\0", 0));
        std::fprintf(stderr, "[projects] %02d:%02d:%02d.%03d %.*s\n", now.wHour, now.wMinute,
                     now.wSecond, now.wMilliseconds, static_cast<int>(std::min<size_t>(path.size(), 80)),
                     path.data());
    }
    name = name.substr(0, name.find('/'));
    static constexpr std::string_view menus[] = {
        "gtmode", "arcade", "campaign", "garage", "brand_central", "gallery", "livery2",
        "photo", "play_photo", "option", "ps_store", "car_select", "course_select",
        "event_setting", "preview", "car_preview", "online", "boot", "demo"};
    // "race" itself is the races' shared script, which the game also loads at boot (before the
    // gift car's thumbnail is drawn); a race runs one of the race_* projects.
    const bool race = name.starts_with("race_");
    if (!race && std::find(std::begin(menus), std::end(menus), name) == std::end(menus)) {
        return;
    }
    if (g_in_race.exchange(race) != race) {
        std::fprintf(stderr, "[patch] FIX-110 %.*s: %s\n", static_cast<int>(name.size()),
                     name.data(), race ? "race, new pipelines are built in the background"
                                       : "menus, draws wait for new pipelines");
    }
}

// DIAG-163: the first 60 scripts the game loads, with their source path and the time since the
// first one, so a boot that ends early shows how far its scripts got. RYGT_LOG_BOOT=0 turns it off.
void NoteBootScript(const u8* data, s64 read) {
    static const bool off = std::getenv("RYGT_LOG_BOOT") && std::getenv("RYGT_LOG_BOOT")[0] == '0';
    static std::atomic<int> count{0};
    static const auto first = std::chrono::steady_clock::now();
    if (off || read < 16 || std::memcmp(data, "ADCH012", 8) != 0) {
        return;
    }
    const int number = ++count;
    if (number > 60) {
        return;
    }
    // After the header: the symbol count (one byte, plus one more per leading 1 bit of the first,
    // as ScriptKey reads it), then length-prefixed names; the first is the compiled unit
    // (_build/.works/gt7sp/network/network.adl, out/.work/projects/gt7sp/gallery/_tmp_gallery.adi).
    size_t at = 9;
    for (u8 b = data[8]; b & 0x80; b <<= 1) {
        ++at;
    }
    std::string name = "?";
    if (at < size_t(read) && !(data[at] & 0x80) && at + 1 + data[at] <= size_t(read)) {
        name.assign(reinterpret_cast<const char*>(data + at + 1), data[at]);
    }
    for (const std::string_view prefix : {"_build/.works/", "out/.work/"}) {
        if (name.starts_with(prefix)) {
            name.erase(0, prefix.size());
        }
    }
    for (char& c : name) {
        c = (c >= 32 && c < 127) ? c : '.';
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - first)
                        .count();
    std::fprintf(stderr, "[boot] script %d (+%lld ms): %s\n", number, static_cast<long long>(ms),
                 name.c_str());
}

s64 PS4_SYSV_ABI StreamRefillHook(void* file, GuestIoVec* iov) {
    const s64 read = g_file_read(file, iov);
    if (read <= 0 || !iov || !iov->base) {
        return read;
    }
    u8* const begin = iov->base;
    u8* const end = begin + read;
    if (!g_script_replacements.empty()) {
        ServeReplacement(file, begin, read);
    }
    NoteBootScript(begin, read); // DIAG-163
    NoteProjectScript(begin, read); // FIX-110
    static const bool log_scripts = std::getenv("RYGT_LOG_SCRIPTS") != nullptr;
    if (log_scripts) {
        // DIAG-107: the start of each script chunk read (first 48 printable bytes).
        std::string text;
        for (s64 i = 0; i < std::min<s64>(read, 160) && text.size() < 120; ++i) {
            text += (begin[i] >= 32 && begin[i] < 127) ? static_cast<char>(begin[i]) : '.';
        }
        std::fprintf(stderr, "[scripts] file %p read %lld: %s\n", file, static_cast<long long>(read), text.c_str());
    }
    for (const auto& patch : g_script_patches) {
        u8* found = std::search(begin, end, patch.original.begin(), patch.original.end());
        if (found != end) {
            std::memcpy(found + patch.offset, patch.replacement.data(), patch.replacement.size());
            std::fprintf(stderr, "[patch] %s applied to the menu script\n", patch.name);
        }
    }
    return read;
}

void InstallScriptPatches(u8* base) {
    const char* livery = std::getenv("RYGT_LIVERY");
    if (!(livery && livery[0] == '0')) {
        // STRING_CONST 'disable', ARRAY_CONST 1, VARIABLE_EVAL enable_menu [7], UNARY_OPERATOR
        // __not__, LOGICAL_OR, VARIABLE_EVAL VariableManager.get, STRING_CONST
        // 'GameDemo::DisableLivery', CALL 1. enable_menu [7] becomes completed_playgo [2].
        g_script_patches.push_back(
            {"GTS-003 Livery Editor offline",
             ParseHex("940200001d811e940200003201000000940200002601000000811c070000009402000022160000"
                      "0000407001000094020000260300000076777800000000940200001d8142940200000601000000"),
             25, ParseHex("811302000000")});
    }
    if (!(std::getenv("RYGT_SCRIPTS") && std::getenv("RYGT_SCRIPTS")[0] == '0')) {
        LoadScriptReplacements();
    }
    // The hook stays in without patches too: FIX-110 reads which project the game loads.
    u8* stubs = AllocNear(base);
    if (!stubs) {
        std::fprintf(stderr, "[patch] GTS-003: no memory near the game's code for the hooks\n");
        return;
    }
    g_file_read = reinterpret_cast<FileReadFn>(base + kFileRead + kTextRvaShift);
    for (size_t i = 0; i < std::size(kStreamRefillCalls); ++i) {
        RedirectCall(base, stubs + i * 16, kStreamRefillCalls[i], kFileRead,
                     reinterpret_cast<void*>(&StreamRefillHook), "GTS-003 script reads");
    }
}

// DIAG-GT7-043: RYGT_DUMP_SCRIPTS=<folder> saves every Adhoc script GT7 loads, one file each, with
// the time it was loaded in stderr. Since version 15 a script's body is encrypted: the loader reads
// the "ADCH015" header and 16 bytes, makes a decrypting stream from them and reads through it. What
// that stream's read returns is saved, so this sees what the game itself gets.
constexpr u64 kGt7MakeScriptStream = 0x2bec690;     // void make(Stream** out, Stream** in, u8* iv)
constexpr u64 kGt7MakeScriptStreamCall = 0x2b7e712; // its call in the script loader
constexpr u64 kGt7ScriptStreamRead = 0x2beb2f0;     // u64 read(Stream*, u8* to, u64 size)
constexpr u8 kGt7ScriptStreamReadStart[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                            0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x78};
using MakeScriptStreamFn = void(PS4_SYSV_ABI*)(void**, void*, void*);
using ScriptStreamReadFn = u64(PS4_SYSV_ABI*)(void*, u8*, u64);
MakeScriptStreamFn g_make_script_stream = nullptr;
ScriptStreamReadFn g_script_stream_read = nullptr;
std::string g_script_dump_folder;
std::mutex g_script_dump_mutex;
std::vector<std::pair<void*, FILE*>> g_script_dumps;
int g_script_dump_count = 0;

// FIX-GT7-060: Arcade opens on the same course, weather and car at every launch: it keeps them in
// its context for one session only. As the game reads Arcade's scripts, this adds to their bytecode:
// the top screen writes the choices to a file, and the context's first read of the course and of the
// car takes them from it. Nothing changes on disk but that file. Every function touched is checked
// by name and sizes first; if one differs, the script stays as it is. RYGT_ARCADE_MEMORY=0 turns it off.
constexpr const char* kArcadeMemoryFile = "/APP_DATA/arcade_last.txt";

// Operands of each Adhoc instruction: s symbol, S count and symbols, i 32 bits, b 8 bits, h 16 bits,
// q 64 bits, F a code frame. An opcode without an entry stops the reading.
constexpr std::array<const char*, 76> kAdhocOperands = [] {
    std::array<const char*, 76> a{};
    a[2] = "s";    a[3] = "S";   a[4] = "s";   a[5] = "s";   a[6] = "i";   a[7] = "sS";  a[8] = "";
    a[9] = "i";    a[10] = "sF"; a[11] = "Sss"; a[12] = "i"; a[13] = "i";  a[14] = "i";  a[15] = "i";
    a[20] = "sF";  a[21] = "S";  a[22] = "";   a[23] = "";   a[25] = "i";  a[26] = "S";  a[28] = "s";
    a[29] = "s";   a[30] = "i";  a[31] = "";   a[32] = "i";  a[33] = "s";  a[34] = "s";  a[35] = "S";
    a[36] = "Si";  a[37] = "S";  a[38] = "Si"; a[39] = "s";  a[40] = "F";  a[41] = "F";  a[43] = "q";
    a[44] = "";    a[45] = "ib"; a[47] = "s";  a[48] = "s";  a[49] = "ii"; a[50] = "i";  a[51] = "";
    a[52] = "";    a[53] = "";   a[54] = "";   a[55] = "b";  a[56] = "";   a[57] = "";   a[58] = "i";
    a[59] = "q";   a[60] = "q";  a[61] = "";   a[62] = "";   a[63] = "i";  a[64] = "i";  a[65] = "b";
    a[66] = "";    a[67] = "i";  a[69] = "s";  a[70] = "i";  a[71] = "";   a[72] = "b";  a[73] = "b";
    a[74] = "h";   a[75] = "h";
    return a;
}();

void WriteVarint(std::vector<u8>& out, u32 value) {
    if (value < 0x80) {
        out.push_back(static_cast<u8>(value));
    } else if (value < 0x4000) {
        out.push_back(static_cast<u8>(0x80 | (value >> 8)));
        out.push_back(static_cast<u8>(value));
    } else if (value < 0x200000) {
        out.push_back(static_cast<u8>(0xc0 | (value >> 16)));
        out.push_back(static_cast<u8>(value >> 8));
        out.push_back(static_cast<u8>(value));
    } else {
        out.push_back(static_cast<u8>(0xe0 | (value >> 24)));
        out.push_back(static_cast<u8>(value >> 16));
        out.push_back(static_cast<u8>(value >> 8));
        out.push_back(static_cast<u8>(value));
    }
}

// Instructions to add to a function. New symbols are always written in full.
class AdhocCode {
public:
    AdhocCode(u32 at, u32 line) : at_{at}, line_{line} {}

    void Sym(std::string_view name) {
        bytes_.push_back(0);
        WriteVarint(bytes_, static_cast<u32>(name.size()));
        bytes_.insert(bytes_.end(), name.begin(), name.end());
        ++symbols_;
    }
    void Op(u8 op) {
        U32(line_);
        bytes_.push_back(op);
        ++count_;
    }
    void U32(u32 value) {
        for (int i = 0; i < 4; ++i) {
            bytes_.push_back(static_cast<u8>(value >> (i * 8)));
        }
    }
    void Var(u8 op, std::initializer_list<std::string_view> names, u32 slot) {
        Op(op);
        U32(static_cast<u32>(names.size()));
        for (const auto name : names) {
            Sym(name);
        }
        U32(slot);
    }
    void Eval(std::initializer_list<std::string_view> names, u32 slot) { Var(38, names, slot); }
    void Store(std::string_view name, u32 slot) {
        Var(36, {name}, slot);
        Op(57);
    }
    void Attr(u8 op, std::string_view name) {
        Op(op);
        U32(1);
        Sym(name);
    }
    void Get(std::string_view name) { Attr(37, name); }
    void Call(u32 args) {
        Op(6);
        U32(args);
    }
    void Method(std::string_view name) {
        Get(name);
        Call(0);
    }
    void Text(std::string_view text) {
        Op(29);
        Sym(text);
    }
    void Int(s32 value) {
        Op(12);
        U32(static_cast<u32>(value));
    }
    void Binary(std::string_view name) {
        Op(5);
        Sym(name);
    }
    void Element(s32 index, std::string_view array, u32 slot) {
        Eval({array}, slot);
        Int(index);
        Op(62);
    }
    // A jump to fill in later; returns where its target goes.
    size_t Jump(u8 op) {
        Op(op);
        U32(0);
        return bytes_.size() - 4;
    }
    u32 Here() const { return at_ + count_; }
    void Land(size_t jump, u32 target) {
        for (int i = 0; i < 4; ++i) {
            bytes_[jump + i] = static_cast<u8>(target >> (i * 8));
        }
    }
    // A jump to an instruction of the original function, which moves by what is added here.
    void JumpToOriginal(u8 op, u32 original) { later_.emplace_back(Jump(op), original); }
    void Finish() {
        for (const auto& [jump, original] : later_) {
            Land(jump, original + count_);
        }
    }

    const std::vector<u8>& Bytes() const { return bytes_; }
    u32 Count() const { return count_; }
    u32 Symbols() const { return symbols_; }

private:
    u32 at_;
    u32 line_;
    u32 count_{};
    u32 symbols_{};
    std::vector<u8> bytes_;
    std::vector<std::pair<size_t, u32>> later_;
};

// parts = the file's fields; the returned jumps are taken if there's no usable file.
std::vector<size_t> ArcadeReadParts(AdhocCode& code, u32 statics, u32 file, u32 parts) {
    std::vector<size_t> fail;
    code.Eval({"pdistd", "Exist", "pdistd::Exist"}, statics);
    code.Text(kArcadeMemoryFile);
    code.Call(1);
    fail.push_back(code.Jump(15));
    code.Eval({"pdistd", "ReadFile", "pdistd::ReadFile"}, statics + 1);
    code.Text(kArcadeMemoryFile);
    code.Call(1);
    code.Text("buffer");
    code.Op(62);
    code.Store("arcade_file", file);
    code.Eval({"arcade_file"}, file);
    code.Op(22);
    code.Binary("__ne__");
    fail.push_back(code.Jump(15));
    code.Eval({"arcade_file"}, file);
    code.Method("toString");
    code.Get("split");
    code.Text(",");
    code.Call(1);
    code.Store("arcade_parts", parts);
    code.Eval({"arcade_parts"}, parts);
    code.Get("size");
    code.Int(3);
    code.Binary("__ge__");
    fail.push_back(code.Jump(15));
    return fail;
}

struct ArcadeSite {
    std::string_view root;
    std::string_view method;
    u32 count, locals, statics, at;
    void (*make)(AdhocCode&, const ArcadeSite&);
    u32 local; // ctx or ctrl
    u32 after; // what follows the code once it found what it wanted
};

// if (!courseInfo_.nickName_) { <here> list = GetCourseList(); ... }
void MakeArcadeCourse(AdhocCode& code, const ArcadeSite& site) {
    const u32 parts = site.locals + 1;
    const auto fail = ArcadeReadParts(code, site.statics, site.locals, parts);
    code.Eval({"self"}, 0);
    code.Method("GetCourseList");
    code.Get("indexOf");
    code.Element(0, "arcade_parts", parts);
    code.Call(1);
    code.Int(1);
    code.Op(34);
    code.Sym("__uminus__");
    code.Binary("__gt__");
    const size_t unknown = code.Jump(15);
    code.Element(0, "arcade_parts", parts);
    code.Eval({"ctx"}, site.local);
    code.Get("courseInfo_");
    code.Attr(3, "nickName_");
    code.Op(57);
    code.Element(1, "arcade_parts", parts);
    code.Method("toUInt");
    code.Eval({"ctx"}, site.local);
    code.Get("courseInfo_");
    code.Attr(3, "weatherID_");
    code.Op(57);
    code.JumpToOriginal(13, site.after);
    for (const size_t jump : fail) {
        code.Land(jump, code.Here());
    }
    code.Land(unknown, code.Here());
}

// if (!playersState_.cp_) { <here> list = GetPresetCarList(); ... }
void MakeArcadeCar(AdhocCode& code, const ArcadeSite& site) {
    const u32 parts = site.locals + 1;
    const u32 iterator = site.locals + 2;
    const u32 car = site.locals + 3;
    const auto fail = ArcadeReadParts(code, site.statics, site.locals, parts);
    code.Eval({"self"}, 0);
    code.Method("GetPresetCarList");
    code.Get("iterator");
    code.Store("arcade_in", iterator);
    const u32 loop = code.Here();
    code.Eval({"arcade_in"}, iterator);
    code.Get("fetch_next");
    const size_t done = code.Jump(15);
    code.Eval({"arcade_in"}, iterator);
    code.Op(8);
    code.Store("arcade_car", car);
    code.Eval({"arcade_car"}, car);
    code.Method("getCarLabel");
    code.Element(2, "arcade_parts", parts);
    code.Binary("__eq__");
    code.Land(code.Jump(15), loop);
    code.Eval({"arcade_car"}, car);
    code.Eval({"ctx"}, site.local);
    code.Get("playersState_");
    code.Attr(3, "cp_");
    code.Op(57);
    code.JumpToOriginal(13, site.after);
    code.Land(done, code.Here());
    for (const size_t jump : fail) {
        code.Land(jump, code.Here());
    }
}

// At the top screen's refreshes, once ctrl is known and not nil: writes the current choices.
void MakeArcadeSave(AdhocCode& code, const ArcadeSite& site) {
    code.Eval({"ctrl"}, site.local);
    code.Method("GetPlayersState");
    code.Get("cp_");
    code.Op(22);
    code.Binary("__ne__");
    const size_t no_car = code.Jump(15);
    code.Eval({"pdistd", "WriteFile", "pdistd::WriteFile"}, site.statics);
    code.Text(kArcadeMemoryFile);
    code.Eval({"ctrl"}, site.local);
    code.Method("GetCourseInfo");
    code.Get("nickName_");
    code.Text(",");
    code.Eval({"ctrl"}, site.local);
    code.Method("GetCourseInfo");
    code.Get("weatherID_");
    code.Method("toString");
    code.Text(",");
    code.Eval({"ctrl"}, site.local);
    code.Method("GetPlayersState");
    code.Get("cp_");
    code.Method("getCarLabel");
    code.Op(30);
    code.U32(5);
    code.Call(2);
    code.Op(54);
    code.Land(no_car, code.Here());
}

constexpr std::string_view kArcadeControl = "gt7/menu/app/arcade/control.swift";
constexpr std::string_view kArcadeTop = "gt7/menu/app/arcade/events/TopRootEvent.swift";
const ArcadeSite kArcadeSites[] = {
    {kArcadeControl, "GetCourseInfo", 74, 7, 4, 24, MakeArcadeCourse, 2, 71},
    {kArcadeControl, "GetPlayersState", 46, 4, 2, 24, MakeArcadeCar, 2, 43},
    {kArcadeTop, "updateCourseInfo", 160, 9, 20, 16, MakeArcadeSave, 2, 0},
    {kArcadeTop, "updateCarInfo", 259, 15, 18, 16, MakeArcadeSave, 3, 0},
};

// Copies a script file, changing the symbol indices and jumps that the added code moves.
class ArcadeScriptPatch {
public:
    explicit ArcadeScriptPatch(const std::vector<u8>& in) : in_{in} {}

    // False if the file isn't the one with Arcade or anything differs.
    bool Run(std::vector<u8>& out) {
        Varint();
        const u32 frames = Varint();
        out_.reserve(in_.size() + 16384);
        out_.insert(out_.end(), in_.begin(), in_.begin() + pos_);
        for (u32 i = 0; i < frames && ok_; ++i) {
            const size_t start = pos_;
            names_.clear();
            map_.clear();
            mapped_ = 0;
            root_ = RootSource();
            write_ = root_ == kArcadeControl || root_ == kArcadeTop;
            Frame({}, true);
            if (!ok_ || pos_ + 16 > in_.size()) {
                return false;
            }
            if (!write_) {
                out_.insert(out_.end(), in_.begin() + start, in_.begin() + pos_);
            }
            out_.insert(out_.end(), in_.begin() + pos_, in_.begin() + pos_ + 16);
            pos_ += 16;
        }
        if (!ok_ || placed_ != std::size(kArcadeSites)) {
            return false;
        }
        out.swap(out_);
        return true;
    }

    u32 Placed() const { return placed_; }

private:
    u8 U8() {
        if (pos_ >= in_.size()) {
            ok_ = false;
            return 0;
        }
        return in_[pos_++];
    }
    u32 U32() {
        if (pos_ + 4 > in_.size()) {
            ok_ = false;
            return 0;
        }
        u32 value;
        std::memcpy(&value, in_.data() + pos_, 4);
        pos_ += 4;
        return value;
    }
    u32 Varint() {
        u32 value = U8();
        if (value & 0x80) {
            int more = 0;
            u32 mask = 0x80;
            while (value & mask) {
                ++more;
                mask >>= 1;
            }
            value &= mask - 1;
            for (int i = 0; i < more; ++i) {
                value = (value << 8) | U8();
            }
        }
        return value;
    }
    void Out32(u32 value) {
        if (write_) {
            for (int i = 0; i < 4; ++i) {
                out_.push_back(static_cast<u8>(value >> (i * 8)));
            }
        }
    }
    u32 Copy32() {
        const u32 value = U32();
        Out32(value);
        return value;
    }
    void Copy(size_t size) {
        if (pos_ + size > in_.size()) {
            ok_ = false;
            return;
        }
        if (write_) {
            out_.insert(out_.end(), in_.begin() + pos_, in_.begin() + pos_ + size);
        }
        pos_ += size;
    }
    std::string_view RootSource() {
        const size_t start = pos_;
        std::string_view name;
        if (pos_ < in_.size() && in_[pos_] == 0) {
            ++pos_;
            const u32 size = Varint();
            if (pos_ + size <= in_.size()) {
                name = {reinterpret_cast<const char*>(in_.data() + pos_), size};
            }
        }
        pos_ = start;
        return name;
    }
    // A frame of the file numbers its symbols from 1, as each is first written in full.
    std::string_view Symbol() {
        if (pos_ < in_.size() && in_[pos_] != 0) {
            const u32 index = Varint();
            if (!write_) {
                return {};
            }
            if (index == 0 || index > map_.size()) {
                ok_ = false;
                return {};
            }
            WriteVarint(out_, map_[index - 1]);
            return names_[index - 1];
        }
        U8();
        const u32 size = Varint();
        if (pos_ + size > in_.size() || size > 4096) {
            ok_ = false;
            return {};
        }
        const std::string_view name{reinterpret_cast<const char*>(in_.data() + pos_), size};
        pos_ += size;
        if (write_) {
            map_.push_back(++mapped_);
            names_.push_back(name);
            out_.push_back(0);
            WriteVarint(out_, size);
            out_.insert(out_.end(), name.begin(), name.end());
        }
        return name;
    }
    std::string_view Symbols() {
        const u32 count = Copy32();
        if (count > 4096) {
            ok_ = false;
            return {};
        }
        std::string_view first;
        for (u32 i = 0; i < count && ok_; ++i) {
            const auto name = Symbol();
            if (i == 0) {
                first = name;
            }
        }
        return first;
    }
    void Frame(std::string_view method, bool root) {
        Symbol();
        Copy(1);
        for (int list = 0; list < 2; ++list) {
            const u32 count = Copy32();
            for (u32 i = 0; i < count && ok_; ++i) {
                Symbol();
                Copy32();
            }
        }
        Copy32();
        const u32 stack = U32();
        const u32 locals = U32();
        const u32 statics = U32();
        const u32 count = U32();
        if (!ok_ || count > 0x400000) {
            ok_ = false;
            return;
        }
        const ArcadeSite* site = nullptr;
        if (write_ && !root) {
            for (const auto& candidate : kArcadeSites) {
                if (candidate.root == root_ && candidate.method == method &&
                    candidate.count == count && candidate.locals == locals &&
                    candidate.statics == statics) {
                    site = &candidate;
                }
            }
        }
        // The frame's header comes first, so the added code is made now.
        AdhocCode code{site ? site->at : 0, 0};
        if (site) {
            site->make(code, *site);
            code.Finish();
        }
        Out32(stack + (site ? 8 : 0));
        Out32(locals + (site ? 4 : 0));
        Out32(statics + (site ? 2 : 0));
        Out32(count + code.Count());
        for (u32 i = 0; i < count && ok_; ++i) {
            if (site && i == site->at) {
                // Each site's first instruction is a variable's evaluation.
                if (pos_ + 5 > in_.size() || in_[pos_ + 4] != 38) {
                    ok_ = false;
                    return;
                }
                out_.insert(out_.end(), code.Bytes().begin(), code.Bytes().end());
                mapped_ += code.Symbols();
                ++placed_;
            }
            Copy32();
            const u8 op = U8();
            if (write_) {
                out_.push_back(op);
            }
            const char* operands = op < kAdhocOperands.size() ? kAdhocOperands[op] : nullptr;
            if (!operands) {
                ok_ = false;
                return;
            }
            const bool jump = op == 13 || op == 14 || op == 15 || op == 70 || op == 63 || op == 64 ||
                              op == 32;
            std::string_view name;
            for (const char* operand = operands; *operand && ok_; ++operand) {
                switch (*operand) {
                case 's':
                    name = Symbol();
                    break;
                case 'S':
                    name = Symbols();
                    break;
                case 'i':
                    if (jump && site) {
                        const u32 target = U32();
                        Out32(target > site->at ? target + code.Count() : target);
                    } else {
                        Copy32();
                    }
                    break;
                case 'b':
                    Copy(1);
                    break;
                case 'h':
                    Copy(2);
                    break;
                case 'q':
                    Copy(8);
                    break;
                case 'F':
                    Frame(name, false);
                    break;
                default:
                    ok_ = false;
                    break;
                }
            }
        }
    }

    const std::vector<u8>& in_;
    std::vector<u8> out_;
    size_t pos_{};
    bool ok_{true};
    bool write_{};
    std::string_view root_;
    std::vector<std::string_view> names_;
    std::vector<u32> map_;
    u32 mapped_{};
    u32 placed_{};
};

// The game reads its scripts in small pieces through this stream: the first read takes the whole
// script, and the following ones are served from the copy, changed if it's the one with Arcade.
struct ScriptCopy {
    void* stream;
    std::vector<u8> data;
    size_t served;
    bool loaded;
};
bool g_arcade_memory = false;
std::vector<ScriptCopy> g_script_copies;

void PS4_SYSV_ABI MakeScriptStreamHook(void** out, void* in, void* iv) {
    g_make_script_stream(out, in, iv);
    std::scoped_lock lock{g_script_dump_mutex};
    if (g_arcade_memory) {
        const auto copy = std::find_if(g_script_copies.begin(), g_script_copies.end(),
                                       [out](const auto& c) { return c.stream == *out; });
        if (copy != g_script_copies.end()) {
            g_script_copies.erase(copy);
        }
        g_script_copies.push_back({*out, {}, 0, false});
    }
    if (g_script_dump_folder.empty()) {
        return;
    }
    const auto entry = std::find_if(g_script_dumps.begin(), g_script_dumps.end(),
                                    [out](const auto& dump) { return dump.first == *out; });
    if (entry != g_script_dumps.end()) {
        std::fclose(entry->second);
        g_script_dumps.erase(entry);
    }
    const std::string path =
        fmt::format("{}\\{:05}.adc", g_script_dump_folder, ++g_script_dump_count);
    SYSTEMTIME now{};
    GetSystemTime(&now);
    std::fprintf(stderr, "[scripts] %02d:%02d:%02d.%03d UTC #%d\n", now.wHour, now.wMinute,
                 now.wSecond, now.wMilliseconds, g_script_dump_count);
    if (FILE* file = std::fopen(path.c_str(), "wb")) {
        g_script_dumps.emplace_back(*out, file);
    }
}

u64 ReadScriptCopy(ScriptCopy& copy, u8* to, u64 size) {
    if (!copy.loaded) {
        copy.loaded = true;
        static constexpr u64 kChunk = 1u << 20;
        std::vector<u8> data;
        for (;;) {
            const size_t have = data.size();
            data.resize(have + kChunk);
            const s32 got = static_cast<s32>(g_script_stream_read(copy.stream, data.data() + have, kChunk));
            data.resize(have + static_cast<size_t>(std::max(got, 0)));
            if (got <= 0) {
                break;
            }
        }
        ArcadeScriptPatch patch{data};
        std::vector<u8> changed;
        if (patch.Run(changed)) {
            std::fprintf(stderr, "[patch] FIX-GT7-060 Arcade remembers its choices (%zu bytes added)\n",
                         changed.size() - data.size());
            data.swap(changed);
        } else if (patch.Placed() != 0) {
            std::fprintf(stderr, "[patch] FIX-GT7-060: Arcade's scripts differ, not applied\n");
        }
        copy.data.swap(data);
    }
    const u64 count = std::min<u64>(size, copy.data.size() - copy.served);
    if (to && count != 0) {
        std::memcpy(to, copy.data.data() + copy.served, static_cast<size_t>(count));
    }
    copy.served += static_cast<size_t>(count);
    return count;
}

u64 PS4_SYSV_ABI ScriptStreamReadHook(void* stream, u8* to, u64 size) {
    u64 read = 0;
    bool from_copy = false;
    if (g_arcade_memory) {
        std::scoped_lock lock{g_script_dump_mutex};
        const auto copy = std::find_if(g_script_copies.begin(), g_script_copies.end(),
                                       [stream](const auto& c) { return c.stream == stream; });
        if (copy != g_script_copies.end()) {
            read = ReadScriptCopy(*copy, to, size);
            from_copy = true;
        }
    }
    if (!from_copy) {
        read = g_script_stream_read(stream, to, size);
    }
    const s32 count = static_cast<s32>(read);
    if (count > 0 && to && !g_script_dump_folder.empty()) {
        std::scoped_lock lock{g_script_dump_mutex};
        const auto entry = std::find_if(g_script_dumps.begin(), g_script_dumps.end(),
                                        [stream](const auto& dump) { return dump.first == stream; });
        if (entry != g_script_dumps.end()) {
            std::fwrite(to, 1, static_cast<size_t>(count), entry->second);
            std::fflush(entry->second);
        }
    }
    return read;
}

void InstallGt7ScriptDump(u8* base) {
    const char* folder = std::getenv("RYGT_DUMP_SCRIPTS");
    const bool dump = folder && folder[0];
    const char* memory = std::getenv("RYGT_ARCADE_MEMORY");
    u8* const read = base + kGt7ScriptStreamRead + kTextRvaShift;
    constexpr size_t kMoved = sizeof(kGt7ScriptStreamReadStart);
    // Only GT7 has these bytes here: FIX-GT7-060 stays silent elsewhere.
    if (std::memcmp(read, kGt7ScriptStreamReadStart, kMoved) != 0) {
        if (dump) {
            std::fprintf(stderr, "[patch] DIAG-GT7-043: original bytes differ, not applied\n");
        }
        return;
    }
    g_arcade_memory = !(memory && memory[0] == '0');
    if (!dump && !g_arcade_memory) {
        return;
    }
    u8* stubs = AllocNear(base);
    if (!stubs) {
        std::fprintf(stderr, "[patch] DIAG-GT7-043: no memory near the game's code for the hooks\n");
        g_arcade_memory = false;
        return;
    }
    if (dump) {
        g_script_dump_folder = folder;
        CreateDirectoryA(folder, nullptr);
    }
    g_make_script_stream =
        reinterpret_cast<MakeScriptStreamFn>(base + kGt7MakeScriptStream + kTextRvaShift);
    if (!RedirectCall(base, stubs, kGt7MakeScriptStreamCall, kGt7MakeScriptStream,
                      reinterpret_cast<void*>(&MakeScriptStreamHook), "DIAG-GT7-043 script loads")) {
        return;
    }
    // The read's first instructions move to a trampoline; they don't depend on where they run.
    const u8 jump[6] = {0xff, 0x25, 0, 0, 0, 0}; // jmp qword ptr [rip]
    u8* const trampoline = stubs + 16;
    std::memcpy(trampoline, read, kMoved);
    std::memcpy(trampoline + kMoved, jump, 6);
    const u8* const resume = read + kMoved;
    std::memcpy(trampoline + kMoved + 6, &resume, 8);
    FlushInstructionCache(GetCurrentProcess(), trampoline, kMoved + 14);
    g_script_stream_read = reinterpret_cast<ScriptStreamReadFn>(trampoline);
    DWORD old_protect = 0;
    if (!VirtualProtect(read, kMoved, PAGE_EXECUTE_READWRITE, &old_protect)) {
        return;
    }
    const void* const hook = reinterpret_cast<void*>(&ScriptStreamReadHook);
    std::memcpy(read, jump, 6);
    std::memcpy(read + 6, &hook, 8);
    std::memset(read + 14, 0x90, kMoved - 14);
    DWORD ignored = 0;
    VirtualProtect(read, kMoved, old_protect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), read, kMoved);
    std::fprintf(stderr, "[patch] DIAG-GT7-043 script reads applied\n");
}

void InstallMiscHooks(u8* base) {
    const char* diag = Common::DiagEnv("RYGT_ENV_DIAG");
    const char* scale = std::getenv("RYGT_ENV_SCALE");
    if (!(diag && diag[0] == '1') && !scale && !std::getenv("RYGT_ENV_SET")) {
        return;
    }
    std::fill(std::begin(g_misc_scale), std::end(g_misc_scale), 1.0f);
    if (scale) {
        std::string list{scale};
        for (size_t start = 0; start < list.size();) {
            const size_t end = std::min(list.find(',', start), list.size());
            const std::string item = list.substr(start, end - start);
            const size_t eq = item.find('=');
            for (size_t i = 0; eq != std::string::npos && i < std::size(kMiscFields); ++i) {
                if (item.compare(0, eq, kMiscFields[i].name) == 0 &&
                    std::strlen(kMiscFields[i].name) == eq) {
                    g_misc_scale[i] = std::strtof(item.c_str() + eq + 1, nullptr);
                    g_misc_any_scale = true;
                    std::fprintf(stderr, "[env] scaling %s by %g\n", kMiscFields[i].name,
                                 g_misc_scale[i]);
                }
            }
            start = end + 1;
        }
    }
    u8* stubs = AllocNear(base);
    if (!stubs) {
        std::fprintf(stderr, "[env] no memory near the game's code for hook stubs\n");
        return;
    }
    g_misc_ctor = reinterpret_cast<MiscCtorFn>(base + kMiscCtor + kTextRvaShift);
    void* hooks[] = {reinterpret_cast<void*>(&MiscCtorHook<0>),
                     reinterpret_cast<void*>(&MiscCtorHook<1>),
                     reinterpret_cast<void*>(&MiscCtorHook<2>)};
    for (size_t i = 0; i < std::size(kMiscCtorCalls); ++i) {
        u8* call = base + kMiscCtorCalls[i] + kTextRvaShift;
        s32 rel = 0;
        std::memcpy(&rel, call + 1, 4);
        if (call[0] != 0xe8 || call + 5 + rel != base + kMiscCtor + kTextRvaShift) {
            std::fprintf(stderr, "[env] call at %#llx is not the Misc constructor, not hooked\n",
                         static_cast<unsigned long long>(kMiscCtorCalls[i]));
            continue;
        }
        u8* stub = stubs + i * 16;
        const u8 jump[6] = {0xff, 0x25, 0, 0, 0, 0}; // jmp qword ptr [rip]
        std::memcpy(stub, jump, 6);
        std::memcpy(stub + 6, &hooks[i], 8);
        const s32 new_rel = static_cast<s32>(stub - (call + 5));
        DWORD old_protect = 0;
        VirtualProtect(call, 5, PAGE_EXECUTE_READWRITE, &old_protect);
        std::memcpy(call + 1, &new_rel, 4);
        DWORD ignored = 0;
        VirtualProtect(call, 5, old_protect, &ignored);
        FlushInstructionCache(GetCurrentProcess(), call, 5);
    }
    FlushInstructionCache(GetCurrentProcess(), stubs, 0x1000);
    std::fprintf(stderr, "[env] DIAG-097 Misc constructor hooks installed\n");
}

} // namespace

// DIAG-106: the game's text-entry routine (eboot 0xf34700, this, pad slot) opens the keyboard
// dialog or the inline keyboard. It refuses when the slot is above 3, a keyboard is already open
// ([this+0xf4]) or the slot's user id (pad table 0x31a6bc0, 0x550 bytes per slot, +0x40) is -1.
// Its two calls are logged with those values and the result.
constexpr u64 kOpenTextEntry = 0xf34700;
constexpr u64 kOpenTextEntryCalls[] = {0xe8be5e, 0xfbff0d};
constexpr u64 kPadTable = 0x31a6bc0;
using OpenTextEntryFn = s32(PS4_SYSV_ABI*)(u8*, s32);
OpenTextEntryFn g_open_text_entry = nullptr;
u8* g_game_base = nullptr;

s32 PS4_SYSV_ABI OpenTextEntryHook(u8* self, s32 slot) {
    const u8* table = g_game_base + kPadTable + kTextRvaShift;
    s32 users[4];
    for (int i = 0; i < 4; ++i) {
        std::memcpy(&users[i], table + i * 0x550 + 0x40, 4);
    }
    s32 open_state = 0;
    std::memcpy(&open_state, self + 0xf4, 4);
    const s32 result = g_open_text_entry(self, slot);
    std::fprintf(stderr,
                 "[ime] text entry: slot %d, users %d %d %d %d, open state %d, dialog mode %d -> %d\n",
                 slot, users[0], users[1], users[2], users[3], open_state, self[0x108], result);
    return result;
}

void InstallTextEntryDiag(u8* base) {
    g_game_base = base;
    u8* stubs = AllocNear(base);
    if (!stubs) {
        return;
    }
    g_open_text_entry = reinterpret_cast<OpenTextEntryFn>(base + kOpenTextEntry + kTextRvaShift);
    for (size_t i = 0; i < std::size(kOpenTextEntryCalls); ++i) {
        RedirectCall(base, stubs + i * 16, kOpenTextEntryCalls[i], kOpenTextEntry,
                     reinterpret_cast<void*>(&OpenTextEntryHook), "DIAG-106 text entry");
    }
}

// GTS-006: custom decals offline. <exe dir>\decals\*.svg are copied to /APP_CACHE/decals (the
// game's <exe dir>\temp0\decals) as "<id>_<name>.svg"; the patched GRIM script lists them as the
// user's decals and copies each into the decal cache under that id. The id is a hash of the file
// name, so a decal keeps its id (and liveries that use it keep working) across runs.
void SyncDecals() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir{exe};
    dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
    const std::wstring src = dir + L"decals\\";
    const std::wstring dst = dir + L"temp0\\decals\\";
    CreateDirectoryW(src.c_str(), nullptr);
    CreateDirectoryW((dir + L"temp0").c_str(), nullptr);
    CreateDirectoryW(dst.c_str(), nullptr);
    WIN32_FIND_DATAW found{};
    HANDLE find = FindFirstFileW((dst + L"*.svg").c_str(), &found);
    if (find != INVALID_HANDLE_VALUE) {
        do {
            DeleteFileW((dst + found.cFileName).c_str());
        } while (FindNextFileW(find, &found));
        FindClose(find);
    }
    find = FindFirstFileW((src + L"*.svg").c_str(), &found);
    if (find == INVALID_HANDLE_VALUE) {
        return;
    }
    int count = 0;
    do {
        if (found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }
        const std::wstring file{found.cFileName};
        u32 hash = 2166136261u;
        std::string title;
        for (wchar_t c : file.substr(0, file.size() - 4)) {
            const wchar_t lower = towlower(c);
            hash = (hash ^ static_cast<u32>(lower)) * 16777619u;
            title += (c < 128 && (iswalnum(c) || c == L' ' || c == L'-' || c == L'_'))
                         ? static_cast<char>(c) : '_';
        }
        const u32 id = 2000000000u + hash % 100000000u;
        const std::string name = std::to_string(id) + "_" + title + ".svg";
        if (CopyFileW((src + file).c_str(), (dst + std::wstring(name.begin(), name.end())).c_str(),
                      FALSE)) {
            ++count;
        }
    } while (FindNextFileW(find, &found));
    FindClose(find);
    std::fprintf(stderr, "[patch] GTS-006 %d custom decal(s) from decals\\\n", count);
}

// GTS-007: wet-road reflections off. In the rain GT Sport renders the scene a second time into a
// 1024x512 reflection of the wet track (ReflectSetting.enableIfWet of the course settings, checked
// at eboot 0xab27d2): ~24% of a Tokyo rain frame's draws. Ignoring "wet" leaves the pass out
// (Tokyo rain, 3 races each: 40.6 fps vs 31.1). Off when GPU.wet_reflections is false in
// user\custom_configs\CUSA03220.json (the launcher's option) or RYGT_WET_REFLECTIONS=0.
bool WetReflectionsWanted() {
    if (const char* env = std::getenv("RYGT_WET_REFLECTIONS")) {
        return env[0] != '0';
    }
    // FIX-230: Unicode and long-path safe.
    const std::string text = Common::RytyReadGameConfig();
    if (text.empty()) {
        return true;
    }
    const size_t key = text.find("\"wet_reflections\"");
    if (key == std::string::npos) {
        return true;
    }
    const size_t value = text.find_first_not_of(" \t\r\n:", key + 17);
    return value == std::string::npos || text.compare(value, 5, "false") != 0;
}

// GTS-008: rear-view mirror at half rate. The frame render (eboot 0x10ff... ) renders the back
// mirror in a block it skips when the mirror object is null or the scene's mirror-off flag
// ([r13+0x5a8]) is set: "cmp byte ptr [r13+0x5a8], 0; jne skip" at eboot 0x110011a. The cmp becomes
// a call to a stub that also skips every other time, so the mirror is redrawn every second frame
// (its picture is kept in between). GPU.mirror_half_rate=true (launcher) or RYGT_MIRROR_HALF=1.
bool ConfigFlag(const char* key, bool fallback) {
    // FIX-230: Unicode and long-path safe.
    const std::string text = Common::RytyReadGameConfig();
    if (text.empty()) {
        return fallback;
    }
    const std::string quoted = std::string{"\""} + key + "\"";
    const size_t at = text.find(quoted);
    if (at == std::string::npos) {
        return fallback;
    }
    const size_t value = text.find_first_not_of(" \t\r\n:", at + quoted.size());
    if (value == std::string::npos) {
        return fallback;
    }
    return text.compare(value, 4, "true") == 0;
}

// A setting of the GT Sport config as written (a string without its quotes, a number), or "".
static std::string ConfigValue(const char* key) {
    // FIX-230: Unicode and long-path safe.
    const std::string text = Common::RytyReadGameConfig();
    if (text.empty()) {
        return {};
    }
    const std::string quoted = std::string{"\""} + key + "\"";
    const size_t at = text.find(quoted);
    if (at == std::string::npos) {
        return {};
    }
    const size_t value = text.find_first_not_of(" \t\r\n:", at + quoted.size());
    if (value == std::string::npos) {
        return {};
    }
    if (text[value] == '"') {
        const size_t end = text.find('"', value + 1);
        return end == std::string::npos ? std::string{} : text.substr(value + 1, end - value - 1);
    }
    const size_t end = text.find_first_of(",}\r\n", value);
    return text.substr(value, end == std::string::npos ? std::string::npos : end - value);
}

// GTS-009: the camera's aspect ratio for a screen wider (or narrower) than 16:9, from
// GPU.aspect_ratio: "auto" (the shape of the game's window or screen, from the display settings),
// "21:9", "32:9", another width:height or a number. 0 when the game's own 16:9 applies.
// RYGT_ASPECT overrides it.
namespace Vulkan {
float RytyDisplayAspect() {
    static const float aspect = [] {
        std::string value = ConfigValue("aspect_ratio");
        if (const char* env = std::getenv("RYGT_ASPECT")) {
            value = env;
        }
        double ratio = 0.0;
        if (value == "auto") {
            const bool full = ConfigValue("full_screen") == "true";
            const bool exclusive = ConfigValue("full_screen_mode") == "Fullscreen";
            double width = std::atof(ConfigValue("window_width").c_str());
            double height = std::atof(ConfigValue("window_height").c_str());
            if (full && !exclusive) { // borderless: the desktop's resolution
                width = GetSystemMetrics(SM_CXSCREEN);
                height = GetSystemMetrics(SM_CYSCREEN);
            }
            ratio = height > 0.0 ? width / height : 0.0;
        } else if (const size_t colon = value.find(':'); colon != std::string::npos) {
            const double height = std::atof(value.substr(colon + 1).c_str());
            ratio = height > 0.0 ? std::atof(value.substr(0, colon).c_str()) / height : 0.0;
        } else {
            ratio = std::atof(value.c_str());
        }
        if (!(ratio > 0.5 && ratio < 6.0) || std::abs(ratio - 16.0 / 9.0) < 0.01) {
            return 0.0f;
        }
        return static_cast<float>(ratio);
    }();
    return aspect;
}
} // namespace Vulkan

// GTS-009: the race camera's view is set up with a 16:9 aspect
// (eboot 0x5e717f: mov dword [rbx+0x8d464], 1.7777778f); the wider aspect widens its view.
static void InstallWideCamera(u8* base) {
    const float aspect = Vulkan::RytyDisplayAspect();
    if (aspect <= 0.0f) {
        return;
    }
    const float original = 16.0f / 9.0f;
    std::vector<u8> from(4), to(4);
    std::memcpy(from.data(), &original, 4);
    std::memcpy(to.data(), &aspect, 4);
    if (Apply(base, Patch{0x5f7185, from, to, fmt::format("GTS-009 camera aspect {:.3f}", aspect)})) {
        std::fprintf(stderr, "[patch] GTS-009: picture stretched to the window\n");
    }
}

// DLSS-GT7-005: the game's own 4K interface images with temporal DLSS. GT7 (CUSA24769 v1.71) keeps
// a flag ([0x1462fda71]) that says its output is high resolution: set, the interface loads its
// /4k/ images and draws its maps larger (513x326 for 308x196); clear, it rewrites /4k/ paths to
// /2k/. The temporal DLSS path draws the HUD at twice the game's size (NativeRenderer::HudScale),
// where the 2k images were merely magnified: icons, gauges and maps came out soft beside a build
// that selects the 4K ones (measured on 2026-11-14: bottom HUD detail 11.7 -> 15.8). The flag is
// forced where the game computes it (its first value and the one after a display change, keeping
// the latter's comparison) and set now. The idea and the three addresses are Gabarsolon's, from
// RyGT-Team/shadgt7 PR 42 (InstallHdUiAssetsGt7); rewritten here on this file's Apply.
// Only where temporal DLSS is asked for and can run, decided as ryty_upscale.h's TemporalWanted
// and TemporalRequested decide it, from the settings file (the renderer's settings are not loaded
// yet): NVIDIA DLSS chosen, Streamline's sl.dlss.dll there and not switched off, no frame
// generation, not the PS4 Pro mode on a 4K TV (the game picks its own images there). What cannot
// be known here is whether Streamline will start: a GPU that falls back from DLSS gets the 4K
// images minified. RYGT_DLSS_TEMPORAL_UI_4K=0 leaves the game's flag alone, =1 forces it whatever
// the upscaler. Costs texture memory and loading, neither measured; menus draw the 4K images
// into the 1080p picture, minified, as does a GPU without 32768-wide viewports in a race.
void InstallUi4kImagesGt7(u8* base) {
    const char* own = std::getenv("RYGT_DLSS_TEMPORAL_UI_4K");
    if (own && own[0] == '0') {
        return;
    }
    if (!(own && own[0] == '1')) {
        const char* env = std::getenv("RYGT_DLSS_TEMPORAL");
        const char* streamline = std::getenv("RYGT_DLSS_STREAMLINE");
        const bool plugin =
            !(streamline && streamline[0] == '0') &&
            GetFileAttributesW((Common::RytyExeDirW() + L"dlss\\sl.dlss.dll").c_str()) !=
                INVALID_FILE_ATTRIBUTES;
        const bool temporal = env && (env[0] == '0' || env[0] == '1')
                                  ? env[0] == '1'
                                  : ConfigFlag("dlss_enabled", false) && plugin &&
                                        !ConfigFlag("dlssg_enabled", false) &&
                                        !ConfigFlag("framegen_enabled", false);
        const char* neo = std::getenv("RYGT_NEO");
        const char* tv = std::getenv("RYGT_NEO_TV");
        const bool pro_4k =
            neo && neo[0] == '1' && neo[1] == '\0' && !(tv && std::string_view{tv} == "1080");
        if (!temporal || pro_4k) {
            return;
        }
    }
    const Patch patches[] = {
        {0x181bc7c, {0x20, 0xd0, 0x88, 0x05, 0xed, 0x1d, 0xae, 0x04},
         {0xb0, 0x01, 0x88, 0x05, 0xed, 0x1d, 0xae, 0x04},
         "DLSS-GT7-005 4K interface images (1/2)"},
        {0x18f3751, {0x20, 0xca, 0x40, 0x38, 0xd6, 0x0f, 0x85, 0xc2, 0x02, 0x00, 0x00},
         {0xb2, 0x01, 0x40, 0x38, 0xd6, 0x0f, 0x85, 0xc2, 0x02, 0x00, 0x00},
         "DLSS-GT7-005 4K interface images (2/2)"},
    };
    constexpr u64 flag_rva = 0x62fda71;
    // Both writers and the flag are checked before anything changes: another executable must not
    // get half of this.
    for (const Patch& patch : patches) {
        if (!ReadableRange(base + patch.rva, patch.original.size()) ||
            std::memcmp(base + patch.rva, patch.original.data(), patch.original.size()) != 0) {
            std::fprintf(stderr,
                         "[patch] DLSS-GT7-005 4K interface images: the bytes at %#llx differ, "
                         "not applied\n",
                         static_cast<unsigned long long>(patch.rva));
            return;
        }
    }
    if (!ReadableRange(base + flag_rva, 1) || base[flag_rva] > 1) {
        std::fprintf(stderr, "[patch] DLSS-GT7-005 4K interface images: the flag is unreadable or "
                             "not a flag, not applied\n");
        return;
    }
    for (const Patch& patch : patches) {
        if (!Apply(base, patch)) {
            return;
        }
    }
    DWORD old_protect = 0;
    if (VirtualProtect(base + flag_rva, 1, PAGE_READWRITE, &old_protect)) {
        base[flag_rva] = 1;
        DWORD ignored = 0;
        VirtualProtect(base + flag_rva, 1, old_protect, &ignored);
        std::fprintf(stderr, "[patch] DLSS-GT7-005: the game's 4K interface images are selected\n");
    }
}

void InstallMirrorHalfRate(u8* base) {
    bool wanted = ConfigFlag("mirror_half_rate", false);
    if (const char* env = std::getenv("RYGT_MIRROR_HALF")) {
        wanted = env[0] == '1';
    }
    if (!wanted) {
        return;
    }
    constexpr u64 kSite = 0x111011a; // RVA of "cmp byte ptr [r13+0x5a8], 0"
    static const u8 kOriginal[] = {0x41, 0x80, 0xbd, 0xa8, 0x05, 0x00, 0x00, 0x00};
    u8* site = base + kSite;
    if (std::memcmp(site, kOriginal, sizeof(kOriginal)) != 0) {
        std::fprintf(stderr, "[patch] GTS-008: original bytes differ, not applied\n");
        return;
    }
    u8* stub = AllocNear(base);
    if (!stub) {
        std::fprintf(stderr, "[patch] GTS-008: no memory near the game's code\n");
        return;
    }
    // cmp byte [r13+0x5a8],0 ; jne skip ; xor byte [rip+toggle],1 ; ret ; skip: ret ; toggle: 0
    const u8 code[] = {0x41, 0x80, 0xbd, 0xa8, 0x05, 0x00, 0x00, 0x00, // +0  cmp
                       0x75, 0x08,                                     // +8  jne +18
                       0x80, 0x35, 0x02, 0x00, 0x00, 0x00, 0x01,       // +10 xor [rip+2],1
                       0xc3,                                           // +17 ret (ZF from xor)
                       0xc3,                                           // +18 ret (ZF=0: skip)
                       0x00};                                          // +19 toggle
    std::memcpy(stub, code, sizeof(code));
    u8 call[8] = {0xe8, 0, 0, 0, 0, 0x90, 0x90, 0x90};
    const s32 rel = static_cast<s32>(reinterpret_cast<s64>(stub) - reinterpret_cast<s64>(site + 5));
    std::memcpy(call + 1, &rel, 4);
    Apply(base, Patch{kSite, {std::begin(kOriginal), std::end(kOriginal)},
                      {std::begin(call), std::end(call)}, "GTS-008 rear-view mirror at half rate"});
}

namespace {
using TelemetryUpdateFn = void(PS4_SYSV_ABI*)(void*, void*);
TelemetryUpdateFn g_telemetry_update{};
void PS4_SYSV_ABI TelemetryUpdateHook(void* service, void* context) {
    g_telemetry_update(service, context);
    Input::TriggerFeedback::Publish(static_cast<const u8*>(service) + 0x1dc);
}
void InstallTriggerTelemetry(u8* base) {
    // CUSA03220 v1.69: SimulatorInterface::update virtual slot, converter RVA.
    // Validate both the slot and packet-builder magic before touching the table.
    auto* slot = reinterpret_cast<TelemetryUpdateFn*>(base + 0x2858390);
    auto original = reinterpret_cast<TelemetryUpdateFn>(base + 0x68cf20);
    const u8 magic[] = {0xc7, 0x87, 0xdc, 0x01, 0, 0, 0x30, 0x53, 0x37, 0x47};
    if (*slot != original || std::memcmp(base + 0x68cf53, magic, sizeof(magic)) != 0) return;
    const char* enabled = std::getenv("RYGT_ADAPTIVE_TRIGGERS");
    if (enabled && enabled[0] == '0') return;
    DWORD protection;
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &protection)) return;
    g_telemetry_update = original;
    *slot = &TelemetryUpdateHook;
    DWORD ignored;
    VirtualProtect(slot, sizeof(*slot), protection, &ignored);
}

// GT7 (CUSA24769 v1.71): the same Simulator Interface packet (magic, layout and the service's
// +0x1dc offset as GT Sport's), from GT7's own SimulatorInterface::update; on while the launcher's
// trigger strength slider is above 0%. xacidr4inx's trigger feedback, ported.
TelemetryUpdateFn g_telemetry_update_gt7{};
void PS4_SYSV_ABI TelemetryUpdateHookGt7(void* service, void* context) {
    g_telemetry_update_gt7(service, context);
    const auto* packet = static_cast<const u8*>(service) + 0x1dc;
    Input::TriggerFeedback::Publish(packet);
    static const bool log = [] {
        const char* env = std::getenv("RYGT_ADAPTIVE_TRIGGERS_LOG");
        return env && env[0] == '1';
    }();
    static u64 calls = 0;
    if (log && ++calls % 30 == 1) {
        float speed = 0;
        std::memcpy(&speed, packet + 0x4c, sizeof(speed));
        u32 id = 0;
        u16 flags = 0;
        float omega = 0, radius = 0;
        std::memcpy(&id, packet + 0x70, sizeof(id));
        std::memcpy(&flags, packet + 0x8e, sizeof(flags));
        std::memcpy(&omega, packet + 0xa4, sizeof(omega));
        std::memcpy(&radius, packet + 0xb4, sizeof(radius));
        // Wheel speed against car speed: lock (below 1) and spin (above 1), the worst wheel.
        float lock = 0, spin = 0;
        for (int i = 0; i < 4 && speed > 1; ++i) {
            float w = 0, r = 0;
            std::memcpy(&w, packet + 0xa4 + 4 * i, sizeof(w));
            std::memcpy(&r, packet + 0xb4 + 4 * i, sizeof(r));
            const float ratio = std::abs(w) * r / speed;
            lock = std::max(lock, 1 - ratio);
            spin = std::max(spin, ratio - 1);
        }
        std::fprintf(stderr, "[triggers] GT7 telemetry #%llu: packet %u flags %#x speed %.1f m/s "
                     "throttle %u brake %u wheel0 %.1f rad/s r %.3f lock %.3f spin %.3f\n",
                     static_cast<unsigned long long>(calls), id, flags, speed, packet[0x91],
                     packet[0x92], omega, radius, lock, spin);
    }
}
void InstallTriggerTelemetryGt7(u8* base) {
    // Converter RVAs: the virtual slot, the update function, and its packet-magic store.
    auto* slot = reinterpret_cast<TelemetryUpdateFn*>(base + 0x53efcb0);
    auto original = reinterpret_cast<TelemetryUpdateFn>(base + 0x1d10f10);
    const u8 magic[] = {0xc7, 0x87, 0xdc, 0x01, 0, 0, 0x30, 0x53, 0x37, 0x47};
    // FIX-GT7-030: this runs for every title: another executable may have no readable page there.
    const auto readable = [](const u8* address, size_t size) {
        MEMORY_BASIC_INFORMATION info{};
        constexpr DWORD ReadAccess = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                                     PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                                     PAGE_EXECUTE_WRITECOPY;
        return VirtualQuery(address, &info, sizeof(info)) != 0 && info.State == MEM_COMMIT &&
               (info.Protect & ReadAccess) != 0 && (info.Protect & PAGE_GUARD) == 0 &&
               address + size <= static_cast<const u8*>(info.BaseAddress) + info.RegionSize;
    };
    if (!readable(base + 0x53efcb0, sizeof(*slot)) || *slot != original ||
        !readable(base + 0x1d10f53, sizeof(magic)) ||
        std::memcmp(base + 0x1d10f53, magic, sizeof(magic)) != 0) {
        return;
    }
    // On whenever the launcher's trigger strength slider is above 0% (the bridge scales the effects
    // by it and releases the triggers at 0%, also while the game runs); RYGT_ADAPTIVE_TRIGGERS=0
    // forces it off, =1 on even at 0%.
    const char* enabled = std::getenv("RYGT_ADAPTIVE_TRIGGERS");
    if (enabled && enabled[0] == '0') return;
    DWORD protection;
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &protection)) return;
    g_telemetry_update_gt7 = original;
    *slot = &TelemetryUpdateHookGt7;
    DWORD ignored;
    VirtualProtect(slot, sizeof(*slot), protection, &ignored);
    std::fprintf(stderr, "[triggers] GT7 adaptive triggers on (telemetry hook installed)\n");
}
} // namespace

namespace {
// DIAG-161: GT Sport's fatal-error routine (eboot 0x1796f70: source file, line, message) throws the
// error that ends the game: a script's System.ErrorAbort, a failed native assert. The game shows
// the message nowhere and then shuts down as if the player quit (a player's window closed right
// after the boot screen with nothing in the logs). Its entry now prints the message to stderr
// first. The message is the game's string: an 8-byte header, then the text inline (capacity under
// 16) or a pointer to it, the size at +0x18 and the capacity at +0x20.
constexpr u64 kFatalError = 0x1796f70;
constexpr u8 kFatalErrorPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                      0x56, 0x41, 0x55, 0x41, 0x54, 0x53};

void PS4_SYSV_ABI FatalErrorHook(const char* file, s32 line, const u8* message) {
    static std::atomic<u32> reported{0};
    if (reported++ >= 20) {
        return;
    }
    char where[256] = "?";
    if (file) {
        for (size_t i = 0; i + 1 < sizeof(where) && Core::RytyMemory::GuardedCopy(where + i, file + i, 1); ++i) {
            if (where[i] == 0) {
                break;
            }
            where[i + 1] = 0;
        }
    }
    char text[2048] = "";
    u64 header[5] = {};
    if (message && Core::RytyMemory::GuardedCopy(header, message, sizeof(header))) {
        const u64 size = std::min<u64>(header[3], sizeof(text) - 1);
        const char* data = header[4] >= 0x10 ? reinterpret_cast<const char*>(header[1])
                                             : reinterpret_cast<const char*>(message + 8);
        if (size != 0 && Core::RytyMemory::GuardedCopy(text, data, size)) {
            text[size] = 0;
        }
    }
    std::fprintf(stderr, "[game] fatal error at %s(%d): %s\n", where, line, text);
    std::fflush(stderr);
}

void InstallFatalErrorDiag(u8* base) {
    u8* entry = base + kFatalError + kTextRvaShift;
    if (std::memcmp(entry, kFatalErrorPrologue, sizeof(kFatalErrorPrologue)) != 0) {
        std::fprintf(stderr, "[patch] DIAG-161: original bytes differ, not applied\n");
        return;
    }
    u8* stub = AllocNear(base);
    if (!stub) {
        std::fprintf(stderr, "[patch] DIAG-161: no memory near the game's code\n");
        return;
    }
    // Saves the argument registers, calls the hook with the stack aligned, restores them, runs the
    // 13 prologue bytes the jump replaced and jumps back after them.
    std::vector<u8> code = {0x57, 0x56, 0x52, 0x51, 0x41, 0x50, 0x41, 0x51, // push rdi..r9
                            0x48, 0x83, 0xec, 0x08,                         // sub rsp, 8
                            0x48, 0xb8};                                    // mov rax, imm64
    const u64 hook = reinterpret_cast<u64>(&FatalErrorHook);
    code.insert(code.end(), reinterpret_cast<const u8*>(&hook), reinterpret_cast<const u8*>(&hook) + 8);
    code.insert(code.end(), {0xff, 0xd0,                                    // call rax
                             0x48, 0x83, 0xc4, 0x08,                        // add rsp, 8
                             0x41, 0x59, 0x41, 0x58, 0x59, 0x5a, 0x5e, 0x5f}); // pop r9..rdi
    code.insert(code.end(), std::begin(kFatalErrorPrologue), std::end(kFatalErrorPrologue));
    code.insert(code.end(), {0xff, 0x25, 0, 0, 0, 0});                      // jmp [rip]
    const u64 back = reinterpret_cast<u64>(entry + sizeof(kFatalErrorPrologue));
    code.insert(code.end(), reinterpret_cast<const u8*>(&back), reinterpret_cast<const u8*>(&back) + 8);
    std::memcpy(stub, code.data(), code.size());
    FlushInstructionCache(GetCurrentProcess(), stub, code.size());
    u8 jump[sizeof(kFatalErrorPrologue)];
    std::memset(jump, 0x90, sizeof(jump));
    jump[0] = 0xe9;
    const s32 rel = static_cast<s32>(stub - (entry + 5));
    std::memcpy(jump + 1, &rel, 4);
    DWORD old_protect = 0;
    if (!VirtualProtect(entry, sizeof(jump), PAGE_EXECUTE_READWRITE, &old_protect)) {
        return;
    }
    std::memcpy(entry, jump, sizeof(jump));
    DWORD ignored = 0;
    VirtualProtect(entry, sizeof(jump), old_protect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), entry, sizeof(jump));
    std::fprintf(stderr, "[patch] DIAG-161 fatal-error message applied at %#llx\n",
                 static_cast<unsigned long long>(kFatalError));
}
} // namespace


// PERF-GT7-113: GT7's file cipher. The game decrypts what it reads (ChaCha20: 20 rounds, a 32-bit
// block counter in state word 12, 16 state words at +8 of the cipher object) one 64-byte block at
// a time and XORs the keystream one byte at a time: while a race loads its file thread (FexpR)
// spends ~65% of its time there (1.44 s frame after Enter on Monza: ~0.9 s of it). The same
// cipher here makes 8 blocks at once (AVX2, or 4 with SSE) and XORs 8 bytes at a time, with the
// same output and the same counter left in the object. Hooked where the default and the custom
// seek paths join (the three instructions moved to the stub are position independent), so a
// cipher object with its own block-step method still runs the game's code. Applied only when the
// function's bytes are the expected ones and the self-test passes. RYGT_FAST_CIPHER=0: off.
// RYGT_FAST_CIPHER_VERIFY=<n>: the first n calls also run the game's own code on a copy of the
// data and compare. RYGT_DIAG_CIPHER=1: calls, MB and time a second on stderr.
namespace Gt7Cipher {

constexpr u64 kXor = 0x3326e10;         // void xor(Cipher* self, u8* data, s64 size, u64 offset)
constexpr u64 kLoop = 0x3326f4b;        // keystream merge and the byte loop
constexpr u64 kJoin = 0x3326e50;        // the block computation, after the seek
constexpr u64 kResume = 0x3326e5f;      // after the three loads moved to the stub
constexpr u64 kEpilogue = 0x3327092;    // stack check and return
constexpr u64 kDefaultSeek = 0x2bec400; // vtable +0x8 that the code inlines
constexpr u64 kDefaultNext = 0x2bec410; // vtable +0x10 that the code inlines

constexpr u8 kEntryBytes[] = {
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83,
    0xec, 0x48, 0x4c, 0x8b, 0x35, 0xb8, 0x7e, 0x24, 0x02, 0x4c, 0x8d, 0x0d, 0xd1, 0x55, 0x8c,
    0xff, 0x49, 0x8b, 0x06, 0x48, 0x89, 0x45, 0xd0, 0x48, 0x89, 0xc8, 0x4c, 0x8b, 0x07, 0x48,
    0xc1, 0xe8, 0x06, 0x4d, 0x8b, 0x40, 0x08, 0x4d, 0x39, 0xc8, 0x0f, 0x85, 0xa1, 0x02, 0x00,
    0x00, 0x89, 0x47, 0x38, 0xc5, 0xfa, 0x6f, 0x47, 0x08, 0xc5, 0xfa, 0x6f, 0x4f, 0x18, 0xc5,
    0xfa, 0x6f, 0x57, 0x28, 0xc5, 0xfa, 0x6f, 0x5f, 0x38, 0xc5, 0x7a, 0x6f, 0x0d, 0xa4, 0xb4,
    0xc1, 0x00, 0xc5, 0x79, 0x6f, 0x15, 0x4c, 0x53, 0xc2, 0x00};
constexpr u8 kLoopBytes[] = {
    0xc5, 0xd9, 0xfe, 0xc0, 0x48, 0x85, 0xd2, 0xc5, 0xc9, 0xfe, 0xc9, 0xc5, 0xd1, 0xfe, 0xd2,
    0xc5, 0xc1, 0xfe, 0xdb, 0xc5, 0xfa, 0x7f, 0x45, 0x90, 0xc5, 0xfa, 0x7f, 0x4d, 0xa0, 0xc5,
    0xfa, 0x7f, 0x55, 0xb0, 0xc5, 0xfa, 0x7f, 0x5d, 0xc0, 0x0f, 0x8e, 0x1a, 0x01, 0x00, 0x00,
    0x48, 0x01, 0xf2, 0x83, 0xe1, 0x3f, 0x4c, 0x8d, 0x3d, 0x8b, 0x54, 0x8c, 0xff, 0x8a, 0x44,
    0x0d, 0x90, 0x30, 0x06, 0x48, 0x83, 0xf9, 0x3f, 0x0f, 0x83, 0x15, 0x01, 0x00, 0x00, 0x48,
    0xff, 0xc1, 0x48, 0xff, 0xc6, 0x48, 0x39, 0xd6, 0x72, 0xe5, 0xe9, 0xed, 0x00, 0x00, 0x00};
constexpr u8 kTailBytes[] = {
    0x49, 0x8b, 0x06, 0x48, 0x3b, 0x45, 0xd0, 0x75, 0x4c, 0x48, 0x83, 0xc4, 0x48, 0x5b, 0x41,
    0x5c, 0x41, 0x5d, 0x41, 0x5e, 0x41, 0x5f, 0x5d, 0xc3, 0x48, 0x8b, 0x07, 0x48, 0x8b, 0x40,
    0x10, 0x4c, 0x39, 0xf8, 0x75, 0x5b, 0xff, 0x47, 0x38, 0xc5, 0xfa, 0x6f, 0x47, 0x08, 0xc5,
    0xfa, 0x6f, 0x4f, 0x18, 0xc5, 0xfa, 0x6f, 0x57, 0x28, 0xc5, 0xfa, 0x6f, 0x5f, 0x38, 0xb8,
    0x0a, 0x00, 0x00, 0x00, 0xc5, 0xf9, 0x6f, 0xfb, 0xc5, 0xf9, 0x6f, 0xea, 0xc5, 0xf9, 0x6f,
    0xf1, 0xc5, 0xf9, 0x6f, 0xe0, 0xe9, 0xbe, 0xfe, 0xff, 0xff, 0xe8, 0xb4, 0x18, 0xa3, 0x00,
    0x0f, 0x0b};

using XorFn = void(PS4_SYSV_ABI*)(void*, u8*, s64, u64);
XorFn g_original = nullptr;
u64 g_default_seek = 0;
bool g_avx2 = false;
bool g_stats = false;
std::atomic<s64> g_verify_left{0};
std::atomic<u64> g_verify_same{0}, g_verify_differ{0}, g_verify_ns_original{0},
    g_verify_ns_fast{0}, g_verify_bytes{0};
thread_local bool t_in_original = false;
std::atomic<u64> g_calls{0}, g_bytes{0}, g_ns{0}, g_last_report{0};

#define RYGT_CHACHA_QR(a, b, c, d)                                                                 \
    a += b;                                                                                        \
    d ^= a;                                                                                        \
    d = (d << 16) | (d >> 16);                                                                     \
    c += d;                                                                                        \
    b ^= c;                                                                                        \
    b = (b << 12) | (b >> 20);                                                                     \
    a += b;                                                                                        \
    d ^= a;                                                                                        \
    d = (d << 8) | (d >> 24);                                                                      \
    c += d;                                                                                        \
    b ^= c;                                                                                        \
    b = (b << 7) | (b >> 25);

#define RYGT_CHACHA_ROUNDS(x)                                                                      \
    for (int round = 0; round < 10; ++round) {                                                     \
        RYGT_CHACHA_QR(x[0], x[4], x[8], x[12])                                                    \
        RYGT_CHACHA_QR(x[1], x[5], x[9], x[13])                                                    \
        RYGT_CHACHA_QR(x[2], x[6], x[10], x[14])                                                   \
        RYGT_CHACHA_QR(x[3], x[7], x[11], x[15])                                                   \
        RYGT_CHACHA_QR(x[0], x[5], x[10], x[15])                                                   \
        RYGT_CHACHA_QR(x[1], x[6], x[11], x[12])                                                   \
        RYGT_CHACHA_QR(x[2], x[7], x[8], x[13])                                                    \
        RYGT_CHACHA_QR(x[3], x[4], x[9], x[14])                                                    \
    }

// One block: the 16 words of the object with `counter` in word 12.
void Block(const u32* state, u32 counter, u8* out) {
    u32 s[16];
    std::memcpy(s, state, sizeof(s));
    s[12] = counter;
    u32 x[16];
    std::memcpy(x, s, sizeof(x));
    RYGT_CHACHA_ROUNDS(x)
    for (int j = 0; j < 16; ++j) {
        x[j] += s[j];
    }
    std::memcpy(out, x, 64);
}

// N blocks with counters counter..counter+N-1 (32-bit wrap, like the game), one vector lane each.
template <int N>
[[gnu::always_inline]] inline void Blocks(const u32* state, u32 counter, u8* out) {
    typedef u32 V __attribute__((vector_size(N * 4)));
    V s[16], x[16];
    for (int j = 0; j < 16; ++j) {
        for (int lane = 0; lane < N; ++lane) {
            s[j][lane] = state[j];
        }
    }
    for (int lane = 0; lane < N; ++lane) {
        s[12][lane] = counter + static_cast<u32>(lane);
    }
    for (int j = 0; j < 16; ++j) {
        x[j] = s[j];
    }
    RYGT_CHACHA_ROUNDS(x)
    u32 words[16][N];
    for (int j = 0; j < 16; ++j) {
        x[j] += s[j];
        std::memcpy(words[j], &x[j], sizeof(words[j]));
    }
    u32* out32 = reinterpret_cast<u32*>(out);
    for (int lane = 0; lane < N; ++lane) {
        for (int j = 0; j < 16; ++j) {
            out32[lane * 16 + j] = words[j][lane];
        }
    }
}

[[gnu::always_inline]] inline void XorBytes(u8* data, const u8* key, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        u64 a, b;
        std::memcpy(&a, data + i, 8);
        std::memcpy(&b, key + i, 8);
        a ^= b;
        std::memcpy(data + i, &a, 8);
    }
    for (; i < size; ++i) {
        data[i] ^= key[i];
    }
}

// Whole groups of 8 (AVX2) or 4 blocks from `p`; returns the counter after them.
__attribute__((target("avx2"))) u32 Wide8(const u32* state, u32 counter, u8*& p, u8* end) {
    alignas(64) u8 key[512];
    while (end - p >= 512) {
        Blocks<8>(state, counter, key);
        XorBytes(p, key, 512);
        counter += 8;
        p += 512;
    }
    return counter;
}

u32 Wide4(const u32* state, u32 counter, u8*& p, u8* end) {
    alignas(64) u8 key[256];
    while (end - p >= 256) {
        Blocks<4>(state, counter, key);
        XorBytes(p, key, 256);
        counter += 4;
        p += 256;
    }
    return counter;
}

// The game's function from its join point: word 12 already holds the block of `offset`.
void Fast(u8* self, u8* data, s64 size, u64 offset) {
    if (size <= 0) {
        return;
    }
    u32* state = reinterpret_cast<u32*>(self + 8);
    u32 counter = state[12];
    alignas(64) u8 key[64];
    u8* p = data;
    u8* const end = data + size;
    if (const u32 pos = static_cast<u32>(offset & 63)) {
        Block(state, counter, key);
        const size_t n = std::min<size_t>(64 - pos, static_cast<size_t>(end - p));
        XorBytes(p, key + pos, n);
        p += n;
        if (pos + n == 64) {
            ++counter;
        }
    }
    counter = g_avx2 ? Wide8(state, counter, p, end) : Wide4(state, counter, p, end);
    while (end - p >= 64) {
        Block(state, counter, key);
        XorBytes(p, key, 64);
        ++counter;
        p += 64;
    }
    if (p < end) {
        Block(state, counter, key);
        XorBytes(p, key, static_cast<size_t>(end - p));
    }
    state[12] = counter;
}

u64 NowNs() {
    LARGE_INTEGER now, frequency;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    return static_cast<u64>(static_cast<double>(now.QuadPart) * 1e9 /
                            static_cast<double>(frequency.QuadPart));
}

void Report() {
    const u64 now = NowNs();
    u64 last = g_last_report.load(std::memory_order_relaxed);
    if (last == 0) {
        g_last_report.compare_exchange_strong(last, now);
        return;
    }
    if (now - last < 1000000000ULL || !g_last_report.compare_exchange_strong(last, now)) {
        return;
    }
    const u64 calls = g_calls.exchange(0), bytes = g_bytes.exchange(0), ns = g_ns.exchange(0);
    if (calls) {
        std::fprintf(stderr, "[cipher] PERF-GT7-113: %llu calls, %.1f MB, %.1f ms (%.0f MB/s)\n",
                     static_cast<unsigned long long>(calls), bytes / 1048576.0, ns / 1e6,
                     ns ? bytes / 1048576.0 / (ns / 1e9) : 0.0);
    }
}

// Called by the stub; false: run the game's own code (the stub's fallback).
bool PS4_SYSV_ABI Hook(u8* self, u8* data, s64 size, u64 offset) {
    if (t_in_original) {
        return false;
    }
    if (g_verify_left.load(std::memory_order_relaxed) > 0 && size > 0 &&
        *reinterpret_cast<const u64*>(*reinterpret_cast<const u64*>(self) + 8) ==
            g_default_seek &&
        g_verify_left.fetch_sub(1) > 0) {
        u32* state = reinterpret_cast<u32*>(self + 8);
        const u32 counter = state[12];
        std::vector<u8> copy(data, data + size);
        const u64 t0 = NowNs();
        t_in_original = true;
        g_original(self, copy.data(), size, offset);
        t_in_original = false;
        const u64 t1 = NowNs();
        const u32 counter_after = state[12];
        state[12] = counter;
        Fast(self, data, size, offset);
        const u64 t2 = NowNs();
        g_verify_ns_original += t1 - t0;
        g_verify_ns_fast += t2 - t1;
        g_verify_bytes += static_cast<u64>(size);
        const bool same = std::memcmp(copy.data(), data, static_cast<size_t>(size)) == 0 &&
                          state[12] == counter_after;
        if (same) {
            ++g_verify_same;
        } else {
            if (++g_verify_differ <= 10) {
                std::fprintf(stderr,
                             "[cipher] PERF-GT7-113: differs from the game's code (size %lld, "
                             "offset %#llx, counter %u vs %u)\n",
                             static_cast<long long>(size), static_cast<unsigned long long>(offset),
                             state[12], counter_after);
            }
            std::memcpy(data, copy.data(), static_cast<size_t>(size)); // the game's result
            state[12] = counter_after;
        }
        if (g_verify_left.load(std::memory_order_relaxed) <= 0) {
            static std::atomic<bool> told{false};
            if (!told.exchange(true)) {
                std::fprintf(stderr,
                             "[cipher] PERF-GT7-113: verified %llu calls (%.1f MB): %llu same, "
                             "%llu different; game's code %.1f ms, this one %.1f ms\n",
                             static_cast<unsigned long long>(g_verify_same + g_verify_differ),
                             g_verify_bytes / 1048576.0,
                             static_cast<unsigned long long>(g_verify_same.load()),
                             static_cast<unsigned long long>(g_verify_differ.load()),
                             g_verify_ns_original / 1e6, g_verify_ns_fast / 1e6);
            }
        }
        return true;
    }
    if (!g_stats) {
        Fast(self, data, size, offset);
        return true;
    }
    const u64 t0 = NowNs();
    Fast(self, data, size, offset);
    g_ns += NowNs() - t0;
    ++g_calls;
    g_bytes += size > 0 ? static_cast<u64>(size) : 0;
    Report();
    return true;
}

bool SelfTest() {
    // RFC 8439 2.3.2: key 00..1f, nonce 00000009 0000004a 00000000, block count 1.
    const u32 rfc[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, 0x03020100, 0x07060504,
                         0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
                         0x00000001, 0x09000000, 0x4a000000, 0x00000000};
    const u8 expected[16] = {0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15,
                             0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4};
    u8 out[64];
    Block(rfc, 1, out);
    if (std::memcmp(out, expected, sizeof(expected)) != 0) {
        return false;
    }
    // The wide paths against single blocks, across a counter wrap.
    u32 state[16];
    for (int j = 0; j < 16; ++j) {
        state[j] = 0x9e3779b9u * static_cast<u32>(j + 1);
    }
    alignas(64) u8 one[64 * 8], wide[64 * 8];
    const u32 start = 0xfffffffdu;
    for (u32 b = 0; b < 8; ++b) {
        Block(state, start + b, one + b * 64);
    }
    u8 data[64 * 8 + 37];
    for (size_t i = 0; i < sizeof(data); ++i) {
        data[i] = static_cast<u8>(i * 7);
    }
    std::memcpy(wide, data, sizeof(wide));
    u8* p = wide;
    const u32 next = g_avx2 ? Wide8(state, start, p, wide + 512) : Wide4(state, start, p, wide + 512);
    if (p != wide + 512 || next != start + 8u) {
        return false;
    }
    for (size_t i = 0; i < 512; ++i) {
        if (wide[i] != static_cast<u8>(data[i] ^ one[i])) {
            return false;
        }
    }
    // Fast() from an unaligned offset: byte by byte like the game.
    u8 object[8 + 64] = {};
    std::memcpy(object + 8, state, 64);
    const u64 offset = 64ull * 5 + 13;
    reinterpret_cast<u32*>(object + 8)[12] = static_cast<u32>(offset >> 6);
    u8 fast[sizeof(data)];
    std::memcpy(fast, data, sizeof(data));
    Fast(object, fast, sizeof(fast), offset);
    u32 counter = static_cast<u32>(offset >> 6);
    u8 key[64];
    Block(state, counter, key);
    u32 pos = static_cast<u32>(offset & 63);
    for (size_t i = 0; i < sizeof(data); ++i) {
        if (fast[i] != static_cast<u8>(data[i] ^ key[pos])) {
            return false;
        }
        if (pos == 63) {
            ++counter;
            Block(state, counter, key);
            pos = 0;
        } else {
            ++pos;
        }
    }
    return reinterpret_cast<u32*>(object + 8)[12] == counter;
}

void Install(u8* base) {
    const char* env = std::getenv("RYGT_FAST_CIPHER");
    const bool asked = env && env[0];
    if (asked && env[0] == '0') {
        return;
    }
    u8* const entry = base + kXor + kTextRvaShift;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(entry, &info, sizeof(info)) || info.State != MEM_COMMIT ||
        std::memcmp(entry, kEntryBytes, sizeof(kEntryBytes)) != 0 ||
        std::memcmp(base + kLoop + kTextRvaShift, kLoopBytes, sizeof(kLoopBytes)) != 0 ||
        std::memcmp(base + kEpilogue + kTextRvaShift, kTailBytes, sizeof(kTailBytes)) != 0) {
        if (asked) {
            std::fprintf(stderr, "[patch] PERF-GT7-113: original bytes differ, not applied\n");
        }
        return;
    }
    int regs[4] = {};
    __cpuidex(regs, 7, 0);
    g_avx2 = (regs[1] & (1 << 5)) != 0;
    if (const char* wide = std::getenv("RYGT_FAST_CIPHER_SSE"); wide && wide[0] == '1') {
        g_avx2 = false;
    }
    if (!SelfTest()) {
        std::fprintf(stderr, "[patch] PERF-GT7-113: self-test failed, not applied\n");
        return;
    }
    u8* stub = AllocNear(base);
    if (!stub) {
        std::fprintf(stderr, "[patch] PERF-GT7-113: no memory for the stub, not applied\n");
        return;
    }
    g_original = reinterpret_cast<XorFn>(entry);
    g_default_seek = reinterpret_cast<u64>(base + kDefaultSeek + kTextRvaShift);
    const u64 default_next = reinterpret_cast<u64>(base + kDefaultNext + kTextRvaShift);
    const u64 hook = reinterpret_cast<u64>(&Hook);
    const u64 epilogue = reinterpret_cast<u64>(base + kEpilogue + kTextRvaShift);
    const u64 resume = reinterpret_cast<u64>(base + kResume + kTextRvaShift);
    std::vector<u8> code;
    const auto bytes = [&](std::initializer_list<u8> list) { code.insert(code.end(), list); };
    const auto imm64 = [&](u64 value) {
        for (int i = 0; i < 8; ++i) code.push_back(static_cast<u8>(value >> (8 * i)));
    };
    bytes({0x4c, 0x8b, 0x17});       // mov r10, [rdi]          (vtable)
    bytes({0x4d, 0x8b, 0x52, 0x10}); // mov r10, [r10 + 0x10]   (block step)
    bytes({0x49, 0xbb});             // mov r11, default step
    imm64(default_next);
    bytes({0x4d, 0x39, 0xda}); // cmp r10, r11
    bytes({0x75, 0x00});       // jne fallback (patched below)
    const size_t jne_at = code.size() - 1;
    bytes({0x57, 0x56, 0x52, 0x51}); // push rdi, rsi, rdx, rcx (rsp stays 16-aligned)
    bytes({0x49, 0xbb});             // mov r11, Hook
    imm64(hook);
    bytes({0x41, 0xff, 0xd3});       // call r11
    bytes({0x59, 0x5a, 0x5e, 0x5f}); // pop rcx, rdx, rsi, rdi
    bytes({0x84, 0xc0});             // test al, al
    bytes({0x74, 0x00});             // jz fallback (patched below)
    const size_t jz_at = code.size() - 1;
    bytes({0x49, 0xbb}); // mov r11, epilogue
    imm64(epilogue);
    bytes({0x41, 0xff, 0xe3}); // jmp r11
    const size_t fallback = code.size();
    bytes({0xc5, 0xfa, 0x6f, 0x47, 0x08}); // vmovdqu xmm0, [rdi + 0x08]  (moved from the game)
    bytes({0xc5, 0xfa, 0x6f, 0x4f, 0x18}); // vmovdqu xmm1, [rdi + 0x18]
    bytes({0xc5, 0xfa, 0x6f, 0x57, 0x28}); // vmovdqu xmm2, [rdi + 0x28]
    bytes({0x49, 0xbb});                   // mov r11, resume
    imm64(resume);
    bytes({0x41, 0xff, 0xe3}); // jmp r11
    code[jne_at] = static_cast<u8>(fallback - (jne_at + 1));
    code[jz_at] = static_cast<u8>(fallback - (jz_at + 1));
    std::memcpy(stub, code.data(), code.size());
    FlushInstructionCache(GetCurrentProcess(), stub, code.size());

    u8* const join = base + kJoin + kTextRvaShift;
    u8 jump[15] = {0xff, 0x25, 0, 0, 0, 0}; // jmp [rip]; address; int3
    std::memcpy(jump + 6, &stub, 8);
    jump[14] = 0xcc;
    DWORD old_protect = 0;
    if (!VirtualProtect(join, sizeof(jump), PAGE_EXECUTE_READWRITE, &old_protect)) {
        return;
    }
    std::memcpy(join, jump, sizeof(jump));
    DWORD ignored = 0;
    VirtualProtect(join, sizeof(jump), old_protect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), join, sizeof(jump));
    if (const char* verify = std::getenv("RYGT_FAST_CIPHER_VERIFY"); verify && verify[0]) {
        g_verify_left = std::strtoll(verify, nullptr, 0);
    }
    if (const char* stats = std::getenv("RYGT_DIAG_CIPHER"); stats && stats[0] == '1') {
        g_stats = true;
    }
    std::fprintf(stderr, "[patch] PERF-GT7-113 fast file cipher applied (%s)\n",
                 g_avx2 ? "AVX2, 8 blocks" : "SSE, 4 blocks");
}

} // namespace Gt7Cipher

void RytySetGt7HdrStatusCompatibility(bool available) {
    Gt7HdrStatus::g_available.store(available, std::memory_order_relaxed);
}

// NEO-GT7-014: GT7's graphics presets, option by option. GT7 fills its preset in one function
// (GT7.exe+0x1bcc7c0: GT7_PS4, GT7_PS4Pro_4K for a Pro on a TV above 1080p, GT7_PS4Pro_2K for a
// Pro on a 1080p TV) and copies it into GTRender::GraphicsConfig (+0x1bccba0, called from
// +0x1bcc4bd and +0x173149c). That copy is hooked: with RYGT_NEO_EXTRAS set, five values the Pro
// presets differ on are forced in whichever preset the game picked, each on or off on its own:
//   blur        motion blur in races (race effect flags 0x08, motion blur off, and 0x10, a
//               second blur off, cleared; off: both set, as in the PS4 and Pro 4K presets)
//   replays60   replays and photo mode at 60 fps (frame divisor 1 for modes 1, 3 and 6; off: 2)
//   replayblur  the second blur in replays and photo mode (flag 0x10 cleared in modes 1, 2, 3
//               and 6; off: set)
//   garage60    the high-quality garage scenes (mode 5) at 60 fps (divisor 1; off: 2)
//   lod         level of detail kept 1.3 times farther (the preset's detail scale 1.3; off: 1.0)
// RYGT_NEO_EXTRAS=blur,lod turns those two on and the three others off; "all" and "none" too
// (gt7_preset_options.h). Unset: the game's presets as they are (the Pro 1080p one has all five,
// the others none). Preset layout (0x350 bytes, read from +0x1bcc7c0 and +0x1bccba0): +0x18
// detail scale (float), mode m at 0x48 + 0x60 m with its frame divisor at +0 (u32) and effect
// flags at +0x50 (u64), preset number at +0x348. Modes: 0 race, 1 and 3 replay, 2 replay
// variant, 4 garage, 5 high-quality garage, 6 photo, 7 Scapes. Every option on costs GPU time;
// checked in play on an RTX 4070 only, with GT7.exe 01.00 (the one this file's RVAs are from).
namespace Gt7Presets {

using namespace Gt7PresetOptions;

constexpr u64 kPresetCopy = 0x1bccba0; // GT7.exe RVA (the PE's own, no kTextRvaShift)
// mov eax, [rsi + 0x348]; xor edx, edx; mov [rdi + 0x4a4], eax: 14 bytes that don't depend on
// where they run, then xor eax, eax at +0xe. They hold the offsets of the preset's number and of
// its copy in the config, and appear once in GT7.exe 01.00.
constexpr u8 kPresetCopyStart[14] = {0x8b, 0x86, 0x48, 0x03, 0x00, 0x00, 0x31,
                                     0xd2, 0x89, 0x87, 0xa4, 0x04, 0x00, 0x00};

using PresetCopyFn = void(PS4_SYSV_ABI*)(u8*, u8*);
PresetCopyFn g_copy = nullptr;
u32 g_on = 0;

u8* Mode(u8* preset, int mode) {
    return preset + 0x48 + 0x60 * mode;
}

void SetFlag(u8* preset, int mode, u64 flags, bool set) {
    u64 value = 0;
    std::memcpy(&value, Mode(preset, mode) + 0x50, 8);
    value = set ? (value | flags) : (value & ~flags);
    std::memcpy(Mode(preset, mode) + 0x50, &value, 8);
}

void SetDivisor(u8* preset, int mode, u32 divisor) {
    std::memcpy(Mode(preset, mode), &divisor, 4);
}

const char* OnOff(u32 bit) {
    return (g_on & bit) ? "on" : "off";
}

void PS4_SYSV_ABI PresetCopyHook(u8* config, u8* preset) {
    SetFlag(preset, 0, 0x18, (g_on & Blur) == 0);
    for (const int mode : {1, 3, 6}) {
        SetDivisor(preset, mode, (g_on & Replays60) ? 1 : 2);
    }
    for (const int mode : {1, 2, 3, 6}) {
        SetFlag(preset, mode, 0x10, (g_on & ReplayBlur) == 0);
    }
    SetDivisor(preset, 5, (g_on & Garage60) ? 1 : 2);
    const float scale = (g_on & Lod) ? 1.3f : 1.0f;
    std::memcpy(preset + 0x18, &scale, 4);
    s32 number = 0;
    std::memcpy(&number, preset + 0x348, 4);
    std::fprintf(stderr,
                 "[patch] NEO-GT7-014: preset %d: blur %s, replays60 %s, replayblur %s, "
                 "garage60 %s, lod %s\n",
                 number, OnOff(Blur), OnOff(Replays60), OnOff(ReplayBlur), OnOff(Garage60),
                 OnOff(Lod));
    g_copy(config, preset);
}

// The game's code (.elf0) as [begin, end), when all of it is committed and readable; else empty.
std::pair<u8*, u8*> GameCode(u8* base) {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    const IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);
    for (u32 i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        if (std::memcmp(section->Name, ".elf0", 6) != 0) {
            continue;
        }
        u8* const begin = base + section->VirtualAddress;
        u8* const end = begin + section->Misc.VirtualSize;
        for (u8* at = begin; at < end;) {
            MEMORY_BASIC_INFORMATION info{};
            if (!VirtualQuery(at, &info, sizeof(info)) || info.State != MEM_COMMIT ||
                (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
                return {};
            }
            at = reinterpret_cast<u8*>(info.BaseAddress) + info.RegionSize;
        }
        return {begin, end};
    }
    return {};
}

// The copy at its RVA in GT7.exe 01.00; in another version, the one place in the game's code
// (.elf0) that starts a function (16-byte aligned) with these 14 bytes. None or several, or no
// readable .elf0 (another program): null.
u8* FindPresetCopy(u8* base) {
    const auto [begin, end] = GameCode(base);
    if (!begin || end - begin < static_cast<std::ptrdiff_t>(sizeof(kPresetCopyStart))) {
        return nullptr;
    }
    u8* const known = base + kPresetCopy;
    if (known >= begin && known + sizeof(kPresetCopyStart) <= end &&
        std::memcmp(known, kPresetCopyStart, sizeof(kPresetCopyStart)) == 0) {
        return known;
    }
    u8* found = nullptr;
    for (u8* at = begin; at + sizeof(kPresetCopyStart) <= end; at += 16) {
        if (std::memcmp(at, kPresetCopyStart, sizeof(kPresetCopyStart)) == 0) {
            if (found) {
                return nullptr;
            }
            found = at;
        }
    }
    return found;
}

void Install(u8* base) {
    const char* env = std::getenv("RYGT_NEO_EXTRAS");
    if (!env || !*env) {
        return;
    }
    const u32 on = Parse(env, [](std::string_view item) {
        std::fprintf(stderr, "[patch] NEO-GT7-014: unknown option '%.*s' in RYGT_NEO_EXTRAS, "
                             "ignored\n",
                     static_cast<int>(item.size()), item.data());
    });
    u8* const copy = FindPresetCopy(base);
    if (!copy) {
        std::fprintf(stderr, "[patch] NEO-GT7-014: preset copy not found (another GT7 version), "
                             "RYGT_NEO_EXTRAS not applied\n");
        return;
    }
    if (copy != base + kPresetCopy) {
        std::fprintf(stderr, "[patch] NEO-GT7-014: preset copy found at GT7.exe+%#llx\n",
                     static_cast<unsigned long long>(copy - base));
    }
    u8* const trampoline = AllocNear(base);
    if (!trampoline) {
        std::fprintf(stderr, "[patch] NEO-GT7-014: no memory near the game's code for the hook\n");
        return;
    }
    // The copy's first instructions move to the trampoline, then jump back after them.
    const u8 jump[6] = {0xff, 0x25, 0, 0, 0, 0}; // jmp qword ptr [rip]
    std::memcpy(trampoline, copy, sizeof(kPresetCopyStart));
    std::memcpy(trampoline + sizeof(kPresetCopyStart), jump, 6);
    const u8* const resume = copy + sizeof(kPresetCopyStart);
    std::memcpy(trampoline + sizeof(kPresetCopyStart) + 6, &resume, 8);
    FlushInstructionCache(GetCurrentProcess(), trampoline, sizeof(kPresetCopyStart) + 14);
    g_on = on;
    g_copy = reinterpret_cast<PresetCopyFn>(trampoline);
    DWORD old_protect = 0;
    if (!VirtualProtect(copy, sizeof(kPresetCopyStart), PAGE_EXECUTE_READWRITE, &old_protect)) {
        std::fprintf(stderr, "[patch] NEO-GT7-014: code not writable (error %lu), not applied\n",
                     GetLastError());
        return;
    }
    const void* const hook = reinterpret_cast<void*>(&PresetCopyHook);
    std::memcpy(copy, jump, 6);
    std::memcpy(copy + 6, &hook, 8);
    DWORD ignored = 0;
    VirtualProtect(copy, sizeof(kPresetCopyStart), old_protect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), copy, sizeof(kPresetCopyStart));
    std::fprintf(stderr, "[patch] NEO-GT7-014 preset options applied (RYGT_NEO_EXTRAS=%s)\n", env);
}

} // namespace Gt7Presets

// FIX-GT7-162: launching a replay crashed once (a read of address 0 on a Job thread at GT7.exe
// +0x1eaac3b). Each frame of an event, GT7's work module org_race (the std::function of
// GTFramework::Organizer::Impl::initialize's lambda $_7, GT7.exe+0x1eaa5d0) sets the race sound
// (bgm_effect, sfx_attenuation) from the Organizer's GTRace::RaceContext and reads the first
// 32-bit field of the object the context's shared_ptr at +0x58 points to, without a check. The
// race sequence (GTRaceApp::Sequence::Impl, module race_app) empties that shared_ptr during an
// event (GT7.exe+0x1f251e0) and it is filled again shortly after. Both modules have the same order
// key in the frame's first phase, and the game's executor only waits for its jobs between
// different keys, so the two run as concurrent jobs of one frame: nothing in the game's code
// orders the read and the emptying. The callback's two loads of the shared_ptr (control block,
// then pointer) go to a stub that tests the pointer first: null, the function leaves through its
// own end before any reference is taken, so that frame skips its race sound settings; set, the
// game's code goes on as before (with or without a control block). This also covers an emptying
// that lands between the two loads; one that lands after both and before the reference is taken
// is still the game's own, narrower, window. Applied only where every byte checked below is GT7's
// (no other game or version is changed). RYGT_RACE_CONTEXT_GUARD=0 turns it off.
namespace Gt7RaceContextGuard {

// mov rbx,[r13+0x60] (control block); mov r15,[r13+0x58] (pointer); test rbx,rbx; jz kNoRef.
// The two loads (8 bytes) become a jmp to the stub.
constexpr u64 kLoad = 0x1eaa889;
constexpr u8 kLoadBytes[] = {0x49, 0x8b, 0x5d, 0x60, 0x4d, 0x8b, 0x7d, 0x58, 0x48,
                             0x85, 0xdb, 0x0f, 0x84, 0xa1, 0x03, 0x00, 0x00};
constexpr size_t kLoadsSize = 8;
// With a control block: a reference taken (lea r14,[rbx+8]; ... call), the read, the release.
constexpr u64 kRefPath = 0x1eaa89a;
constexpr u8 kRefPathBytes[] = {0x4c, 0x8d, 0x73, 0x08, 0x4c, 0x89, 0xf7, 0xbe, 0x01, 0x00, 0x00,
                                0x00, 0x31, 0xd2, 0xe8, 0x33, 0xab, 0xeb, 0x01, 0x45, 0x8b, 0x3f};
// Without one: mov r15d,[r15] (the read that faulted); jmp kResume.
constexpr u64 kNoRef = 0x1eaac3b;
constexpr u8 kNoRefBytes[] = {0x45, 0x8b, 0x3f, 0xe9, 0x88, 0xfc, 0xff, 0xff};
// cmp r15d,10: where both paths go on.
constexpr u64 kResume = 0x1eaa8cb;
constexpr u8 kResumeBytes[] = {0x41, 0x83, 0xff, 0x0a};
// The function's end, also reached from other paths: rbx reloaded with the stack guard's address,
// guard check against [rbp-0x30], add rsp,0x18, pops of rbx, r12-r15 and rbp, ret.
constexpr u64 kEnd = 0x1eaaae8;
constexpr u8 kEndBytes[] = {0x48, 0x8b, 0x1d, 0xf1, 0x41, 0x6d, 0x03, 0x48, 0x8b, 0x03, 0x48,
                            0x3b, 0x45, 0xd0, 0x75, 0x5b, 0x48, 0x83, 0xc4, 0x18, 0x5b, 0x41,
                            0x5c, 0x41, 0x5d, 0x41, 0x5e, 0x41, 0x5f, 0x5d, 0xc3};

std::atomic<u64> g_skipped{0};

// Called from the stub on the game's Job thread when the pointer is null.
void PS4_SYSV_ABI Skipped(const void* control_block) {
    const u64 count = g_skipped.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 8 || (count & (count - 1)) == 0) {
        std::fprintf(stderr,
                     "[patch] FIX-GT7-162: race context empty (%s), this frame's race sound "
                     "settings skipped (%llu so far)\n",
                     control_block ? "pointer cleared, control block still read" : "emptied",
                     static_cast<unsigned long long>(count));
    }
}

bool Matches(u8* base, u64 rva, const u8* bytes, size_t size) {
    return ReadableRange(base + rva, size) && std::memcmp(base + rva, bytes, size) == 0;
}

void Install(u8* base) {
    const char* env = std::getenv("RYGT_RACE_CONTEXT_GUARD");
    if (env && env[0] == '0') {
        return;
    }
    if (!Matches(base, kLoad, kLoadBytes, sizeof(kLoadBytes)) ||
        !Matches(base, kRefPath, kRefPathBytes, sizeof(kRefPathBytes)) ||
        !Matches(base, kNoRef, kNoRefBytes, sizeof(kNoRefBytes)) ||
        !Matches(base, kResume, kResumeBytes, sizeof(kResumeBytes)) ||
        !Matches(base, kEnd, kEndBytes, sizeof(kEndBytes))) {
        // GT Sport, or another GT7 build: silent unless the switch was set.
        if (env && env[0]) {
            std::fprintf(stderr, "[patch] FIX-GT7-162: original bytes differ, not applied\n");
        }
        return;
    }
    u8* const stub = AllocNear(base);
    if (!stub) {
        std::fprintf(stderr, "[patch] FIX-GT7-162: no memory near the game's code, not applied\n");
        return;
    }
    std::vector<u8> code;
    const auto bytes = [&](std::initializer_list<u8> list) { code.insert(code.end(), list); };
    const auto imm64 = [&](u64 value) {
        for (int i = 0; i < 8; ++i) code.push_back(static_cast<u8>(value >> (8 * i)));
    };
    const auto jump = [&](u64 rva) { // jmp [rip] -> the game's code
        bytes({0xff, 0x25, 0, 0, 0, 0});
        imm64(reinterpret_cast<u64>(base + rva));
    };
    code.insert(code.end(), kLoadBytes, kLoadBytes + kLoadsSize); // the game's two loads
    bytes({0x4d, 0x85, 0xff});       // test r15, r15
    bytes({0x74, 0x00});             // jz empty (set below)
    const size_t jz_empty = code.size() - 1;
    bytes({0x48, 0x85, 0xdb});       // test rbx, rbx
    bytes({0x74, 0x0e});             // jz no_ref (over the next jump: 6 + 8 bytes)
    jump(kRefPath);                  // a control block: reference, read, release, as the game does
    bytes({0x45, 0x8b, 0x3f});       // no_ref: mov r15d, [r15]   (the game's kNoRef read)
    jump(kResume);
    const size_t empty = code.size();
    // The function's end reloads rbx and pops every callee-saved register from its frame, so the
    // stub may use rbx to keep rsp across a 16-byte aligned call of Skipped(control block).
    bytes({0x48, 0x89, 0xdf});       // mov rdi, rbx
    bytes({0x48, 0x89, 0xe3});       // mov rbx, rsp
    bytes({0x48, 0x83, 0xe4, 0xf0}); // and rsp, -16
    bytes({0x48, 0xb8});             // mov rax, Skipped
    imm64(reinterpret_cast<u64>(&Skipped));
    bytes({0xff, 0xd0});             // call rax
    bytes({0x48, 0x89, 0xdc});       // mov rsp, rbx
    jump(kEnd);
    code[jz_empty] = static_cast<u8>(empty - (jz_empty + 1));
    std::memcpy(stub, code.data(), code.size());

    const auto distance =
        reinterpret_cast<intptr_t>(stub) - reinterpret_cast<intptr_t>(base + kLoad + 5);
    DWORD previous = 0;
    if (distance < INT32_MIN || distance > INT32_MAX ||
        !VirtualProtect(stub, 0x1000, PAGE_EXECUTE_READ, &previous)) {
        std::fprintf(stderr, "[patch] FIX-GT7-162: stub out of reach or not executable, not applied\n");
        VirtualFree(stub, 0, MEM_RELEASE);
        return;
    }
    FlushInstructionCache(GetCurrentProcess(), stub, code.size());
    // jmp stub; int3 x3 (nothing jumps inside the two loads).
    const std::vector<u8> original(kLoadBytes, kLoadBytes + kLoadsSize);
    std::vector<u8> replacement{0xe9, 0, 0, 0, 0, 0xcc, 0xcc, 0xcc};
    const s32 relative = static_cast<s32>(distance);
    std::memcpy(replacement.data() + 1, &relative, sizeof(relative));
    if (!Apply(base, Patch{kLoad, original, replacement, "FIX-GT7-162 race context guard"})) {
        VirtualFree(stub, 0, MEM_RELEASE);
    }
}

} // namespace Gt7RaceContextGuard

void RytyApplyGamePatches() {
    Gt7HdrStatus::Install(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    Gt7Presets::Install(reinterpret_cast<u8*>(GetModuleHandleA(nullptr))); // NEO-GT7-014
    InstallWideCamera(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallTriggerTelemetry(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallTriggerTelemetryGt7(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallMirrorHalfRate(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallUi4kImagesGt7(reinterpret_cast<u8*>(GetModuleHandleA(nullptr))); // DLSS-GT7-005
    if (!WetReflectionsWanted()) {
        // jne (reflect when wet) -> two nops
        Apply(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)),
              Patch{0xac27db, {0x75, 0x77}, {0x90, 0x90}, "GTS-007 wet-road reflections off"});
    }
    InstallMiscHooks(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallLewisChallenge(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallScriptPatches(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallGt7ScriptDump(reinterpret_cast<u8*>(GetModuleHandleA(nullptr))); // DIAG-GT7-043
    Gt7Cipher::Install(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));     // PERF-GT7-113
    Gt7RaceContextGuard::Install(reinterpret_cast<u8*>(GetModuleHandleA(nullptr))); // FIX-GT7-162
    InstallTextEntryDiag(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    InstallFatalErrorDiag(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    if (!(std::getenv("RYGT_DECALS") && std::getenv("RYGT_DECALS")[0] == '0')) {
        SyncDecals();
    }
    auto* base = reinterpret_cast<u8*>(GetModuleHandleA(nullptr));
    char buffer[4096];
    const DWORD length = GetEnvironmentVariableA("RYGT_PATCH", buffer, sizeof(buffer));
    if (length == 0 || length >= sizeof(buffer)) {
        return;
    }
    std::string list{buffer, length};
    size_t start = 0;
    while (start < list.size()) {
        const size_t end = std::min(list.find(',', start), list.size());
        const std::string item = list.substr(start, end - start);
        const size_t a = item.find(':');
        const size_t b = item.find(':', a + 1);
        if (a != std::string::npos && b != std::string::npos) {
            Patch patch{std::strtoull(item.substr(0, a).c_str(), nullptr, 16),
                        ParseHex(item.substr(a + 1, b - a - 1)), ParseHex(item.substr(b + 1)),
                        "RYGT_PATCH " + item.substr(0, a)};
            Apply(base, patch);
        }
        start = end + 1;
    }
}

namespace Vulkan {
// FIX-110: true while a race project runs (see NoteProjectScript).
bool RytyGtsInRace() {
    return g_in_race.load(std::memory_order_relaxed);
}
} // namespace Vulkan
