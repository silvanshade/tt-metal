// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t rows = get_compile_time_arg_val(0);  // live rows, at most 8
constexpr uint32_t tiles = get_compile_time_arg_val(1);  // column tiles in the shard
constexpr uint32_t per = get_compile_time_arg_val(2);    // column tiles per core
constexpr uint32_t grid_x = get_compile_time_arg_val(3);
constexpr uint32_t half = get_compile_time_arg_val(4);  // 0: lower tiles of the core's range, 1: upper
constexpr uint32_t slice = rows * 32;                   // bytes of the live rows in one BF16 face

// Order-preserving key of a BF16 value, sign-extended by the load: negatives invert, non-negatives
// set bit 15, and -0 (key 0x7FFF) joins +0 (0x8000), as argmax compares them equal.
inline uint32_t key(int32_t bits) {
    const uint32_t k = static_cast<uint32_t>(bits ^ ((bits >> 15) | 0x8000)) & 0xFFFFu;
    return k + (k == 0x7FFFu);
}
}  // namespace

// Partial p = 2 * core + half scans its column tiles of the BF16 logits [1, 1, rows, 32 * tiles]
// (half 0 the lower part of the core's range) and writes, for each live row, its maximum and the
// first column holding it, as FP32, to face row p % 16 of face (p % 32 / 16) * 2 of partial tile
// p / 32: [values(rows) | columns(rows) | 0...]. Columns are visited in increasing order and
// replaced only on a strictly greater key, so partials in p order are in vocabulary order.
void kernel_main() {
    constexpr auto la = TensorAccessorArgs<5>();
    constexpr auto pa = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t first = core * per;
    const uint32_t last = first + per < tiles ? first + per : tiles;
    const uint32_t middle = first + (last - first + 1) / 2;
    const uint32_t begin = half == 0 ? first : middle;
    const uint32_t end = half == 0 ? middle : last;
    const auto logits = TensorAccessor(la, get_common_arg_val<uint32_t>(0), 2048);
    const auto partials = TensorAccessor(pa, get_common_arg_val<uint32_t>(1), 4096);
    DataflowBuffer scratch(half);
    const uint32_t base = get_write_ptr(half);
    Noc noc;
    for (uint32_t t = begin; t < end; ++t) {
        for (uint32_t face = 0; face < 2; ++face) {
            noc.async_read(
                logits,
                scratch,
                slice,
                {.page_id = t, .offset_bytes = face * 512},
                {.offset_bytes = ((t - begin) * 2 + face) * slice});
        }
    }
    noc.async_read_barrier();

    // Every real BF16 value keys above 0, so the first visited value always replaces the start.
    uint32_t best[rows];
    uint32_t column[rows];
    int32_t value[rows];
    for (uint32_t r = 0; r < rows; ++r) {
        best[r] = 0;
        column[r] = begin * 32;
        value[r] = -1;
    }
    for (uint32_t t = begin; t < end; ++t) {
        for (uint32_t face = 0; face < 2; ++face) {
            const auto* part = reinterpret_cast<const int16_t*>(base + ((t - begin) * 2 + face) * slice);
            const uint32_t start = t * 32 + face * 16;
            for (uint32_t r = 0; r < rows; ++r) {
                const int16_t* row = part + r * 16;
                uint32_t top = best[r];
                uint32_t at = 16;
                for (uint32_t c = 0; c < 16; ++c) {
                    const uint32_t k = key(row[c]);
                    if (k > top) {
                        top = k;
                        at = c;
                    }
                }
                if (at != 16) {
                    best[r] = top;
                    column[r] = start + at;
                    value[r] = row[at];
                }
            }
        }
    }

    auto* out = reinterpret_cast<uint32_t*>(base);
    for (uint32_t i = 0; i < 16; ++i) {
        out[i] = 0;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        out[r] = static_cast<uint32_t>(value[r] & 0xFFFF) << 16;
        const float index = static_cast<float>(column[r]);
        std::memcpy(&out[rows + r], &index, sizeof(index));
    }
    const uint32_t p = core * 2 + half;
    const uint32_t offset = ((p % 32) / 16) * 2 * 1024 + (p % 16) * 64;
    noc.async_write(scratch, partials, 64, {}, {.page_id = p / 32, .offset_bytes = offset});
    noc.async_write_barrier();
}
