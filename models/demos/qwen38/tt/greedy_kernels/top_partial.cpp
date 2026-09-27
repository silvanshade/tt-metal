// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include "top.hpp"

namespace {
constexpr uint32_t rows = get_compile_time_arg_val(0);  // live rows, at most 8
constexpr uint32_t tiles = get_compile_time_arg_val(1);  // column tiles in the shard
constexpr uint32_t per = get_compile_time_arg_val(2);    // column tiles per core
constexpr uint32_t grid_x = get_compile_time_arg_val(3);
constexpr uint32_t half = get_compile_time_arg_val(4);  // 0: lower tiles of the core's range, 1: upper
constexpr uint32_t count = get_compile_time_arg_val(5);  // candidates kept per row
constexpr uint32_t slice = rows * 32;                   // bytes of the live rows in one BF16 face

// Namespace scope puts the lists in the processor's local memory, not L1.
top::List<count> lists[rows];
}  // namespace

// Partial p = 2 * core + half scans its column tiles of the BF16 logits [1, 1, rows, 32 * tiles]
// (half 0 the lower part of the core's range) and keeps, per live row, the `count` largest values
// with ties to the lower column. Row r's list goes to page p * rows + r of the row-major FP32
// partials [partials * rows, 2 * count]: [values(count) | columns(count)], sorted by value
// descending, then column ascending.
void kernel_main() {
    constexpr auto la = TensorAccessorArgs<6>();
    constexpr auto pa = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t first = core * per;
    const uint32_t last = first + per < tiles ? first + per : tiles;
    const uint32_t middle = first + (last - first + 1) / 2;
    const uint32_t begin = half == 0 ? first : middle;
    const uint32_t end = half == 0 ? middle : last;
    const auto logits = TensorAccessor(la, get_common_arg_val<uint32_t>(0), 2048);
    const auto partials = TensorAccessor(pa, get_common_arg_val<uint32_t>(1), count * 8);
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

    for (uint32_t r = 0; r < rows; ++r) {
        lists[r].reset();
    }
    for (uint32_t t = begin; t < end; ++t) {
        for (uint32_t face = 0; face < 2; ++face) {
            const auto* part = reinterpret_cast<const int16_t*>(base + ((t - begin) * 2 + face) * slice);
            const uint32_t start = t * 32 + face * 16;
            for (uint32_t r = 0; r < rows; ++r) {
                const int16_t* row = part + r * 16;
                top::List<count>& list = lists[r];
                // Columns only increase here, so an equal key never displaces a kept one.
                uint32_t low = list.low_key;
                for (uint32_t c = 0; c < 16; ++c) {
                    const int32_t bits = row[c];
                    const uint32_t k = top::key(bits);
                    if (k > low) {
                        list.offer(k, start + c, bits);
                        low = list.low_key;
                    }
                }
            }
        }
    }

    // Output rows reuse the start of scratch, after every slice was read.
    const uint32_t out = base;
    for (uint32_t r = 0; r < rows; ++r) {
        lists[r].sort();
        lists[r].store(reinterpret_cast<uint32_t*>(out + r * count * 8));
    }
    const uint32_t p = core * 2 + half;
    for (uint32_t r = 0; r < rows; ++r) {
        noc.async_write(scratch, partials, count * 8, {.offset_bytes = r * count * 8}, {.page_id = p * rows + r});
    }
    noc.async_write_barrier();
}
