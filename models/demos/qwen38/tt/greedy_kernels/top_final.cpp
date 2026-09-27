// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include "top.hpp"

namespace {
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t partials = get_compile_time_arg_val(1);
constexpr uint32_t count = get_compile_time_arg_val(2);  // candidates per row, at most 32
constexpr uint32_t grid_x = get_compile_time_arg_val(3);
constexpr uint32_t page = count * 8;  // one partial row: [values(count) | columns(count)] FP32

// Namespace scope puts the list in the processor's local memory, not L1.
top::List<count> merged;
}  // namespace

// Core r merges row r of every partial list (each sorted descending), in vocabulary order, into the `count` largest values
// with ties to the lower column, and writes them to row r of two FP32 tiles [1, 1, rows, 32]:
// values (then -inf) and shard-local columns (then 0), in no particular order.
void kernel_main() {
    constexpr auto pa = TensorAccessorArgs<4>();
    constexpr auto va = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    const uint32_t r = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto lists = TensorAccessor(pa, get_common_arg_val<uint32_t>(0), page);
    const auto values = TensorAccessor(va, get_common_arg_val<uint32_t>(1), 4096);
    const auto indices = TensorAccessor(ia, get_common_arg_val<uint32_t>(2), 4096);
    DataflowBuffer scratch(0);
    const uint32_t base = get_write_ptr(0);
    Noc noc;
    for (uint32_t p = 0; p < partials; ++p) {
        noc.async_read(lists, scratch, page, {.page_id = p * rows + r}, {.offset_bytes = p * page});
    }
    noc.async_read_barrier();

    top::List<count>* list = &merged;
    list->reset();
    for (uint32_t p = 0; p < partials; ++p) {
        const auto* entry = reinterpret_cast<const int32_t*>(base + p * page);
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t bits = entry[i] >> 16;
            float column;
            std::memcpy(&column, &entry[count + i], sizeof(column));
            const uint32_t k = top::key(bits);
            if (k < list->low_key) {
                break;  // the partial is sorted descending: nothing after this can enter
            }
            list->offer(k, static_cast<uint32_t>(column), bits);
        }
    }

    // Row r of each output tile: [values | -inf...] and [columns | 0...], columns 0..15 in face 0
    // and 16..31 in face 1.
    auto* out = reinterpret_cast<uint32_t*>(base + partials * page);
    for (uint32_t i = 0; i < 32; ++i) {
        out[i] = i < count ? static_cast<uint32_t>(list->bits[i] & 0xFFFF) << 16 : 0xFF800000u;
        const float column = i < count ? static_cast<float>(list->columns[i]) : 0.0f;
        std::memcpy(&out[32 + i], &column, sizeof(column));
    }
    for (uint32_t face = 0; face < 2; ++face) {
        noc.async_write(scratch, values, 64, {.offset_bytes = partials * page + face * 64}, {.page_id = 0, .offset_bytes = face * 1024 + r * 64});
        noc.async_write(scratch, indices, 64, {.offset_bytes = partials * page + 128 + face * 64}, {.page_id = 0, .offset_bytes = face * 1024 + r * 64});
    }
    noc.async_write_barrier();
}
