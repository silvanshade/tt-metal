// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t kv_heads = get_compile_time_arg_val(1);
constexpr uint32_t columns = get_compile_time_arg_val(2);  // tiles across the head dimension
constexpr uint32_t block = get_compile_time_arg_val(3);    // positions per cache page
constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
constexpr uint32_t row_bytes = get_compile_time_arg_val(5);  // value bytes per 16-value face row
constexpr uint32_t grid_x = get_compile_time_arg_val(6);
constexpr uint32_t position_page = get_compile_time_arg_val(7);  // aligned page sizes of the
constexpr uint32_t table_page = get_compile_time_arg_val(8);     // positions and page table
constexpr uint32_t position_slot = get_compile_time_arg_val(9);  // their 64 B aligned scratch slots
constexpr uint32_t table_slot = get_compile_time_arg_val(10);
constexpr uint32_t exponents = get_compile_time_arg_val(11);  // exponent section bytes, 0 for BF16

// Tile row `row`, column half `half` is face row ((row / 16) * 2 + half) * 16 + row % 16. A
// block-float tile starts with one exponent byte per face row (faces in order); then, as in BF16,
// come each face row's values.
constexpr uint32_t face_row(uint32_t row, uint32_t half) { return ((row / 16) * 2 + half) * 16 + row % 16; }

void copy_row(uint32_t source, uint32_t from, uint32_t destination, uint32_t to) {
    const auto* in = reinterpret_cast<const uint8_t*>(source);
    auto* out = reinterpret_cast<uint8_t*>(destination);
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t s = face_row(from, half), d = face_row(to, half);
        if constexpr (exponents != 0) {
            out[d] = in[s];
        }
        std::memcpy(out + exponents + d * row_bytes, in + exponents + s * row_bytes, row_bytes);
    }
}

// Scratch: every row's input tile, the cache tile being edited, the positions, page-table row 0.
template <typename Input, typename Target, typename Positions, typename Table>
void place(const Input& input, const Target& target, const Positions& positions, const Table& table, uint32_t head, uint32_t column) {
    DataflowBuffer scratch(0);
    const uint32_t base = get_write_ptr(0);
    constexpr uint32_t tile_offset = rows * tile_bytes;
    constexpr uint32_t position_offset = tile_offset + tile_bytes;
    constexpr uint32_t table_offset = position_offset + position_slot;
    Noc noc;
    for (uint32_t row = 0; row < rows; ++row) {
        noc.async_read(input, scratch, tile_bytes, {.page_id = row * columns + column}, {.offset_bytes = row * tile_bytes});
    }
    noc.async_read(positions, scratch, position_page, {.page_id = 0}, {.offset_bytes = position_offset});
    noc.async_read(table, scratch, table_page, {.page_id = 0}, {.offset_bytes = table_offset});
    noc.async_read_barrier();
    const auto* position = reinterpret_cast<const int32_t*>(base + position_offset);
    const auto* page = reinterpret_cast<const int32_t*>(base + table_offset);

    uint32_t open = ~0u;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t p = static_cast<uint32_t>(position[row]);
        const uint32_t within = p % block;
        const uint32_t id =
            ((static_cast<uint32_t>(page[p / block]) * kv_heads + head) * (block / 32) + within / 32) * columns + column;
        if (id != open) {
            if (open != ~0u) {
                noc.async_write(scratch, target, tile_bytes, {.offset_bytes = tile_offset}, {.page_id = open});
                noc.async_write_barrier();
            }
            noc.async_read(target, scratch, tile_bytes, {.page_id = id}, {.offset_bytes = tile_offset});
            noc.async_read_barrier();
            open = id;
        }
        copy_row(base + row * tile_bytes, head, base + tile_offset, within % 32);
    }
    noc.async_write(scratch, target, tile_bytes, {.offset_bytes = tile_offset}, {.page_id = open});
    noc.async_write_barrier();
}
}  // namespace

// Core (cache, head, column) owns column tile `column` of `head` in the K (cache 0) or V (cache 1)
// paged cache. Row r's input is row `head` of its tile r * columns + column; it lands on row p % 32
// of the cache tile holding position p = positions[r]. Each touched cache tile is read once, edited
// and written back by its one owner, and rows of a block-float tile carry their own exponents, so
// the placement is exact and race free.
void kernel_main() {
    constexpr auto ka = TensorAccessorArgs<12>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto kca = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto vca = TensorAccessorArgs<kca.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<vca.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t head = core / columns % kv_heads;
    const uint32_t column = core % columns;
    const auto positions = TensorAccessor(pa, get_common_arg_val<uint32_t>(4), position_page);
    const auto table = TensorAccessor(ta, get_common_arg_val<uint32_t>(5), table_page);
    if (core < kv_heads * columns) {
        const auto input = TensorAccessor(ka, get_common_arg_val<uint32_t>(0), tile_bytes);
        const auto target = TensorAccessor(kca, get_common_arg_val<uint32_t>(2), tile_bytes);
        place(input, target, positions, table, head, column);
    } else {
        const auto input = TensorAccessor(va, get_common_arg_val<uint32_t>(1), tile_bytes);
        const auto target = TensorAccessor(vca, get_common_arg_val<uint32_t>(3), tile_bytes);
        place(input, target, positions, table, head, column);
    }
}
