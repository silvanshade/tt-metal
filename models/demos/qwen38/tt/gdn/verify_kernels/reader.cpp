// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// One core owns `columns` value-column tiles of one head. Gather that head's row into row zero of
// `tiles` consecutive tiles starting at `first`; zero padding keeps other rows out of the matmuls.
template <typename Accessor>
void read_row(Noc& noc, const Accessor& tensor, uint32_t cb, uint32_t page, uint32_t row, uint32_t first, uint32_t tiles) {
    DataflowBuffer buffer(cb);
    buffer.reserve_back(tiles);
    DataflowBuffer dest(cb);
    auto* words = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb));
    for (uint32_t i = 0; i < tiles * 1024; ++i) {
        words[i] = 0;
    }
    const uint32_t offset = ((row / 16) * 512 + (row % 16) * 16) * 4;
    for (uint32_t t = 0; t < tiles; ++t) {
        for (uint32_t face = 0; face < 2; ++face) {
            noc.async_read(
                tensor,
                dest,
                64,
                {.page_id = page + first + t, .offset_bytes = offset + face * 1024},
                {.offset_bytes = t * 4096 + face * 1024});
        }
    }
    noc.async_read_barrier();
    buffer.push_back(tiles);
}

template <typename Accessor>
void read_scalar(Noc& noc, const Accessor& tensor, uint32_t cb, uint32_t page, uint32_t col) {
    DataflowBuffer scratch(18);
    noc.async_read(tensor, scratch, 4096, {.page_id = page}, {});
    noc.async_read_barrier();
    const auto* source = reinterpret_cast<volatile uint32_t*>(get_write_ptr(18));
    const uint32_t value = source[(col / 16) * 256 + col % 16];
    DataflowBuffer buffer(cb);
    buffer.reserve_back(1);
    auto* words = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb));
    for (uint32_t i = 0; i < 1024; ++i) {
        words[i] = value;
    }
    buffer.push_back(1);
}

void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr auto qa = TensorAccessorArgs<2>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(3), 4096);
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(4), 4096);
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(5), 4096);
    const auto g = TensorAccessor(ga, get_arg_val<uint32_t>(6), 4096);
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(7), 4096);
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(8), 4096);
    Noc noc;
    // State column block, row-major over (row tile, owned column tile).
    cb_reserve_back(5, 4 * columns);
    DataflowBuffer target(5);
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < columns; ++c) {
            noc.async_read(
                state, target, 4096, {.page_id = head * 16 + r * 4 + first + c}, {.offset_bytes = (r * columns + c) * 4096});
        }
    }
    noc.async_read_barrier();
    cb_push_back(5, 4 * columns);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t page = (row * head_tiles + head / 32) * 4;
        read_row(noc, q, 0, page, head % 32, 0, 4);
        read_row(noc, k, 1, page, head % 32, 0, 4);
        read_row(noc, v, 2, page, head % 32, first, columns);
        read_scalar(noc, g, 3, row * head_tiles + head / 32, head % 32);
        read_scalar(noc, beta, 4, row * head_tiles + head / 32, head % 32);
    }
}
