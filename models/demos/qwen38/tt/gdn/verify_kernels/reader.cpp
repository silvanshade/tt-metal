// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
// Byte offset of element row `row` of a 32 x 32 FP32 tile, left face.
constexpr uint32_t row_offset(uint32_t row) { return ((row / 16) * 512 + (row % 16) * 16) * 4; }

void zero(uint32_t address, uint32_t tiles) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t chunk = 0; chunk < tiles * 4096 / MEM_ZEROS_SIZE; ++chunk) {
        noc_async_read(zeros, address + chunk * MEM_ZEROS_SIZE, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
}

// One core owns `tiles` consecutive column tiles (from `first`) of one head. Row r of the CB tiles
// receives this head's row of token r; rows past the block stay zero so they never reach a sum.
template <typename Accessor>
void gather(Noc& noc, const Accessor& tensor, uint32_t cb, uint32_t rows, uint32_t head_tiles, uint32_t head,
            uint32_t first, uint32_t tiles) {
    DataflowBuffer buffer(cb);
    buffer.reserve_back(tiles);
    zero(get_write_ptr(cb), tiles);
    DataflowBuffer dest(cb);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t page = (row * head_tiles + head / 32) * 4 + first;
        for (uint32_t t = 0; t < tiles; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc.async_read(
                    tensor,
                    dest,
                    64,
                    {.page_id = page + t, .offset_bytes = row_offset(head % 32) + face * 1024},
                    {.offset_bytes = t * 4096 + row_offset(row) + face * 1024});
            }
        }
    }
    noc.async_read_barrier();
    buffer.push_back(tiles);
}

template <typename Accessor>
uint32_t scalar(Noc& noc, const Accessor& tensor, uint32_t page, uint32_t col) {
    DataflowBuffer scratch(18);
    noc.async_read(tensor, scratch, 4096, {.page_id = page}, {});
    noc.async_read_barrier();
    const auto* source = reinterpret_cast<volatile uint32_t*>(get_write_ptr(18));
    return source[(col / 16) * 256 + col % 16];
}
}  // namespace

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
    // Initial state column block, (row tile, column) order. It gets its own CB: the resident state
    // CB is produced by the packer, whose private push count would overwrite this core's.
    DataflowBuffer initial(5);
    initial.reserve_back(4 * columns);
    DataflowBuffer target(5);
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < columns; ++c) {
            noc.async_read(
                state, target, 4096, {.page_id = head * 16 + r * 4 + first + c}, {.offset_bytes = (r * columns + c) * 4096});
        }
    }
    noc.async_read_barrier();
    initial.push_back(4 * columns);
    gather(noc, k, 1, rows, head_tiles, head, 0, 4);
    gather(noc, q, 0, rows, head_tiles, head, 0, 4);
    gather(noc, v, 2, rows, head_tiles, head, first, columns);

    // Per row: the decay argument broadcast over a tile (exponentiated on the SFPU), and beta on
    // the row's own line of an otherwise zero tile, which also masks the delta to that row.
    // The beta CB's two pages alternate; each keeps at most one nonzero line.
    uint32_t written[2] = {0, 0};
    bool fresh[2] = {true, true};
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t page = row * head_tiles + head / 32;
        const uint32_t decay = scalar(noc, g, page, head % 32);
        DataflowBuffer decays(3);
        decays.reserve_back(1);
        auto* words = reinterpret_cast<volatile uint32_t*>(get_write_ptr(3));
        for (uint32_t i = 0; i < 1024; ++i) {
            words[i] = decay;
        }
        decays.push_back(1);

        const uint32_t value = scalar(noc, beta, page, head % 32);
        DataflowBuffer betas(4);
        betas.reserve_back(1);
        const uint32_t slot = row % 2;
        const uint32_t base = get_write_ptr(4);
        if (fresh[slot]) {
            zero(base, 1);
            fresh[slot] = false;
        }
        auto* line = reinterpret_cast<volatile uint32_t*>(base);
        for (uint32_t face = 0; face < 2; ++face) {
            const uint32_t old = (row_offset(written[slot]) + face * 1024) / 4;
            const uint32_t now = (row_offset(row) + face * 1024) / 4;
            for (uint32_t i = 0; i < 16; ++i) {
                line[old + i] = 0;
            }
            for (uint32_t i = 0; i < 16; ++i) {
                line[now + i] = value;
            }
        }
        written[slot] = row;
        betas.push_back(1);
    }
}
