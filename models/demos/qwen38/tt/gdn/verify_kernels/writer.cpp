// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t columns = get_compile_time_arg_val(1);
    constexpr uint32_t heads = get_compile_time_arg_val(2);
    constexpr auto oa = TensorAccessorArgs<3>();
    constexpr auto sa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(3), 4096);
    const auto snapshots = TensorAccessor(sa, get_arg_val<uint32_t>(4), 4096);
    Noc noc;
    const uint32_t offset = (((head % 32) / 16) * 512 + (head % 16) * 16) * 4;
    for (uint32_t row = 0; row < rows; ++row) {
        // Output: row zero of each owned column tile lands in this head's row of the output tile.
        const uint32_t page = (row * head_tiles + head / 32) * 4 + first;
        DataflowBuffer out(14);
        out.wait_front(columns);
        DataflowBuffer source(14);
        for (uint32_t t = 0; t < columns; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc.async_write(
                    source,
                    output,
                    64,
                    {.offset_bytes = t * 4096 + face * 1024},
                    {.page_id = page + t, .offset_bytes = offset + face * 1024});
            }
        }
        // Snapshot: state after this row, [row, head] of [rows, heads, 128, 128].
        DataflowBuffer state(16);
        state.wait_front(4 * columns);
        DataflowBuffer block(16);
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < columns; ++c) {
                noc.async_write(
                    block,
                    snapshots,
                    4096,
                    {.offset_bytes = (r * columns + c) * 4096},
                    {.page_id = (row * heads + head) * 16 + r * 4 + first + c});
            }
        }
        noc.async_write_barrier();
        out.pop_front(columns);
        state.pop_front(4 * columns);
    }
}
