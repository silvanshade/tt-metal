// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <typename Accessor>
void write_row(Noc& noc, const Accessor& tensor, uint32_t cb, uint32_t page, uint32_t row) {
    DataflowBuffer buffer(cb);
    buffer.wait_front(4);
    DataflowBuffer source(cb);
    const uint32_t offset = ((row / 16) * 512 + (row % 16) * 16) * 4;
    for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t face = 0; face < 2; ++face) {
            noc.async_write(
                source,
                tensor,
                64,
                {.offset_bytes = t * 4096 + face * 1024},
                {.page_id = page + t, .offset_bytes = offset + face * 1024});
        }
    }
    noc.async_write_barrier();
    buffer.pop_front(4);
}

void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    constexpr auto da = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(2), 4096);
    const auto delta = TensorAccessor(da, get_arg_val<uint32_t>(3), 4096);
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(4), 4096);
    Noc noc;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t page = (row * head_tiles + head / 32) * 4;
        // Delta is produced first; draining it permits the recurrence to finish this row.
        write_row(noc, delta, 17, page, head % 32);
        write_row(noc, output, 14, page, head % 32);
    }
    cb_wait_front(16, 16);
    DataflowBuffer source(16);
    for (uint32_t t = 0; t < 16; ++t) {
        noc.async_write(source, state, 4096, {.offset_bytes = t * 4096}, {.page_id = head * 16 + t});
    }
    noc.async_write_barrier();
    cb_pop_front(16, 16);
}
