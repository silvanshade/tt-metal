// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Each channel tile's history (BF16) and activation (FP32) land on page c of their tensors.
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t per = get_compile_time_arg_val(1);
    constexpr uint32_t grid_x = get_compile_time_arg_val(2);
    constexpr auto ha = TensorAccessorArgs<3>();
    constexpr auto aa = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto history = TensorAccessor(ha, get_common_arg_val<uint32_t>(0), 2048);
    const auto activated = TensorAccessor(aa, get_common_arg_val<uint32_t>(1), 4096);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t begin = core * per;
    const uint32_t end = begin + per < tiles ? begin + per : tiles;
    Noc noc;
    for (uint32_t c = begin; c < end; ++c) {
        DataflowBuffer rows(4);
        rows.wait_front(1);
        DataflowBuffer row(4);
        noc.async_write(row, history, 2048, {}, {.page_id = c});
        DataflowBuffer out(6);
        out.wait_front(1);
        DataflowBuffer value(6);
        noc.async_write(value, activated, 4096, {}, {.page_id = c});
        noc.async_write_barrier();
        rows.pop_front(1);
        out.pop_front(1);
    }
}
