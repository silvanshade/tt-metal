// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Core h writes value head h's four gated tiles.
void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr auto ga = TensorAccessorArgs<1>();
    const uint32_t head = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto gated = TensorAccessor(ga, get_common_arg_val<uint32_t>(0), 4096);
    Noc noc;
    DataflowBuffer out(7);
    out.wait_front(4);
    DataflowBuffer source(7);
    for (uint32_t t = 0; t < 4; ++t) {
        noc.async_write(source, gated, 4096, {.offset_bytes = t * 4096}, {.page_id = head * 4 + t});
    }
    noc.async_write_barrier();
    out.pop_front(4);
}
