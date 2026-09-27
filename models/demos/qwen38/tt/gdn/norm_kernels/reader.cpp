// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
template <typename Accessor>
void read_tiles(Noc& noc, const Accessor& tensor, uint32_t cb, uint32_t first, uint32_t bytes) {
    DataflowBuffer buffer(cb);
    buffer.reserve_back(4);
    DataflowBuffer dest(cb);
    for (uint32_t t = 0; t < 4; ++t) {
        noc.async_read(tensor, dest, bytes, {.page_id = first + t}, {.offset_bytes = t * bytes});
    }
    noc.async_read_barrier();
    buffer.push_back(4);
}
}  // namespace

// Core h of the row-major grid owns value head h: its four output and z tiles (token t on row t),
// the four norm-weight tiles, and a ones tile for the row sums.
void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    constexpr auto za = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<za.next_compile_time_args_offset()>();
    const uint32_t head = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto output = TensorAccessor(oa, get_common_arg_val<uint32_t>(0), 4096);
    const auto z = TensorAccessor(za, get_common_arg_val<uint32_t>(1), 2048);
    const auto weight = TensorAccessor(wa, get_common_arg_val<uint32_t>(2), 2048);
    Noc noc;
    read_tiles(noc, z, 1, head * 4, 2048);
    read_tiles(noc, weight, 2, 0, 2048);
    DataflowBuffer ones(3);
    ones.reserve_back(1);
    auto* one = reinterpret_cast<volatile uint32_t*>(get_write_ptr(3));
    for (uint32_t i = 0; i < 1024; ++i) {
        one[i] = 0x3F800000;
    }
    ones.push_back(1);
    read_tiles(noc, output, 0, head * 4, 4096);
}
