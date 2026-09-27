// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t kPage = 2048;  // BF16 tile
}  // namespace

// Core i of the row-major grid convolves channel tiles [i * per, min((i + 1) * per, tiles)). Per
// tile it takes the four tap states and the block's rows (one tile each: states on row 0, token t
// on row t) and the four tap diagonals; the selectors arrive once.
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t per = get_compile_time_arg_val(1);
    constexpr uint32_t grid_x = get_compile_time_arg_val(2);
    constexpr auto sa = TensorAccessorArgs<3>();
    constexpr auto xa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto a0 = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    const auto selectors = TensorAccessor(sa, get_common_arg_val<uint32_t>(0), kPage);
    const auto x = TensorAccessor(xa, get_common_arg_val<uint32_t>(1), kPage);
    const auto s0 = TensorAccessor(a0, get_common_arg_val<uint32_t>(2), kPage);
    const auto s1 = TensorAccessor(a1, get_common_arg_val<uint32_t>(3), kPage);
    const auto s2 = TensorAccessor(a2, get_common_arg_val<uint32_t>(4), kPage);
    const auto s3 = TensorAccessor(a3, get_common_arg_val<uint32_t>(5), kPage);
    const auto diagonals = TensorAccessor(da, get_common_arg_val<uint32_t>(6), kPage);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t begin = core * per;
    const uint32_t end = begin + per < tiles ? begin + per : tiles;
    Noc noc;

    DataflowBuffer resident(0);
    resident.reserve_back(9);
    DataflowBuffer select(0);
    for (uint32_t i = 0; i < 9; ++i) {
        noc.async_read(selectors, select, kPage, {.page_id = i}, {.offset_bytes = i * kPage});
    }
    noc.async_read_barrier();
    resident.push_back(9);

    for (uint32_t c = begin; c < end; ++c) {
        DataflowBuffer inputs(1);
        inputs.reserve_back(5);
        DataflowBuffer in(1);
        noc.async_read(s0, in, kPage, {.page_id = c}, {.offset_bytes = 0 * kPage});
        noc.async_read(s1, in, kPage, {.page_id = c}, {.offset_bytes = 1 * kPage});
        noc.async_read(s2, in, kPage, {.page_id = c}, {.offset_bytes = 2 * kPage});
        noc.async_read(s3, in, kPage, {.page_id = c}, {.offset_bytes = 3 * kPage});
        noc.async_read(x, in, kPage, {.page_id = c}, {.offset_bytes = 4 * kPage});
        DataflowBuffer taps(2);
        taps.reserve_back(4);
        DataflowBuffer tap(2);
        for (uint32_t j = 0; j < 4; ++j) {
            noc.async_read(diagonals, tap, kPage, {.page_id = j * tiles + c}, {.offset_bytes = j * kPage});
        }
        noc.async_read_barrier();
        inputs.push_back(5);
        taps.push_back(4);
    }
}
