// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Commit the recurrent state after `accepted` verifier inputs into one decode slot: the checkpoint
// for zero, else snapshot `accepted - 1`. Whole-tile copies of this core's page range.
void kernel_main() {
    const uint32_t fixed = get_arg_val<uint32_t>(0);  // accepted count, or ~0u to read `count`
    const uint32_t slot = get_arg_val<uint32_t>(1);
    const uint32_t begin = get_arg_val<uint32_t>(2);
    const uint32_t end = get_arg_val<uint32_t>(3);
    constexpr uint32_t pages = get_compile_time_arg_val(0);  // tiles per slot state
    constexpr uint32_t batch = get_compile_time_arg_val(1);
    constexpr auto ca = TensorAccessorArgs<2>();
    constexpr auto na = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<na.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto count = TensorAccessor(ca, get_arg_val<uint32_t>(4), 4096);
    const auto snapshots = TensorAccessor(na, get_arg_val<uint32_t>(5), 4096);
    const auto checkpoint = TensorAccessor(pa, get_arg_val<uint32_t>(6), 4096);
    const auto state = TensorAccessor(da, get_arg_val<uint32_t>(7), 4096);
    Noc noc;
    DataflowBuffer scratch(0);
    uint32_t accepted = fixed;
    if (fixed == ~0u) {
        noc.async_read(count, scratch, 64, {.page_id = 0}, {});
        noc.async_read_barrier();
        float value;
        std::memcpy(&value, reinterpret_cast<const void*>(get_write_ptr(0)), sizeof(value));
        accepted = static_cast<uint32_t>(value + 0.5f);
    }
    for (uint32_t first = begin; first < end; first += batch) {
        const uint32_t tiles = end - first < batch ? end - first : batch;
        for (uint32_t t = 0; t < tiles; ++t) {
            if (accepted == 0) {
                noc.async_read(checkpoint, scratch, 4096, {.page_id = first + t}, {.offset_bytes = t * 4096});
            } else {
                noc.async_read(
                    snapshots, scratch, 4096, {.page_id = (accepted - 1) * pages + first + t}, {.offset_bytes = t * 4096});
            }
        }
        noc.async_read_barrier();
        for (uint32_t t = 0; t < tiles; ++t) {
            noc.async_write(scratch, state, 4096, {.offset_bytes = t * 4096}, {.page_id = slot * pages + first + t});
        }
        noc.async_write_barrier();
    }
}
