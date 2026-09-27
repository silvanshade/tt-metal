// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
// Byte offset of one row's 16-element BF16 half-row within a 32x32 tile.
constexpr uint32_t row_offset(uint32_t row, uint32_t face) { return ((row / 16) * 2 + face) * 512 + (row % 16) * 32; }

template <typename Accessor>
void forward(Noc& noc, DataflowBuffer& scratch, const Accessor& state, uint32_t tile, uint32_t local, uint32_t slot, uint32_t face) {
    noc.async_write(scratch, state, 32, {.offset_bytes = local}, {.page_id = tile, .offset_bytes = row_offset(slot, face)});
}
}  // namespace

// Convolution taps after `accepted` inputs: tap m is history row accepted + m, where history holds
// the checkpoint taps followed by the verified inputs, one BF16 row each within a single tile row.
// Each row moves into row `slot` of tap m's [1, B, width] state, face row by face row.
// DRAM reads need 64 B alignment, so read the aligned pair of 32 B face rows and forward one.
// All of a core's reads, then all of its writes, share one barrier each: the fold is latency-bound.
void kernel_main() {
    const uint32_t fixed = get_arg_val<uint32_t>(0);  // accepted count, or ~0u to read `count`
    const uint32_t slot = get_arg_val<uint32_t>(1);
    const uint32_t begin = get_arg_val<uint32_t>(2);
    const uint32_t end = get_arg_val<uint32_t>(3);
    constexpr auto ca = TensorAccessorArgs<0>();
    constexpr auto ha = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto s0 = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto s1 = TensorAccessorArgs<s0.next_compile_time_args_offset()>();
    constexpr auto s2 = TensorAccessorArgs<s1.next_compile_time_args_offset()>();
    constexpr auto s3 = TensorAccessorArgs<s2.next_compile_time_args_offset()>();
    const auto count = TensorAccessor(ca, get_arg_val<uint32_t>(4), 4096);
    const auto history = TensorAccessor(ha, get_arg_val<uint32_t>(5), 2048);
    const auto state0 = TensorAccessor(s0, get_arg_val<uint32_t>(6), 2048);
    const auto state1 = TensorAccessor(s1, get_arg_val<uint32_t>(7), 2048);
    const auto state2 = TensorAccessor(s2, get_arg_val<uint32_t>(8), 2048);
    const auto state3 = TensorAccessor(s3, get_arg_val<uint32_t>(9), 2048);
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
    const auto local = [&](uint32_t tile, uint32_t m, uint32_t face) {
        return ((tile - begin) * 8 + m * 2 + face) * 64 + (row_offset(accepted + m, face) & 63u);
    };
    for (uint32_t tile = begin; tile < end; ++tile) {
        for (uint32_t m = 0; m < 4; ++m) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc.async_read(
                    history,
                    scratch,
                    64,
                    {.page_id = tile, .offset_bytes = row_offset(accepted + m, face) & ~63u},
                    {.offset_bytes = ((tile - begin) * 8 + m * 2 + face) * 64});
            }
        }
    }
    noc.async_read_barrier();
    for (uint32_t tile = begin; tile < end; ++tile) {
        for (uint32_t face = 0; face < 2; ++face) {
            forward(noc, scratch, state0, tile, local(tile, 0, face), slot, face);
            forward(noc, scratch, state1, tile, local(tile, 1, face), slot, face);
            forward(noc, scratch, state2, tile, local(tile, 2, face), slot, face);
            forward(noc, scratch, state3, tile, local(tile, 3, face), slot, face);
        }
    }
    noc.async_write_barrier();
}
