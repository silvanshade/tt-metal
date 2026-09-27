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
}  // namespace

// Scatter `rows` consecutive BF16 rows of one head into a ring of `ring` rows: source row i lands in
// ring row (base + i) mod ring, where base is a device FP32 scalar. Source [1, H, 32, W] (rows < 32)
// sits in L1, so half-row reads need no DRAM alignment; each destination half-row is one write.
void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t ring = get_compile_time_arg_val(1);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(2);
    constexpr auto ba = TensorAccessorArgs<3>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto base = TensorAccessor(ba, get_arg_val<uint32_t>(1), 4096);
    const auto source = TensorAccessor(sa, get_arg_val<uint32_t>(2), 2048);
    const auto ring_tensor = TensorAccessor(da, get_arg_val<uint32_t>(3), 2048);
    Noc noc;
    DataflowBuffer scratch(0);
    noc.async_read(base, scratch, 64, {.page_id = 0}, {});
    noc.async_read_barrier();
    float value;
    std::memcpy(&value, reinterpret_cast<const void*>(get_write_ptr(0)), sizeof(value));
    const uint32_t first = static_cast<uint32_t>(value + 0.5f) % ring;
    for (uint32_t c = 0; c < width_tiles; ++c) {
        noc.async_read(source, scratch, 2048, {.page_id = head * width_tiles + c}, {.offset_bytes = 64 + c * 2048});
    }
    noc.async_read_barrier();
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t slot = (first + i) % ring;
        const uint32_t page = (head * (ring / 32) + slot / 32) * width_tiles;
        for (uint32_t c = 0; c < width_tiles; ++c) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc.async_write(
                    scratch,
                    ring_tensor,
                    32,
                    {.offset_bytes = 64 + c * 2048 + row_offset(i, face)},
                    {.page_id = page + c, .offset_bytes = row_offset(slot % 32, face)});
            }
        }
    }
    noc.async_write_barrier();
}
