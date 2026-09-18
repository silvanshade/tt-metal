// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

/// Gather standard BF16 tiles into one-face H128 operands.
///
/// # Specification
/// - requires: CB0/CB1 are BF16 faces; CB2 holds four standard BF16 tiles; assigned blocks do not overlap.
/// - ensures: publishes H16 once and 32 zero-padded face operands per assigned block.
/// - panics: none.
///
/// # Adequacy
/// Basis and multi-batch partial-row witnesses distinguish tile/face addressing and sign faults.
void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto source = TensorAccessor(args, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer tiles(2);
    cb_reserve_back(1, 1);
    auto* h = reinterpret_cast<volatile uint16_t*>(get_write_ptr(1));
    for (uint32_t r = 0; r < 16; ++r) {
        for (uint32_t c = 0; c < 16; ++c) {
            uint32_t bits = r & c;
            bits ^= bits >> 2;
            bits ^= bits >> 1;
            h[r * 16 + c] = (bits & 1) ? 0xbf80 : 0x3f80;
        }
    }
    cb_push_back(1, 1);
    for (uint32_t block = first; block < first + count; ++block) {
        tiles.reserve_back(4);
        for (uint32_t t = 0; t < 4; ++t) {
            noc.async_read(source, tiles, 2048, {.page_id = block * 4 + t}, {.offset_bytes = t * 2048});
        }
        noc.async_read_barrier();
        tiles.push_back(4);
        tiles.wait_front(4);
        const auto* src = reinterpret_cast<volatile const uint16_t*>(get_read_ptr(2));
        for (uint32_t row = 0; row < 32; ++row) {
            cb_reserve_back(0, 1);
            auto* dst = reinterpret_cast<volatile uint16_t*>(get_write_ptr(0));
            // Four standard tiles become the first eight rows of one face.
            for (uint32_t chunk = 0; chunk < 8; ++chunk) {
                const uint32_t offset = (chunk / 2) * 1024 + (row / 16) * 512 + (chunk % 2) * 256 + (row % 16) * 16;
                for (uint32_t col = 0; col < 16; ++col) {
                    dst[chunk * 16 + col] = src[offset + col];
                    dst[128 + chunk * 16 + col] = 0;
                }
            }
            cb_push_back(0, 1);
        }
        tiles.pop_front(4);
    }
}
