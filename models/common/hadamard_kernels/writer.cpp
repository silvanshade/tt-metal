// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

/// Assemble BFP8 faces into whole standard output tiles.
///
/// # Specification
/// - requires: CB16 contains 272-byte faces; CB17 holds four 1088-byte tiles; assigned blocks are disjoint.
/// - ensures: preserves exponent and mantissa bytes and writes four complete tiles per block.
/// - panics: none.
///
/// # Adequacy
/// Exponent-scaled and partial-row witnesses distinguish exponent-offset and face-placement faults.
void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto destination = TensorAccessor(args, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer tiles(17);
    for (uint32_t block = first; block < first + count; ++block) {
        tiles.reserve_back(4);
        auto* dst = reinterpret_cast<volatile uint8_t*>(get_write_ptr(17));
        for (uint32_t row = 0; row < 32; ++row) {
            cb_wait_front(16, 1);
            const auto* src = reinterpret_cast<volatile const uint8_t*>(get_read_ptr(16));
            // Preserve the primitive's BFP8 exponents and mantissas verbatim.
            // A single face has 16 exponent bytes; a full tile has 64.
            for (uint32_t chunk = 0; chunk < 8; ++chunk) {
                const uint32_t tile = (chunk / 2) * 1088;
                const uint32_t face = (row / 16) * 2 + chunk % 2;
                dst[tile + face * 16 + row % 16] = src[chunk];
                for (uint32_t col = 0; col < 16; ++col) {
                    dst[tile + 64 + face * 256 + (row % 16) * 16 + col] = src[16 + chunk * 16 + col];
                }
            }
            cb_pop_front(16, 1);
        }
        tiles.push_back(4);
        tiles.wait_front(4);
        for (uint32_t t = 0; t < 4; ++t) {
            noc.async_write(tiles, destination, 1088, {.offset_bytes = t * 1088}, {.page_id = block * 4 + t});
        }
        noc.async_write_barrier();
        tiles.pop_front(4);
    }
}
