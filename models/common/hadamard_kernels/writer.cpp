// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

/// Write each finished output tile to its own page.
///
/// # Specification
/// - requires: CB16 carries `group_tiles` standard output tiles per assigned group.
/// - ensures: one whole-tile write per output column, at the page its row block owns.
/// - panics: none.
///
/// # Adequacy
/// Multi-block and split witnesses distinguish page-addressing faults; a poisoned
/// destination distinguishes a skipped write from a stale one.
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t group_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t split = get_compile_time_arg_val(2);
    constexpr auto args = TensorAccessorArgs<3>();
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const auto destination = TensorAccessor(args, get_arg_val<uint32_t>(0));
    const uint32_t tile_bytes = get_tile_size(16);
    Noc noc;
    DataflowBuffer results(16);

    for (uint32_t group = first; group < first + count; ++group) {
        const uint32_t block = group / split;
        const uint32_t column = (group % split) * group_tiles;
        results.wait_front(group_tiles);
        for (uint32_t t = 0; t < group_tiles; ++t) {
            noc.async_write(
                results,
                destination,
                tile_bytes,
                {.offset_bytes = t * tile_bytes},
                {.page_id = block * tiles + column + t});
        }
        noc.async_write_barrier();
        results.pop_front(group_tiles);
    }
}
