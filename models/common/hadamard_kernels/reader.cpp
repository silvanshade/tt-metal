// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

/// Stream whole input tiles and the stage pair into L1.
///
/// # Specification
/// - requires: CB0 holds standard tiles, CB1 two stage tiles; assigned groups exist; dynamic NoC mode.
/// - ensures: alternates activation and stage reads over both NoCs; publishes only after both barriers.
/// - panics: none.
///
/// # Adequacy
/// Multi-block and partial-row witnesses distinguish page-addressing faults; the
/// stage pair's sign is distinguished by the basis witnesses.
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t split = get_compile_time_arg_val(2);
    constexpr auto source_args = TensorAccessorArgs<3>();
    constexpr auto stage_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0));
    const auto stage = TensorAccessor(stage_args, get_arg_val<uint32_t>(1));
    const uint32_t tile_bytes = get_tile_size(0);
    Noc nocs[2] = {Noc(0), Noc(1)};
    DataflowBuffer input(0);
    DataflowBuffer stages(1);

    // The stage pair is resident for the program's lifetime: every group reuses it.
    stages.reserve_back(2);
    for (uint32_t page = 0; page < 2; ++page) {
        nocs[page & 1].async_read(stage, stages, tile_bytes, {.page_id = page}, {.offset_bytes = page * tile_bytes});
    }
    nocs[0].async_read_barrier();
    nocs[1].async_read_barrier();
    stages.push_back(2);

    for (uint32_t group = first; group < first + count; ++group) {
        // Every output tile of a row block reads all of the block's tiles, so a
        // split block is re-read rather than exchanged.
        const uint32_t block = group / split;
        input.reserve_back(tiles);
        for (uint32_t k = 0; k < tiles; ++k) {
            nocs[k & 1].async_read(source, input, tile_bytes, {.page_id = block * tiles + k}, {.offset_bytes = k * tile_bytes});
        }
        nocs[0].async_read_barrier();
        nocs[1].async_read_barrier();
        input.push_back(tiles);
    }
}
