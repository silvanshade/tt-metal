// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t height_tiles = get_arg(args::in0_h_tiles);
    constexpr uint32_t width_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t heads = get_arg(args::in0_c);
#ifndef ARCH_QUASAR
    constexpr uint32_t read_group_tiles = get_arg(args::read_group_tiles);
#endif
    constexpr uint32_t output_row_tiles = heads * width_tiles;
    constexpr uint32_t batch_tiles = height_tiles * output_row_tiles;

    DataflowBuffer buffer(dfb::in0);
    const uint32_t tile_bytes = buffer.get_entry_size();
    const auto source = TensorAccessor(tensor::src);

    for (uint32_t done = 0; done < num_pages;) {
#ifdef ARCH_QUASAR
        // Quasar requires the cached DFB endpoint; advance it once per read.
        const uint32_t count = 1;
#else
        const uint32_t remaining = num_pages - done;
        const uint32_t count = remaining < read_group_tiles ? remaining : read_group_tiles;
#endif
        buffer.reserve_back(count);
#ifndef ARCH_QUASAR
        // Reacquire after every reservation: the next group can wrap the circular buffer.
        uint32_t write_address = buffer.get_write_ptr();
#endif
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t output_tile = start_id + done + i;
            const uint32_t batch = output_tile / batch_tiles;
            const uint32_t height = (output_tile / output_row_tiles) % height_tiles;
            const uint32_t head = (output_tile / width_tiles) % heads;
            const uint32_t width = output_tile % width_tiles;
            const uint32_t input_tile = ((batch * heads + head) * height_tiles + height) * width_tiles + width;
#ifdef ARCH_QUASAR
            noc.async_read(source, buffer, tile_bytes, {.page_id = input_tile}, {});
#else
            noc.async_read(source, CoreLocalMem<uint32_t>(write_address), tile_bytes, {.page_id = input_tile}, {});
            write_address += tile_bytes;
#endif
        }
        noc.async_read_barrier();
        buffer.push_back(count);
        done += count;
    }
}
