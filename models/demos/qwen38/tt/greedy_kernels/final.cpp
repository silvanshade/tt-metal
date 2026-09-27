// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t partials = get_compile_time_arg_val(1);  // partial rows per device
constexpr uint32_t devices = get_compile_time_arg_val(2);   // shards gathered along columns
constexpr uint32_t shard = get_compile_time_arg_val(3);     // vocabulary columns per shard
constexpr uint32_t grid_x = get_compile_time_arg_val(4);
constexpr uint32_t tile_rows = (partials + 31) / 32;

inline uint32_t key(int32_t bits) {
    const uint32_t k = static_cast<uint32_t>(bits ^ ((bits >> 15) | 0x8000)) & 0xFFFFu;
    return k + (k == 0x7FFFu);
}
}  // namespace

// Core r reduces row r over every device's partials [values(rows) | columns(rows)] (FP32, one face
// row per partial; device d's partials are column tile d of each tile row) in vocabulary order,
// device then partial, replacing only on a strictly greater key: the first index of the maximum,
// as argmax over the gathered logits. It writes token = column + d * shard as FP32 to column 0 of
// face row r of the output tile, and zeros to the rest of that face row.
void kernel_main() {
    constexpr auto ga = TensorAccessorArgs<5>();
    constexpr auto oa = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const uint32_t r = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto gathered = TensorAccessor(ga, get_common_arg_val<uint32_t>(0), 4096);
    const auto output = TensorAccessor(oa, get_common_arg_val<uint32_t>(1), 4096);
    DataflowBuffer scratch(0);
    const uint32_t base = get_write_ptr(0);
    Noc noc;
    for (uint32_t t = 0; t < tile_rows; ++t) {
        for (uint32_t d = 0; d < devices; ++d) {
            noc.async_read(gathered, scratch, 4096, {.page_id = t * devices + d}, {.offset_bytes = (d * tile_rows + t) * 4096});
        }
    }
    noc.async_read_barrier();

    uint32_t best = 0;
    float token = 0.0f;
    for (uint32_t d = 0; d < devices; ++d) {
        for (uint32_t p = 0; p < partials; ++p) {
            const uint32_t offset = (d * tile_rows + p / 32) * 4096 + ((p % 32) / 16) * 2 * 1024 + (p % 16) * 64;
            const auto* partial = reinterpret_cast<const int32_t*>(base + offset);
            const uint32_t candidate = key(partial[r] >> 16);
            if (candidate > best) {
                float column;
                std::memcpy(&column, &partial[rows + r], sizeof(column));
                best = candidate;
                token = column + static_cast<float>(d * shard);
            }
        }
    }

    const uint32_t out = base + devices * tile_rows * 4096;
    auto* row = reinterpret_cast<uint32_t*>(out);
    for (uint32_t i = 0; i < 16; ++i) {
        row[i] = 0;
    }
    std::memcpy(row, &token, sizeof(token));
    noc.async_write(scratch, output, 64, {.offset_bytes = devices * tile_rows * 4096}, {.page_id = 0, .offset_bytes = r * 64});
    noc.async_write_barrier();
}
