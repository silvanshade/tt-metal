// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "nlp_concat_heads.hpp"

namespace ttnn::experimental {

ttnn::Tensor nlp_concat_heads(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    const auto& shape = input_tensor.logical_shape();
    TT_FATAL(shape.rank() == 4, "Input tensor must have rank 4. Shape: {}", shape);
    const auto& padded_shape = input_tensor.padded_shape();
    if (shape[3] % ttnn::types::TILE_SIZE == 0 && shape[3] == padded_shape[3] &&
        shape[1] == padded_shape[1] && shape[0] == padded_shape[0]) {
        return ttnn::prim::nlp_concat_heads(input_tensor, memory_config);
    }

    TT_FATAL(input_tensor.storage_type() == ttnn::StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have an allocated device buffer");
    TT_FATAL(input_tensor.layout() == tt::tt_metal::Layout::TILE, "Input tensor must use TILE layout");
    const auto dtype = input_tensor.dtype();
    TT_FATAL(
        dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B,
        "Unsupported data format");

    // Whole-tile copies cannot remove padding between logical heads. Use generic compaction
    // with interleaved scratch storage; the final conversion validates the requested placement.
    const auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    const MemoryConfig scratch_memory_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto decoded = ttnn::to_memory_config(input_tensor, scratch_memory_config);
    if (dtype == DataType::BFLOAT8_B) {
        // Keep both layout operations in BF16: intermediate BF8 packing would round twice.
        decoded = ttnn::typecast(decoded, DataType::BFLOAT16, scratch_memory_config);
    }
    auto token_major = ttnn::permute(decoded, {0, 2, 1, 3}, scratch_memory_config);
    auto flattened = ttnn::reshape(
        token_major,
        Shape{shape[0], 1, shape[2], shape[1] * shape[3]},
        scratch_memory_config,
        PadValue{0.0f},
        TileReshapeMapMode::CACHE,
        std::nullopt,
        false);
    if (dtype == DataType::BFLOAT8_B) {
        // Explicit zero padding above prevents shared exponents from observing uninitialized lanes.
        flattened = ttnn::typecast(flattened, dtype, scratch_memory_config);
    }
    return ttnn::to_memory_config(flattened, output_memory_config);
}

}  // namespace ttnn::experimental
