// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

template <uint32_t block_ct, uint32_t sequence_tiles>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    constexpr uint32_t tap_count = 4;
    constexpr uint32_t scalar_one = 0x3f800000;
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer broadcast_weight(dfb::broadcast_weight);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer output(dfb::output);

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        if (item == 0 || work % sequence_tiles == 0) {
            weights.wait_front(tap_count * block_ct);
        }
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
            activation.wait_front(block_ct);
            if (tap != 0) {
                partial.wait_front(block_ct);
            }
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                // Match the row-broadcast addcmul kernel: expand BF16 weights before SFPU copies.
                broadcast_weight.reserve_back(1);
                reconfig_data_format(dfb::weights, dfb::weights);
                pack_reconfig_data_format(dfb::broadcast_weight);
                unary_bcast_init<BroadcastType::ROW>(dfb::weights);
                tile_regs_acquire();
                unary_bcast<BroadcastType::ROW>(dfb::weights, tap * block_ct + ct, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb::broadcast_weight);
                tile_regs_release();
                broadcast_weight.push_back(1);
                broadcast_weight.wait_front(1);

                partial.reserve_back(1);
                tile_regs_acquire();
                if (tap != 0) {
                    reconfig_data_format_srca(dfb::partial);
                    copy_init(dfb::partial);
                    copy_tile(dfb::partial, 0, 0);
                }
                reconfig_data_format_srca(dfb::act_tile);
                copy_init(dfb::act_tile);
                copy_tile(dfb::act_tile, ct, 1);
                reconfig_data_format_srca(dfb::broadcast_weight);
                copy_init(dfb::broadcast_weight);
                copy_tile(dfb::broadcast_weight, 0, 2);
                if (tap == 0) {
                    mul_binary_tile_init();
                    mul_binary_tile(1, 2, 0);
                } else {
                    addcmul_tile_init();
                    addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, scalar_one);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(dfb::partial);
                pack_tile(0, dfb::partial);
                tile_regs_release();
                partial.push_back(1);
                if (tap != 0) {
                    partial.pop_front(1);
                }
                broadcast_weight.pop_front(1);
            }
            activation.pop_front(block_ct);
        }

        // The final convolution sum crosses the same BF16 pack/unpack boundary as standalone SiLU.
        partial.wait_front(block_ct);
        reconfig_data_format_srca(dfb::partial);
        copy_init(dfb::partial);
        silu_tile_init();
        pack_reconfig_data_format(dfb::output);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            output.reserve_back(1);
            tile_regs_acquire();
            copy_tile(dfb::partial, 0, 0);
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::output);
            tile_regs_release();
            output.push_back(1);
            partial.pop_front(1);
        }
        if (item + 1 == wi_count || (work + 1) % sequence_tiles == 0) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
