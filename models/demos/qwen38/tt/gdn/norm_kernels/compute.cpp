// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
namespace {
// BF16: kZ, kWeight. FP32: the rest.
constexpr uint32_t kOut = 0, kZ = 1, kWeight = 2, kOnes = 3, kSquares = 4, kInverse = 5, kGate = 6, kGated = 7;
}  // namespace

// One value head: 128 output columns in four tiles, token t on row t of each.
// gated = out * sqrt(128) / sqrt(sum of the row's squares + 128 eps) * weight * silu(z), which is
// out / sqrt(mean(out^2) + eps) * weight * silu(z). A matmul of the squares against ones broadcasts
// each row's sum across the row, so no reduce or broadcast op runs.
void kernel_main() {
    constexpr uint32_t eps = get_compile_time_arg_val(0);    // FP32 bits of 128 * eps
    constexpr uint32_t scale = get_compile_time_arg_val(1);  // FP32 bits of sqrt(128)
    compute_kernel_hw_startup(kZ, kWeight, kGate);

    // weight * silu(z), from the two BF16 inputs.
    cb_wait_front(kZ, 4);
    cb_wait_front(kWeight, 4);
    cb_reserve_back(kGate, 4);
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        copy_init(kZ);
        copy_tile(kZ, t, 0);
        silu_tile_init();
        silu_tile(0);
        copy_init(kWeight);
        copy_tile(kWeight, t, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kGate, t);
        tile_regs_release();
    }
    cb_push_back(kGate, 4);
    cb_pop_front(kZ, 4);
    cb_pop_front(kWeight, 4);

    // Every operand from here on is FP32.
    reconfig_data_format(kOut, kOnes);
    cb_wait_front(kOut, 4);
    cb_reserve_back(kSquares, 4);
    copy_init(kOut);
    square_tile_init();
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        copy_tile(kOut, t, 0);
        square_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kSquares, t);
        tile_regs_release();
    }
    cb_push_back(kSquares, 4);

    cb_wait_front(kOnes, 1);
    cb_wait_front(kSquares, 4);
    cb_reserve_back(kInverse, 1);
    tile_regs_acquire();
    matmul_init(kSquares, kOnes);
    for (uint32_t t = 0; t < 4; ++t) {
        matmul_tiles(kSquares, kOnes, t, 0, 0);
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, kInverse);
    tile_regs_release();
    cb_push_back(kInverse, 1);
    cb_pop_front(kSquares, 4);

    cb_wait_front(kInverse, 1);
    cb_wait_front(kGate, 4);
    cb_reserve_back(kGated, 4);
    copy_init(kOut);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        copy_tile(kOut, t, 0);
        copy_tile(kInverse, 0, 1);
        mul_binary_tile(0, 1, 0);
        copy_tile(kGate, t, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kGated, t);
        tile_regs_release();
    }
    cb_push_back(kGated, 4);
    cb_pop_front(kInverse, 1);
    cb_pop_front(kGate, 4);
    cb_pop_front(kOut, 4);
    cb_pop_front(kOnes, 1);
}
