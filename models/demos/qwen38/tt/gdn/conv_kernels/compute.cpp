// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
namespace {
constexpr uint32_t kSelectors = 0, kInputs = 1, kDiagonals = 2, kHistory = 3, kHistoryOut = 4, kShifted = 5;
constexpr uint32_t kActivated = 6;
// Selector tiles: 0..3 place tap state m on row m, 4 moves token t to row 4 + t, 5..8 lift row
// t + s to row t for s = 1..4.
constexpr uint32_t kPlace = 0, kTokens = 4, kShift = 5;
}  // namespace

// Every step is a matmul with a 0/1 selector or a diagonal, so the history and the shifted rows
// stay exact in BF16 and the four taps accumulate in FP32 DST:
//   history = sum_m P_m @ state_m + D @ x          (rows: tap states, then tokens)
//   conv    = sum_j (S_{j+1} @ history) @ diag(w_j) = sum_j w_j * history[t + j + 1]
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t per = get_compile_time_arg_val(1);
    constexpr uint32_t grid_x = get_compile_time_arg_val(2);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t begin = core * per;
    const uint32_t end = begin + per < tiles ? begin + per : tiles;
    compute_kernel_hw_startup(kSelectors, kInputs, kHistory);
    cb_wait_front(kSelectors, 9);
    for (uint32_t c = begin; c < end; ++c) {
        cb_wait_front(kInputs, 5);
        cb_reserve_back(kHistory, 1);
        cb_reserve_back(kHistoryOut, 1);
        pack_reconfig_data_format(kHistory);
        matmul_init(kSelectors, kInputs);
        tile_regs_acquire();
        for (uint32_t m = 0; m < 4; ++m) {
            matmul_tiles(kSelectors, kInputs, kPlace + m, m, 0);
        }
        matmul_tiles(kSelectors, kInputs, kTokens, 4, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kHistory);
        pack_tile(0, kHistoryOut);
        tile_regs_release();
        cb_push_back(kHistory, 1);
        cb_push_back(kHistoryOut, 1);
        cb_pop_front(kInputs, 5);

        cb_wait_front(kHistory, 1);
        cb_reserve_back(kShifted, 4);
        matmul_init(kSelectors, kHistory);
        tile_regs_acquire();
        for (uint32_t j = 0; j < 4; ++j) {
            matmul_tiles(kSelectors, kHistory, kShift + j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < 4; ++j) {
            pack_tile(j, kShifted, j);
        }
        tile_regs_release();
        cb_push_back(kShifted, 4);
        cb_pop_front(kHistory, 1);

        cb_wait_front(kShifted, 4);
        cb_wait_front(kDiagonals, 4);
        cb_reserve_back(kActivated, 1);
        matmul_init(kShifted, kDiagonals);
        tile_regs_acquire();
        for (uint32_t j = 0; j < 4; ++j) {
            matmul_tiles(kShifted, kDiagonals, j, j, 0);
        }
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(kActivated);
        pack_tile(0, kActivated);
        tile_regs_release();
        cb_push_back(kActivated, 1);
        cb_pop_front(kShifted, 4);
        cb_pop_front(kDiagonals, 4);
    }
    cb_pop_front(kSelectors, 9);
}
