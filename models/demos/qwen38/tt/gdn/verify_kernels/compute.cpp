// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
namespace {
void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

void store(uint32_t cb, uint32_t tile = 0) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb, tile);
    tile_regs_release();
}

// SFPU arithmetic preserves FP32 state mantissas, unlike the FPU's elementwise path.
template <int Op>
void binary(uint32_t a, uint32_t b, uint32_t out, uint32_t tiles, bool scalar = false) {
    cb_wait_front(a, tiles);
    cb_wait_front(b, scalar ? 1 : tiles);
    cb_reserve_back(out, tiles);
    pack_reconfig_data_format(out);
    for (uint32_t t = 0; t < tiles; ++t) {
        tile_regs_acquire();
        load(a, t, 0);
        load(b, scalar ? 0 : t, 1);
        if constexpr (Op == 0) {
            mul_binary_tile(0, 1, 0);
        }
        if constexpr (Op == 1) {
            sub_binary_tile(0, 1, 0);
        }
        if constexpr (Op == 2) {
            add_binary_tile(0, 1, 0);
        }
        store(out, t);
    }
    cb_push_back(out, tiles);
}

void copy(uint32_t in, uint32_t out, uint32_t tiles) {
    cb_wait_front(in, tiles);
    cb_reserve_back(out, tiles);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_init(in);
    for (uint32_t t = 0; t < tiles; ++t) {
        tile_regs_acquire();
        copy_tile(in, t, 0);
        store(out, t);
    }
    cb_push_back(out, tiles);
}

void matmul(uint32_t a, uint32_t b, uint32_t out, uint32_t m, uint32_t n, uint32_t k) {
    cb_wait_front(a, m * k);
    cb_wait_front(b, k * n);
    cb_reserve_back(out, m * n);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    pack_reconfig_data_format(out);
    matmul_init(a, b);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            tile_regs_acquire();
            for (uint32_t p = 0; p < k; ++p) {
                matmul_tiles(a, b, i * k + p, p * n + j, 0);
            }
            store(out, i * n + j);
        }
    }
    cb_push_back(out, m * n);
}
}  // namespace

// Each value column of a head's state advances independently, so one core owns a 128 x (32 *
// columns) block: decay, k.h, delta, rank-one write and q.h touch only that block. Every row's
// state block leaves through CB 16; the host selects any accepted prefix from those snapshots.
void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    constexpr uint32_t n = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(5, 6);
    copy(5, 6, 4 * n);
    cb_pop_front(5, 4 * n);
    for (uint32_t row = 0; row < rows; ++row) {
        cb_wait_front(3, 1);
        cb_reserve_back(15, 1);
        pack_reconfig_data_format(15);
        tile_regs_acquire();
        load(3, 0, 0);
        exp_tile_init<false>();
        exp_tile<false>(0);
        store(15);
        cb_push_back(15, 1);
        cb_pop_front(3, 1);

        binary<0>(6, 15, 7, 4 * n, true);  // decayed state
        cb_pop_front(6, 4 * n);
        cb_pop_front(15, 1);
        matmul(1, 7, 8, 1, n, 4);  // k @ h
        binary<1>(2, 8, 9, n);     // delta = v - k @ h
        cb_pop_front(2, n);
        cb_pop_front(8, n);

        cb_reserve_back(10, 4);
        reconfig_data_format_srca(1);
        pack_reconfig_data_format(10);
        transpose_init(1);
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            transpose_tile(1, t, 0);
            store(10, t);
        }
        cb_push_back(10, 4);
        matmul(10, 9, 11, 4, n, 1);  // k^T delta
        cb_pop_front(10, 4);
        cb_pop_front(9, n);
        binary<0>(11, 4, 12, 4 * n, true);  // beta k^T delta
        cb_pop_front(11, 4 * n);
        cb_pop_front(4, 1);
        binary<2>(7, 12, 6, 4 * n);  // state remains resident between rows
        cb_pop_front(7, 4 * n);
        cb_pop_front(12, 4 * n);
        matmul(0, 6, 14, 1, n, 4);  // q @ h
        cb_pop_front(0, 4);
        cb_pop_front(1, 4);
        copy(6, 16, 4 * n);  // this row's state snapshot
    }
}
