// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
namespace {
constexpr uint32_t kQ = 0, kK = 1, kV = 2, kDecay = 3, kBeta = 4, kInitial = 5, kState = 6, kDecayed = 7;
constexpr uint32_t kProjected = 8, kDelta = 9, kKT = 10, kOut = 14, kExp = 15, kSnapshot = 16;
// FP32 tiles in one half of DST.
constexpr uint32_t kDst = 4;

// Every CB here is FP32, so unpack and pack formats never change after startup.
void matmul(uint32_t a, uint32_t b, uint32_t out, uint32_t n, uint32_t k) {
    cb_reserve_back(out, n);
    matmul_init(a, b);
    for (uint32_t j = 0; j < n; ++j) {
        tile_regs_acquire();
        for (uint32_t p = 0; p < k; ++p) {
            matmul_tiles(a, b, p, p * n + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, j);
        tile_regs_release();
    }
    cb_push_back(out, n);
}
}  // namespace

// One core owns a 128 x (32 * n) value-column block of one head's state. Q, K and V arrive once
// with token r on row r of their tiles, so K^T is formed once and each row's matmuls read the
// resident tiles; row r's output is row r of Q @ S. The beta tile carries beta_r on row r only,
// so the delta it scales is zero on every other row and K^T @ delta is k_r^T delta_r.
// Every row's state block leaves through CB 16; the host selects any accepted prefix from those.
void kernel_main() {
    const uint32_t rows = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blocks = 4 * n;
    compute_kernel_hw_startup(kK, kKT);

    cb_wait_front(kK, 4);
    cb_reserve_back(kKT, 4);
    transpose_init(kK);
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        transpose_tile(kK, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kKT, t);
        tile_regs_release();
    }
    cb_push_back(kKT, 4);
    cb_wait_front(kKT, 4);
    cb_wait_front(kQ, 4);
    cb_wait_front(kV, n);

    for (uint32_t row = 0; row < rows; ++row) {
        // Decay factor, then the decayed state S * exp(g_r).
        cb_wait_front(kDecay, 1);
        cb_reserve_back(kExp, 1);
        tile_regs_acquire();
        copy_init(kDecay);
        copy_tile(kDecay, 0, 0);
        exp_tile_init<false>();
        exp_tile<false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kExp);
        tile_regs_release();
        cb_push_back(kExp, 1);
        cb_pop_front(kDecay, 1);

        // Row 0 reads the reader's initial state; later rows the state the packer left resident.
        const uint32_t source = row == 0 ? kInitial : kState;
        cb_wait_front(kExp, 1);
        cb_wait_front(source, blocks);
        cb_reserve_back(kDecayed, blocks);
        copy_init(source);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < blocks; ++t) {
            tile_regs_acquire();
            copy_tile(source, t, 0);
            copy_tile(kExp, 0, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, kDecayed, t);
            tile_regs_release();
        }
        cb_push_back(kDecayed, blocks);
        cb_pop_front(source, blocks);
        cb_pop_front(kExp, 1);

        // delta = beta_r (V - K @ S), nonzero on row r only.
        cb_wait_front(kDecayed, blocks);
        matmul(kK, kDecayed, kProjected, n, 4);
        cb_wait_front(kProjected, n);
        cb_wait_front(kBeta, 1);
        cb_reserve_back(kDelta, n);
        copy_init(kV);
        sub_binary_tile_init();
        for (uint32_t j = 0; j < n; ++j) {
            tile_regs_acquire();
            copy_tile(kV, j, 0);
            copy_tile(kProjected, j, 1);
            sub_binary_tile(0, 1, 0);
            copy_tile(kBeta, 0, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, kDelta, j);
            tile_regs_release();
        }
        cb_push_back(kDelta, n);
        cb_pop_front(kProjected, n);
        cb_pop_front(kBeta, 1);

        // S = decayed S + K^T @ delta, accumulated onto the reloaded decayed block in DST; the
        // same registers pack the resident state and this row's snapshot.
        cb_wait_front(kDelta, n);
        cb_reserve_back(kState, blocks);
        cb_reserve_back(kSnapshot, blocks);
        for (uint32_t first = 0; first < blocks; first += kDst) {
            const uint32_t count = blocks - first < kDst ? blocks - first : kDst;
            tile_regs_acquire();
            copy_init(kDecayed);
            for (uint32_t d = 0; d < count; ++d) {
                copy_tile(kDecayed, first + d, d);
            }
            matmul_init(kKT, kDelta);
            for (uint32_t d = 0; d < count; ++d) {
                matmul_tiles(kKT, kDelta, (first + d) / n, (first + d) % n, d);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < count; ++d) {
                pack_tile(d, kState, first + d);
                pack_tile(d, kSnapshot, first + d);
            }
            tile_regs_release();
        }
        cb_push_back(kState, blocks);
        cb_push_back(kSnapshot, blocks);
        cb_pop_front(kDecayed, blocks);
        cb_pop_front(kDelta, n);

        // Row r of Q @ S is this row's output; the writer extracts it.
        cb_wait_front(kState, blocks);
        matmul(kQ, kState, kOut, n, 4);
    }
    cb_pop_front(kState, blocks);
    cb_pop_front(kKT, 4);
    cb_pop_front(kQ, 4);
    cb_pop_front(kK, 4);
    cb_pop_front(kV, n);
}
