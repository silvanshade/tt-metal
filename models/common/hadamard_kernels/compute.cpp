// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/experimental/hadamard.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"

/// Compute normalized H128 rows and pack BFP8 faces.
///
/// # Specification
/// - requires: BF16 face operands in CB0/CB1, BFP8 faces in CB16, full destination synchronization.
/// - ensures: packs one normalized transform per row; calls the primitive uninitializer.
/// - panics: none.
///
/// # Adequacy
/// Host references distinguish normalization faults; repeated rows expose stale destination accumulation.
void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 16);
    hadamard_h128_init<true>(0, 1, 16);
    cb_wait_front(1, 1);
    for (uint32_t row = 0; row < rows; ++row) {
        cb_wait_front(0, 1);
        cb_reserve_back(16, 1);
        tile_regs_acquire();
        MATH(TTI_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, ADDR_MOD_7, 0));
        hadamard_h128_tile<true>(0, 1, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
        cb_pop_front(0, 1);
    }
    cb_pop_front(1, 1);
    // The primitive owns no restoration; subsequent kernels initialize their
    // own address modifiers and SFPU state rather than inheriting this setup.
    hadamard_h128_uninit();
}
