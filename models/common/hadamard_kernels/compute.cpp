// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/experimental/hadamard.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_api.h"

/// Accumulate each output tile from the row block's tiles and the stage pair.
///
/// H_w = H_(w/32) (x) H_32, so output tile j of a row block is
/// sum_k X_k * (H_m[j, k] H_32), and H_m[j, k] is the parity of the bits j and k
/// share. Every term is one tile matmul accumulating into the same FP32
/// destination, which keeps the served matmul's single-rounding arithmetic.
///
/// The stage operand's entries are signed powers of two, so only two of the four
/// fidelity phases can contribute. `stage_mop` selects the Hadamard stage
/// primitive, which issues exactly those two; the stock matmul path issues
/// HiFi3's three for the same result and is kept selectable so the two can be
/// measured against each other.
///
/// # Specification
/// - requires: CB0 carries `tiles` input tiles per group, CB1 the +/- stage pair,
///   CB16 accepts `group_tiles` results; FP32 destination accumulation.
/// - ensures: one packed output tile per assigned output column, input consumed per group.
/// - panics: none.
///
/// # Adequacy
/// Float64 basis witnesses distinguish a stage-sign or column-order fault;
/// repeated groups expose stale destination accumulation; running both paths
/// against the same golden distinguishes a phase-selection fault in the stage MOP.
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t group_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t split = get_compile_time_arg_val(2);
    constexpr uint32_t scale = get_compile_time_arg_val(3);
    constexpr bool stage_mop = get_compile_time_arg_val(4) != 0;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(0, 16);
    reconfig_data_format<SrcOrder::Reverse>(0, 1);
    pack_reconfig_data_format(16);
    if constexpr (stage_mop) {
        hadamard_stage_init(0, 1);
    } else {
        matmul_init(0, 1);
    }
    cb_wait_front(1, 2);

    for (uint32_t group = first; group < first + count; ++group) {
        const uint32_t column = (group % split) * group_tiles;
        cb_wait_front(0, tiles);
        cb_reserve_back(16, group_tiles);
        for (uint32_t t = 0; t < group_tiles; ++t) {
            const uint32_t j = column + t;
            tile_regs_acquire();
            for (uint32_t k = 0; k < tiles; ++k) {
                // Stage page 0 is +H_32 and page 1 is -H_32; H_m[j, k] = (-1)^|j & k|.
                const uint32_t sign = __builtin_parity(j & k);
                if constexpr (stage_mop) {
                    hadamard_stage_tile(0, 1, k, sign, 0);
                } else {
                    matmul_tiles(0, 1, k, sign, 0);
                }
            }
            if constexpr (scale != 0) {
                // Widths whose normalization is not a power of two cannot ride on the
                // stage tile in bfloat16; they take an exact FP32 scale on dest instead.
                mul_unary_tile(0, scale);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 16, t);
            tile_regs_release();
        }
        cb_push_back(16, group_tiles);
        cb_pop_front(0, tiles);
    }
    cb_pop_front(1, 2);
    if constexpr (stage_mop) {
        hadamard_stage_uninit();
    }
}
