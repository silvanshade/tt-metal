// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK math primitive for one Hadamard stage: a 32x32 tile matmul whose SrcA
// operand is a signed power of two, run in exactly the two fidelity phases
// that can contribute.
//
// The Hadamard rotation of a 32m-wide row block factors as H_32m = H_m (x)
// H_32, so every term is X_k * (+/-H_32/norm): a full tile matmul against a
// constant whose entries are +/-2^-e. The matrix engine splits each operand's
// mantissa and accumulates one product per fidelity phase:
//
//   phase 0 = srcA_hi * srcB_hi
//   phase 1 = srcA_lo * srcB_hi
//   phase 2 = srcA_hi * srcB_lo
//   phase 3 = srcA_lo * srcB_lo
//
// Matmul routes in0 (the activation) to SrcB and in1 (the stage tile) to SrcA.
// A signed power of two has an all-zero mantissa below the hidden bit, so
// srcA_lo is identically zero and phases 1 and 3 add nothing: phases {0, 2}
// are exact for a bfloat16 activation, whose 8 mantissa bits need both SrcB
// halves. Cumulative fidelity cannot express that set — LoFi runs {0}, HiFi2
// {0,1}, HiFi3 {0,1,2}, HiFi4 all four — so the cheapest exact stock setting
// is HiFi3, which spends one pass computing zero. This primitive keeps the
// matmul MOP and steps the phase counter by two instead of one, so the same
// exact result costs two passes rather than three (and half of HiFi4's four).
//
// Everything else is the stock 32x32 matmul: the same address mods, the same
// 16-MVMUL replay sequence, the same destination walk. Only ADDR_MOD_5's
// fidelity increment and the MOP's inner-loop count differ.

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_matmul.h"

using namespace ckernel;

// Number of MVMULs the stock full-tile (32x32 by 32x32, whole faces) matmul
// records in the replay buffer; see matmul_configure_mop.
constexpr std::uint32_t kHadamardStageReplayLength = 16;

// Fidelity phases this primitive issues, and the step between them.
constexpr std::uint32_t kHadamardStagePhases   = 2;
constexpr std::uint8_t kHadamardStagePhaseStep = 2;

// Configure the math thread for Hadamard stage tiles. Call once per kernel,
// after the one-time HW configuration and before any stage tile.
//
// Precondition: both operands are whole 32x32 tiles, no transpose, and the
// SrcA operand's entries are signed powers of two. A stage tile that is not
// (a rounded 1/sqrt(w) folded into it, say) makes phase 1 significant and this
// primitive inexact; normalize on dest instead.
inline void _llk_math_hadamard_stage_init_()
{
    // Stock matmul address mods for one whole-tile output, no transpose.
    matmul_configure_addrmod<MathFidelity::HiFi3, 0>(false /*transpose*/, TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /*partial_face*/);

    // Re-set the end-of-pass mod with a phase step of two: pass 1 runs at
    // phase 0 and pass 2 at phase 2, skipping the zero-valued phase 1.
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 1},
        .srcb     = {.incr = 0, .clr = 1, .cr = 1},
        .dest     = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = kHadamardStagePhaseStep, .clr = 0},
    }
        .set(ADDR_MOD_5);

    // Load the stock replay buffer, then re-program the MOP over it with two
    // inner loops instead of the fidelity enum's three.
    matmul_configure_mop<MathFidelity::HiFi3>(1 /*ct_dim*/, 1 /*rt_dim*/, TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /*partial_face*/);
    ckernel_template mop(1 /*outer loop*/, kHadamardStagePhases, lltt::replay_insn(ckernel::math::replay_buf_offset, kHadamardStageReplayLength));
    // ct_dim >= rt_dim, so the stage tile in SrcA is the reused operand.
    mop.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
    mop.program();

    math::reset_counters(p_setrwc::SET_ABD_F);

    // Matmul runs in the operand-driven DEFAULT zero-substitution state and
    // never sets the flag itself; re-establish it exactly as the stock init does.
    math::_configure_default_zero_flag_state_();
}

// Accumulate one stage term into the destination tile at dst_index.
// The unpack thread must have fed the activation to SrcB and the stage tile to
// SrcA, as for a stock matmul.
inline void _llk_math_hadamard_stage_(const std::uint32_t dst_index)
{
    _llk_math_matmul_<MathFidelity::HiFi3, 0>(dst_index, 1 /*ct_dim*/, 1 /*rt_dim*/);
}

// The primitive restores nothing: it leaves the matmul address mods and a MOP
// programmed, exactly as the stock matmul init does, and every math LLK
// programs the mods and MOP it needs in its own init.
inline void _llk_math_hadamard_stage_uninit_()
{
}
