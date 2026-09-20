// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_hadamard_stage.h"
#include "sanitizer/api.h"

inline void llk_math_hadamard_stage_init() {
    SAN_HOOK(unsupported());
    _llk_math_hadamard_stage_init_();
}

inline void llk_math_hadamard_stage(uint32_t dst_index) {
    SAN_HOOK(unsupported());
    LLK_ASSERT(
        (dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "llk_math_hadamard_stage: destination tile index exceeds the acquired destination capacity");
    _llk_math_hadamard_stage_(dst_index);
}

inline void llk_math_hadamard_stage_uninit() {
    SAN_HOOK(unsupported());
    _llk_math_hadamard_stage_uninit_();
}
