// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <tt_stl/span.hpp>
#include <vector>
#include <tt_stl/assert.hpp>
#include "blockfloat_common.hpp"
#include "blockfloat_pack.hpp"
#include "constants.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "tile.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include "tt_backend_api_types.hpp"

uint32_t get_byte(uint32_t word, uint32_t index) {
    TT_ASSERT(index < 4);
    uint32_t mask = 0xff << (8 * index);
    uint32_t masked = word & mask;
    masked = masked >> (8 * index);
    return masked;
}

uint32_t convert_bfp_to_u32(tt::DataFormat bfp_format, uint8_t data, uint8_t shared_exp, bool is_exp_a) {
    uint32_t exp = shared_exp;
    uint32_t out_num = 0;
    if ((bfp_format == tt::DataFormat::Bfp2_b) || (bfp_format == tt::DataFormat::Bfp2)) {
        uint32_t sign = data >> 1;
        uint32_t man = data & 0x1;

        // Shift mantissa up until there is a 1 in bit 1
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            // shift again to put first non-hidden mantissa
            // bit in bit 1
            man = man << 1;
            man = man & 0x1;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 22);
    } else if ((bfp_format == tt::DataFormat::Bfp4_b) || (bfp_format == tt::DataFormat::Bfp4)) {
        uint32_t sign = data >> 3;
        uint32_t man = data & 0x7;

        // Shift mantissa up until there is a 1 in bit 3
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            while ((man & 0x04) == 0) {
                man = man << 1;
                shift_cnt++;
            }
            // shift one more time and zero the
            // hidden top mantissa bit
            // shift again to put first non-hidden mantissa
            // bit in bit 3
            man = man << 1;
            man = man & 0x7;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 20);
    } else if ((bfp_format == tt::DataFormat::Bfp8_b) || (bfp_format == tt::DataFormat::Bfp8)) {
        uint32_t sign = data >> 7;
        uint32_t man = data & 0x7f;

        // Shift mantissa up until there is a 1 in bit 6
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            // shift_cnt = 6 - (31 - __builtin_clz(man));
            // man = (man << (shift_cnt + 1)) & 0x7f;
            while ((man & 0x40) == 0) {
                man = man << 1;
                shift_cnt++;
            }
            // shift one more time and zero the
            // hidden top mantissa bit
            // shift again to put first non-hidden mantissa
            // bit in bit 7
            man = man << 1;
            man = man & 0x7f;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 16);
    }
    return out_num;
}

/// Convert one binary32 encoding using the shared scalar packing arithmetic.
/// # Specification
/// - requires: BfpFormat is a supported BFP format; shared_exp is its group exponent.
/// - ensures: Preserve exponent rebias, flushing, sign, saturation and selected rounding mode.
/// - panics: none.
/// # Adequacy
/// Numerical boundary comparisons distinguish nearest-even from truncation.
template <tt::DataFormat BfpFormat, bool truncate_bfp_mantissa>
uint8_t convert_u32_to_bfp(uint32_t input, uint32_t shared_exp, bool is_exp_a) {
    constexpr int bits = BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b   ? 2
                         : BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b ? 4
                                                                                                    : 8;
    return tt::tt_metal::detail::bfp::convert<bits, truncate_bfp_mantissa>(input, shared_exp, is_exp_a);
}

template <tt::DataFormat BfpFormat>
std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles(
    ttsl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return pack_as_bfp_tiles<BfpFormat, float>(fp32_vec, row_major_input, is_exp_a, tile);
}

/// Pack host values into independent BFP tiles.
/// # Specification
/// - requires: Supported BFP format and tile shape; input contains whole tiles.
/// - ensures: Preserve packed byte layout and numerical semantics in both input layouts.
/// - fails: Allocation and scheduler exceptions propagate.
/// - panics: Assertion failure for incomplete tiles.
/// # Adequacy
/// Packed-output comparisons cover numerical boundaries, layouts, partial faces and parallel calls.
template <tt::DataFormat BfpFormat, typename T>
std::vector<uint32_t> pack_as_bfp_tiles(
    ttsl::Span<const T> input_data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile) {
    TTZoneScopedD(DATA_FORMAT);
    constexpr int bits = BfpFormat == tt::DataFormat::Bfp2 || BfpFormat == tt::DataFormat::Bfp2_b   ? 2
                         : BfpFormat == tt::DataFormat::Bfp4 || BfpFormat == tt::DataFormat::Bfp4_b ? 4
                                                                                                    : 8;
    const auto height = tile ? tile->get_tile_shape()[0] : tt::constants::TILE_HEIGHT;
    const auto width = tile ? tile->get_tile_shape()[1] : tt::constants::TILE_WIDTH;
    const auto face_height = tile ? tile->get_face_shape()[0] : tt::constants::FACE_HEIGHT;
    TT_ASSERT(input_data.size() % (height * width) == 0);
    const auto alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);
    return tt::tt_metal::detail::bfp::pack_tiles<bits>(
        std::span<const T>(input_data.data(), input_data.size()),
        height,
        width,
        face_height,
        alignment,
        row_major_input,
        is_exp_a);
}

// Explicit instantiations
// clang-format off

// truncate_bfp_mantissa = false
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2_b, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4_b, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8_b, false>(uint32_t input, uint32_t shared_exp, bool is_exp_a);

// truncate_bfp_mantissa = true
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp2_b, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp4_b, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);
template uint8_t convert_u32_to_bfp<tt::DataFormat::Bfp8_b, true>(uint32_t input, uint32_t shared_exp, bool is_exp_a);

template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const float> fp32_vec, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const float> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const bfloat16> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const int32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const uint32_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);


template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const uint16_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);


template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const uint8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp2_b>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp4_b>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);
template std::vector<uint32_t> pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(ttsl::Span<const int8_t> input_data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile);

// clang-format on
