// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <oneapi/tbb/task_arena.h>
#include "impl/data_format/blockfloat_common.hpp"
#include "impl/data_format/blockfloat_pack.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>
#include "jit_build/data_format.hpp"

TEST(HostBfpPack, NumericalBoundariesPreservePackedWords) {
    namespace bfp = tt::tt_metal::detail::bfp;
    auto check = []<int Bits>() {
        for (bool exp_a : {false, true}) {
            for (uint32_t exponent : {0u, 1u, 111u, 112u, 113u, 126u, 127u, 143u, 144u, 254u, 255u}) {
                // Same-exponent rounding ties, opposite signs, zero/subnormal flushing,
                // exponent rebias saturation and shifts beyond the source word width.
                constexpr unsigned shift = 25 - Bits;
                std::array<uint32_t, 16> raw{
                    0u,
                    0x80000000u,
                    1u,
                    0x80000001u,
                    (exponent << 23),
                    (exponent << 23) | 0x80000000u,
                    (exponent << 23) | ((1u << (shift - 1)) - 1),
                    (exponent << 23) | (1u << (shift - 1)),
                    (exponent << 23) | ((1u << (shift - 1)) + 1),
                    (exponent << 23) | 0x7fffffu,
                    (exponent << 23) | 0x807fffffu,
                    0x00800000u,
                    0x37800000u,
                    0x38000000u,
                    0x3f800000u,
                    0xbf800000u};
                unsigned shared = 0;
                for (auto value : raw) {
                    shared = std::max(shared, (value >> 23) & 255);
                }
                if (exp_a) {
                    shared = std::clamp(int(shared) - 112, 0, 31);
                }
                std::vector<uint32_t> expected(4 + Bits / 2, 0);
                expected[0] = shared;
                for (unsigned lane = 0; lane < raw.size(); ++lane) {
                    const auto value = raw[lane];
                    int exp = (value >> 23) & 255;
                    unsigned quantized = 0;
                    if (exp != 0) {
                        unsigned significand = value & 0x7fffff;
                        if (exp_a) {
                            exp -= 112;
                            if (exp < 0) {
                                exp = 0;
                                significand = 0;
                            }
                            if (exp > 31) {
                                exp = 31;
                                significand = 0x7fffff;
                            }
                        }
                        significand += 0x800000;
                        while (unsigned(exp) < shared) {
                            significand /= 2;
                            ++exp;
                        }
                        const unsigned divisor = 1u << shift;
                        quantized = significand / divisor;
                        const auto remainder = significand % divisor;
                        if (remainder > divisor / 2 || (remainder == divisor / 2 && quantized % 2)) {
                            ++quantized;
                        }
                        quantized = std::min(quantized, (1u << (Bits - 1)) - 1);
                        if (quantized) {
                            quantized |= (value >> 31) << (Bits - 1);
                        }
                    }
                    expected[4 + lane / (32 / Bits)] |= quantized << ((lane % (32 / Bits)) * Bits);
                }
                std::array<float, 16> input;
                std::transform(
                    raw.begin(), raw.end(), input.begin(), [](uint32_t v) { return std::bit_cast<float>(v); });
                EXPECT_EQ(bfp::pack_tiles<Bits>(std::span<const float>(input), 1, 16, 1, 16, true, exp_a), expected);
            }
        }
    };
    check.template operator()<2>();
    check.template operator()<4>();
    check.template operator()<8>();
}

TEST(HostBfpPack, FaceOrderPaddingAndNestedCallers) {
    namespace bfp = tt::tt_metal::detail::bfp;
    oneapi::tbb::task_arena arena(2);
    for (unsigned height : {1u, 8u, 32u}) {
        for (unsigned width : {16u, 32u}) {
            const unsigned face_height = std::min(height, 16u);
            const unsigned elements = height * width;
            const unsigned tile_words = 16 + elements / 4;
            std::vector<float> row_major(65 * elements), face_major(row_major.size());
            std::vector<uint32_t> expected(65 * tile_words, 0);
            for (unsigned tile = 0; tile < 65; ++tile) {
                unsigned group = 0;
                for (unsigned fy = 0; fy < height; fy += face_height) {
                    for (unsigned fx = 0; fx < width; fx += 16) {
                        for (unsigned y = 0; y < face_height; ++y, ++group) {
                            unsigned exponent = 100 + (tile + (fy + y) * 2 + fx / 16) % 60;
                            expected[tile * tile_words + group / 4] |= exponent << (8 * (group % 4));
                            for (unsigned lane = 0; lane < 16; ++lane) {
                                unsigned magnitude = 64 + lane;
                                unsigned sign = lane % 2;
                                float value = std::bit_cast<float>((sign << 31) | (exponent << 23) | (lane << 17));
                                row_major[tile * elements + (fy + y) * width + fx + lane] = value;
                                face_major[tile * elements + group * 16 + lane] = value;
                                expected[tile * tile_words + 16 + group * 4 + lane / 4] |= (magnitude | (sign << 7))
                                                                                           << (8 * (lane % 4));
                            }
                        }
                    }
                }
            }
            arena.execute([&] {
                oneapi::tbb::parallel_for(0, 4, [&](int caller) {
                    bool is_row_major = caller % 2 == 0;
                    const auto& input = is_row_major ? row_major : face_major;
                    EXPECT_EQ(
                        bfp::pack_tiles<8>(
                            std::span<const float>(input), height, width, face_height, 64, is_row_major, false),
                        expected);
                });
            });
        }
    }
}

namespace {

void roundtrip_test_for_mantissa_rounding_with_bfp8(
    float float_input, uint8_t expected_mantissa, float expected_float_output) {
    auto uint32_input = std::bit_cast<uint32_t>(float_input);
    // Set shared exponent as original float exponent (ie. skip logic for handling shared exponents)
    auto shared_exp = uint32_input >> 23 & 0xFF;

    auto output_mantissa = convert_u32_to_bfp<tt::DataFormat::Bfp8_b, false>(uint32_input, shared_exp, false);
    EXPECT_EQ(output_mantissa, expected_mantissa);

    uint32_t uint32_output = convert_bfp_to_u32(tt::DataFormat::Bfp8_b, output_mantissa, shared_exp, false);
    float float_output = std::bit_cast<float>(uint32_output);
    EXPECT_EQ(float_output, expected_float_output);
};

}  // namespace

struct ConvertU32ToBfpParams {
    float float_input = 0;
    uint32_t expected_mantissa = 0;
    float expected_float_output = 0;
};

class ConvertU32ToBfpTests : public ::testing::TestWithParam<ConvertU32ToBfpParams> {};

TEST_P(ConvertU32ToBfpTests, CPU_MantissaRoundingWithPositiveFloat) {
    const auto& params = GetParam();
    roundtrip_test_for_mantissa_rounding_with_bfp8(
        params.float_input, params.expected_mantissa, params.expected_float_output);
}

TEST_P(ConvertU32ToBfpTests, CPU_MantissaRoundingWithNegativeFloat) {
    const auto& params = GetParam();
    const auto float_input = -1 * params.float_input;
    const auto expected_mantissa = params.expected_mantissa | 0x80;
    const auto expected_float_output = -1 * params.expected_float_output;

    roundtrip_test_for_mantissa_rounding_with_bfp8(float_input, expected_mantissa, expected_float_output);
}

INSTANTIATE_TEST_SUITE_P(
    BlockfloatCommonTests,
    ConvertU32ToBfpTests,
    // clang-format off
    // See tests/tt_metal/tt_metal/api/test_blockfloat_common.cpp for explanation of rounding
    // NOTE: These float values are cherry-picked such that:
    // - The mantissa hits the 4 cases for rounding
    // - The float values match behaviour of round(float) (assuming same spec of ties round to even)
    ::testing::Values(
        // Round up always
        ConvertU32ToBfpParams{
            .float_input = 64.75,  // Mantissa is 0x18000
            .expected_mantissa = 0x41,
            .expected_float_output = 65,
        },
        // Round down always
        ConvertU32ToBfpParams{
            .float_input = 65.25,  // Mantissa is 0x28000
            .expected_mantissa = 0x41,
            .expected_float_output = 65,
        },
        // Tie: round down to nearest even
        ConvertU32ToBfpParams{
            .float_input = 64.5,  // Mantissa is 0x10000
            .expected_mantissa = 0x40,
            .expected_float_output = 64,
        },
        // Tie: round up to nearest even
        ConvertU32ToBfpParams{
            .float_input = 65.5,  // Mantissa is 0x30000
            .expected_mantissa = 0x42,
            .expected_float_output = 66,
        }
    )  // Values
    // clang-format on
);

// FP8_E4M3 is supported on Blackhole and Quasar but not Wormhole. Verify the arch guard in
// get_single_pack_src_format() matches that: QUASAR and BLACKHOLE pass, WORMHOLE_B0 throws.
// Host-only: calls the public get_pack_src_formats() wrapper, no device required.
TEST(DataFormatFp8ArchGuard, Fp8E4m3PackSrcFormatPerArch) {
    const std::array<tt::DataFormat, 1> fp8_formats{tt::DataFormat::Fp8_e4m3};
    constexpr auto unpack_dst = tt::DataFormat::Float16_b;

    EXPECT_NO_THROW(tt::get_pack_src_formats(
        fp8_formats,
        unpack_dst,
        /*fp32_dest_acc_en=*/true,
        /*bfp8_pack_precise=*/false,
        /*int_fpu_en=*/false,
        tt::ARCH::QUASAR));

    EXPECT_NO_THROW(tt::get_pack_src_formats(fp8_formats, unpack_dst, true, false, false, tt::ARCH::BLACKHOLE));

    EXPECT_ANY_THROW(tt::get_pack_src_formats(fp8_formats, unpack_dst, true, false, false, tt::ARCH::WORMHOLE_B0));
}
