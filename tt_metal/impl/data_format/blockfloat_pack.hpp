// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TT_BFP_X86_DISPATCH 1
#endif

namespace tt::tt_metal::detail::bfp {

/// Convert one binary32 encoding to a sign-magnitude BFP value.
/// # Specification
/// - requires: Bits is 2, 4 or 8; shared_exp is the group's maximum adjusted exponent.
/// - ensures: Flush exponent-zero inputs; round aligned magnitude to nearest even and saturate.
/// - panics: none.
/// # Adequacy
/// Boundary comparisons must distinguish rebias clamps, discarded shifts, ties and negative zero.
template <int Bits, bool Truncate = false>
inline uint8_t convert(uint32_t input, uint32_t shared_exp, bool exp_a) {
    static_assert(Bits == 2 || Bits == 4 || Bits == 8);
    constexpr unsigned shift = 25 - Bits;
    constexpr unsigned maximum = (1u << (Bits - 1)) - 1;
    unsigned exponent = (input >> 23) & 255;
    if (exponent == 0) {
        return 0;
    }
    unsigned magnitude = input & 0x7fffff;
    if (exp_a) {
        if (exponent > 143) {
            magnitude = 0x7fffff;
        } else if (exponent < 112) {
            magnitude = 0;
        }
        exponent = std::clamp(int(exponent) - 112, 0, 31);
    }
    magnitude |= 0x800000;
    const unsigned distance = shared_exp > exponent ? shared_exp - exponent : 0;
    magnitude = distance >= 32 ? 0 : magnitude >> distance;
    if constexpr (Truncate) {
        magnitude >>= shift;
    } else {
        magnitude = std::min((magnitude + ((1u << (shift - 1)) - 1) + ((magnitude >> shift) & 1)) >> shift, maximum);
    }
    return magnitude == 0 ? 0 : magnitude | ((input >> 31) << (Bits - 1));
}

/// Pack one logical group without an ISA requirement.
/// # Specification
/// - requires: input contains 16 encodings; output has Bits/2 words.
/// - ensures: Return adjusted shared exponent; write little-lane-order packed magnitudes.
/// - panics: none.
/// # Adequacy
/// Independent packed-output comparisons observe every word and exponent.
template <int Bits>
inline uint8_t scalar_row(const void* input, uint32_t* output, bool exp_a) {
    // Byte loads accept both float storage and converted uint32_t encodings without aliasing.
    const auto load = [&](unsigned lane) {
        uint32_t value;
        std::memcpy(&value, static_cast<const std::byte*>(input) + lane * sizeof(value), sizeof(value));
        return value;
    };
    unsigned exponent = 0;
    for (unsigned i = 0; i < 16; ++i) {
        exponent = std::max(exponent, (load(i) >> 23) & 255);
    }
    if (exp_a) {
        exponent = std::clamp(int(exponent) - 112, 0, 31);
    }
    for (unsigned word = 0; word < Bits / 2; ++word) {
        uint32_t packed = 0;
        for (unsigned lane = 0; lane < 32 / Bits; ++lane) {
            packed |= uint32_t(convert<Bits>(load(word * (32 / Bits) + lane), exponent, exp_a)) << (lane * Bits);
        }
        output[word] = packed;
    }
    return exponent;
}

#if defined(TT_BFP_X86_DISPATCH)
/// Narrow sixteen sign-magnitude bytes into their format's words.
/// # Specification
/// - requires: AVX2; bytes fit Bits; output has Bits/2 words.
/// - ensures: Preserve input lane order in low-to-high packed bits.
/// - panics: none.
/// # Adequacy
/// Distinct lane values distinguish pack permutation and shift errors.
template <int Bits>
__attribute__((target("avx2"))) inline void store_bytes(__m128i bytes, uint32_t* output) {
    if constexpr (Bits == 8) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output), bytes);
    } else {
        const auto pairs = _mm_maddubs_epi16(bytes, _mm_set1_epi16((1 << (Bits + 8)) | 1));
        if constexpr (Bits == 4) {
            const auto packed = _mm_packus_epi16(pairs, pairs);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(output), packed);
        } else {
            const auto groups = _mm_madd_epi16(pairs, _mm_set1_epi32(0x00100001));
            const auto words = _mm_packus_epi32(groups, groups);
            output[0] = uint32_t(_mm_cvtsi128_si32(_mm_packus_epi16(words, words)));
        }
    }
}

/// Convert eight lanes using integer arithmetic only.
/// # Specification
/// - requires: AVX2; exponent is the adjusted maximum of the logical sixteen-lane group.
/// - ensures: Match convert for each lane, including shifts exceeding 31.
/// - panics: none.
/// # Adequacy
/// Biased numerical inputs distinguish rounding, saturation and zero masks.
template <int Bits>
__attribute__((target("avx2"))) inline __m256i convert8(__m256i input, unsigned shared_exp, bool exp_a) {
    constexpr unsigned shift = 25 - Bits;
    const auto zero = _mm256_setzero_si256();
    auto exponent = _mm256_and_si256(_mm256_srli_epi32(input, 23), _mm256_set1_epi32(255));
    const auto nonzero = _mm256_cmpgt_epi32(exponent, zero);
    auto magnitude = _mm256_and_si256(input, _mm256_set1_epi32(0x7fffff));
    if (exp_a) {
        magnitude = _mm256_blendv_epi8(
            magnitude, _mm256_set1_epi32(0x7fffff), _mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(143)));
        magnitude = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(112), exponent), magnitude);
        exponent = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_sub_epi32(exponent, _mm256_set1_epi32(112)), zero), _mm256_set1_epi32(31));
    }
    magnitude = _mm256_or_si256(magnitude, _mm256_set1_epi32(0x800000));
    magnitude = _mm256_srlv_epi32(magnitude, _mm256_sub_epi32(_mm256_set1_epi32(shared_exp), exponent));
    const auto odd = _mm256_and_si256(_mm256_srli_epi32(magnitude, shift), _mm256_set1_epi32(1));
    magnitude = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_add_epi32(magnitude, _mm256_set1_epi32((1u << (shift - 1)) - 1)), odd), shift);
    magnitude = _mm256_min_epu32(magnitude, _mm256_set1_epi32((1u << (Bits - 1)) - 1));
    magnitude = _mm256_and_si256(magnitude, nonzero);
    const auto sign = _mm256_slli_epi32(_mm256_srli_epi32(input, 31), Bits - 1);
    return _mm256_or_si256(magnitude, _mm256_and_si256(sign, _mm256_cmpgt_epi32(magnitude, zero)));
}

/// Pack one logical group in two eight-lane vectors.
/// # Specification
/// - requires: AVX2; input contains 16 encodings; output has Bits/2 words.
/// - ensures: Match scalar_row exactly.
/// - panics: none.
/// # Adequacy
/// Compare packed words against independent scalar arithmetic, not another SIMD backend.
template <int Bits>
__attribute__((target("avx2"))) inline uint8_t avx2_row(const void* input, uint32_t* output, bool exp_a) {
    const auto lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    const auto hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const std::byte*>(input) + 32));
    auto exponents = _mm256_max_epu32(
        _mm256_and_si256(_mm256_srli_epi32(lo, 23), _mm256_set1_epi32(255)),
        _mm256_and_si256(_mm256_srli_epi32(hi, 23), _mm256_set1_epi32(255)));
    auto maximum = _mm_max_epu32(_mm256_castsi256_si128(exponents), _mm256_extracti128_si256(exponents, 1));
    maximum = _mm_max_epu32(maximum, _mm_shuffle_epi32(maximum, 0x4e));
    maximum = _mm_max_epu32(maximum, _mm_shuffle_epi32(maximum, 0xb1));
    unsigned exponent = _mm_cvtsi128_si32(maximum);
    if (exp_a) {
        exponent = std::clamp(int(exponent) - 112, 0, 31);
    }
    auto words = _mm256_packs_epi32(convert8<Bits>(lo, exponent, exp_a), convert8<Bits>(hi, exponent, exp_a));
    words = _mm256_permute4x64_epi64(words, 0xd8);
    store_bytes<Bits>(_mm_packus_epi16(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1)), output);
    return exponent;
}

/// Pack one logical group in one sixteen-lane vector.
/// # Specification
/// - requires: AVX512F and AVX2; input contains 16 encodings; output has Bits/2 words.
/// - ensures: Match scalar_row exactly without requiring AVX512BW, DQ or VL.
/// - panics: none.
/// # Adequacy
/// Numerical boundaries and lane-distinct inputs observe masked conversion and narrowing.
template <int Bits>
__attribute__((target("avx512f,avx2"))) inline uint8_t avx512_row(const void* input, uint32_t* output, bool exp_a) {
    constexpr unsigned shift = 25 - Bits;
    const auto raw = _mm512_loadu_si512(input);
    const auto zero = _mm512_setzero_si512();
    auto exponent = _mm512_and_si512(_mm512_srli_epi32(raw, 23), _mm512_set1_epi32(255));
    const auto nonzero = _mm512_cmpneq_epi32_mask(exponent, zero);
    auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi32(0x7fffff));
    if (exp_a) {
        magnitude = _mm512_mask_mov_epi32(
            magnitude, _mm512_cmpgt_epi32_mask(exponent, _mm512_set1_epi32(143)), _mm512_set1_epi32(0x7fffff));
        magnitude = _mm512_mask_mov_epi32(magnitude, _mm512_cmplt_epi32_mask(exponent, _mm512_set1_epi32(112)), zero);
        exponent = _mm512_min_epi32(
            _mm512_max_epi32(_mm512_sub_epi32(exponent, _mm512_set1_epi32(112)), zero), _mm512_set1_epi32(31));
    }
    const unsigned shared_exp = _mm512_reduce_max_epu32(exponent);
    magnitude = _mm512_or_si512(magnitude, _mm512_set1_epi32(0x800000));
    magnitude = _mm512_srlv_epi32(magnitude, _mm512_sub_epi32(_mm512_set1_epi32(shared_exp), exponent));
    const auto odd = _mm512_and_si512(_mm512_srli_epi32(magnitude, shift), _mm512_set1_epi32(1));
    magnitude = _mm512_srli_epi32(
        _mm512_add_epi32(_mm512_add_epi32(magnitude, _mm512_set1_epi32((1u << (shift - 1)) - 1)), odd), shift);
    magnitude = _mm512_maskz_mov_epi32(nonzero, _mm512_min_epu32(magnitude, _mm512_set1_epi32((1u << (Bits - 1)) - 1)));
    const auto sign = _mm512_slli_epi32(_mm512_srli_epi32(raw, 31), Bits - 1);
    magnitude = _mm512_mask_or_epi32(magnitude, _mm512_cmpneq_epi32_mask(magnitude, zero), magnitude, sign);
    store_bytes<Bits>(_mm512_cvtepi32_epi8(magnitude), output);
    return shared_exp;
}
#endif

/// Choose a row kernel once per call, without global CPU compiler flags.
/// # Specification
/// - ensures: Returned kernel uses only instructions supported by CPU and OS vector state.
/// - panics: none.
/// # Adequacy
/// Baseline compilation and direct supported-backend comparisons supplement runtime dispatch.
template <int Bits>
inline auto row_kernel() -> uint8_t (*)(const void*, uint32_t*, bool) {
#if defined(TT_BFP_X86_DISPATCH)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx2")) {
        return avx512_row<Bits>;
    }
    if (__builtin_cpu_supports("avx2")) {
        return avx2_row<Bits>;
    }
#endif
    return scalar_row<Bits>;
}

/// Pack independent tiles directly into the final allocation.
/// # Specification
/// - requires: Supported tile shape with 16-column faces; input contains whole tiles; alignment is a multiple of four.
/// - ensures: Preserve tile/face/row order; zero exponent padding; disjoint tile writes.
/// - fails: Allocation or scheduler failures propagate to the caller.
/// - panics: none.
/// # Adequacy
/// Layout, partial-tile, empty-input and nested-arena comparisons observe output and completion.
template <int Bits, typename T>
inline std::vector<uint32_t> pack_tiles(
    std::span<const T> input,
    unsigned height,
    unsigned width,
    unsigned face_height,
    unsigned alignment,
    bool row_major,
    bool exp_a) {
    const size_t tile_elements = size_t(height) * width;
    const size_t tiles = input.size() / tile_elements;
    const size_t rows = tile_elements / 16;
    const size_t exponent_words = std::max(rows, size_t(alignment)) / 4;
    const size_t tile_words = exponent_words + tile_elements * Bits / 32;
    std::vector<uint32_t> output(tiles * tile_words, 0);
    const auto kernel = row_kernel<Bits>();
    auto pack_range = [&](size_t begin, size_t end) {
        std::array<uint32_t, 16> row;
        for (size_t tile = begin; tile < end; ++tile) {
            auto* destination = output.data() + tile * tile_words;
            size_t r = 0;
            for (size_t face_y = 0; face_y < height; face_y += face_height) {
                for (size_t face_x = 0; face_x < width; face_x += 16) {
                    for (size_t y = 0; y < face_height; ++y, ++r) {
                        const size_t offset =
                            tile * tile_elements + (row_major ? (face_y + y) * width + face_x : r * 16);
                        const void* source;
                        if constexpr (std::is_same_v<T, float>) {
                            source = input.data() + offset;
                        } else {
                            for (size_t lane = 0; lane < 16; ++lane) {
                                row[lane] = std::bit_cast<uint32_t>(static_cast<float>(input[offset + lane]));
                            }
                            source = row.data();
                        }
                        const uint32_t exponent = kernel(source, destination + exponent_words + r * (Bits / 2), exp_a);
                        destination[r / 4] |= exponent << ((r % 4) * 8);
                    }
                }
            }
        }
    };
    // Keep scheduler setup off small calls; grain scales with work, not tile shape.
    const size_t grain = std::max(size_t(1), size_t(32768) / tile_elements);
    if (input.size() <= 65536) {
        pack_range(0, tiles);
    } else {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, tiles, grain), [&](const auto& range) {
            pack_range(range.begin(), range.end());
        });
    }
    return output;
}

}  // namespace tt::tt_metal::detail::bfp

#undef TT_BFP_X86_DISPATCH
