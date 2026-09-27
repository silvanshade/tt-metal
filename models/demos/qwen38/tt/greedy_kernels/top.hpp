// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstring>

namespace top {

// Order-preserving key of a BF16 value held sign-extended in an int32: negatives invert,
// non-negatives set bit 15, and -0 (key 0x7FFF) joins +0 (0x8000). Every real value keys above 0.
inline uint32_t key(int32_t bits) {
    const uint32_t k = static_cast<uint32_t>(bits ^ ((bits >> 15) | 0x8000)) & 0xFFFFu;
    return k + (k == 0x7FFFu);
}

// The `N` largest offered values, ties to the lower column, as a binary heap whose root is the
// entry the next better offer evicts: the smallest key, and among equal keys the largest column.
// `low_key` / `low_column` mirror the root once full (0 / ~0 before). Kernels keep lists at
// namespace scope, in the processor's local memory, so `reset` replaces a constructor.
template <uint32_t N>
struct List {
    uint32_t keys[N];
    uint32_t columns[N];
    int32_t bits[N];
    uint32_t size;
    uint32_t low_key;
    uint32_t low_column;

    void reset() {
        for (uint32_t i = 0; i < N; ++i) {
            keys[i] = 0;
            columns[i] = ~0u;
            bits[i] = -1;
        }
        size = 0;
        low_key = 0;
        low_column = ~0u;
    }

    // Entry a leaves before entry b.
    bool before(uint32_t a, uint32_t b) const {
        return keys[a] < keys[b] || (keys[a] == keys[b] && columns[a] > columns[b]);
    }

    void swap(uint32_t a, uint32_t b) {
        const uint32_t k = keys[a], c = columns[a];
        const int32_t v = bits[a];
        keys[a] = keys[b];
        columns[a] = columns[b];
        bits[a] = bits[b];
        keys[b] = k;
        columns[b] = c;
        bits[b] = v;
    }

    void offer(uint32_t k, uint32_t column, int32_t value) {
        if (size < N) {
            uint32_t i = size++;
            keys[i] = k;
            columns[i] = column;
            bits[i] = value;
            while (i > 0 && before(i, (i - 1) / 2)) {
                swap(i, (i - 1) / 2);
                i = (i - 1) / 2;
            }
        } else {
            if (k < keys[0] || (k == keys[0] && column >= columns[0])) {
                return;
            }
            keys[0] = k;
            columns[0] = column;
            bits[0] = value;
            uint32_t i = 0;
            while (true) {
                const uint32_t left = 2 * i + 1, right = left + 1;
                uint32_t first = i;
                if (left < N && before(left, first)) {
                    first = left;
                }
                if (right < N && before(right, first)) {
                    first = right;
                }
                if (first == i) {
                    break;
                }
                swap(i, first);
                i = first;
            }
        }
        if (size == N) {
            low_key = keys[0];
            low_column = columns[0];
        }
    }

    // Order entries by key descending, then column ascending, so a merge can stop reading a list
    // at its first entry below its threshold. The list is no longer a heap afterwards.
    void sort() {
        for (uint32_t i = 1; i < N; ++i) {
            const uint32_t k = keys[i], column = columns[i];
            const int32_t value = bits[i];
            uint32_t j = i;
            while (j > 0 && (keys[j - 1] < k || (keys[j - 1] == k && columns[j - 1] > column))) {
                keys[j] = keys[j - 1];
                columns[j] = columns[j - 1];
                bits[j] = bits[j - 1];
                --j;
            }
            keys[j] = k;
            columns[j] = column;
            bits[j] = value;
        }
    }

    // [values(N) | columns(N)] as FP32 words; a value is its BF16 bits in the high half.
    void store(uint32_t* out) const {
        for (uint32_t i = 0; i < N; ++i) {
            out[i] = static_cast<uint32_t>(bits[i] & 0xFFFF) << 16;
            const float column = static_cast<float>(columns[i]);
            std::memcpy(&out[N + i], &column, sizeof(column));
        }
    }
};

}  // namespace top
