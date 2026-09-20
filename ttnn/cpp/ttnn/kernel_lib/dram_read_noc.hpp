// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/noc.h"

namespace dataflow_kernel_lib {

// Blackhole admits roughly one DRAM read request per 16.4 cycles per endpoint per NoC, so a
// per-tile reader pinned to one NoC stalls well below channel bandwidth
// (tenstorrent/tt-metal#55898). Under NOC_MODE::DM_DYNAMIC_NOC either data-movement RISC may issue
// on either NoC, so alternating consecutive per-tile requests across the two NoCs doubles the
// request rate admitted at one endpoint.
//
// A read names a source and a destination and nothing else. A multicast write additionally names a
// rectangle whose corners are ordered by the NoC's own traversal direction, which is why a host
// factory swaps the start and end corners when it puts a multicasting kernel on NOC1. So moving a
// read between NoCs cannot change which bytes land where, and outputs stay bit-identical to the
// single-NoC path, while writes and multicasts keep the kernel's own NoC.
//
// DRAM_READ_BOTH_NOC is emitted by a program factory only when it also puts the kernel group in
// DM_DYNAMIC_NOC. Without it this collapses to the kernel's configured NoC, and the alternation
// state compiles away.
class DramReadNoc {
public:
    // The NoC to issue the next per-tile read request on.
    FORCE_INLINE const Noc& next() {
#ifdef DRAM_READ_BOTH_NOC
        // One counter for the whole read rather than one per row, so an odd tile count per row
        // cannot bias one NoC.
        return (request_count_++ & 1u) ? alt_noc_ : noc_;
#else
        return noc_;
#endif
    }

    // Waits for every read issued through next(), on each NoC that carried one. Under
    // DM_DYNAMIC_NOC the wait compares the NIU status register against the counters of both
    // data-movement processors, so it also covers the other RISC's reads on that NoC.
    FORCE_INLINE void read_barrier() const {
        noc_.async_read_barrier();
#ifdef DRAM_READ_BOTH_NOC
        alt_noc_.async_read_barrier();
#endif
    }

private:
    Noc noc_{noc_index};
#ifdef DRAM_READ_BOTH_NOC
    Noc alt_noc_{static_cast<uint8_t>(1 - noc_index)};
    uint32_t request_count_ = 0;
#endif
};

}  // namespace dataflow_kernel_lib
