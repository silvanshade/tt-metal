# Real-Time Profiler: Dispatch Core, Profiler Core, and Host Interaction

This document describes how the **dispatch core** (dispatch_s), **real-time profiler core**, and **host** interact to stream program execution timestamps and metadata to the host for profiling (e.g. Tracy).

---

## 1. High-Level Architecture

```
+-----------------------------------------------------------------------------+
| HOST                                                                        |
|                                                                             |
|   +-------------------+  +------------------+  +-------------------------+  |
|   | Init/Calibration  |  | D2H Socket       |  | Receiver thread         |  |
|   | - Pick profiler   |  | - Config buffer  |  | - wait_for_pages()      |  |
|   |   core            |  | - Page flow      |  | - Parse timestamps      |  |
|   | - Create D2H      |  |   (PCIe)         |  | - InvokeProgramRealtime |  |
|   |   socket          |  |                  |  |   Callbacks()           |  |
|   | - Run sync        |  |                  |  |                         |  |
|   | - Start recv      |  |                  |  |                         |  |
|   +--------+----------+  +--------+---------+  +------------+------------+  |
|            |                     |                          |               |
|            | L1 writes           | PCIe read                | PCIe read     |
|            | (sync_request,      | (timestamp pages)        | (timestamp    |
|            |  sync_host_ts,      |                          |  pages)       |
|            |  config_buffer_addr)|                          |               |
+------------+---------------------+--------------------------+---------------+
             |                     |                          |               |
             v                     |                          |               |
+-----------------------------------------------------------------------------+
| DEVICE (per chip)                                                           |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   | REAL-TIME PROFILER CORE (Tensix, closest to PCIe)                   |   |
|   | Kernels: cq_realtime_profiler{,_push}.cpp                            |   |
|   |                                                                     |   |
|   |   +---------------+  +-------------------------------------------+  |   |
|   |   | Mailbox (L1)  |  | Loop:                                     |  |   |
|   |   | - config_buf  |  |   unchanged state + sync_request -> sync  |  |   |
|   |   |   _addr       |  |   new PUSH_A -> read A -> L1 ring        |  |   |
|   |   | - state (R)   |  |   new PUSH_B -> read B -> L1 ring        |  |   |
|   |   | - sync_req    |  |   TERMINATE_A/B -> final drain -> exit   |  |   |
|   |   | - sync_host_ts|  |                                           |  |   |
|   |   +-------+-------+  +-------------------------------------------+  |   |
|   |           |                        ^ NOC read (timestamp data)      |   |
|   +-----------+------------------------+--------------------------------+   |
|               |                        |                                    |
|               | state (PUSH_A/B or TERMINATE_A/B)                           |
|               | NOC write              |                                    |
|               v                        |                                    |
|   +---------------------------------------------------------------------+   |
|   | DISPATCH CORE (dispatch_s)                                          |   |
|   | Kernel: cq_dispatch_subordinate.cpp                                 |   |
|   |                                                                     |   |
|   |   L1 carve-out realtime_profiler_msg_t:                              |   |
|   |     Ping-pong: kernel_start_a/b, kernel_end_a/b                     |   |
|   |     program_id_fifo, realtime_profiler_core_noc_xy,                 |   |
|   |     realtime_profiler_remote_state_addr                             |   |
|   |                                                                     |   |
|   |   Per-command: record start ts, FIFO program id, process cmd,       |   |
|   |     record end ts, signal_realtime_profiler_and_switch()            |   |
|   |     ... process command ...                                         |   |
|   |     record_realtime_timestamp(false); signal_realtime_profiler_and_ |   |
|   |     switch();  (NOC-write state to profiler core)                   |   |
|   +---------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------+
```

---

## 2. Data Flow: Program Timestamp to Host

```
  DISPATCH_S                 REAL-TIME PROFILER CORE              HOST
  (dispatch_s)               (cq_realtime_profiler)               (receiver thread)

       |                              |                                  |
       | 1. Record start ts,          |                                  |
       |    program_id into           |                                  |
       |    mailbox buf A or B        |                                  |
       | 2. Process command           |                                  |
       | 3. Record end ts             |                                  |
       | 4. Update state PUSH_A/B     |                                  |
       | 5. NOC write state --------> |                                  |
       |                              | 6. See state PUSH_A or PUSH_B    |
       |                              | 7. NOC read timestamp data       |
       | <----------------------------|    from dispatch_s L1 (buf A/B)  |
       |                              | 8. Push page to D2H socket       |
       |                              |    (PCIe write to host buffer)   |
       |                              | -------------------------------> | 9. wait_for_pages
       |                              |                                  |    get_read_ptr
       |                              |                                  | 10. Parse start/end ts,
       |                              |                                  |     program_id
       |                              |                                  | 11. InvokeProgramRealtime
       |                              |                                  |     Callbacks(record)
       |                              | <------------------------------- | pop_pages, notify_sender
```

---

## 3. Sync (Timestamp Calibration)

Host and device timestamps are aligned so that Tracy (or other consumers) can relate device cycles to host time.

```
  HOST                              REAL-TIME PROFILER CORE

    |  Write sync_request = 1 (L1)        |
    | ---------------------------------> |  Poll sync_request
    |  Write sync_host_timestamp = T     |
    | ---------------------------------> |  See host_ts > 0
    |                                    |  Capture device wall clock (D)
    |                                    |  Push page: (D_hi, D_lo, T,
    |                                    |    REALTIME_PROFILER_SYNC_MARKER_ID)
    |                                    |  Clear sync_host_timestamp
    |  wait_for_pages(1)                 |
    | <--------------------------------- |  (D2H page arrives)
    |  Parse device_time D, host_time T  |
    |  Repeat for N samples              |
    |  Write sync_request = 0 (L1)       |
    | ---------------------------------> |  Exit sync loop
    |  Linear regression -> frequency,   |
    |  first_timestamp for this device   |
```

---

## 4. Carve-out layout (conceptual)

| Location | Contents (`realtime_profiler_msg_t`) |
|----------|----------------------------------------|
| **Dispatch_s L1** | Ping-pong buffers, program_id_fifo, **realtime_profiler_core_noc_xy**, **realtime_profiler_remote_state_addr**, realtime_profiler_state. Host writes NOC XY and the profiler tensix L1 address of `realtime_profiler_state` for NOC signaling. |
| **Profiler tensix L1** | **config_buffer_addr**, **realtime_profiler_state**, sync_request, sync_host_timestamp. |

Layout: `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`. HAL: `tt::tt_metal::realtime_profiler_msgs`. Not in `mailboxes_t`.

---

## 5. Shutdown and DMA Ownership

Closing a mesh drains its logical command queues before stopping its profiler. Physical dispatch stays alive: unit meshes created together share physical-device ownership, so closing one must not terminate a sibling mesh's dispatch.

1. `RealtimeProfilerManager::shutdown()` queues a terminal `RT_PROFILER_FLUSH` on each device's CQ0. Dispatch sends one `TERMINATE_A` or `TERMINATE_B` message selecting the final timestamp buffer, then disables further profiler writes. Its local termination state also releases the companion dispatch TRISC, even when profiling was never enabled.
2. Dispatch owns the profiler state mailbox after initialization. BRISC remembers the last state it consumed instead of clearing the mailbox to `IDLE`, which could overwrite a concurrent terminal message. On terminal state it drains an unobserved predecessor buffer, if any, then the final buffer. Only BRISC publishes ring termination, after completing these reads and enqueues.
3. NCRISC samples termination before the ring indices. It exits only when it has observed termination and an empty ring, so a stale empty-index snapshot cannot skip the producer's final entry. D2H writes complete before kernel exit.
4. The host waits for the profiler program's completion while the receiver still drains the D2H FIFO. It then joins the receiver, publishes remaining records, and joins consumers before releasing the socket and its DMA pin. An empty FIFO alone is not a device-completion acknowledgment.

If a device stop fails, close reports failure and logs the error. The failed device's socket is retained until process exit rather than unpinning a buffer that device code might still access.

`RealtimeProfilerSanity.CloseDrainsRegisteredCallback` exercises close without an explicit quiesce or grace sleep. `RealtimeProfilerSanity.ClosingUnitMeshPreservesSiblingProfiler` exercises a sibling enqueue after the first unit mesh closes. Both require hardware execution; compilation alone does not establish the shutdown contract.

---

## 6. File / Component Reference

| Component | File(s) |
|-----------|--------|
| Dispatch_s (timestamp record + signal) | `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`, `realtime_profiler.hpp` |
| Real-time profiler producer and D2H pusher | `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp`, `cq_realtime_profiler_push.cpp` |
| Host init, sync, receiver thread | `mesh_device.cpp`, `realtime_profiler_manager.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Callbacks (Tracy, user) | `tt_metal/impl/dispatch/data_collector.cpp`, `realtime_profiler_tracy_handler.cpp` |
