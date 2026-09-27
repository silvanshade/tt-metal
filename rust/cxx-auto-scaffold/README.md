# tt-metal host API scaffold

Machine-readable inventory of the tt-metal host API surface that `libtt-metal-cxx` bridges, with measured C++ type traits and cxx-auto export stubs. Built by walking the ten bridge translation units in `libtt-metal-cxx` and resolving every callable and type to its declaration in the pinned `tt-metal` tree.

## Inputs

| Input | Pin |
|---|---|
| `libtt-metal-cxx` (bridge sources) | `3f62391` |
| `tt-metal` (declarations) | `72592c1e661` |
| build tree | `build_clang22_integration` |

## Counts

- Functions: **202** (deduped by qualified name + signature).
- Types: **72** (deduped by qualified name): 33 class, 20 struct, 11 enum, 8 alias.
- Measured traits: 64 C++ types (72 minus 8 Rust-side cxx repr structs); all complete.
- cxx-auto stubs: 44 `CXX_AUTO_PRELUDE` / `CXX_AUTO_EXPORT` (one per complete class with a non-empty namespace).

## Per-subsystem

| Subsystem | Functions | Types |
|---|---|---|
| `runtime.cc` | 2 | 0 |
| `device.cc` | 4 | 3 |
| `program.cc` | 5 | 8 |
| `kernel.cc` | 5 | 8 |
| `runtime_args.cc` | 4 | 1 |
| `buffer.cc` | 43 | 15 |
| `mesh_buffer.cc` | 6 | 5 |
| `distributed.cc` | 18 | 4 |
| `tilize.cc` | 14 | 11 |
| `ffi.rs` | 101 | 17 |
| **Total** | **202** | **72** |

### Types by subsystem

#### `device.cc`

| Type | Kind | Declaration |
|---|---|---|
| `IDevice` | class | `tt_metal/api/tt-metalium/device.hpp:63` |
| `ChipId` | alias | `tt_metal/api/tt-metalium/device_types.hpp:11` |
| `DispatchCoreConfig` | class | `tt_metal/api/tt-metalium/dispatch_core_common.hpp:18` |

#### `program.cc`

| Type | Kind | Declaration |
|---|---|---|
| `Program` | class | `tt_metal/api/tt-metalium/program.hpp:25` |
| `CoreCoord` | alias | `tt_metal/api/tt-metalium/core_coord.hpp:32` |
| `CoreRange` | class | `tt_metal/api/tt-metalium/core_coord.hpp:38` |
| `CoreRangeSet` | class | `tt_metal/api/tt-metalium/core_coord.hpp:111` |
| `DataMovementConfig` | struct | `tt_metal/api/tt-metalium/kernel_types.hpp:64` |
| `ComputeConfig` | struct | `tt_metal/api/tt-metalium/kernel_types.hpp:105` |
| `KernelHandle` | alias | `tt_metal/api/tt-metalium/kernel_types.hpp:51` |
| `ProgramId` | alias | `tt_metal/api/tt-metalium/program.hpp:23` |

#### `kernel.cc`

| Type | Kind | Declaration |
|---|---|---|
| `ReaderDataMovementConfig` | struct | `tt_metal/api/tt-metalium/kernel_types.hpp:87` |
| `WriterDataMovementConfig` | struct | `tt_metal/api/tt-metalium/kernel_types.hpp:96` |
| `MathFidelity` | enum | `tt_metal/api/tt-metalium/base_types.hpp:12` |
| `UnpackToDestMode` | enum | `tt_metal/api/tt-metalium/base_types.hpp:39` |
| `KernelBuildOptLevel` | enum | `tt_metal/api/tt-metalium/kernel_types.hpp:54` |
| `DataMovementProcessor` | enum | `tt_metal/api/tt-metalium/kernel_types.hpp:22` |
| `NOC` | enum | `tt_metal/api/tt-metalium/kernel_types.hpp:33` |
| `NOC_MODE` | enum | `tt_metal/api/tt-metalium/kernel_types.hpp:40` |

#### `runtime_args.cc`

| Type | Kind | Declaration |
|---|---|---|
| `RuntimeArgsData` | struct | `tt_metal/api/tt-metalium/runtime_args_data.hpp:21` |

#### `buffer.cc`

| Type | Kind | Declaration |
|---|---|---|
| `Buffer` | class | `tt_metal/api/tt-metalium/buffer.hpp:158` |
| `BufferConfig` | struct | `tt_metal/api/tt-metalium/buffer.hpp:95` |
| `InterleavedBufferConfig` | alias | `tt_metal/api/tt-metalium/buffer.hpp:102` |
| `ShardedBufferConfig` | struct | `tt_metal/api/tt-metalium/buffer.hpp:107` |
| `ShardSpec` | struct | `tt_metal/api/tt-metalium/buffer.hpp:38` |
| `ShardSpecBuffer` | struct | `tt_metal/api/tt-metalium/buffer.hpp:68` |
| `BufferType` | enum | `tt_metal/api/tt-metalium/buffer_types.hpp:37` |
| `TensorMemoryLayout` | enum | `tt_metal/api/tt-metalium/buffer_types.hpp:11` |
| `ShardOrientation` | enum | `tt_metal/api/tt-metalium/buffer_types.hpp:21` |
| `SubDeviceId` | alias | `tt_metal/api/tt-metalium/sub_device_types.hpp:13` |
| `DeviceAddr` | alias | `tt_metal/api/tt-metalium/hal_types.hpp:11` |
| `CBHandle` | alias | `tt_metal/api/tt-metalium/circular_buffer_config.hpp:33` |
| `Tile` | struct | `tt_metal/api/tt-metalium/tile.hpp:23` |
| `CircularBufferConfig` | class | `tt_metal/api/tt-metalium/circular_buffer_config.hpp:35` |
| `CircularBufferConfig::Builder` | class | `tt_metal/api/tt-metalium/circular_buffer_config.hpp:112` |

#### `mesh_buffer.cc`

| Type | Kind | Declaration |
|---|---|---|
| `ReplicatedBufferConfig` | struct | `tt_metal/api/tt-metalium/mesh_buffer.hpp:41` |
| `DeviceLocalBufferConfig` | struct | `tt_metal/api/tt-metalium/mesh_buffer.hpp:24` |
| `MeshBuffer` | class | `tt_metal/api/tt-metalium/mesh_buffer.hpp:92` |
| `TensorAccessorArgs` | class | `tt_metal/api/tt-metalium/tensor_accessor_args.hpp:18` |
| `MeshDevice` | class | `tt_metal/api/tt-metalium/mesh_device.hpp:74` |

#### `distributed.cc`

| Type | Kind | Declaration |
|---|---|---|
| `MeshWorkload` | class | `tt_metal/api/tt-metalium/mesh_workload.hpp:20` |
| `MeshCommandQueue` | class | `tt_metal/api/tt-metalium/mesh_command_queue.hpp:58` |
| `MeshShape` | class | `tt_metal/api/tt-metalium/mesh_coord.hpp:21` |
| `MeshCoordinateRange` | class | `tt_metal/api/tt-metalium/mesh_coord.hpp:122` |

#### `tilize.cc`

| Type | Kind | Declaration |
|---|---|---|
| `DataType` | enum | `tt_metal/api/tt-metalium/tensor/tensor_types.hpp:26` |
| `TensorSpec` | class | `tt_metal/api/tt-metalium/tensor/spec/tensor_spec.hpp:19` |
| `TensorLayout` | class | `tt_metal/api/tt-metalium/tensor/spec/layout/tensor_layout.hpp:32` |
| `PageConfig` | class | `tt_metal/api/tt-metalium/tensor/spec/layout/page_config.hpp:29` |
| `MemoryConfig` | class | `tt_metal/api/tt-metalium/tensor/spec/layout/memory_config/memory_config.hpp:28` |
| `Layout` | enum | `tt_metal/api/tt-metalium/tensor/spec/layout/layout.hpp:11` |
| `Shape` | class | `tt_metal/api/tt-metalium/shape.hpp:21` |
| `HostTensor` | class | `tt_metal/api/tt-metalium/tensor/host_tensor.hpp:40` |
| `HostBuffer` | class | `tt_metal/api/tt-metalium/host_buffer.hpp:29` |
| `TensorTopology` | class | `tt_metal/api/tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp:12` |
| `bfloat16` | class | `tt_metal/api/tt-metalium/bfloat16.hpp:17` |

#### `ffi.rs`

| Type | Kind | Declaration |
|---|---|---|
| `DeviceHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/device.hpp:17` |
| `BufferHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/buffer.hpp:29` |
| `CircularBufferConfigHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/buffer.hpp:53` |
| `ProgramHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/program.hpp:29` |
| `ComputeKernelConfigHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/kernel.hpp:15` |
| `DataMovementKernelConfigHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/kernel.hpp:42` |
| `MeshBufferHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/mesh_buffer.hpp:17` |
| `MeshDeviceHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/distributed.hpp:21` |
| `MeshWorkloadHandle` | class | `libtt-metal-cxx/include/tt_metal_cxx/distributed.hpp:62` |
| `CoreRangeRepr` | struct | `libtt-metal-cxx/src/ffi.rs:4` |
| `InterleavedBufferConfigRepr` | struct | `libtt-metal-cxx/src/ffi.rs:11` |
| `ShardedBufferConfigRepr` | struct | `libtt-metal-cxx/src/ffi.rs:17` |
| `BufferCreateOptionsRepr` | struct | `libtt-metal-cxx/src/ffi.rs:31` |
| `BufferInfoRepr` | struct | `libtt-metal-cxx/src/ffi.rs:38` |
| `ShardSpecBufferMetadataRepr` | struct | `libtt-metal-cxx/src/ffi.rs:49` |
| `CircularBufferMetadataRepr` | struct | `libtt-metal-cxx/src/ffi.rs:59` |
| `CircularBufferIndexConfigRepr` | struct | `libtt-metal-cxx/src/ffi.rs:69` |

### Functions by subsystem

#### `runtime.cc` — 2 functions

`SetRootDir`, `ReleaseOwnership`

#### `device.cc` — 4 functions

`GetNumAvailableDevices`, `GetNumPCIeDevices`, `CreateDevice`, `CloseDevice`

#### `program.cc` — 5 functions

`CreateProgram`, `CreateKernel`, `CreateKernelFromString`, `get_runtime_id`, `set_runtime_id`

#### `kernel.cc` — 5 functions

`ComputeConfig::(ctor)`, `DataMovementConfig::(ctor)`, `DataMovementConfig::(ctor)`, `ReaderDataMovementConfig::(ctor)`, `WriterDataMovementConfig::(ctor)`

#### `runtime_args.cc` — 4 functions

`SetRuntimeArgs`, `GetRuntimeArgs`, `SetCommonRuntimeArgs`, `GetCommonRuntimeArgs`

#### `buffer.cc` — 43 functions

`CreateBuffer`, `CreateBuffer`, `CreateBuffer`, `CreateBuffer`, `CreateBuffer`, `CreateBuffer`, `DeallocateBuffer`, `AssignGlobalBufferToProgram`, `CreateCircularBuffer`, `GetCircularBufferConfig`, `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, `UpdateDynamicCircularBufferAddress`, `UpdateDynamicCircularBufferAddress`, `UpdateDynamicCircularBufferAddressAndTotalSize`, `CreateSemaphore`, `sub_device_id`, `is_allocated`, `address`, `size`, `page_size`, `buffer_type`, `buffer_layout`, `has_shard_spec`, `shard_spec`, `grid`, `shape`, `orientation`, `ranges`, `total_size`, `globally_allocated_address`, `buffer_indices`, `index`, `remote_index`, `get_transpose_within_face`, `get_transpose_of_faces`, `Tile::(ctor)`, `SubDeviceId::(ctor)`, `ShardSpecBuffer::(ctor)`, `ShardSpec::(ctor)`, `CoreRangeSet::(ctor)`, `CoreRange::(ctor)`, `CircularBufferConfig::(ctor)`

#### `mesh_buffer.cc` — 6 functions

`create`, `address`, `size`, `is_allocated`, `TensorAccessorArgs::(ctor)`, `get_compile_time_args`

#### `distributed.cc` — 18 functions

`close`, `enable_program_cache`, `clear_program_cache`, `disable_and_clear_program_cache`, `num_program_cache_entries`, `num_devices`, `num_rows`, `num_cols`, `compute_with_storage_grid_size`, `mesh_command_queue`, `shape`, `create_unit_mesh`, `EnqueueMeshWorkload`, `MeshWorkload::(ctor)`, `add_program`, `enqueue_write_mesh_buffer`, `enqueue_read_mesh_buffer`, `MeshCoordinateRange::(ctor)`

#### `tilize.cc` — 14 functions

`tilize_nfaces`, `untilize_nfaces`, `TensorSpec::(ctor)`, `TensorLayout::(ctor)`, `PageConfig::(ctor)`, `MemoryConfig::(ctor)`, `from_vector`, `get_host_buffer`, `view_bytes`, `HostBuffer::(ctor)`, `HostTensor::(ctor)`, `to_vector`, `compute_packed_buffer_size_bytes`, `TensorTopology::(ctor)`

#### `ffi.rs` — 101 functions

`throw_invalid_argument`, `create_device`, `create_interleaved_buffer`, `create_sharded_buffer`, `create_circular_buffer_config`, `create_program`, `create_compute_kernel_config`, `create_data_movement_kernel_config`, `create_reader_data_movement_kernel_config`, `create_writer_data_movement_kernel_config`, `create_unit_mesh`, `create_mesh_workload`, `get_num_available_devices`, `get_num_pcie_devices`, `tilize`, `untilize`, `tilize_with_data_format`, `untilize_with_data_format`, `create_replicated_mesh_buffer`, `close`, `is_open`, `device_id`, `info`, `deallocate`, `has_shard_spec`, `shard_spec_metadata`, `shard_spec_core_ranges`, `set_total_size`, `set_address_offset`, `set_globally_allocated_address`, `set_globally_allocated_address_and_total_size`, `set_index_data_format`, `set_index_total_size`, `set_index_page_size`, `set_index_tile`, `runtime_id`, `set_runtime_id`, `set_runtime_args`, `get_runtime_args`, `set_common_runtime_args`, `get_common_runtime_args`, `create_compute_kernel`, `create_compute_kernel_from_string`, `create_compute_kernel_with_config`, `create_compute_kernel_from_string_with_config`, `create_compute_kernel_from_string_with_config_range`, `create_compute_kernel_from_string_with_config_ranges`, `create_data_movement_kernel`, `create_data_movement_kernel_from_string`, `create_data_movement_kernel_with_config`, `create_data_movement_kernel_from_string_with_config`, `create_data_movement_kernel_from_string_with_config_range`, `create_data_movement_kernel_from_string_with_config_ranges`, `assign_global_buffer`, `create_circular_buffer`, `get_circular_buffer_metadata`, `get_circular_buffer_indices`, `update_circular_buffer_total_size`, `update_circular_buffer_page_size`, `update_dynamic_circular_buffer_address`, `update_dynamic_circular_buffer_address_with_offset`, `update_dynamic_circular_buffer_address_and_total_size`, `create_semaphore`, `set_math_fidelity`, `set_fp32_dest_acc_en`, `set_dst_full_sync_en`, `fill_unpack_to_dest_modes`, `set_bfp8_pack_precise`, `set_math_approx_mode`, `add_compile_arg`, `add_define`, `add_named_compile_arg`, `set_opt_level`, `set_processor`, `set_noc`, `set_noc_mode`, `add_compile_arg`, `add_define`, `add_named_compile_arg`, `set_opt_level`, `close`, `is_open`, `device_id`, `num_devices`, `num_rows`, `num_cols`, `compute_with_storage_grid_size_x`, `compute_with_storage_grid_size_y`, `enable_program_cache`, `clear_program_cache`, `disable_and_clear_program_cache`, `num_program_cache_entries`, `enqueue_workload`, `write_mesh_buffer`, `read_mesh_buffer`, `add_program_to_full_mesh`, `program_count`, `address`, `size`, `is_allocated`, `compute_compile_args`

## Type traits

Measured by `probes/type_traits.cpp` (clang 22, `-std=c++20`, against the pinned build tree); output in `probes/type_traits.txt`. Columns: `size`, `align`, `trivially_copyable`, `move_constructible`, `destructible`, `default_constructible`, `copy_constructible`, the `nothrow_*` variants, `copy_assignable`, `move_assignable`, `standard_layout`. Full values are in `types.json` under each type's `traits` field.

29 of 64 measured types are trivially copyable and safe to cross the bridge by value: `ChipId`, `DispatchCoreConfig`, `CoreCoord`, `CoreRange`, `KernelHandle`, `ProgramId`, `MathFidelity`, `UnpackToDestMode`, `KernelBuildOptLevel`, `DataMovementProcessor`, `NOC`, `NOC_MODE`, `RuntimeArgsData`, `BufferConfig`, `InterleavedBufferConfig`, `BufferType`, `TensorMemoryLayout`, `ShardOrientation`, `SubDeviceId`, `DeviceAddr`, `CBHandle`, `Tile`, `CircularBufferConfig::Builder`, `ReplicatedBufferConfig`, `TensorAccessorArgs`, `DataType`, `PageConfig`, `Layout`, `bfloat16`. The rest are move-only or non-trivial and must cross by handle or explicit copy.

## Hard cases

Types and call sites that cxx-auto cannot express as-is, with the measured or declared reason.

### By-value non-relocatable (not move-constructible)

13 complete classes are not move-constructible (measured). cxx by-value bridging requires move construction; these need a `UniquePtr` handle instead. All 9 `tt_metal_cxx` handle classes are in this set, as are the four tt-metal device/buffer classes:

- `tt::tt_metal::IDevice` — `tt_metal/api/tt-metalium/device.hpp:63`
- `tt::tt_metal::Buffer` — `tt_metal/api/tt-metalium/buffer.hpp:158`
- `tt::tt_metal::distributed::MeshDevice` — `tt_metal/api/tt-metalium/mesh_device.hpp:74`
- `tt::tt_metal::distributed::MeshCommandQueue` — `tt_metal/api/tt-metalium/mesh_command_queue.hpp:58`
- `tt_metal_cxx::DeviceHandle` — `libtt-metal-cxx/include/tt_metal_cxx/device.hpp:17`
- `tt_metal_cxx::BufferHandle` — `libtt-metal-cxx/include/tt_metal_cxx/buffer.hpp:29`
- `tt_metal_cxx::CircularBufferConfigHandle` — `libtt-metal-cxx/include/tt_metal_cxx/buffer.hpp:53`
- `tt_metal_cxx::ProgramHandle` — `libtt-metal-cxx/include/tt_metal_cxx/program.hpp:29`
- `tt_metal_cxx::ComputeKernelConfigHandle` — `libtt-metal-cxx/include/tt_metal_cxx/kernel.hpp:15`
- `tt_metal_cxx::DataMovementKernelConfigHandle` — `libtt-metal-cxx/include/tt_metal_cxx/kernel.hpp:42`
- `tt_metal_cxx::MeshBufferHandle` — `libtt-metal-cxx/include/tt_metal_cxx/mesh_buffer.hpp:17`
- `tt_metal_cxx::MeshDeviceHandle` — `libtt-metal-cxx/include/tt_metal_cxx/distributed.hpp:21`
- `tt_metal_cxx::MeshWorkloadHandle` — `libtt-metal-cxx/include/tt_metal_cxx/distributed.hpp:62`

### Global namespace

- `bfloat16` (`tt_metal/api/tt-metalium/bfloat16.hpp:17`) is declared in the global namespace. `cxx_auto::TypeSpec.cxx_namespace` is required non-empty (`cxx_auto.cppm:564`), so a global-namespace type cannot be named in a spec. It is used only as a template argument to `tilize_nfaces`/`untilize_nfaces`, never passed across the bridge by value, so it needs no PRELUDE.
- `tilize_nfaces` and `untilize_nfaces` (`tilize_utils.hpp:89,92`) are global-namespace function templates, and the bridge calls them as `::tilize_nfaces` and `::untilize_nfaces` (`tilize.cc:55,62`). `functions.json` records them under their global names.

### Nested type

- `tt::tt_metal::CircularBufferConfig::Builder` (`circular_buffer_config.hpp:112`) is a nested class. It is preluded with `cxx_namespace = "tt::tt_metal::CircularBufferConfig"` (the enclosing class), but the cxx-auto generator's handling of a nested-type alias is not exercised by the fixtures — confirm it emits a nested `type` alias before relying on this export.

### shared_ptr / unique_ptr parameters

Several tt-metal APIs take or return `std::shared_ptr`/`std::unique_ptr` (e.g. `MeshDevice` is held as `shared_ptr` in `MeshDeviceHandle`, `MeshBuffer` as `shared_ptr` in `MeshBufferHandle`). cxx-auto bridges by value or by `UniquePtr` of a complete type; `shared_ptr` ownership must be wrapped in a handle class, which the `tt_metal_cxx` layer already does.

### variant / optional

- `PageConfig` holds a `std::variant<RowMajorPageConfig, TilePageConfig>` (`page_config.hpp:31`). cxx-auto has no variant support; the variant must be flattened to a tagged union or an enum + payload on the Rust side.
- `MeshDevice::mesh_command_queue` takes an `std::optional<uint8_t>` parameter (`mesh_device.hpp:312`). `std::optional` must be mapped to a Rust `Option`.

### Overload sets

Several functions are overloaded; cxx-auto names one binding per export. The overload sets in this scaffold:

- `CreateBuffer` — 6 overloads
- `DataMovementConfig::DataMovementConfig` — 2 overloads
- `UpdateDynamicCircularBufferAddress` — 2 overloads

### Templates

Six records are templates, marked `template: true`: `tilize_nfaces`, `untilize_nfaces`, the `SubDeviceId` constructor, `HostTensor::from_vector`, `HostTensor::to_vector`, and the `HostBuffer` constructors. cxx-auto exports one concrete instantiation per export; the Rust side instantiates the needed `T` (e.g. `bfloat16`, `float`).

### Unresolved declarations

Three call sites reference declarations that do not exist in the pinned tree:

- `HostTensor::from_vector` is called with three arguments (`tilize.cc:148`) but only the two-argument overloads exist (`host_tensor.hpp:119,127`).
- `HostTensor` is constructed from `(HostBuffer, TensorSpec, TensorTopology)` (`tilize.cc:170`) but no such public constructor is declared (`host_tensor.hpp:40-204`).
- `BufferHandle::info()` calls `buffer.sub_device_id()` (`buffer.cc:201`), but the public `Buffer` declares no such member (`buffer.hpp:158-209`); only `BufferImpl` does (`buffer_impl.hpp:67`).

All three are recorded in `functions.json` with `unresolved: true` and the reason. They must be resolved against the actual `libtt-metal-cxx` build before the bridge can compile.

## Files

| File | Contents |
|---|---|
| `functions.json` | 202 function records (qualified name, header, line, signature, params, return, used_by). |
| `types.json` | 72 type records (qualified name, header, line, kind, measured traits, used_by). |
| `probes/type_traits.cpp` | The trait-measurement probe. |
| `probes/type_traits.txt` | Measured trait output (one line per type). |
| `cxx-auto/proxy.hxx` | 44 `CXX_AUTO_PRELUDE` stubs, one per complete class, each in its own namespace. |
| `cxx-auto/export.cxx` | 44 `CXX_AUTO_EXPORT` stubs, each naming its type's proxy namespace. |
| `LEDGER.md` | One line per step, with evidence. |
