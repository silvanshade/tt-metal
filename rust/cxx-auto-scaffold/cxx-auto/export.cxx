#include "proxy.hxx"

// One CXX_AUTO_EXPORT per complete class. ID is unique per type.
// IDevice: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  iDevice,
  tt_metal_cxx_auto::proxy::iDevice,
  .rust_path = "iDevice",
  .rust_name = "IDevice",
  .cxx_name = "IDevice",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  dispatchCoreConfig,
  tt_metal_cxx_auto::proxy::dispatchCoreConfig,
  .rust_path = "dispatchCoreConfig",
  .rust_name = "DispatchCoreConfig",
  .cxx_name = "DispatchCoreConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  program,
  tt_metal_cxx_auto::proxy::program,
  .rust_path = "program",
  .rust_name = "Program",
  .cxx_name = "Program",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  coreRange,
  tt_metal_cxx_auto::proxy::coreRange,
  .rust_path = "coreRange",
  .rust_name = "CoreRange",
  .cxx_name = "CoreRange",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  coreRangeSet,
  tt_metal_cxx_auto::proxy::coreRangeSet,
  .rust_path = "coreRangeSet",
  .rust_name = "CoreRangeSet",
  .cxx_name = "CoreRangeSet",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  dataMovementConfig,
  tt_metal_cxx_auto::proxy::dataMovementConfig,
  .rust_path = "dataMovementConfig",
  .rust_name = "DataMovementConfig",
  .cxx_name = "DataMovementConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  computeConfig,
  tt_metal_cxx_auto::proxy::computeConfig,
  .rust_path = "computeConfig",
  .rust_name = "ComputeConfig",
  .cxx_name = "ComputeConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  readerDataMovementConfig,
  tt_metal_cxx_auto::proxy::readerDataMovementConfig,
  .rust_path = "readerDataMovementConfig",
  .rust_name = "ReaderDataMovementConfig",
  .cxx_name = "ReaderDataMovementConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  writerDataMovementConfig,
  tt_metal_cxx_auto::proxy::writerDataMovementConfig,
  .rust_path = "writerDataMovementConfig",
  .rust_name = "WriterDataMovementConfig",
  .cxx_name = "WriterDataMovementConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  runtimeArgsData,
  tt_metal_cxx_auto::proxy::runtimeArgsData,
  .rust_path = "runtimeArgsData",
  .rust_name = "RuntimeArgsData",
  .cxx_name = "RuntimeArgsData",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

// Buffer: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  buffer,
  tt_metal_cxx_auto::proxy::buffer,
  .rust_path = "buffer",
  .rust_name = "Buffer",
  .cxx_name = "Buffer",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  bufferConfig,
  tt_metal_cxx_auto::proxy::bufferConfig,
  .rust_path = "bufferConfig",
  .rust_name = "BufferConfig",
  .cxx_name = "BufferConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  shardedBufferConfig,
  tt_metal_cxx_auto::proxy::shardedBufferConfig,
  .rust_path = "shardedBufferConfig",
  .rust_name = "ShardedBufferConfig",
  .cxx_name = "ShardedBufferConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  shardSpec,
  tt_metal_cxx_auto::proxy::shardSpec,
  .rust_path = "shardSpec",
  .rust_name = "ShardSpec",
  .cxx_name = "ShardSpec",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  shardSpecBuffer,
  tt_metal_cxx_auto::proxy::shardSpecBuffer,
  .rust_path = "shardSpecBuffer",
  .rust_name = "ShardSpecBuffer",
  .cxx_name = "ShardSpecBuffer",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  tile,
  tt_metal_cxx_auto::proxy::tile,
  .rust_path = "tile",
  .rust_name = "Tile",
  .cxx_name = "Tile",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  circularBufferConfig,
  tt_metal_cxx_auto::proxy::circularBufferConfig,
  .rust_path = "circularBufferConfig",
  .rust_name = "CircularBufferConfig",
  .cxx_name = "CircularBufferConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

// Builder: nested class — cxx_namespace points at the enclosing class; confirm the cxx-auto generator emits a nested-type alias.
CXX_AUTO_EXPORT(
  builder,
  tt_metal_cxx_auto::proxy::builder,
  .rust_path = "builder",
  .rust_name = "Builder",
  .cxx_name = "Builder",
  .cxx_namespace = "tt::tt_metal::CircularBufferConfig",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  replicatedBufferConfig,
  tt_metal_cxx_auto::proxy::replicatedBufferConfig,
  .rust_path = "replicatedBufferConfig",
  .rust_name = "ReplicatedBufferConfig",
  .cxx_name = "ReplicatedBufferConfig",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  deviceLocalBufferConfig,
  tt_metal_cxx_auto::proxy::deviceLocalBufferConfig,
  .rust_path = "deviceLocalBufferConfig",
  .rust_name = "DeviceLocalBufferConfig",
  .cxx_name = "DeviceLocalBufferConfig",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  meshBuffer,
  tt_metal_cxx_auto::proxy::meshBuffer,
  .rust_path = "meshBuffer",
  .rust_name = "MeshBuffer",
  .cxx_name = "MeshBuffer",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  tensorAccessorArgs,
  tt_metal_cxx_auto::proxy::tensorAccessorArgs,
  .rust_path = "tensorAccessorArgs",
  .rust_name = "TensorAccessorArgs",
  .cxx_name = "TensorAccessorArgs",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

// MeshDevice: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  meshDevice,
  tt_metal_cxx_auto::proxy::meshDevice,
  .rust_path = "meshDevice",
  .rust_name = "MeshDevice",
  .cxx_name = "MeshDevice",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  meshWorkload,
  tt_metal_cxx_auto::proxy::meshWorkload,
  .rust_path = "meshWorkload",
  .rust_name = "MeshWorkload",
  .cxx_name = "MeshWorkload",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

// MeshCommandQueue: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  meshCommandQueue,
  tt_metal_cxx_auto::proxy::meshCommandQueue,
  .rust_path = "meshCommandQueue",
  .rust_name = "MeshCommandQueue",
  .cxx_name = "MeshCommandQueue",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  meshShape,
  tt_metal_cxx_auto::proxy::meshShape,
  .rust_path = "meshShape",
  .rust_name = "MeshShape",
  .cxx_name = "MeshShape",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  meshCoordinateRange,
  tt_metal_cxx_auto::proxy::meshCoordinateRange,
  .rust_path = "meshCoordinateRange",
  .rust_name = "MeshCoordinateRange",
  .cxx_name = "MeshCoordinateRange",
  .cxx_namespace = "tt::tt_metal::distributed",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  tensorSpec,
  tt_metal_cxx_auto::proxy::tensorSpec,
  .rust_path = "tensorSpec",
  .rust_name = "TensorSpec",
  .cxx_name = "TensorSpec",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  tensorLayout,
  tt_metal_cxx_auto::proxy::tensorLayout,
  .rust_path = "tensorLayout",
  .rust_name = "TensorLayout",
  .cxx_name = "TensorLayout",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  pageConfig,
  tt_metal_cxx_auto::proxy::pageConfig,
  .rust_path = "pageConfig",
  .rust_name = "PageConfig",
  .cxx_name = "PageConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  memoryConfig,
  tt_metal_cxx_auto::proxy::memoryConfig,
  .rust_path = "memoryConfig",
  .rust_name = "MemoryConfig",
  .cxx_name = "MemoryConfig",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  shape,
  tt_metal_cxx_auto::proxy::shape,
  .rust_path = "shape",
  .rust_name = "Shape",
  .cxx_name = "Shape",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  hostTensor,
  tt_metal_cxx_auto::proxy::hostTensor,
  .rust_path = "hostTensor",
  .rust_name = "HostTensor",
  .cxx_name = "HostTensor",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  hostBuffer,
  tt_metal_cxx_auto::proxy::hostBuffer,
  .rust_path = "hostBuffer",
  .rust_name = "HostBuffer",
  .cxx_name = "HostBuffer",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

CXX_AUTO_EXPORT(
  tensorTopology,
  tt_metal_cxx_auto::proxy::tensorTopology,
  .rust_path = "tensorTopology",
  .rust_name = "TensorTopology",
  .cxx_name = "TensorTopology",
  .cxx_namespace = "tt::tt_metal",
  .cxx_proxy_include = "proxy.hxx"
)

// DeviceHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  deviceHandle,
  tt_metal_cxx_auto::proxy::deviceHandle,
  .rust_path = "deviceHandle",
  .rust_name = "DeviceHandle",
  .cxx_name = "DeviceHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// BufferHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  bufferHandle,
  tt_metal_cxx_auto::proxy::bufferHandle,
  .rust_path = "bufferHandle",
  .rust_name = "BufferHandle",
  .cxx_name = "BufferHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// CircularBufferConfigHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  circularBufferConfigHandle,
  tt_metal_cxx_auto::proxy::circularBufferConfigHandle,
  .rust_path = "circularBufferConfigHandle",
  .rust_name = "CircularBufferConfigHandle",
  .cxx_name = "CircularBufferConfigHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// ProgramHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  programHandle,
  tt_metal_cxx_auto::proxy::programHandle,
  .rust_path = "programHandle",
  .rust_name = "ProgramHandle",
  .cxx_name = "ProgramHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// ComputeKernelConfigHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  computeKernelConfigHandle,
  tt_metal_cxx_auto::proxy::computeKernelConfigHandle,
  .rust_path = "computeKernelConfigHandle",
  .rust_name = "ComputeKernelConfigHandle",
  .cxx_name = "ComputeKernelConfigHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// DataMovementKernelConfigHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  dataMovementKernelConfigHandle,
  tt_metal_cxx_auto::proxy::dataMovementKernelConfigHandle,
  .rust_path = "dataMovementKernelConfigHandle",
  .rust_name = "DataMovementKernelConfigHandle",
  .cxx_name = "DataMovementKernelConfigHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// MeshBufferHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  meshBufferHandle,
  tt_metal_cxx_auto::proxy::meshBufferHandle,
  .rust_path = "meshBufferHandle",
  .rust_name = "MeshBufferHandle",
  .cxx_name = "MeshBufferHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// MeshDeviceHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  meshDeviceHandle,
  tt_metal_cxx_auto::proxy::meshDeviceHandle,
  .rust_path = "meshDeviceHandle",
  .rust_name = "MeshDeviceHandle",
  .cxx_name = "MeshDeviceHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

// MeshWorkloadHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
CXX_AUTO_EXPORT(
  meshWorkloadHandle,
  tt_metal_cxx_auto::proxy::meshWorkloadHandle,
  .rust_path = "meshWorkloadHandle",
  .rust_name = "MeshWorkloadHandle",
  .cxx_name = "MeshWorkloadHandle",
  .cxx_namespace = "tt_metal_cxx",
  .cxx_proxy_include = "proxy.hxx"
)

