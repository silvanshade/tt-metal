#pragma once

// cxx-auto proxy for the tt-metal host API scaffold.
// One CXX_AUTO_PRELUDE per complete class, each in its own namespace:
// a prelude declares `Self` there, so two cannot share one. Enums and aliases cross as
// plain values and are listed in the trailing comment, not preluded.
#include "cxx-auto/cxx/include/cxx-auto.hxx"
#include "tt-metalium/device.hpp"
#include "tt-metalium/dispatch_core_common.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/runtime_args_data.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/tile.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "tt-metalium/tensor/spec/tensor_spec.hpp"
#include "tt-metalium/tensor/spec/layout/tensor_layout.hpp"
#include "tt-metalium/tensor/spec/layout/page_config.hpp"
#include "tt-metalium/tensor/spec/memory_config/memory_config.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tensor/host_tensor.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp"
#include "tt_metal_cxx/device.hpp"
#include "tt_metal_cxx/buffer.hpp"
#include "tt_metal_cxx/program.hpp"
#include "tt_metal_cxx/kernel.hpp"
#include "tt_metal_cxx/mesh_buffer.hpp"
#include "tt_metal_cxx/distributed.hpp"

namespace tt_metal_cxx_auto::proxy {
// IDevice: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace iDevice {
CXX_AUTO_PRELUDE(IDevice, tt::tt_metal::IDevice)
} // namespace iDevice
namespace dispatchCoreConfig {
CXX_AUTO_PRELUDE(DispatchCoreConfig, tt::tt_metal::DispatchCoreConfig)
} // namespace dispatchCoreConfig
namespace program {
CXX_AUTO_PRELUDE(Program, tt::tt_metal::Program)
} // namespace program
namespace coreRange {
CXX_AUTO_PRELUDE(CoreRange, tt::tt_metal::CoreRange)
} // namespace coreRange
namespace coreRangeSet {
CXX_AUTO_PRELUDE(CoreRangeSet, tt::tt_metal::CoreRangeSet)
} // namespace coreRangeSet
namespace dataMovementConfig {
CXX_AUTO_PRELUDE(DataMovementConfig, tt::tt_metal::DataMovementConfig)
} // namespace dataMovementConfig
namespace computeConfig {
CXX_AUTO_PRELUDE(ComputeConfig, tt::tt_metal::ComputeConfig)
} // namespace computeConfig
namespace readerDataMovementConfig {
CXX_AUTO_PRELUDE(ReaderDataMovementConfig, tt::tt_metal::ReaderDataMovementConfig)
} // namespace readerDataMovementConfig
namespace writerDataMovementConfig {
CXX_AUTO_PRELUDE(WriterDataMovementConfig, tt::tt_metal::WriterDataMovementConfig)
} // namespace writerDataMovementConfig
namespace runtimeArgsData {
CXX_AUTO_PRELUDE(RuntimeArgsData, tt::tt_metal::RuntimeArgsData)
} // namespace runtimeArgsData
// Buffer: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace buffer {
CXX_AUTO_PRELUDE(Buffer, tt::tt_metal::Buffer)
} // namespace buffer
namespace bufferConfig {
CXX_AUTO_PRELUDE(BufferConfig, tt::tt_metal::BufferConfig)
} // namespace bufferConfig
namespace shardedBufferConfig {
CXX_AUTO_PRELUDE(ShardedBufferConfig, tt::tt_metal::ShardedBufferConfig)
} // namespace shardedBufferConfig
namespace shardSpec {
CXX_AUTO_PRELUDE(ShardSpec, tt::tt_metal::ShardSpec)
} // namespace shardSpec
namespace shardSpecBuffer {
CXX_AUTO_PRELUDE(ShardSpecBuffer, tt::tt_metal::ShardSpecBuffer)
} // namespace shardSpecBuffer
namespace tile {
CXX_AUTO_PRELUDE(Tile, tt::tt_metal::Tile)
} // namespace tile
namespace circularBufferConfig {
CXX_AUTO_PRELUDE(CircularBufferConfig, tt::tt_metal::CircularBufferConfig)
} // namespace circularBufferConfig
// Builder: nested class — cxx_namespace points at the enclosing class; confirm the cxx-auto generator emits a nested-type alias.
namespace builder {
CXX_AUTO_PRELUDE(Builder, tt::tt_metal::CircularBufferConfig::Builder)
} // namespace builder
namespace replicatedBufferConfig {
CXX_AUTO_PRELUDE(ReplicatedBufferConfig, tt::tt_metal::distributed::ReplicatedBufferConfig)
} // namespace replicatedBufferConfig
namespace deviceLocalBufferConfig {
CXX_AUTO_PRELUDE(DeviceLocalBufferConfig, tt::tt_metal::distributed::DeviceLocalBufferConfig)
} // namespace deviceLocalBufferConfig
namespace meshBuffer {
CXX_AUTO_PRELUDE(MeshBuffer, tt::tt_metal::distributed::MeshBuffer)
} // namespace meshBuffer
namespace tensorAccessorArgs {
CXX_AUTO_PRELUDE(TensorAccessorArgs, tt::tt_metal::TensorAccessorArgs)
} // namespace tensorAccessorArgs
// MeshDevice: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace meshDevice {
CXX_AUTO_PRELUDE(MeshDevice, tt::tt_metal::distributed::MeshDevice)
} // namespace meshDevice
namespace meshWorkload {
CXX_AUTO_PRELUDE(MeshWorkload, tt::tt_metal::distributed::MeshWorkload)
} // namespace meshWorkload
// MeshCommandQueue: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace meshCommandQueue {
CXX_AUTO_PRELUDE(MeshCommandQueue, tt::tt_metal::distributed::MeshCommandQueue)
} // namespace meshCommandQueue
namespace meshShape {
CXX_AUTO_PRELUDE(MeshShape, tt::tt_metal::distributed::MeshShape)
} // namespace meshShape
namespace meshCoordinateRange {
CXX_AUTO_PRELUDE(MeshCoordinateRange, tt::tt_metal::distributed::MeshCoordinateRange)
} // namespace meshCoordinateRange
namespace tensorSpec {
CXX_AUTO_PRELUDE(TensorSpec, tt::tt_metal::TensorSpec)
} // namespace tensorSpec
namespace tensorLayout {
CXX_AUTO_PRELUDE(TensorLayout, tt::tt_metal::TensorLayout)
} // namespace tensorLayout
namespace pageConfig {
CXX_AUTO_PRELUDE(PageConfig, tt::tt_metal::PageConfig)
} // namespace pageConfig
namespace memoryConfig {
CXX_AUTO_PRELUDE(MemoryConfig, tt::tt_metal::MemoryConfig)
} // namespace memoryConfig
namespace shape {
CXX_AUTO_PRELUDE(Shape, tt::tt_metal::Shape)
} // namespace shape
namespace hostTensor {
CXX_AUTO_PRELUDE(HostTensor, tt::tt_metal::HostTensor)
} // namespace hostTensor
namespace hostBuffer {
CXX_AUTO_PRELUDE(HostBuffer, tt::tt_metal::HostBuffer)
} // namespace hostBuffer
namespace tensorTopology {
CXX_AUTO_PRELUDE(TensorTopology, tt::tt_metal::TensorTopology)
} // namespace tensorTopology
// DeviceHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace deviceHandle {
CXX_AUTO_PRELUDE(DeviceHandle, tt_metal_cxx::DeviceHandle)
} // namespace deviceHandle
// BufferHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace bufferHandle {
CXX_AUTO_PRELUDE(BufferHandle, tt_metal_cxx::BufferHandle)
} // namespace bufferHandle
// CircularBufferConfigHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace circularBufferConfigHandle {
CXX_AUTO_PRELUDE(CircularBufferConfigHandle, tt_metal_cxx::CircularBufferConfigHandle)
} // namespace circularBufferConfigHandle
// ProgramHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace programHandle {
CXX_AUTO_PRELUDE(ProgramHandle, tt_metal_cxx::ProgramHandle)
} // namespace programHandle
// ComputeKernelConfigHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace computeKernelConfigHandle {
CXX_AUTO_PRELUDE(ComputeKernelConfigHandle, tt_metal_cxx::ComputeKernelConfigHandle)
} // namespace computeKernelConfigHandle
// DataMovementKernelConfigHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace dataMovementKernelConfigHandle {
CXX_AUTO_PRELUDE(DataMovementKernelConfigHandle, tt_metal_cxx::DataMovementKernelConfigHandle)
} // namespace dataMovementKernelConfigHandle
// MeshBufferHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace meshBufferHandle {
CXX_AUTO_PRELUDE(MeshBufferHandle, tt_metal_cxx::MeshBufferHandle)
} // namespace meshBufferHandle
// MeshDeviceHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace meshDeviceHandle {
CXX_AUTO_PRELUDE(MeshDeviceHandle, tt_metal_cxx::MeshDeviceHandle)
} // namespace meshDeviceHandle
// MeshWorkloadHandle: not move-constructible (measured) — by-value cxx bridging is impossible; needs a UniquePtr handle. See README hard cases.
namespace meshWorkloadHandle {
CXX_AUTO_PRELUDE(MeshWorkloadHandle, tt_metal_cxx::MeshWorkloadHandle)
} // namespace meshWorkloadHandle
} // namespace tt_metal_cxx_auto::proxy

// --- enums (plain values; no PRELUDE) ---
// tt::tt_metal::MathFidelity  (tt_metal/api/tt-metalium/base_types.hpp:12)
// tt::tt_metal::UnpackToDestMode  (tt_metal/api/tt-metalium/base_types.hpp:39)
// tt::tt_metal::KernelBuildOptLevel  (tt_metal/api/tt-metalium/kernel_types.hpp:54)
// tt::tt_metal::DataMovementProcessor  (tt_metal/api/tt-metalium/kernel_types.hpp:22)
// tt::tt_metal::NOC  (tt_metal/api/tt-metalium/kernel_types.hpp:33)
// tt::tt_metal::NOC_MODE  (tt_metal/api/tt-metalium/kernel_types.hpp:40)
// tt::tt_metal::BufferType  (tt_metal/api/tt-metalium/buffer_types.hpp:37)
// tt::tt_metal::TensorMemoryLayout  (tt_metal/api/tt-metalium/buffer_types.hpp:11)
// tt::tt_metal::ShardOrientation  (tt_metal/api/tt-metalium/buffer_types.hpp:21)
// tt::tt_metal::DataType  (tt_metal/api/tt-metalium/tensor/tensor_types.hpp:26)
// tt::tt_metal::Layout  (tt_metal/api/tt-metalium/tensor/spec/layout/layout.hpp:11)
// --- aliases (plain values; no PRELUDE) ---
// tt::ChipId  (tt_metal/api/tt-metalium/device_types.hpp:11)
// tt::tt_metal::CoreCoord  (tt_metal/api/tt-metalium/core_coord.hpp:32)
// tt::tt_metal::KernelHandle  (tt_metal/api/tt-metalium/kernel_types.hpp:51)
// tt::tt_metal::ProgramId  (tt_metal/api/tt-metalium/program.hpp:23)
// tt::tt_metal::InterleavedBufferConfig  (tt_metal/api/tt-metalium/buffer.hpp:102)
// tt::tt_metal::SubDeviceId  (tt_metal/api/tt-metalium/sub_device_types.hpp:13)
// tt::tt_metal::DeviceAddr  (tt_metal/api/tt-metalium/hal_types.hpp:11)
// tt::tt_metal::CBHandle  (tt_metal/api/tt-metalium/circular_buffer_config.hpp:33)
// --- hard cases (no PRELUDE) ---
// bfloat16  (tt_metal/api/tt-metalium/bfloat16.hpp:17) — global namespace: TypeSpec.cxx_namespace is required non-empty (cxx_auto.cppm:564); cannot express a global-namespace type
