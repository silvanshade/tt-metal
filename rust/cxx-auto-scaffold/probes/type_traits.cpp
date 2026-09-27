#include <type_traits>
#include <cstdio>
#include <cstddef>
#include "tt-metalium/device.hpp"
#include "tt-metalium/device_types.hpp"
#include "tt-metalium/dispatch_core_common.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/runtime_args_data.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/sub_device_types.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/tile.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "tt-metalium/tensor/tensor_types.hpp"
#include "tt-metalium/tensor/spec/tensor_spec.hpp"
#include "tt-metalium/tensor/spec/layout/tensor_layout.hpp"
#include "tt-metalium/tensor/spec/layout/page_config.hpp"
#include "tt-metalium/tensor/spec/memory_config/memory_config.hpp"
#include "tt-metalium/tensor/spec/layout/layout.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tensor/host_tensor.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt_metal_cxx/device.hpp"
#include "tt_metal_cxx/buffer.hpp"
#include "tt_metal_cxx/program.hpp"
#include "tt_metal_cxx/kernel.hpp"
#include "tt_metal_cxx/mesh_buffer.hpp"
#include "tt_metal_cxx/distributed.hpp"

template<class T, class = void> struct is_complete : std::false_type {};
template<class T> struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

struct Row { const char* name; int complete; unsigned long long size; unsigned long long align;
  int tc,mv,dt,dc,cc,ntc,nmv,ndt,ndc,ncc,ca,ma,nca,nma,sl; };

template<class T>
Row probe(const char* name){
  if(!is_complete<T>::value) return {name,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  return {name,1,(unsigned long long)sizeof(T),(unsigned long long)alignof(T),
    (int)std::is_trivially_copyable<T>::value,
    (int)std::is_move_constructible<T>::value,
    (int)std::is_destructible<T>::value,
    (int)std::is_default_constructible<T>::value,
    (int)std::is_copy_constructible<T>::value,
    (int)std::is_trivially_copyable<T>::value,
    (int)std::is_nothrow_move_constructible<T>::value,
    (int)std::is_nothrow_destructible<T>::value,
    (int)std::is_nothrow_default_constructible<T>::value,
    (int)std::is_nothrow_copy_constructible<T>::value,
    (int)std::is_copy_assignable<T>::value,
    (int)std::is_move_assignable<T>::value,
    (int)std::is_nothrow_copy_assignable<T>::value,
    (int)std::is_nothrow_move_assignable<T>::value,
    (int)std::is_standard_layout<T>::value};
}

int main(){
  { auto r0 = probe<tt::tt_metal::IDevice>("tt::tt_metal::IDevice");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r0.name,r0.complete,r0.size,r0.align,r0.tc,r0.mv,r0.dt,r0.dc,r0.cc,r0.ntc,r0.nmv,r0.ndt,r0.ndc,r0.ncc,r0.ca,r0.ma,r0.nca,r0.nma,r0.sl); }
  { auto r1 = probe<tt::ChipId>("tt::ChipId");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r1.name,r1.complete,r1.size,r1.align,r1.tc,r1.mv,r1.dt,r1.dc,r1.cc,r1.ntc,r1.nmv,r1.ndt,r1.ndc,r1.ncc,r1.ca,r1.ma,r1.nca,r1.nma,r1.sl); }
  { auto r2 = probe<tt::tt_metal::DispatchCoreConfig>("tt::tt_metal::DispatchCoreConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r2.name,r2.complete,r2.size,r2.align,r2.tc,r2.mv,r2.dt,r2.dc,r2.cc,r2.ntc,r2.nmv,r2.ndt,r2.ndc,r2.ncc,r2.ca,r2.ma,r2.nca,r2.nma,r2.sl); }
  { auto r3 = probe<tt::tt_metal::Program>("tt::tt_metal::Program");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r3.name,r3.complete,r3.size,r3.align,r3.tc,r3.mv,r3.dt,r3.dc,r3.cc,r3.ntc,r3.nmv,r3.ndt,r3.ndc,r3.ncc,r3.ca,r3.ma,r3.nca,r3.nma,r3.sl); }
  { auto r4 = probe<tt::tt_metal::CoreCoord>("tt::tt_metal::CoreCoord");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r4.name,r4.complete,r4.size,r4.align,r4.tc,r4.mv,r4.dt,r4.dc,r4.cc,r4.ntc,r4.nmv,r4.ndt,r4.ndc,r4.ncc,r4.ca,r4.ma,r4.nca,r4.nma,r4.sl); }
  { auto r5 = probe<tt::tt_metal::CoreRange>("tt::tt_metal::CoreRange");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r5.name,r5.complete,r5.size,r5.align,r5.tc,r5.mv,r5.dt,r5.dc,r5.cc,r5.ntc,r5.nmv,r5.ndt,r5.ndc,r5.ncc,r5.ca,r5.ma,r5.nca,r5.nma,r5.sl); }
  { auto r6 = probe<tt::tt_metal::CoreRangeSet>("tt::tt_metal::CoreRangeSet");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r6.name,r6.complete,r6.size,r6.align,r6.tc,r6.mv,r6.dt,r6.dc,r6.cc,r6.ntc,r6.nmv,r6.ndt,r6.ndc,r6.ncc,r6.ca,r6.ma,r6.nca,r6.nma,r6.sl); }
  { auto r7 = probe<tt::tt_metal::DataMovementConfig>("tt::tt_metal::DataMovementConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r7.name,r7.complete,r7.size,r7.align,r7.tc,r7.mv,r7.dt,r7.dc,r7.cc,r7.ntc,r7.nmv,r7.ndt,r7.ndc,r7.ncc,r7.ca,r7.ma,r7.nca,r7.nma,r7.sl); }
  { auto r8 = probe<tt::tt_metal::ComputeConfig>("tt::tt_metal::ComputeConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r8.name,r8.complete,r8.size,r8.align,r8.tc,r8.mv,r8.dt,r8.dc,r8.cc,r8.ntc,r8.nmv,r8.ndt,r8.ndc,r8.ncc,r8.ca,r8.ma,r8.nca,r8.nma,r8.sl); }
  { auto r9 = probe<tt::tt_metal::KernelHandle>("tt::tt_metal::KernelHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r9.name,r9.complete,r9.size,r9.align,r9.tc,r9.mv,r9.dt,r9.dc,r9.cc,r9.ntc,r9.nmv,r9.ndt,r9.ndc,r9.ncc,r9.ca,r9.ma,r9.nca,r9.nma,r9.sl); }
  { auto r10 = probe<tt::tt_metal::ProgramId>("tt::tt_metal::ProgramId");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r10.name,r10.complete,r10.size,r10.align,r10.tc,r10.mv,r10.dt,r10.dc,r10.cc,r10.ntc,r10.nmv,r10.ndt,r10.ndc,r10.ncc,r10.ca,r10.ma,r10.nca,r10.nma,r10.sl); }
  { auto r11 = probe<tt::tt_metal::ReaderDataMovementConfig>("tt::tt_metal::ReaderDataMovementConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r11.name,r11.complete,r11.size,r11.align,r11.tc,r11.mv,r11.dt,r11.dc,r11.cc,r11.ntc,r11.nmv,r11.ndt,r11.ndc,r11.ncc,r11.ca,r11.ma,r11.nca,r11.nma,r11.sl); }
  { auto r12 = probe<tt::tt_metal::WriterDataMovementConfig>("tt::tt_metal::WriterDataMovementConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r12.name,r12.complete,r12.size,r12.align,r12.tc,r12.mv,r12.dt,r12.dc,r12.cc,r12.ntc,r12.nmv,r12.ndt,r12.ndc,r12.ncc,r12.ca,r12.ma,r12.nca,r12.nma,r12.sl); }
  { auto r13 = probe<tt::tt_metal::MathFidelity>("tt::tt_metal::MathFidelity");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r13.name,r13.complete,r13.size,r13.align,r13.tc,r13.mv,r13.dt,r13.dc,r13.cc,r13.ntc,r13.nmv,r13.ndt,r13.ndc,r13.ncc,r13.ca,r13.ma,r13.nca,r13.nma,r13.sl); }
  { auto r14 = probe<tt::tt_metal::UnpackToDestMode>("tt::tt_metal::UnpackToDestMode");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r14.name,r14.complete,r14.size,r14.align,r14.tc,r14.mv,r14.dt,r14.dc,r14.cc,r14.ntc,r14.nmv,r14.ndt,r14.ndc,r14.ncc,r14.ca,r14.ma,r14.nca,r14.nma,r14.sl); }
  { auto r15 = probe<tt::tt_metal::KernelBuildOptLevel>("tt::tt_metal::KernelBuildOptLevel");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r15.name,r15.complete,r15.size,r15.align,r15.tc,r15.mv,r15.dt,r15.dc,r15.cc,r15.ntc,r15.nmv,r15.ndt,r15.ndc,r15.ncc,r15.ca,r15.ma,r15.nca,r15.nma,r15.sl); }
  { auto r16 = probe<tt::tt_metal::DataMovementProcessor>("tt::tt_metal::DataMovementProcessor");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r16.name,r16.complete,r16.size,r16.align,r16.tc,r16.mv,r16.dt,r16.dc,r16.cc,r16.ntc,r16.nmv,r16.ndt,r16.ndc,r16.ncc,r16.ca,r16.ma,r16.nca,r16.nma,r16.sl); }
  { auto r17 = probe<tt::tt_metal::NOC>("tt::tt_metal::NOC");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r17.name,r17.complete,r17.size,r17.align,r17.tc,r17.mv,r17.dt,r17.dc,r17.cc,r17.ntc,r17.nmv,r17.ndt,r17.ndc,r17.ncc,r17.ca,r17.ma,r17.nca,r17.nma,r17.sl); }
  { auto r18 = probe<tt::tt_metal::NOC_MODE>("tt::tt_metal::NOC_MODE");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r18.name,r18.complete,r18.size,r18.align,r18.tc,r18.mv,r18.dt,r18.dc,r18.cc,r18.ntc,r18.nmv,r18.ndt,r18.ndc,r18.ncc,r18.ca,r18.ma,r18.nca,r18.nma,r18.sl); }
  { auto r19 = probe<tt::tt_metal::RuntimeArgsData>("tt::tt_metal::RuntimeArgsData");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r19.name,r19.complete,r19.size,r19.align,r19.tc,r19.mv,r19.dt,r19.dc,r19.cc,r19.ntc,r19.nmv,r19.ndt,r19.ndc,r19.ncc,r19.ca,r19.ma,r19.nca,r19.nma,r19.sl); }
  { auto r20 = probe<tt::tt_metal::Buffer>("tt::tt_metal::Buffer");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r20.name,r20.complete,r20.size,r20.align,r20.tc,r20.mv,r20.dt,r20.dc,r20.cc,r20.ntc,r20.nmv,r20.ndt,r20.ndc,r20.ncc,r20.ca,r20.ma,r20.nca,r20.nma,r20.sl); }
  { auto r21 = probe<tt::tt_metal::BufferConfig>("tt::tt_metal::BufferConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r21.name,r21.complete,r21.size,r21.align,r21.tc,r21.mv,r21.dt,r21.dc,r21.cc,r21.ntc,r21.nmv,r21.ndt,r21.ndc,r21.ncc,r21.ca,r21.ma,r21.nca,r21.nma,r21.sl); }
  { auto r22 = probe<tt::tt_metal::InterleavedBufferConfig>("tt::tt_metal::InterleavedBufferConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r22.name,r22.complete,r22.size,r22.align,r22.tc,r22.mv,r22.dt,r22.dc,r22.cc,r22.ntc,r22.nmv,r22.ndt,r22.ndc,r22.ncc,r22.ca,r22.ma,r22.nca,r22.nma,r22.sl); }
  { auto r23 = probe<tt::tt_metal::ShardedBufferConfig>("tt::tt_metal::ShardedBufferConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r23.name,r23.complete,r23.size,r23.align,r23.tc,r23.mv,r23.dt,r23.dc,r23.cc,r23.ntc,r23.nmv,r23.ndt,r23.ndc,r23.ncc,r23.ca,r23.ma,r23.nca,r23.nma,r23.sl); }
  { auto r24 = probe<tt::tt_metal::ShardSpec>("tt::tt_metal::ShardSpec");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r24.name,r24.complete,r24.size,r24.align,r24.tc,r24.mv,r24.dt,r24.dc,r24.cc,r24.ntc,r24.nmv,r24.ndt,r24.ndc,r24.ncc,r24.ca,r24.ma,r24.nca,r24.nma,r24.sl); }
  { auto r25 = probe<tt::tt_metal::ShardSpecBuffer>("tt::tt_metal::ShardSpecBuffer");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r25.name,r25.complete,r25.size,r25.align,r25.tc,r25.mv,r25.dt,r25.dc,r25.cc,r25.ntc,r25.nmv,r25.ndt,r25.ndc,r25.ncc,r25.ca,r25.ma,r25.nca,r25.nma,r25.sl); }
  { auto r26 = probe<tt::tt_metal::BufferType>("tt::tt_metal::BufferType");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r26.name,r26.complete,r26.size,r26.align,r26.tc,r26.mv,r26.dt,r26.dc,r26.cc,r26.ntc,r26.nmv,r26.ndt,r26.ndc,r26.ncc,r26.ca,r26.ma,r26.nca,r26.nma,r26.sl); }
  { auto r27 = probe<tt::tt_metal::TensorMemoryLayout>("tt::tt_metal::TensorMemoryLayout");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r27.name,r27.complete,r27.size,r27.align,r27.tc,r27.mv,r27.dt,r27.dc,r27.cc,r27.ntc,r27.nmv,r27.ndt,r27.ndc,r27.ncc,r27.ca,r27.ma,r27.nca,r27.nma,r27.sl); }
  { auto r28 = probe<tt::tt_metal::ShardOrientation>("tt::tt_metal::ShardOrientation");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r28.name,r28.complete,r28.size,r28.align,r28.tc,r28.mv,r28.dt,r28.dc,r28.cc,r28.ntc,r28.nmv,r28.ndt,r28.ndc,r28.ncc,r28.ca,r28.ma,r28.nca,r28.nma,r28.sl); }
  { auto r29 = probe<tt::tt_metal::SubDeviceId>("tt::tt_metal::SubDeviceId");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r29.name,r29.complete,r29.size,r29.align,r29.tc,r29.mv,r29.dt,r29.dc,r29.cc,r29.ntc,r29.nmv,r29.ndt,r29.ndc,r29.ncc,r29.ca,r29.ma,r29.nca,r29.nma,r29.sl); }
  { auto r30 = probe<tt::tt_metal::DeviceAddr>("tt::tt_metal::DeviceAddr");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r30.name,r30.complete,r30.size,r30.align,r30.tc,r30.mv,r30.dt,r30.dc,r30.cc,r30.ntc,r30.nmv,r30.ndt,r30.ndc,r30.ncc,r30.ca,r30.ma,r30.nca,r30.nma,r30.sl); }
  { auto r31 = probe<tt::tt_metal::CBHandle>("tt::tt_metal::CBHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r31.name,r31.complete,r31.size,r31.align,r31.tc,r31.mv,r31.dt,r31.dc,r31.cc,r31.ntc,r31.nmv,r31.ndt,r31.ndc,r31.ncc,r31.ca,r31.ma,r31.nca,r31.nma,r31.sl); }
  { auto r32 = probe<tt::tt_metal::Tile>("tt::tt_metal::Tile");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r32.name,r32.complete,r32.size,r32.align,r32.tc,r32.mv,r32.dt,r32.dc,r32.cc,r32.ntc,r32.nmv,r32.ndt,r32.ndc,r32.ncc,r32.ca,r32.ma,r32.nca,r32.nma,r32.sl); }
  { auto r33 = probe<tt::tt_metal::CircularBufferConfig>("tt::tt_metal::CircularBufferConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r33.name,r33.complete,r33.size,r33.align,r33.tc,r33.mv,r33.dt,r33.dc,r33.cc,r33.ntc,r33.nmv,r33.ndt,r33.ndc,r33.ncc,r33.ca,r33.ma,r33.nca,r33.nma,r33.sl); }
  { auto r34 = probe<tt::tt_metal::CircularBufferConfig::Builder>("tt::tt_metal::CircularBufferConfig::Builder");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r34.name,r34.complete,r34.size,r34.align,r34.tc,r34.mv,r34.dt,r34.dc,r34.cc,r34.ntc,r34.nmv,r34.ndt,r34.ndc,r34.ncc,r34.ca,r34.ma,r34.nca,r34.nma,r34.sl); }
  { auto r35 = probe<tt::tt_metal::distributed::ReplicatedBufferConfig>("tt::tt_metal::distributed::ReplicatedBufferConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r35.name,r35.complete,r35.size,r35.align,r35.tc,r35.mv,r35.dt,r35.dc,r35.cc,r35.ntc,r35.nmv,r35.ndt,r35.ndc,r35.ncc,r35.ca,r35.ma,r35.nca,r35.nma,r35.sl); }
  { auto r36 = probe<tt::tt_metal::distributed::DeviceLocalBufferConfig>("tt::tt_metal::distributed::DeviceLocalBufferConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r36.name,r36.complete,r36.size,r36.align,r36.tc,r36.mv,r36.dt,r36.dc,r36.cc,r36.ntc,r36.nmv,r36.ndt,r36.ndc,r36.ncc,r36.ca,r36.ma,r36.nca,r36.nma,r36.sl); }
  { auto r37 = probe<tt::tt_metal::distributed::MeshBuffer>("tt::tt_metal::distributed::MeshBuffer");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r37.name,r37.complete,r37.size,r37.align,r37.tc,r37.mv,r37.dt,r37.dc,r37.cc,r37.ntc,r37.nmv,r37.ndt,r37.ndc,r37.ncc,r37.ca,r37.ma,r37.nca,r37.nma,r37.sl); }
  { auto r38 = probe<tt::tt_metal::TensorAccessorArgs>("tt::tt_metal::TensorAccessorArgs");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r38.name,r38.complete,r38.size,r38.align,r38.tc,r38.mv,r38.dt,r38.dc,r38.cc,r38.ntc,r38.nmv,r38.ndt,r38.ndc,r38.ncc,r38.ca,r38.ma,r38.nca,r38.nma,r38.sl); }
  { auto r39 = probe<tt::tt_metal::distributed::MeshDevice>("tt::tt_metal::distributed::MeshDevice");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r39.name,r39.complete,r39.size,r39.align,r39.tc,r39.mv,r39.dt,r39.dc,r39.cc,r39.ntc,r39.nmv,r39.ndt,r39.ndc,r39.ncc,r39.ca,r39.ma,r39.nca,r39.nma,r39.sl); }
  { auto r40 = probe<tt::tt_metal::distributed::MeshWorkload>("tt::tt_metal::distributed::MeshWorkload");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r40.name,r40.complete,r40.size,r40.align,r40.tc,r40.mv,r40.dt,r40.dc,r40.cc,r40.ntc,r40.nmv,r40.ndt,r40.ndc,r40.ncc,r40.ca,r40.ma,r40.nca,r40.nma,r40.sl); }
  { auto r41 = probe<tt::tt_metal::distributed::MeshCommandQueue>("tt::tt_metal::distributed::MeshCommandQueue");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r41.name,r41.complete,r41.size,r41.align,r41.tc,r41.mv,r41.dt,r41.dc,r41.cc,r41.ntc,r41.nmv,r41.ndt,r41.ndc,r41.ncc,r41.ca,r41.ma,r41.nca,r41.nma,r41.sl); }
  { auto r42 = probe<tt::tt_metal::distributed::MeshShape>("tt::tt_metal::distributed::MeshShape");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r42.name,r42.complete,r42.size,r42.align,r42.tc,r42.mv,r42.dt,r42.dc,r42.cc,r42.ntc,r42.nmv,r42.ndt,r42.ndc,r42.ncc,r42.ca,r42.ma,r42.nca,r42.nma,r42.sl); }
  { auto r43 = probe<tt::tt_metal::distributed::MeshCoordinateRange>("tt::tt_metal::distributed::MeshCoordinateRange");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r43.name,r43.complete,r43.size,r43.align,r43.tc,r43.mv,r43.dt,r43.dc,r43.cc,r43.ntc,r43.nmv,r43.ndt,r43.ndc,r43.ncc,r43.ca,r43.ma,r43.nca,r43.nma,r43.sl); }
  { auto r44 = probe<tt::tt_metal::DataType>("tt::tt_metal::DataType");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r44.name,r44.complete,r44.size,r44.align,r44.tc,r44.mv,r44.dt,r44.dc,r44.cc,r44.ntc,r44.nmv,r44.ndt,r44.ndc,r44.ncc,r44.ca,r44.ma,r44.nca,r44.nma,r44.sl); }
  { auto r45 = probe<tt::tt_metal::TensorSpec>("tt::tt_metal::TensorSpec");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r45.name,r45.complete,r45.size,r45.align,r45.tc,r45.mv,r45.dt,r45.dc,r45.cc,r45.ntc,r45.nmv,r45.ndt,r45.ndc,r45.ncc,r45.ca,r45.ma,r45.nca,r45.nma,r45.sl); }
  { auto r46 = probe<tt::tt_metal::TensorLayout>("tt::tt_metal::TensorLayout");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r46.name,r46.complete,r46.size,r46.align,r46.tc,r46.mv,r46.dt,r46.dc,r46.cc,r46.ntc,r46.nmv,r46.ndt,r46.ndc,r46.ncc,r46.ca,r46.ma,r46.nca,r46.nma,r46.sl); }
  { auto r47 = probe<tt::tt_metal::PageConfig>("tt::tt_metal::PageConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r47.name,r47.complete,r47.size,r47.align,r47.tc,r47.mv,r47.dt,r47.dc,r47.cc,r47.ntc,r47.nmv,r47.ndt,r47.ndc,r47.ncc,r47.ca,r47.ma,r47.nca,r47.nma,r47.sl); }
  { auto r48 = probe<tt::tt_metal::MemoryConfig>("tt::tt_metal::MemoryConfig");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r48.name,r48.complete,r48.size,r48.align,r48.tc,r48.mv,r48.dt,r48.dc,r48.cc,r48.ntc,r48.nmv,r48.ndt,r48.ndc,r48.ncc,r48.ca,r48.ma,r48.nca,r48.nma,r48.sl); }
  { auto r49 = probe<tt::tt_metal::Layout>("tt::tt_metal::Layout");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r49.name,r49.complete,r49.size,r49.align,r49.tc,r49.mv,r49.dt,r49.dc,r49.cc,r49.ntc,r49.nmv,r49.ndt,r49.ndc,r49.ncc,r49.ca,r49.ma,r49.nca,r49.nma,r49.sl); }
  { auto r50 = probe<tt::tt_metal::Shape>("tt::tt_metal::Shape");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r50.name,r50.complete,r50.size,r50.align,r50.tc,r50.mv,r50.dt,r50.dc,r50.cc,r50.ntc,r50.nmv,r50.ndt,r50.ndc,r50.ncc,r50.ca,r50.ma,r50.nca,r50.nma,r50.sl); }
  { auto r51 = probe<tt::tt_metal::HostTensor>("tt::tt_metal::HostTensor");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r51.name,r51.complete,r51.size,r51.align,r51.tc,r51.mv,r51.dt,r51.dc,r51.cc,r51.ntc,r51.nmv,r51.ndt,r51.ndc,r51.ncc,r51.ca,r51.ma,r51.nca,r51.nma,r51.sl); }
  { auto r52 = probe<tt::tt_metal::HostBuffer>("tt::tt_metal::HostBuffer");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r52.name,r52.complete,r52.size,r52.align,r52.tc,r52.mv,r52.dt,r52.dc,r52.cc,r52.ntc,r52.nmv,r52.ndt,r52.ndc,r52.ncc,r52.ca,r52.ma,r52.nca,r52.nma,r52.sl); }
  { auto r53 = probe<tt::tt_metal::TensorTopology>("tt::tt_metal::TensorTopology");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r53.name,r53.complete,r53.size,r53.align,r53.tc,r53.mv,r53.dt,r53.dc,r53.cc,r53.ntc,r53.nmv,r53.ndt,r53.ndc,r53.ncc,r53.ca,r53.ma,r53.nca,r53.nma,r53.sl); }
  { auto r54 = probe<bfloat16>("bfloat16");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r54.name,r54.complete,r54.size,r54.align,r54.tc,r54.mv,r54.dt,r54.dc,r54.cc,r54.ntc,r54.nmv,r54.ndt,r54.ndc,r54.ncc,r54.ca,r54.ma,r54.nca,r54.nma,r54.sl); }
  { auto r55 = probe<tt_metal_cxx::DeviceHandle>("tt_metal_cxx::DeviceHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r55.name,r55.complete,r55.size,r55.align,r55.tc,r55.mv,r55.dt,r55.dc,r55.cc,r55.ntc,r55.nmv,r55.ndt,r55.ndc,r55.ncc,r55.ca,r55.ma,r55.nca,r55.nma,r55.sl); }
  { auto r56 = probe<tt_metal_cxx::BufferHandle>("tt_metal_cxx::BufferHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r56.name,r56.complete,r56.size,r56.align,r56.tc,r56.mv,r56.dt,r56.dc,r56.cc,r56.ntc,r56.nmv,r56.ndt,r56.ndc,r56.ncc,r56.ca,r56.ma,r56.nca,r56.nma,r56.sl); }
  { auto r57 = probe<tt_metal_cxx::CircularBufferConfigHandle>("tt_metal_cxx::CircularBufferConfigHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r57.name,r57.complete,r57.size,r57.align,r57.tc,r57.mv,r57.dt,r57.dc,r57.cc,r57.ntc,r57.nmv,r57.ndt,r57.ndc,r57.ncc,r57.ca,r57.ma,r57.nca,r57.nma,r57.sl); }
  { auto r58 = probe<tt_metal_cxx::ProgramHandle>("tt_metal_cxx::ProgramHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r58.name,r58.complete,r58.size,r58.align,r58.tc,r58.mv,r58.dt,r58.dc,r58.cc,r58.ntc,r58.nmv,r58.ndt,r58.ndc,r58.ncc,r58.ca,r58.ma,r58.nca,r58.nma,r58.sl); }
  { auto r59 = probe<tt_metal_cxx::ComputeKernelConfigHandle>("tt_metal_cxx::ComputeKernelConfigHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r59.name,r59.complete,r59.size,r59.align,r59.tc,r59.mv,r59.dt,r59.dc,r59.cc,r59.ntc,r59.nmv,r59.ndt,r59.ndc,r59.ncc,r59.ca,r59.ma,r59.nca,r59.nma,r59.sl); }
  { auto r60 = probe<tt_metal_cxx::DataMovementKernelConfigHandle>("tt_metal_cxx::DataMovementKernelConfigHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r60.name,r60.complete,r60.size,r60.align,r60.tc,r60.mv,r60.dt,r60.dc,r60.cc,r60.ntc,r60.nmv,r60.ndt,r60.ndc,r60.ncc,r60.ca,r60.ma,r60.nca,r60.nma,r60.sl); }
  { auto r61 = probe<tt_metal_cxx::MeshBufferHandle>("tt_metal_cxx::MeshBufferHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r61.name,r61.complete,r61.size,r61.align,r61.tc,r61.mv,r61.dt,r61.dc,r61.cc,r61.ntc,r61.nmv,r61.ndt,r61.ndc,r61.ncc,r61.ca,r61.ma,r61.nca,r61.nma,r61.sl); }
  { auto r62 = probe<tt_metal_cxx::MeshDeviceHandle>("tt_metal_cxx::MeshDeviceHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r62.name,r62.complete,r62.size,r62.align,r62.tc,r62.mv,r62.dt,r62.dc,r62.cc,r62.ntc,r62.nmv,r62.ndt,r62.ndc,r62.ncc,r62.ca,r62.ma,r62.nca,r62.nma,r62.sl); }
  { auto r63 = probe<tt_metal_cxx::MeshWorkloadHandle>("tt_metal_cxx::MeshWorkloadHandle");
    printf("%s\t%d\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
      r63.name,r63.complete,r63.size,r63.align,r63.tc,r63.mv,r63.dt,r63.dc,r63.cc,r63.ntc,r63.nmv,r63.ndt,r63.ndc,r63.ncc,r63.ca,r63.ma,r63.nca,r63.nma,r63.sl); }
  return 0;
}
