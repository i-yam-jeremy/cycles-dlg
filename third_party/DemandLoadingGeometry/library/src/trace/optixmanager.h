#pragma once

#include "asset.h"
#include "optixconnector.h"
#include <DemandLoadingGeometry.hpp>
#include <array>
#include <cuda_runtime.h>
#include <device_memory_wrapper.h>
#include <memory>
#include <mutex>
#include <optix_types.h>
#include <sharedtypes/chunk.h>
#include <span>
#include <string>
#include <vector>

namespace glow::optix {

struct ASData {
  OptixTraversableHandle gasHandle = 0;
  std::shared_ptr<glow::memory::DevicePtr<char>> gasBuffer = nullptr;
};

class OptixManager {
 public:
  OptixManager(const std::shared_ptr<glow::optix::OptixConnector> optix,
               OptixDeviceContext context);
  result<OptixProgramGroupDesc, Err> createProgramGroup(
      const OptixPipelineCompileOptions &pipeline_compile_options,
      const OptixModuleCompileOptions &module_compile_options);
  result<OptixTraversableHandle, Err> createTopLevelTraversable(
      const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &topLevelAssets,
      unsigned int baseDlgSbtOffset);
  void updateTopLevelAS(const std::span<OptixTraversableHandle> &topLevelAssets,
                        cudaStream_t stream);
  result<std::tuple<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err>
  createChunkAS(const demandLoadingGeometry::Chunk &chunk,
                const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &assets,
                const std::unordered_map<int, std::shared_ptr<glow::pipeline::render::IAsset>>
                    &assetDependencies,
                float rayDifferential,
                int instanceCount,
                cudaStream_t stream);
  result<std::shared_ptr<ASData>, Err> buildAS(const demandLoadingGeometry::Mesh &mesh,
                                               int sbtOffset,
                                               cudaStream_t stream);
  result<OptixAccelBufferSizes, Err> computeMemoryUsage(const OptixBuildInput &buildInput);
  result<OptixAccelBufferSizes, Err> computeMemoryUsage(
      const std::vector<OptixBuildInput> &buildInputs);

 private:
  result<std::tuple<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err>
  createTopLevelAS(const std::vector<demandLoadingGeometry::Chunk::InstanceList> &instanceXforms,
                   const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &assets,
                   unsigned int baseDlgSbtOffset,
                   cudaStream_t stream);
  result<std::pair<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err>
  buildASandCompact(const OptixBuildInput &buildInput,
                    cudaStream_t stream,
                    bool allowUpdates = false);
  result<std::pair<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err>
  buildASandCompact(const std::vector<OptixBuildInput> &buildInputs,
                    cudaStream_t stream,
                    bool allowUpdates = false);
  result<void, Err> updateAS(OptixTraversableHandle handle,
                             glow::memory::DevicePtr<char> &buffer,
                             const OptixBuildInput &buildInput,
                             cudaStream_t stream);
  result<void, Err> updateAS(OptixTraversableHandle handle,
                             glow::memory::DevicePtr<char> &buffer,
                             const std::vector<OptixBuildInput> &buildInputs,
                             cudaStream_t stream);
  result<OptixTraversableHandle, Err> getAssetAabbAS(
      int assetId,
      const std::shared_ptr<glow::pipeline::render::IAsset> asset,
      cudaStream_t stream);

  const std::shared_ptr<glow::optix::OptixConnector> optix;
  OptixDeviceContext context = nullptr;
  OptixModule m_module{};
  OptixTraversableHandle topLevelAABBHandle = 0;
  std::shared_ptr<glow::memory::DevicePtr<char>> topLevelAABBBuffer = nullptr;
  std::unordered_map<
      int,
      std::pair<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>>
      aabbASes;
  std::unordered_map<int, std::mutex> aabbASMutexes;
  std::unordered_map<int, std::vector<OptixInstance>> aabbInstances;

  std::vector<OptixInstance> m_topLevelChunkInstancesHost;
  std::shared_ptr<glow::memory::DevicePtr<OptixInstance>> m_topLevelChunkInstances;

  std::mutex mutex;
  std::vector<char> log;  // For error reporting from OptiX creation functions
  OptixAccelBuildOptions accel_options = {OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                              OPTIX_BUILD_FLAG_PREFER_FAST_BUILD |
                                              OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
                                          OPTIX_BUILD_OPERATION_BUILD,
                                          OptixMotionOptions{1, OPTIX_MOTION_FLAG_NONE, 0.f, 0.f}};
};
};  // namespace glow::optix