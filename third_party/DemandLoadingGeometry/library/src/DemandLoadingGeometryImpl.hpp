#pragma once

#include <DemandLoadingGeometry.hpp>
#include <cuda.h>
#include <memory>
#include <mutex>
#include <scenepartioning/instance_partitioner.h>
#include <sharedtypes/chunk.h>
#include <trace/AssetCache.hpp>
#include <trace/optixmanager.h>

namespace demandLoadingGeometry {

class GeometryDemandLoaderImpl {
 public:
  GeometryDemandLoaderImpl(
      std::unique_ptr<glow::pipeline::sceneloader::partition::InstancePartitioner>
          instancePartitioner,
      const Options &options,
      CUcontext cuContext,
      OptixDeviceContext context);
  ~GeometryDemandLoaderImpl();

  // Package internal OptiX shaders for user to link into their pipeline
  std::optional<OptixProgramGroupDesc> getOptixProgramGroup(
      const OptixPipelineCompileOptions &pipeline_compile_options,
      const OptixModuleCompileOptions &module_compile_options);

  // Scene Building API
  // void reserveSpaceForNewInstances(size_t instanceCount);
  MeshHandle addMesh(const Mesh &mesh, const std::optional<OptixAabb> &aabb);
  void addInstance(MeshHandle meshHandle, const AffineXform &xform, uint32_t instanceId);
  OptixTraversableHandle updateScene(unsigned int baseDlgSbtOffset);

  std::unique_ptr<SBTBuffer> getInternalApiHitgroupSbtEntries(size_t sizeOfUserSbtStruct,
                                                              uint32_t maxTraceSbtOffset);

  // Tracing API
  demandLoadingGeometry::LaunchData preLaunch(demandLoadingGeometry::RayIndex *d_endOfUserRayQueue,
                                              cudaStream_t stream);
  void postLaunch(cudaStream_t stream);

 private:
  CUcontext m_cuContext{};
  std::shared_ptr<glow::memory::DevicePtr<GeometryDeviceContext>> m_deviceContext = nullptr;
  std::unique_ptr<glow::pipeline::sceneloader::partition::InstancePartitioner>
      m_instancePartitioner = nullptr;
  Options m_options;

  std::vector<Mesh> m_meshes;
  std::vector<Chunk> m_chunks;
  std::shared_ptr<glow::optix::OptixManager> m_optixManager = nullptr;
  std::shared_ptr<AssetCache> m_assetCache = nullptr;
  std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> m_chunkAssets;
  std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> m_assets;

  std::shared_ptr<glow::memory::DevicePtr<internal::RayCount>> m_assetRayCounts = nullptr;
  std::vector<RayIndex *> m_stalledRayQueues;

  void partition();
  void updateAssets();
  OptixTraversableHandle createTopLevelTraversable(unsigned int baseDlgSbtOffset);
  void updateAssetCache();
  void clearAssetRayCounts();
};

}  // namespace demandLoadingGeometry