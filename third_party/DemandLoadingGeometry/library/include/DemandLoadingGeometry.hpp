#pragma once

#include <MeshHandle.hpp>
#include <MeshTypes.h>
#include <affinexform.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <internal/structs.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <optix.h>
#include <string>
#include <tuple>
#include <vector>
#include <cuda.h>

// Temporary (while refactoring is in progress)
enum class InstancePartitionerType {
  OCTREE,
  KDTREE,
};
// End Temporary

#define DLG_CYCLES_PRIM_TYPE (1 << 6)

namespace demandLoadingGeometry {

constexpr OptixTraversableHandle NULL_TRAVERSABLE_HANDLE = 0;

class GeometryDemandLoader;
class GeometryDemandLoaderImpl;
class GeometryDeviceContext;
struct MeshHandle;
struct AssetEntry;

template<typename T> struct SBTRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  union {
    T data;
    internal::ChunkSBTEntry __chunkSbtData;
  };
};

struct Options {
  // ----------------- Scene Loading & Partitioning Options -----------------
  // Maximum instance count per partition. Any partitions with a larger
  // instance count will be subdivided further.
  size_t maxPartitionInstanceCount;

  // ----------------- Runtime/Trace-time Demand Loading Geometry Options -----------------

  // Maximum memory (in bytes) allowed by DemandLoadingGeometry.
  // This memory estimate includes internal buffers, so the actual
  // amount used for geometry will be less.
  size_t maxMemory;

  // The maximum possible number of rays in-flight at once. For example,
  // if the user is tracing one ray per pixel of a 1920x1080 image, then
  // this value would be 1920*1080. This information is required to
  // allocate internal buffers.
  size_t maxActiveRayCount;

  // Geometry will start loading immediately after
  // DemandLoadingGeometry context creation.
  // This is useful to reduce load times for scenes that fit in-core.
  // For out-of-core scenes, this can potentially hurt performance
  // as the loaded geometry will likely be immediately evicted.
  bool greedyLoading;

  InstancePartitionerType instancePartitionerType;
};

GeometryDemandLoader *createDemandLoader(Options &options, CUcontext cuContext, OptixDeviceContext optixContext);

struct SBTBuffer {
  ~SBTBuffer();
  void *data;
  size_t numElements;
  size_t sizeInBytes;
};

struct LaunchData {
  GeometryDeviceContext context;
  demandLoadingGeometry::RayIndex numNewRaysAddedToUserRayQueue;
  demandLoadingGeometry::RayIndex numStalledRays;
};

class GeometryDemandLoader {
 public:
  ~GeometryDemandLoader();

  // Package internal OptiX shaders for user to link into their pipeline
  std::optional<OptixProgramGroupDesc> getOptixProgramGroup(
      const OptixPipelineCompileOptions &pipeline_compile_options,
      const OptixModuleCompileOptions &module_compile_options);
  
  // Register path queues to resume stalled rays
  void registerPathQueues(int** d_pathQueues, int** d_pathQueueSizes);

  // Scene Building API
  // void reserveSpaceForNewInstances(size_t instanceCount);
  MeshHandle addMesh(const Mesh &mesh, const std::optional<OptixAabb> &aabb = {});
  void addInstance(MeshHandle meshHandle, const AffineXform &xform);
  OptixTraversableHandle updateScene(unsigned int baseDlgSbtOffset);

  std::unique_ptr<SBTBuffer> getInternalApiHitgroupSbtEntries(size_t sizeOfUserSbtStruct,
                                                              uint32_t maxSbtTraceOffset);

  // Tracing API
  demandLoadingGeometry::LaunchData preLaunch(demandLoadingGeometry::RayIndex *d_endOfUserRayQueue,
                                              cudaStream_t stream);
  void postLaunch(cudaStream_t stream);

 private:
  friend GeometryDemandLoader *createDemandLoader(Options &options,
                                                  CUcontext cuContext, 
                                                  OptixDeviceContext optixContext);
  GeometryDemandLoader(GeometryDemandLoaderImpl *impl);

  GeometryDemandLoaderImpl *m_impl = nullptr;
};

};  // namespace demandLoadingGeometry