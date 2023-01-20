
#include <DemandLoadingGeometry.hpp>

#include <cuda_runtime.h>
#include <optix.h>
#include <stdlib.h>

constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * KB;
constexpr size_t GB = 1024 * MB;

const size_t WIDTH = 1920;
const size_t HEIGHT = 1080;

struct MeshSBTRecordEntry {
  float3 *vertexColors;
};

int main() {
  // // Create DLG Context
  // demandLoadingGeometry::Options options{
  //     .maxPartitionInstanceCount = 10000,
  //     .maxMemory = 1 * GB,
  //     .maxActiveRayCount = WIDTH * HEIGHT,
  //     .greedyLoading = false,
  // };
  // auto geoDemandLoader = demandLoadingGeometry::createDemandLoader(options);

  // auto triBuildInput = std::make_unique<OptixBuildInput>();
  // auto triHandle = geoDemandLoader->addBuildInput(std::move(triBuildInput));

  // constexpr size_t instanceCount = 100000;
  // geoDemandLoader->reserveSpaceForNewInstances(instanceCount); // Just an optimization to avoid vector resizes
  // for (size_t i = 0; i < instanceCount; i++) {
  //   auto instance = std::make_unique<OptixInstance>();
  //   geoDemandLoader->addInstance(std::move(instance), triHandle);
  // }

  // geoDemandLoader->partition(); // (maybe this can be done without SBT using instance ID of the custom AABB primitive instance?? and instance ID == asset Id)

  // // Standard OptiX setup with
  // auto dlgModule = geoDemandLoader->getOptixModule();
  // // TODO link optix pipeline (may also need to return program groups from DLG)

  // cudaStream_t stream;
  // CUDA_CHECK(cudaStreamCreate(&stream));

  // bool anyRaysActive = true;
  // while (anyRaysActive) {
  //   geoDemandLoader->preLaunch(stream);
  //   anyRaysActive = traceRays(geoDemandLoader->getDeviceContext(), stream);
  // }
}