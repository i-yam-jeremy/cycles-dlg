#include <DemandLoadingGeometry.hpp>

#include <ShaderNames.hpp>
#include <optix.h>

namespace demandLoadingGeometry {

constexpr double EPSILON = 0.1f; // NOTE: 0.1f for Moana island, 0.001f for buddha fractal scene

extern "C" __global__ void DEMANDLOADINGGEOMETRY_CHUNK_CH_SHADER_NAME() {
  // Note: optixGetSbtDataPointer() points to the SBT entry data after the SBT entry header
  auto &dlgContext = reinterpret_cast<internal::ChunkSBTEntry *>(optixGetSbtDataPointer())->context;
  const auto assetIndex = optixGetInstanceId();

  // Store ray index to internal buffer to stall ray eval until the asset has been loaded
  const auto index = atomicAdd(&(dlgContext.d_assetRayCountBuffer[assetIndex]), 1);
  const auto rayIndex = optixGetPayload_0();
  dlgContext.d_stalledRayIndices[assetIndex][index] = rayIndex;

  // user CH shader needs to write it's output to a ray queue of ray indices (needs to be double buffered for writing output without corrupting inputs).
  // In the prelaunch method, if there are any non-empty stalled ray queues for resident assets, those are copied/appended on the user stream so they happen before
  // the user's launch method (unsure how to do this, since user launch needs to know how many rays were added. Maybe prelaunch also returns how many rays were added so the launch is correct?)

  optixSetPayload_0(__float_as_int(max(0.0f, optixGetRayTmax() - EPSILON))); // Intersection t-value so the user can update their ray/path state's min t-value so for a wavefront approach they don't rely on re-tracing the same chunk over and over again
}

} // namespace demandLoadingGeometry