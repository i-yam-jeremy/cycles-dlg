#include <DemandLoadingGeometry.hpp>

#include <ShaderNames.hpp>
#include <optix.h>

namespace demandLoadingGeometry {

// TODO(jberchtold) move this into internal library header so it can be shared on geo demand loader
// impl (e.g. geo demand loader impl will need to allocate ray states of this type, instead of just
// integer index. So I guess this will replace the RayIndex struct)
struct StalledPathState {
  unsigned int kernelIndex : 5;
  unsigned int pathIndex : 27;
};
static_assert(sizeof(StalledPathState) == 4, "sizeof(StalledPathState) must be only 4 bytes");

constexpr double EPSILON = 0.1f;  // NOTE: 0.1f for Moana island, 0.001f for buddha fractal scene

extern "C" __global__ void DEMANDLOADINGGEOMETRY_CHUNK_AH_SHADER_NAME()
{
  // printf("Func: %s\n", __PRETTY_FUNCTION__);
  // printf("Hello!!!\n");  // This function is just for debugging
}

extern "C" __global__ void DEMANDLOADINGGEOMETRY_CHUNK_CH_SHADER_NAME()
{
  // printf("Func: %s\n", __PRETTY_FUNCTION__);
  // printf("Hello: %d, %d\n", optixGetPayload_8(), optixGetPayload_9());
  // Note: optixGetSbtDataPointer() points to the SBT entry data after the SBT entry header
  // auto &dlgContext =
  //     reinterpret_cast<internal::ChunkSBTEntry *>(optixGetSbtDataPointer())->context;
  // const auto assetIndex = optixGetInstanceId();

  // // Store ray index to internal buffer to stall ray eval until the asset has been loaded
  // const auto index = atomicAdd(&(dlgContext.d_assetRayCountBuffer[assetIndex]), 1);
  // const StalledPathState stalledPathState{
  //   .kernelIndex = optixGetPayload_9(),
  //   .pathIndex = optixGetPayload_8()
  // };
  // dlgContext.d_stalledPathStates[assetIndex][index] = stalledPathState;

  // // // user CH shader needs to write it's output to a ray queue of ray indices (needs to be
  // double
  // // // buffered for writing output without corrupting inputs). In the prelaunch method, if
  // there are
  // // // any non-empty stalled ray queues for resident assets, those are copied/appended on the
  // user
  // // // stream so they happen before the user's launch method (unsure how to do this, since user
  // // // launch needs to know how many rays were added. Maybe prelaunch also returns how many
  // rays were
  // // // added so the launch is correct?)

  optixSetPayload_0(__float_as_int(max(0.0f, optixGetRayTmax() - EPSILON)));
  // Intersection t-value so the user can update their ray/path state's
  // min t-value so for a wavefront approach they don't rely on re-tracing
  // the same chunk over and over again

  optixSetPayload_5(DLG_CYCLES_PRIM_TYPE);
  // TODO set prim type to custom DLG type so caller knows to break and early return, but not
  // terminate path

  // TODO(jberchtold) why isnt it printing.Wrong SBT ?
  //     Or wrong bboxes geo or BVH setup that is causing chunk boxes to not be hit ?
  //     Or maybe its somehow traversing the incorrect handle ?
  //     Idk
}

}  // namespace demandLoadingGeometry