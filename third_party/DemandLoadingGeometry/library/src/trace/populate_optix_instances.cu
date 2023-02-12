#include "populate_optix_instances.h"

namespace {
__global__ void populate_optix_instances(OptixInstance *instances,
                                         const demandLoadingGeometry::AffineXform *xforms,
                                         OptixTraversableHandle asHandle,
                                         int sbtOffset,
                                         int instanceCount)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= instanceCount) {
    return;
  }

  const auto &xform = xforms[i];
  auto &instance = instances[i];
  instance.instanceId = 0;
  instance.visibilityMask = 0xFF;
  instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
  memcpy(instance.transform, xform.data, sizeof(instance.transform));
  instance.sbtOffset = sbtOffset;
  instance.traversableHandle = asHandle;
}
}  // namespace

void populateOptixInstances(OptixInstance *d_out_optixInstances,
                            const demandLoadingGeometry::AffineXform *d_xforms,
                            OptixTraversableHandle asHandle,
                            int sbtOffset,
                            int instanceCount,
                            cudaStream_t stream)
{
  dim3 block(32);
  dim3 grid(1 + ((instanceCount - 1) / block.x));
  populate_optix_instances<<<grid, block, 0, stream>>>(
      d_out_optixInstances, d_xforms, asHandle, sbtOffset, instanceCount);
}