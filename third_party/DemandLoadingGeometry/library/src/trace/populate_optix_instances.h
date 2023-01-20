#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sharedtypes/affinexform.h>

void populateOptixInstances(OptixInstance *d_out_optixInstances, const AffineXform *d_xforms, OptixTraversableHandle asHandle, int sbtOffset, int instanceCount, cudaStream_t stream);