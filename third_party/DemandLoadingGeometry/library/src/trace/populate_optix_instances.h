#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <affinexform.h>

void populateOptixInstances(OptixInstance *d_out_optixInstances, const demandLoadingGeometry::AffineXform *d_xforms, OptixTraversableHandle asHandle, int sbtOffset, int instanceCount, cudaStream_t stream);