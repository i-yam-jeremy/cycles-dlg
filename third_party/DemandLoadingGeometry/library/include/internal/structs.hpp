#pragma once

#include <optix.h>

// Internal Data Structures (do not use)
namespace demandLoadingGeometry::internal {
using RayCount = int;
}

namespace demandLoadingGeometry {
using RayIndex = uint32_t;

struct GeometryDeviceContext {
  internal::RayCount *d_assetRayCountBuffer;
  // RayIndex **d_stalledRayIndices;

  uint32_t assetCount;
};

}  // namespace demandLoadingGeometry

// Internal Data Structures (do not use)
namespace demandLoadingGeometry::internal {

struct ChunkSBTEntry {
  demandLoadingGeometry::GeometryDeviceContext *context;
};

}  // namespace demandLoadingGeometry::internal