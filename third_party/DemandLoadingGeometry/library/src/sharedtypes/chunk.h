#pragma once

#include "affinexform.h"
#include <glm/glm.hpp>
#include <optix.h>
#include <vector>

namespace demandLoadingGeometry {

struct Chunk {
  Chunk() {}

  struct InstanceList {
    int assetIndex;
    std::vector<AffineXform> instanceXforms;
    std::vector<uint32_t> instanceIds;
  };

  OptixAabb aabb;
  glm::mat4 xform;
  std::vector<InstanceList> instanceLists;
};

} // namespace demandLoadingGeometry