#pragma once

#include <glm/glm.hpp>
#include <iostream>
#include <memory>
#include <optix.h>
#include <vector>

namespace demandLoadingGeometry {

struct TriangleBuildInput {
  TriangleBuildInput() {}

  std::vector<glm::vec3> positions;
  std::vector<glm::ivec3> indices;
};

struct Mesh {
  Mesh() {}

  OptixAabb aabb;
  std::vector<std::shared_ptr<TriangleBuildInput>> buildInputs;
};

} // namespace demandLoadingGeometry