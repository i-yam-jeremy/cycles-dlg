#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <optix.h>
#include <vector>

namespace demandLoadingGeometry {

struct TriangleBuildInput {
  TriangleBuildInput() {}

  std::vector<float3> positions;
  std::vector<uint3> indices;
};

struct Mesh {
  Mesh() {}
  std::vector<std::shared_ptr<TriangleBuildInput>> buildInputs;
};

} // namespace demandLoadingGeometry