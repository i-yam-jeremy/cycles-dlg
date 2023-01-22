#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace demandLoadingGeometry {

struct __attribute__((packed)) AffineXform {
  __host__ __device__ AffineXform() {}
  explicit AffineXform(const glm::mat4 &m) {
    const auto transpose = glm::transpose(m);
    memcpy(data, &transpose[0][0], sizeof(data));
  }

  __host__ __device__ glm::mat4 toMat() const {
    glm::mat4 m(1);
    memcpy(&m[0][0], data, sizeof(data));
    return glm::transpose(m);
  }

  float data[12];
};

}