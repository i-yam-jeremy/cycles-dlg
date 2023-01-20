#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct __attribute__((packed)) AffineXform {
  __host__ __device__ AffineXform() {}
  explicit AffineXform(const glm::mat4 &m) {
    const auto transpose = glm::transpose(m);
    memcpy(values, &transpose[0][0], sizeof(values));
  }

  __host__ __device__ glm::mat4 toMat() const {
    glm::mat4 m(1);
    memcpy(&m[0][0], values, sizeof(values));
    return glm::transpose(m);
  }

  float values[12];
};