#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <sharedtypes/affinexform.h>
#include <unordered_map>
#include <vector>

namespace glow::pipeline::render {
class Chunk {
public:
  Chunk(const OptixAabb &aabb, const glm::mat4 &xform) : aabb(aabb), xform(xform) {}
  void addInstance(int meshId, const glm::mat4 &instanceXform);
  bool isEmpty() const;
  const std::unordered_map<int, std::vector<AffineXform>> &getInstanceXforms() const;
  const OptixAabb &getAabb() const;
  const glm::mat4 &getXform() const;
  size_t getInstanceCount() const;
  void clear();

  OptixAabb aabb;

private:
  size_t instanceCount = 0;
  std::unordered_map<int, std::vector<AffineXform>> instanceXformsByAsset;
  glm::mat4 xform;
};
} // namespace glow::pipeline::render