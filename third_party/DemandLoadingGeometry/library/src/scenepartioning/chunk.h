#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <affinexform.h>
#include <unordered_map>
#include <vector>

namespace glow::pipeline::render {
class Chunk {
public:
  Chunk(const OptixAabb &aabb, const glm::mat4 &xform) : aabb(aabb), xform(xform) {}
  void addInstance(int meshId, const demandLoadingGeometry::AffineXform &instanceXform, uint32_t instanceId);
  bool isEmpty() const;
  const std::unordered_map<int, std::vector<demandLoadingGeometry::AffineXform>> &getInstanceXforms() const;
  const std::unordered_map<int, std::vector<uint32_t>> &getInstanceIds() const;
  const OptixAabb &getAabb() const;
  const glm::mat4 &getXform() const;
  size_t getInstanceCount() const;
  void clear();

  OptixAabb aabb;

private:
  size_t instanceCount = 0;
  std::unordered_map<int, std::vector<demandLoadingGeometry::AffineXform>> instanceXformsByAsset;
  std::unordered_map<int, std::vector<uint32_t>> instanceIdsByAsset;
  glm::mat4 xform;
};
} // namespace glow::pipeline::render