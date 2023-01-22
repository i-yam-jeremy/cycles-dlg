#include "chunk.h"

void glow::pipeline::render::Chunk::addInstance(int meshId, const demandLoadingGeometry::AffineXform &instanceXform) {
  instanceXformsByAsset[meshId].push_back(instanceXform);
  instanceCount++;
}

bool glow::pipeline::render::Chunk::isEmpty() const {
  return getInstanceXforms().size() == 0;
}

const std::unordered_map<int, std::vector<demandLoadingGeometry::AffineXform>> &glow::pipeline::render::Chunk::getInstanceXforms() const {
  return instanceXformsByAsset;
}

size_t glow::pipeline::render::Chunk::getInstanceCount() const {
  return instanceCount;
}

const OptixAabb &glow::pipeline::render::Chunk::getAabb() const {
  return aabb;
}

const glm::mat4 &glow::pipeline::render::Chunk::getXform() const {
  return xform;
}

void glow::pipeline::render::Chunk::clear() {
  instanceXformsByAsset.clear();
  instanceCount = 0;
}