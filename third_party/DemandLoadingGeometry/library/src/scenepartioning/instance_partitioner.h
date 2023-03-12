#pragma once

#include "chunk.h"
#include "chunk_data.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <affinexform.h>

namespace glow::pipeline::sceneloader::partition {

class InstancePartitioner {
public:
  InstancePartitioner();
  virtual ~InstancePartitioner() = default;
  void setMeshInfo(int meshId, const OptixAabb& aabb, const size_t memoryUsage);
  void add(int meshId, const demandLoadingGeometry::AffineXform &instanceXform, uint32_t instanceId);
  int writeChunks(std::function<void(const std::shared_ptr<glow::pipeline::render::Chunk>)> callback);

protected:
  struct SubChunkStats {
    size_t instanceCount;
    size_t memoryUsage;
  };

  // Sub-class Interface
  virtual bool needsSubChunkEvalPass() = 0;
  virtual void getSubChunkAABBs(std::vector<OptixAabb> &subChunkAabbs, const OptixAabb &chunkAabb) = 0;
  virtual void selectSubChunks(std::vector<OptixAabb> &subChunkAabbs, const std::vector<SubChunkStats> &subChunkStats) {
    throw new std::runtime_error("InstancePartitioner::selectSubChunks: not implemented. Must be overriden if needsSubChunkEvalPass returns true");
  }

  // Internal
  void subdivideChunk(const std::shared_ptr<glow::pipeline::render::Chunk> chunk, int depth, std::function<void(const std::shared_ptr<glow::pipeline::render::Chunk>)> callback);

  std::function<void(const std::shared_ptr<glow::pipeline::render::Chunk>)> callback;
  std::shared_ptr<glow::pipeline::render::Chunk> rootChunk = nullptr;
  std::unordered_map<int, OptixAabb> m_meshAabbs;
  std::unordered_map<int, size_t> m_meshMemoryUsages;
  OptixAabb sceneBounds{1e20, 1e20, 1e20, -1e20, -1e20, -1e20};
  int chunkCount = 0;
  std::vector<std::shared_ptr<glow::pipeline::render::Chunk>> meshChunks;
};

class OctreePartitioner : public InstancePartitioner {
public:
  OctreePartitioner() : InstancePartitioner(){};
  ~OctreePartitioner() override = default;

protected:
  bool needsSubChunkEvalPass() override { return false; }
  void getSubChunkAABBs(std::vector<OptixAabb> &subChunkAabbs, const OptixAabb &chunkAabb) override;
};
class KdTreePartitioner : public InstancePartitioner {
public:
  KdTreePartitioner() : InstancePartitioner(){};
  ~KdTreePartitioner() override = default;

protected:
  bool needsSubChunkEvalPass() override { return true; }
  void getSubChunkAABBs(std::vector<OptixAabb> &subChunkAabbs, const OptixAabb &chunkAabb) override;
  void selectSubChunks(std::vector<OptixAabb> &subChunkAabbs, const std::vector<SubChunkStats> &subChunkStats) override;
};

} // namespace glow::pipeline::sceneloader::partition