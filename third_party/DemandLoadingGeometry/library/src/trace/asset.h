#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>
#include <optix.h>
#include <unordered_map>
#include <unordered_set>
#include <util/monad/error.h>
#include <util/monad/result.h>

namespace glow::pipeline::render {

struct AssetMetrics {
  size_t raysTraced = 0, iterationsUsed = 0, numLoads = 0, raysTracedRequiringLoad = 0;

  AssetMetrics &operator+=(const AssetMetrics &other) {
    this->raysTraced += other.raysTraced;
    this->iterationsUsed += other.iterationsUsed;
    this->numLoads += other.numLoads;
    this->raysTracedRequiringLoad += other.raysTracedRequiringLoad;
    return *this;
  }
};

class IAsset {
public:
  virtual OptixAabb getAABB() const = 0;
  const AssetMetrics &getMetrics() const {
    return metrics;
  }
  void addMetrics(const AssetMetrics &newMetrics) {
    metrics += newMetrics;
  };
  virtual glm::mat4 getChunkXform() const = 0;
  virtual int getSBTOffset() const = 0;
  virtual int getNumSBTEntries() const = 0;
  virtual result<void, Err> build(void *optixManagerPtr, const std::unordered_map<int, std::shared_ptr<IAsset>> &assetDependencies, float rayDifferential, cudaStream_t stream) = 0;
  virtual void free() = 0;
  // Must be thread-safe
  virtual bool isResident() = 0;
  virtual bool hasChildAssets() const = 0;
  virtual const std::unordered_set<int> &getDependencies() const = 0;
  // Must be thread-safe
  virtual OptixTraversableHandle getAS() = 0;
  // Must be thread-safe
  virtual result<size_t, Err> getMemoryFootprintEstimate(void *optixManagerPtr) = 0;

private:
  AssetMetrics metrics = {};
};
} // namespace glow::pipeline::render