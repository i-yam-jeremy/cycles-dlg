#pragma once

#include "asset.h"
#include <sharedtypes/chunk.h>

namespace demandLoadingGeometry {
class ChunkAsset : public glow::pipeline::render::IAsset {
public:
  ChunkAsset(const Chunk *chunk, const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &assets) : chunk(chunk), sbtOffset(sbtOffset), assets(assets) {
    this->instanceCount = 0;
    for (auto &entry : this->chunk->instanceLists) {
      this->instanceCount += entry.instanceXforms.size();
      dependencies.insert(entry.assetIndex);
    }
  }

  OptixAabb getAABB() const override {
    return chunk->aabb;
  }

  glm::mat4 getChunkXform() const override {
    return chunk->xform;
  }

  int getSBTOffset() const override {
    std::cerr << "ChunkAsset::getSBTOffset should never be called\n";
    std::exit(1);
    return -1;
  }

  int getNumSBTEntries() const override {
    return 0;
  }

  const std::unordered_set<int> &getDependencies() const override {
    return dependencies;
  }

  result<void, Err> build(void *optixManagerPtr, const std::unordered_map<int, std::shared_ptr<IAsset>> &assetDependencies, float rayDifferential, cudaStream_t stream) override {
    auto geometryOptixManager = (glow::optix::OptixManager *)optixManagerPtr;
    UNWRAP(as, geometryOptixManager->createChunkAS(*chunk, assets, assetDependencies, rayDifferential, instanceCount, stream));
    this->asData = as;

    size_t size = 0;
    size += std::get<1>(as)->size();

    this->sizeInBytes = size;
    return {};
  }

  void free() override {
    asData = {};
  }

  bool isResident() override {
    std::lock_guard guard(mutex);
    return asData.has_value();
  }

  bool hasChildAssets() const override {
    return false;
  }

  OptixTraversableHandle getAS() override {
    std::lock_guard guard(mutex);
    return asData.has_value() ? std::get<0>(asData.value()) : 0;
  }

  result<size_t, Err> getMemoryFootprintEstimate(void *optixManagerPtr) override {
    std::lock_guard guard(mutex);
    if (sizeInBytes.has_value()) {
      return sizeInBytes.value();
    }
    auto geometryOptixManager = (glow::optix::OptixManager *)optixManagerPtr;

    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = 0;
    input.instanceArray.numInstances = static_cast<unsigned int>(instanceCount);

    UNWRAP(bufferSizes, geometryOptixManager->computeMemoryUsage(input));
    return bufferSizes.outputSizeInBytes;
  }

private:
  const Chunk *chunk;
  int sbtOffset = -1;
  size_t instanceCount = 0;
  const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &assets;

  std::mutex mutex;
  std::optional<std::tuple<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>>
      asData = {};
  std::optional<size_t> sizeInBytes = {};

  std::unordered_set<int> dependencies;
};
} // namespace demandLoadingGeometry