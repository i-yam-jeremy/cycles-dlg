#pragma once

#include "asset.h"
#include <MeshTypes.h>

namespace demandLoadingGeometry {
class GeometryAsset : public glow::pipeline::render::IAsset {
public:
  GeometryAsset(const Mesh *mesh, int sbtOffset) : mesh(mesh), sbtOffset(sbtOffset) {}

  OptixAabb getAABB() const override {
    std::cerr << "Should not be called: GeometryAsset::getAABB()\n";
    std::exit(1);
    return OptixAabb{};
  }

  glm::mat4 getChunkXform() const override {
    std::cerr << "Should not be called: GeometryAsset::getChunkXform\n";
    std::exit(1);
    return glm::mat4(1);
  }

  int getSBTOffset() const override {
    return sbtOffset;
  }

  int getNumSBTEntries() const override {
    return mesh->buildInputs.size();
  }

  const std::unordered_set<int> &getDependencies() const override {
    return dependencies;
  }

  result<void, Err> build(void *optixManagerPtr, const std::unordered_map<int, std::shared_ptr<IAsset>> &assetDependencies, float rayDifferential, cudaStream_t stream) override {
    auto geometryOptixManager = (glow::optix::OptixManager *)optixManagerPtr;
    UNWRAP_ASSIGN(this->asData, geometryOptixManager->buildAS(*mesh, getSBTOffset(), stream));

    return {};
  }

  void free() override {
    asData = nullptr;
  }

  bool isResident() override {
    std::lock_guard guard(mutex);
    return asData != nullptr;
  }

  bool hasChildAssets() const override {
    return false;
  }

  OptixTraversableHandle getAS() override {
    std::lock_guard guard(mutex);
    return asData->gasHandle;
  }

  result<size_t, Err> getMemoryFootprintEstimate(void *optixManagerPtr) override {
    std::lock_guard guard(mutex);
    if (asData != nullptr) {
      return asData->gasBuffer->size();
    } else {
      auto geometryOptixManager = (glow::optix::OptixManager *)optixManagerPtr;

      std::vector<OptixBuildInput> inputs;
      CUdeviceptr pointsBuffers[] = {0};
      uint32_t triangleInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
      for (const auto &buildInput : mesh->buildInputs) {
        OptixBuildInput input = {};

        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = sizeof(float3);
        input.triangleArray.numVertices = (int)buildInput->positions.size();
        input.triangleArray.vertexBuffers = pointsBuffers;

        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
        input.triangleArray.numIndexTriplets = (int)buildInput->indices.size();
        input.triangleArray.indexBuffer = 0;

        input.triangleArray.flags = &triangleInputFlags;
        input.triangleArray.numSbtRecords = 1;
        input.triangleArray.sbtIndexOffsetBuffer = 0;
        input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

        inputs.push_back(input);
      }

      UNWRAP(bufferSizes, geometryOptixManager->computeMemoryUsage(inputs));

      const auto gasEstimate = 1.5 * bufferSizes.outputSizeInBytes;
      const auto attribsSize = 0; // mesh->normals.size() * sizeof(mesh->normals[0]) + mesh->texCoords.size() * sizeof(mesh->texCoords[0]) + mesh->tbns.size() * sizeof(mesh->tbns[0]);
      return gasEstimate + attribsSize;
    }
  }

private:
  const Mesh *mesh;
  int sbtOffset = -1;

  std::mutex mutex;
  std::shared_ptr<glow::optix::ASData> asData = nullptr; // Should only be 1 active instance (this one), but using shared_ptr to avoid unique_ptr/std::move difficulties when returning a result

  std::unordered_set<int> dependencies;
};
} // namespace demandLoadingGeometry