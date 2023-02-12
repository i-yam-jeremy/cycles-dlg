#include "instance_partitioner.h"

#include <execution>
#include <iostream>
#include <optional>
#include <unordered_set>

#include <cuda_check_macros.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "chunk_data.h"
#include "instance_partitioner_cuda.h"

namespace glow::pipeline::sceneloader::partition {

constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * KB;
constexpr size_t meshMemoryUsageThreshold = 1000000000 * MB;
// constexpr size_t meshMemoryUsageThreshold = 250 * MB;
constexpr size_t chunkInstanceCountThreshold =
    1000000;  // 1000 - 2500 works well for budda fractal scene

InstancePartitioner::InstancePartitioner()
{
  rootChunk = std::make_shared<glow::pipeline::render::Chunk>(sceneBounds, glm::mat4(1));
}

namespace {
OptixAabb scaleAABB(const OptixAabb &aabb, float scale)
{
  const glm::vec3 lower(aabb.minX, aabb.minY, aabb.minZ);
  const glm::vec3 upper(aabb.maxX, aabb.maxY, aabb.maxZ);
  const auto center = (lower + upper) / 2.f;
  const auto scaledLower = (lower - center) * scale + center;
  const auto scaledUpper = (upper - center) * scale + center;
  return {
      scaledLower.x, scaledLower.y, scaledLower.z, scaledUpper.x, scaledUpper.y, scaledUpper.z};
}

OptixAabb aabbUnion(const OptixAabb &a, const OptixAabb &b)
{
  OptixAabb out{};
  out.minX = std::min(a.minX, b.minX);
  out.minY = std::min(a.minY, b.minY);
  out.minZ = std::min(a.minZ, b.minZ);
  out.maxX = std::max(a.maxX, b.maxX);
  out.maxY = std::max(a.maxY, b.maxY);
  out.maxZ = std::max(a.maxZ, b.maxZ);
  return out;
}

OptixAabb transformAabb(const OptixAabb &aabb, const demandLoadingGeometry::AffineXform &xform)
{
  const glm::vec3 aabbCorners[] = {glm::vec3(aabb.minX, aabb.minY, aabb.minZ),
                                   glm::vec3(aabb.minX, aabb.minY, aabb.maxZ),
                                   glm::vec3(aabb.minX, aabb.maxY, aabb.minZ),
                                   glm::vec3(aabb.minX, aabb.maxY, aabb.maxZ),
                                   glm::vec3(aabb.maxX, aabb.minY, aabb.minZ),
                                   glm::vec3(aabb.maxX, aabb.minY, aabb.maxZ),
                                   glm::vec3(aabb.maxX, aabb.maxY, aabb.minZ),
                                   glm::vec3(aabb.maxX, aabb.maxY, aabb.maxZ)};

  const auto instanceXform = xform.toMat();
  OptixAabb transformedAabb = {1e20, 1e20, 1e20, -1e20, -1e20, -1e20};
  for (const auto pLocal : aabbCorners) {
    const auto pWorld = glm::vec3(instanceXform * glm::vec4(pLocal, 1));
    transformedAabb = {fmin(transformedAabb.minX, pWorld.x),
                       fmin(transformedAabb.minY, pWorld.y),
                       fmin(transformedAabb.minZ, pWorld.z),
                       fmax(transformedAabb.maxX, pWorld.x),
                       fmax(transformedAabb.maxY, pWorld.y),
                       fmax(transformedAabb.maxZ, pWorld.z)};
  }
  return transformedAabb;
}
}  // namespace

void InstancePartitioner::setMeshInfo(int meshId, const OptixAabb &aabb, const size_t memoryUsage)
{
  m_meshAabbs[meshId] = aabb;
  m_meshMemoryUsages[meshId] = memoryUsage;
}

void InstancePartitioner::add(int meshId, const demandLoadingGeometry::AffineXform &instanceXform)
{
  const auto memoryUsage = m_meshMemoryUsages.find(meshId);
  if (memoryUsage == m_meshMemoryUsages.end()) {
    return;
  }

  const auto &aabb = m_meshAabbs.find(meshId);
  if (aabb == m_meshAabbs.end()) {
    return;
  }

  sceneBounds = aabbUnion(sceneBounds, transformAabb(aabb->second, instanceXform));
  rootChunk->aabb = sceneBounds;

  if (memoryUsage->second >= meshMemoryUsageThreshold) {
    auto meshChunk = std::make_shared<glow::pipeline::render::Chunk>(scaleAABB(aabb->second, 1.),
                                                                     instanceXform.toMat());
    meshChunk->addInstance(meshId, instanceXform);
    meshChunks.push_back(meshChunk);
    return;
  }

  rootChunk->addInstance(meshId, instanceXform);
}

namespace {
size_t estimateIasSize(size_t numInstances)
{
  return numInstances * 231 + 250;  // Heuristic determined by regression line over
                                    // optixAccelComputeMemoryUsage estimates
}
}  // namespace

namespace {

struct OptixAabbHash {
  int operator()(const OptixAabb &aabb) const
  {
    int x = 0;
    for (int i = 0; i < sizeof(OptixAabb) / 4; i++) {
      x ^= ((int *)(&aabb))[i];
    }
    return x;
  }
};

struct OptixAabbEq {
  bool operator()(const OptixAabb &v1, const OptixAabb &v2) const
  {
    const auto eq = [](float a, float b) { return a == b; };
    return eq(v1.minX, v2.minX) && eq(v1.minY, v2.minY) && eq(v1.minZ, v2.minZ) &&
           eq(v1.maxX, v2.maxX) && eq(v1.maxY, v2.maxY) && eq(v1.maxZ, v2.maxZ);
  }
};
}  // namespace

void prefetchInstancesAsync(InstanceList &instances,
                            cudaStream_t stream,
                            std::optional<int> deviceId = {})
{
  // int deviceIdValue;
  // if (deviceId.has_value()) {
  //   deviceIdValue = deviceId.value();
  // } else {
  //   CUDA_CHECK(cudaGetDevice(&deviceIdValue));
  // }

  // CUDA_CHECK(cudaMemPrefetchAsync(instances.rawPtr(), sizeof(instances[0]) *
  // instances.capacity(), deviceIdValue, stream));
}

/*
real    1m2.673s
user    0m32.377s
sys     0m26.466s

Memadvise doesn't seem to do much, but maybe that's because a bulk of the time is scene reading and
writing IO, time actual portion after pbf load and without writing to disk (comment out body of
write to file callback). See if I can somehow limit the GPU unified memory to 24GB? Or try to run
on 3080? And compare to CPU version
*/

void memadviseInstancesReadonly(InstanceList &instances, std::optional<int> deviceId = {})
{
  // if (instances.size() == 0) { // No need to prefetch if no instances
  //   return;
  // }

  // int deviceIdValue;
  // if (deviceId.has_value()) {
  //   deviceIdValue = deviceId.value();
  // } else {
  //   CUDA_CHECK(cudaGetDevice(&deviceIdValue));
  // }
  // CUDA_CHECK(cudaMemAdvise(instances.rawPtr(), sizeof(instances[0]) * instances.capacity(),
  // cudaMemAdviseSetReadMostly, deviceIdValue)); CUDA_CHECK(cudaMemAdvise(instances.rawPtr(),
  // sizeof(instances[0]) * instances.capacity(), cudaMemAdviseSetPreferredLocation,
  // deviceIdValue));
}

std::ostream &operator<<(std::ostream &out, OptixAabb const &aabb)
{
  out << "AABB {min=(";
  out << aabb.minX << ", " << aabb.minY << ", " << aabb.minZ;
  out << "), max=(";
  out << aabb.maxX << ", " << aabb.maxY << ", " << aabb.maxZ;
  out << ")}";
  return out;
}

void InstancePartitioner::subdivideChunk(
    const std::shared_ptr<glow::pipeline::render::Chunk> rootChunk,
    int depth,
    std::function<void(const std::shared_ptr<glow::pipeline::render::Chunk>)> callback)
{

  std::vector<std::shared_ptr<Chunk>> chunks;
  std::unordered_set<OptixAabb, OptixAabbHash, OptixAabbEq> usedChunkAabbs;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const auto completeChunk = [&, this](std::shared_ptr<Chunk> chunk) {
    std::cout << "Completed chunk: " << chunk->getAabb() << std::endl;
    std::cout.flush();
    if (chunk->getInstances().size() == 0) {
      return;
    }
    prefetchInstancesAsync(chunk->getInstances(), stream, cudaCpuDeviceId);
    memadviseInstancesReadonly(chunk->getInstances(), cudaCpuDeviceId);
    // if (usedChunkAabbs.contains(chunk->getAabb())) { // TODO FIXME hack to prevent duplicate
    // chunks
    //   continue;
    // }
    // usedChunkAabbs.insert(chunk->getAabb());
    auto chunkData = std::make_shared<glow::pipeline::render::Chunk>(chunk->getAabb(),
                                                                     glm::mat4(1));
    for (size_t instanceIndex = 0; instanceIndex < chunk->getInstances().size(); instanceIndex++) {
      const auto &instance = chunk->getInstances()[instanceIndex];
      chunkData->addInstance(instance.meshId, instance.xform);
    }
    callback(chunkData);
    this->chunkCount++;
    // size_t meshUsage = 0;
    // for (const auto &entry : chunk->getInstanceXforms()) {
    //   meshUsage += meshMemoryUsages[entry.first];
    // }
    // const auto iasSize = estimateIasSize(chunk->getInstanceCount());
    std::cout << this->chunkCount << ", " << depth << ", " << chunk->getInstances().size()
              << std::endl;  // << ", " << ((meshUsage + iasSize) / MB) << "MB, " << (meshUsage /
                             // MB) << "MB, " << chunk->getInstanceXforms().size() << " unique
                             // meshes" << std::endl;
  };

  // Create buffers
  std::cout << "Root chunk: " << rootChunk->getInstanceCount() << std::endl;
  glow::memory::DevicePtr<char> d_temp_storage(
      1024 +
          rootChunk
              ->getInstanceCount() /*can probably get away with lower than this much temp storage*/
      ,
      stream);
  glow::memory::DevicePtr<int> d_num_selected_out(sizeof(int), stream);
  glow::memory::DevicePtr<OptixAabb> d_meshAabbs(sizeof(OptixAabb) * m_meshAabbs.size(), stream);
  {
    std::vector<OptixAabb> meshAabbVec(m_meshAabbs.size());
    for (const auto &entry : m_meshAabbs) {
      meshAabbVec[entry.first] = entry.second;
    }
    d_meshAabbs.write(meshAabbVec.data());
  }

  {
    chunks.push_back(std::make_shared<Chunk>(rootChunk->aabb, rootChunk->getInstanceCount()));
    auto &instances = chunks.back()->getInstances();
    // CUDA_CHECK(cudaMemAdvise(instances.rawPtr(), sizeof(instances[0]) *
    // rootChunk->getInstanceCount(), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    for (const auto &instanceList : rootChunk->getInstanceXforms()) {
      const auto meshId = instanceList.first;
      for (const auto &xform : instanceList.second) {
        instances.push_back_maybe(Instance{meshId, xform});
      }
    }
    prefetchInstancesAsync(instances, stream);
    memadviseInstancesReadonly(instances);
  }

  if (rootChunk->getInstanceCount() <= chunkInstanceCountThreshold) {
    std::cout << "Early return to process root chunk\n";
    std::cout << "Chunk count: " << chunks.size() << std::endl;
    completeChunk(chunks[0]);
    return;
  }

  while (chunks.size() > 0) {
    const auto chunkCount = chunks.size();
    for (size_t __chunkIndex = 0; __chunkIndex < chunkCount;
         __chunkIndex++) {  // Don't use __chunkIndex. Since chunks are removed from start of list
                            // when subchunks are calculated, the front is always the current chunk
      const auto localChunkPtr = chunks.front();
      chunks.erase(chunks.begin());  // pop front
      const auto &localChunk = *localChunkPtr;

      std::vector<OptixAabb> subChunkAabbs;
      getSubChunkAABBs(subChunkAabbs, localChunk.getAabb());

      // if (needsSubChunkEvalPass()) {
      //   std::vector<SubChunkStats> subChunkStats(subChunkAabbs.size());
      //   std::vector<std::unordered_set<int>> meshIndicesUsed(subChunkAabbs.size());
      //   std::vector<std::mutex> statsMutexes(subChunkAabbs.size());
      //   forEachInstance([&](const size_t subChunkIndex, const int instanceMeshIndex, const int
      //   instanceAffineXformIndex, const OptixAabb &) {
      //     std::lock_guard guard(statsMutexes[subChunkIndex]);
      //     auto &stats = subChunkStats[subChunkIndex];
      //     stats.instanceCount++;
      //     meshIndicesUsed[subChunkIndex].insert(instanceMeshIndex);
      //   });
      //   for (size_t i = 0; i < subChunkStats.size(); i++) {
      //     auto &stats = subChunkStats[i];
      //     stats.memoryUsage = estimateIasSize(stats.instanceCount);
      //     for (const auto meshIndex : meshIndicesUsed[i]) {
      //       stats.memoryUsage += meshMemoryUsages[meshIndex];
      //     }
      //   }
      //   selectSubChunks(subChunkAabbs, subChunkStats);
      // }

      std::vector<std::shared_ptr<Chunk>> subChunks;
      subChunks.reserve(subChunkAabbs.size());
      for (const auto &subAabb : subChunkAabbs) {
        subChunks.push_back(std::make_shared<Chunk>(subAabb, localChunk.getInstances().size()));
      }

      for (auto &subChunk : subChunks) {
        prefetchInstancesAsync(subChunk->getInstances(), stream);
        selectAllInstancesForSubchunk_cuda(subChunk,
                                           localChunk.getInstances(),
                                           d_meshAabbs,
                                           d_num_selected_out,
                                           d_temp_storage,
                                           stream);
        memadviseInstancesReadonly(subChunk->getInstances());
      }

      for (auto &subChunk : subChunks) {
        size_t instanceCount = subChunk->getInstances().size();
        std::cout << "Subchunk: " << instanceCount << std::endl;
        if (instanceCount <= chunkInstanceCountThreshold) {
          completeChunk(subChunk);
        }
        else {
          chunks.push_back(subChunk);
        }
      }
    }

    std::cout << "Chunks[][]: " << chunkCount << " --> " << chunks.size() << std::endl;
  }
}

int InstancePartitioner::writeChunks(
    std::function<void(const std::shared_ptr<glow::pipeline::render::Chunk>)> callback)
{
  // std::vector<glm::mat4> islandXforms = {
  //     glm::mat4(-0.0559, 0.0000, 0.2456, -565.2681, 0.2456, 0.0000, 0.0559, -17283.2910, 0.0000,
  //     0.2519, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(-0.0425, 0.0000,
  //     0.0793, -7905.2397, 0.0793, 0.0000, 0.0425, 34780.8242, 0.0000, 0.0899, -0.0000, 0.0000,
  //     0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.0047, 0.0000, 0.1088, -4806.4370, 0.1088,
  //     -0.0000, -0.0047, 34883.0898, 0.0000, 0.1089, -0.0000, 0.0000, 0.0000, 0.0000,
  //     0.0000, 1.0000), glm::mat4(0.0379, 0.0000, 0.0927, -2147.5254, 0.0927, -0.0000, -0.0379,
  //     34167.2305, 0.0000, 0.1001, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000),
  //     glm::mat4(0.0813, 0.0000, 0.0385, -715.8036, 0.0385, -0.0000, -0.0813, 32428.7129, 0.0000,
  //     0.0899, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.0089, 0.0000,
  //     0.0540, -2147.5251, 0.0540, -0.0000, -0.0089, 14225.3955, 0.0000, 0.0548, -0.0000, 0.0000,
  //     0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.6914, 0.0000, 0.3275, 30032.6738, 0.3275,
  //     -0.0000, -0.6914, 77597.1562, 0.0000, 0.7651, -0.0000, 0.0000, 0.0000, 0.0000,
  //     0.0000, 1.0000), glm::mat4(-0.0059, 0.0000, 0.2966, 12219.7656, 0.2966, 0.0000, 0.0059,
  //     93925.6562, 0.0000, 0.2966, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000),
  //     glm::mat4(0.1213, 0.0000, 0.1414, 16036.8164, 0.1414, -0.0000, -0.1213, 80778.0391,
  //     0.0000, 0.1863, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(-0.0402,
  //     0.0000, 0.1819, -21497.5332, 0.1819, 0.0000, 0.0402, 65085.7109, 0.0000, 0.1863, -0.0000,
  //     0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(-0.0331, 0.0000, 0.1496, -28495.4629,
  //     0.1496, 0.0000, 0.0331, 146304.1094, 0.0000, 0.1532, -0.0000, 0.0000, 0.0000, 0.0000,
  //     0.0000, 1.0000), glm::mat4(-0.0230, 0.0000, 0.1043, -16196.0723, 0.1043, 0.0000, 0.0230,
  //     159027.6094, 0.0000, 0.1068, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000),
  //     glm::mat4(0.0059, 0.0000, 0.0358, 7395.1045, 0.0358, -0.0000, -0.0059, 11680.6943, 0.0000,
  //     0.0363, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(-0.0159, 0.0000,
  //     0.0327, 6348.6191, 0.0327, 0.0000, 0.0159, 11989.8828, 0.0000, 0.0363, -0.0000, 0.0000,
  //     0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(-0.0295, 0.0000, 0.0211, 5278.3501, 0.0211,
  //     0.0000, 0.0295, 11371.5059, 0.0000, 0.0363, -0.0000, 0.0000, 0.0000, 0.0000,
  //     0.0000, 1.0000), glm::mat4(0.0152, 0.0000, 0.0239, 8871.9805, 0.0239, -0.0000, -0.0152,
  //     11626.8467, 0.0000, 0.0283, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000),
  //     glm::mat4(0.0276, 0.0000, 0.0125, 9784.3242, 0.0125, -0.0000, -0.0276, 11214.3418, 0.0000,
  //     0.0303, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.0282, -0.0000,
  //     -0.0109, 10115.3945, -0.0109, -0.0000, -0.0282, 10085.8506, 0.0000, 0.0302, -0.0000,
  //     0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.0561, -0.0000, -0.0216, 10974.0840,
  //     -0.0216, -0.0000, -0.0561, 7277.0840, 0.0000, 0.0601, -0.0000, 0.0000, 0.0000, 0.0000,
  //     0.0000, 1.0000), glm::mat4(0.0339, -0.0000, -0.0130, 11854.6543, -0.0130, -0.0000,
  //     -0.0339, 5125.8906, 0.0000, 0.0363, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000),
  //     glm::mat4(0.0607, 0.0000, 0.0622, 8307.2598, 0.0622, -0.0000, -0.0607, -1497.7896, 0.0000,
  //     0.0869, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.0525, 0.0000,
  //     0.0132, 12622.5850, 0.0132, -0.0000, -0.0525, -11032.6035, 0.0000, 0.0542, -0.0000,
  //     0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.0411, 0.0000, 0.0353, 23431.4492,
  //     0.0353, -0.0000, -0.0411, 13215.4180, 0.0000, 0.0542, -0.0000, 0.0000, 0.0000, 0.0000,
  //     0.0000, 1.0000), glm::mat4(0.0411, 0.0000, 0.0353, -21817.8281, 0.0353, -0.0000, -0.0411,
  //     126071.4688, 0.0000, 0.0542, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000),
  //     glm::mat4(0.0411, 0.0000, 0.0353, -9603.2598, 0.0353, -0.0000, -0.0411, 159407.0625,
  //     0.0000, 0.0542, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000), glm::mat4(0.0411,
  //     0.0000, 0.0353, -12147.9609, 0.0353, -0.0000, -0.0411, 124799.1172, 0.0000, 0.0542,
  //     -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000)};

  // for (auto &xform : islandXforms) {
  //   xform = glm::mat4(1, 0, 0, 0,
  //                     0, 0, 1, 0,
  //                     0, 1, 0, 0,
  //                     0, 0, 0, 1) *
  //           glm::transpose(xform);
  // }

  std::unordered_set<OptixAabb, OptixAabbHash, OptixAabbEq> usedChunkAabbs;
  for (const auto &meshChunk : meshChunks) {
    if (usedChunkAabbs.find(meshChunk->aabb) !=
        usedChunkAabbs.end()) {  // TODO FIXME hack to prevent duplicate chunks
      continue;
    }
    usedChunkAabbs.insert(meshChunk->aabb);
    // for (size_t i = 0; i < islandXforms.size(); i++) {
    //   const auto X = islandXforms[i];
    //   const auto lower = X * glm::vec4(meshChunk->aabb.minX, meshChunk->aabb.minY,
    //   meshChunk->aabb.minZ, 1); const auto upper = X * glm::vec4(meshChunk->aabb.maxX,
    //   meshChunk->aabb.maxY, meshChunk->aabb.maxZ, 1); OptixAabb aabb = {lower.x, lower.y,
    //   lower.z, upper.x, upper.y, upper.z}; auto chunk =
    //   std::make_shared<glow::pipeline::render::Chunk>(aabb, X * meshChunk->getXform()); for
    //   (const auto &entry : meshChunk->getInstanceXforms()) {
    //     for (const auto &xform : entry.second) {
    //       chunk->addInstance(entry.first, xform.toMat());
    //     }
    //   }

    //   callback(chunk);
    //   chunkCount++;
    // }
    callback(meshChunk);
    chunkCount++;
  }
  meshChunks.clear();

  // const int ISLAND_COUNT = islandXforms.size();
  // const OptixAabb aabb = {1e20, 1e20, 1e20, -1e20, -1e20, -1e20};
  // auto chunk = std::make_shared<glow::pipeline::render::Chunk>(aabb, glm::mat4(1));
  // for (size_t i = 0; i < ISLAND_COUNT; i++) {
  //   const auto X = islandXforms[i];
  //   for (const auto &entry : rootChunk->getInstanceXforms()) {
  //     for (const auto &xform : entry.second) {
  //       chunk->addInstance(entry.first, X * xform.toMat());
  //     }
  //   }

  //   const auto lower = X * glm::vec4(rootChunk->aabb.minX, rootChunk->aabb.minY,
  //   rootChunk->aabb.minZ, 1); const auto upper = X * glm::vec4(rootChunk->aabb.maxX,
  //   rootChunk->aabb.maxY, rootChunk->aabb.maxZ, 1); chunk->aabb.minX =
  //   std::min(chunk->aabb.minX, lower.x); chunk->aabb.minY = std::min(chunk->aabb.minY, lower.y);
  //   chunk->aabb.minZ = std::min(chunk->aabb.minZ, lower.z);
  //   chunk->aabb.maxX = std::max(chunk->aabb.maxX, upper.x);
  //   chunk->aabb.maxY = std::max(chunk->aabb.maxY, upper.y);
  //   chunk->aabb.maxZ = std::max(chunk->aabb.maxZ, upper.z);
  // }

  // std::cout << chunk->aabb.minX << ", " << chunk->aabb.minY << ", " << chunk->aabb.minZ << ", "
  // << chunk->aabb.maxX << ", " << chunk->aabb.maxY << ", " << chunk->aabb.maxZ << std::endl;

  // rootChunk->clear();

  subdivideChunk(rootChunk, 0, callback);

  const auto tmp = chunkCount;
  chunkCount = 0;
  return tmp;
}

void OctreePartitioner::getSubChunkAABBs(std::vector<OptixAabb> &subChunkAabbs,
                                         const OptixAabb &chunkAabb)
{
  const glm::vec3 parentChunkMin(chunkAabb.minX, chunkAabb.minY, chunkAabb.minZ);
  const glm::vec3 parentChunkMax(chunkAabb.maxX, chunkAabb.maxY, chunkAabb.maxZ);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        const auto lower = parentChunkMin +
                           glm::vec3(i, j, k) * (parentChunkMax - parentChunkMin) / 2.f;
        const auto upper = parentChunkMin +
                           (glm::vec3(i, j, k) + 1.f) * (parentChunkMax - parentChunkMin) / 2.f;
        OptixAabb subAabb = {lower.x, lower.y, lower.z, upper.x, upper.y, upper.z};
        subChunkAabbs.push_back(subAabb);
      }
    }
  }
}

void KdTreePartitioner::getSubChunkAABBs(std::vector<OptixAabb> &subChunkAabbs,
                                         const OptixAabb &chunkAabb)
{
  constexpr int NUMBER_OF_CUTS = 7;

  // X-axis
  for (size_t i = 0; i < NUMBER_OF_CUTS; i++) {
    const auto splitFraction = static_cast<float>(i + 1) / static_cast<float>(NUMBER_OF_CUTS + 1);
    const auto splitValue = chunkAabb.minX + splitFraction * (chunkAabb.maxX - chunkAabb.minX);
    subChunkAabbs.push_back(OptixAabb{
        chunkAabb.minX,
        chunkAabb.minY,
        chunkAabb.minZ,
        splitValue,
        chunkAabb.maxY,
        chunkAabb.maxZ,
    });
    subChunkAabbs.push_back(OptixAabb{
        splitValue,
        chunkAabb.minY,
        chunkAabb.minZ,
        chunkAabb.maxX,
        chunkAabb.maxY,
        chunkAabb.maxZ,
    });
  }

  // Y-axis
  for (size_t i = 0; i < NUMBER_OF_CUTS; i++) {
    const auto splitFraction = static_cast<float>(i + 1) / static_cast<float>(NUMBER_OF_CUTS + 1);
    const auto splitValue = chunkAabb.minY + splitFraction * (chunkAabb.maxY - chunkAabb.minY);
    subChunkAabbs.push_back(OptixAabb{
        chunkAabb.minX,
        chunkAabb.minY,
        chunkAabb.minZ,
        chunkAabb.maxX,
        splitValue,
        chunkAabb.maxZ,
    });
    subChunkAabbs.push_back(OptixAabb{
        chunkAabb.minX,
        splitValue,
        chunkAabb.minZ,
        chunkAabb.maxX,
        chunkAabb.maxY,
        chunkAabb.maxZ,
    });
  }

  // Z - axis
  for (size_t i = 0; i < NUMBER_OF_CUTS; i++) {
    const auto splitFraction = static_cast<float>(i + 1) / static_cast<float>(NUMBER_OF_CUTS + 1);
    const auto splitValue = chunkAabb.minZ + splitFraction * (chunkAabb.maxZ - chunkAabb.minZ);
    subChunkAabbs.push_back(OptixAabb{
        chunkAabb.minX,
        chunkAabb.minY,
        chunkAabb.minZ,
        chunkAabb.maxX,
        chunkAabb.maxY,
        splitValue,
    });
    subChunkAabbs.push_back(OptixAabb{
        chunkAabb.minX,
        chunkAabb.minY,
        splitValue,
        chunkAabb.maxX,
        chunkAabb.maxY,
        chunkAabb.maxZ,
    });
  }
}

void KdTreePartitioner::selectSubChunks(std::vector<OptixAabb> &subChunkAabbs,
                                        const std::vector<SubChunkStats> &subChunkStats)
{
  // select best splits based on stats (equal memory usage)
  using SubChunkWithStats = std::pair<OptixAabb, SubChunkStats>;
  using Partition = std::pair<SubChunkWithStats, SubChunkWithStats>;
  std::vector<Partition> partitions;
  partitions.reserve(subChunkAabbs.size() / 2);
  for (size_t i = 0; i < subChunkAabbs.size(); i += 2) {
    const SubChunkWithStats subChunk1 = std::make_pair(subChunkAabbs[i], subChunkStats[i]);
    const SubChunkWithStats subChunk2 = std::make_pair(subChunkAabbs[i + 1], subChunkStats[i + 1]);
    partitions.push_back({subChunk1, subChunk2});
  }

  static const auto scoreFunction = [](const Partition &partition) {
    return std::abs(static_cast<int64_t>(partition.first.second.memoryUsage) -
                    static_cast<int64_t>(partition.second.second.memoryUsage));
  };

  std::partial_sort(partitions.begin(),
                    partitions.begin() + 1,
                    partitions.end(),
                    [](const Partition &a, const Partition &b) {
                      // return scoreFunction(a) < scoreFunction(b);
                      return scoreFunction(a) < scoreFunction(b);
                    });

  // for (int i = 0; i < subChunkAabbs.size(); i += 2) {
  //   std::cout << "Subchunks[" << i << ":" << (i + 1) << "]: " << subChunkAabbs.size() << ", " <<
  //   subChunkAabbs[i].minY << ", " << subChunkAabbs[i].maxY << ", " << subChunkAabbs[i + 1].minY
  //   << ", " << subChunkAabbs[i + 1].maxY << std::endl;
  // }

  subChunkAabbs.clear();
  const auto &selectedPartition = partitions[0];
  subChunkAabbs.push_back(selectedPartition.first.first);
  subChunkAabbs.push_back(selectedPartition.second.first);
  // std::cout << "Subchunks: " << subChunkAabbs.size() << ", " << subChunkAabbs[0].minY << ", " <<
  // subChunkAabbs[0].maxY << ", " << subChunkAabbs[1].minY << ", " << subChunkAabbs[1].maxY <<
  // std::endl;
}

}  // namespace glow::pipeline::sceneloader::partition