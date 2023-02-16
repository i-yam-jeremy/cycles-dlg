#include "DemandLoadingGeometryImpl.hpp"

#include <affinexform.h>
#include <trace/ChunkAsset.hpp>
#include <trace/GeometryAsset.hpp>
#include <vector>

#include <fstream>

namespace demandLoadingGeometry {

GeometryDemandLoaderImpl::GeometryDemandLoaderImpl(
    std::unique_ptr<glow::pipeline::sceneloader::partition::InstancePartitioner>
        instancePartitioner,
    const Options &options,
    CUcontext cuContext,
    OptixDeviceContext optixContext)
    : m_cuContext(cuContext),
      m_instancePartitioner(std::move(instancePartitioner)),
      m_options(options),
      m_optixManager(std::make_shared<glow::optix::OptixManager>(
          std::make_shared<glow::optix::OptixConnector>(), optixContext))
{
  m_deviceContext.sceneTraversableHandle = NULL_TRAVERSABLE_HANDLE;
  m_deviceContext.d_assetRayCountBuffer = nullptr;
  m_deviceContext.d_stalledRayIndices = nullptr;
}

GeometryDemandLoaderImpl::~GeometryDemandLoaderImpl()
{
  // m_assetCache->stopThread();

  // // Write chunk metrics
  // std::ofstream outChunkMetrics("chunkMetrics.csv", std::ofstream::out);
  // for (const auto& chunk : m_chunkAssets) {
  //   const auto aabb = chunk->getAABB();
  //   const auto &metrics = chunk->getMetrics();
  //   // TODO chunk xform
  //   outChunkMetrics
  //       << aabb.minX << "," << aabb.minY << "," << aabb.minZ << "," << aabb.maxX << "," <<
  //       aabb.maxY << "," << aabb.maxZ << "," << metrics.raysTraced << "," <<
  //       metrics.iterationsUsed << "," << metrics.numLoads << "," <<
  //       metrics.raysTracedRequiringLoad << std::endl;
  // }
  // outChunkMetrics.close();
}

std::optional<OptixProgramGroupDesc> GeometryDemandLoaderImpl::getOptixProgramGroup(
    const OptixPipelineCompileOptions &pipeline_compile_options,
    const OptixModuleCompileOptions &module_compile_options)
{
  const auto res = m_optixManager->createProgramGroup(pipeline_compile_options,
                                                      module_compile_options);
  if (res.has_error()) {
    return {};
  }
  return res.value();
}

namespace {
OptixAabb calculateMeshAabb(const Mesh &mesh)
{
  OptixAabb aabb{1e20, 1e20, 1e20, -1e20, -1e20, -1e20};
  for (const auto &buildInput : mesh.buildInputs) {
    for (const auto &point : buildInput->positions) {
      aabb.minX = std::min(aabb.minX, point.x);
      aabb.minY = std::min(aabb.minY, point.y);
      aabb.minZ = std::min(aabb.minZ, point.z);
      aabb.maxX = std::max(aabb.maxX, point.x);
      aabb.maxY = std::max(aabb.maxY, point.y);
      aabb.maxZ = std::max(aabb.maxZ, point.z);
    }
  }
  return aabb;
}
}  // namespace

// Scene Building API
// void reserveSpaceForNewInstances(size_t instanceCount);
MeshHandle GeometryDemandLoaderImpl::addMesh(const Mesh &mesh,
                                             const std::optional<OptixAabb> &aabb)
{
  size_t meshIndex = m_meshes.size();
  size_t memoryUsage = 0;
  for (const auto &buildInput : mesh.buildInputs) {
    memoryUsage += buildInput->indices.size() * sizeof(buildInput->indices[0]);
    memoryUsage += buildInput->positions.size() * sizeof(buildInput->positions[0]);
    memoryUsage += (buildInput->indices.size() * 3) * sizeof(glm::vec3);  // Normals
    memoryUsage += (buildInput->indices.size() * 3) * sizeof(glm::vec2);  // Tex coords
  }
  const OptixAabb meshAabb = aabb.has_value() ? aabb.value() : calculateMeshAabb(mesh);
  m_instancePartitioner->setMeshInfo(meshIndex, meshAabb, memoryUsage);
  m_meshes.push_back(mesh);

  return MeshHandle(meshIndex);
}

void GeometryDemandLoaderImpl::addInstance(MeshHandle meshHandle, const AffineXform &xform)
{
  m_instancePartitioner->add(meshHandle.meshIndex, xform);
}

OptixTraversableHandle GeometryDemandLoaderImpl::updateScene(unsigned int baseDlgSbtOffset)
{
  CUDA_CHECK(cudaDeviceSynchronize());

  for (const auto &d_rayQueue : m_stalledRayQueues) {
    CUDA_CHECK(cudaFree(d_rayQueue));
  }
  CUDA_CHECK(cudaFree(m_deviceContext.d_stalledRayIndices));
  m_stalledRayQueues.clear();

  partition();
  updateAssets();
  m_deviceContext.sceneTraversableHandle = createTopLevelTraversable(baseDlgSbtOffset);
  updateAssetCache();

  m_assetRayCounts = std::make_shared<glow::memory::DevicePtr<internal::RayCount>>(
      sizeof(internal::RayCount) * m_assets.size(), (cudaStream_t)0);
  std::vector<internal::RayCount> assetRayCounts(m_assets.size());
  m_assetRayCounts->write(assetRayCounts.data());  // Write zeroes to initialize ray counts
  m_deviceContext.d_assetRayCountBuffer = m_assetRayCounts->rawPtr();

  for (size_t i = 0; i < m_assets.size(); i++) {
    demandLoadingGeometry::RayIndex *d_rayQueue;
    CUDA_CHECK(cudaMallocManaged(
        &d_rayQueue, m_options.maxActiveRayCount * sizeof(demandLoadingGeometry::RayIndex)));
    m_stalledRayQueues.push_back(d_rayQueue);
  }
  CUDA_CHECK(cudaMalloc(&m_deviceContext.d_stalledRayIndices,
                        m_stalledRayQueues.size() * sizeof(m_stalledRayQueues[0])));
  CUDA_CHECK(cudaMemcpy(m_deviceContext.d_stalledRayIndices,
                        m_stalledRayQueues.data(),
                        m_stalledRayQueues.size() * sizeof(m_stalledRayQueues[0]),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0));

  CUDA_CHECK(cudaDeviceSynchronize());

  return m_deviceContext.sceneTraversableHandle;
}

void GeometryDemandLoaderImpl::partition()
{
  m_chunks.clear();
  m_instancePartitioner->writeChunks(
      [this](const std::shared_ptr<glow::pipeline::render::Chunk> chunk) {
        if (chunk != nullptr && !chunk->isEmpty()) {
          Chunk dlgChunk;
          dlgChunk.aabb = chunk->getAabb();
          dlgChunk.xform = chunk->getXform();
          dlgChunk.instanceLists.resize(chunk->getInstanceXforms().size());
          size_t i = 0;
          for (const auto &entry : chunk->getInstanceXforms()) {
            auto &instanceList = dlgChunk.instanceLists[i++];
            const auto meshId = entry.first;
            instanceList.assetIndex = meshId;
            instanceList.instanceXforms.reserve(entry.second.size());
            for (const auto &xform : entry.second) {
              instanceList.instanceXforms.push_back(AffineXform(xform));
            }
          }

          m_chunks.push_back(dlgChunk);
        }
      });

  const size_t meshAssetIndexStart = m_chunks.size();
  for (auto &chunk : m_chunks) {
    for (auto &instanceList : chunk.instanceLists) {
      instanceList.assetIndex += meshAssetIndexStart;
    }
  }
}

namespace {
struct Empty {};
}  // namespace

std::unique_ptr<SBTBuffer> GeometryDemandLoaderImpl::getInternalApiHitgroupSbtEntries(
    size_t sizeOfUserSbtStruct, uint32_t maxTraceSbtOffset)
{
  auto sbtBuffer = std::make_unique<SBTBuffer>();
  sbtBuffer->numElements = maxTraceSbtOffset;
  const auto sbtEntrySize = sizeof(SBTRecord<Empty>) - sizeof(SBTRecord<Empty>::__chunkSbtData) +
                            std::max(sizeOfUserSbtStruct,
                                     sizeof(SBTRecord<Empty>::__chunkSbtData));
  sbtBuffer->sizeInBytes = sbtEntrySize * sbtBuffer->numElements;
  std::cout << "sbtBuffer->sizeInBytes: " << sbtBuffer->sizeInBytes << std::endl;
  sbtBuffer->data = new char[sbtBuffer->sizeInBytes];

  for (size_t i = 0; i < sbtBuffer->numElements; i++) {
    auto *ptr = reinterpret_cast<SBTRecord<Empty> *>(reinterpret_cast<char *>(sbtBuffer->data) +
                                                     i * sbtEntrySize);
    // Note: packing SBT header (with optix program group) is up to the user of the DLG API
    ptr->__chunkSbtData.context = m_deviceContext;
    printf("d_assetRayCountBuffer2: %ld\n", ptr->__chunkSbtData.context.d_assetRayCountBuffer);
  }

  return std::move(sbtBuffer);
}

demandLoadingGeometry::LaunchData GeometryDemandLoaderImpl::preLaunch(
    demandLoadingGeometry::RayIndex *d_endOfUserRayQueue, cudaStream_t stream)
{
  // Lock mutex (then in second implementation, use cuda event to prevent stream sync on host
  // thread)
  CUDA_CHECK(cudaStreamSynchronize(stream));
  m_assetCache->getFreeingAssetsMutex().lock();

  m_assetCache->clearChunkAssetUsages(stream);

  // 1. Ray asset counts
  std::vector<internal::RayCount> assetCounts(m_assetRayCounts->size() /
                                              sizeof(internal::RayCount));
  m_assetRayCounts->read(assetCounts.data(), stream);
  std::cout << "Assets: " << assetCounts.size() << std::endl;

  // // 3. CPU calculation of asset priorities
  std::vector<bool> previousIterAssetResidencies(m_chunkAssets.size());
  std::vector<std::pair<int, double>> assetPriorities(
      m_chunkAssets
          .size());  // Don't include meshes in priorities since they're never set to ray asset IDs
  {
    static const auto priorityHeuristic =
        [](size_t count, bool resident, size_t assetSizeToLoad) -> double {
      if (resident) {
        return 256 * count;
      }
      else {
        // const auto mb = assetSizeToLoad / (1024 * 1024);
        // return double(count) / (mb*mb);
        return count;
      }
    };

    for (int i = 0; i < assetPriorities.size(); i++) {
      const auto assetId = i;
      const auto res = m_assetCache->getSize(assetId);
      if (res.has_error()) {
        std::cerr << "Error: assetCache->getSize\n";
        std::exit(1);
      }
      const auto size = res.value();
      const auto priority = priorityHeuristic(
          assetCounts[i], m_assetCache->isResident(assetId), size);
      assetPriorities[i] = {assetId, priority};
      previousIterAssetResidencies[i] = m_assetCache->isResident(assetId);
    }
    std::sort(assetPriorities.begin(),
              assetPriorities.end(),
              [](const std::pair<int, double> &a, const std::pair<int, double> &b) -> bool {
                return a.second > b.second; /* '>' to sort in descending order */
              });
  }

  // 4. Request asset to load
  float minRayDifferential = 0.0f;
  m_assetCache->queueAsset(minRayDifferential, assetPriorities);

  // 5. If any new assets have been loaded, copy ray indices of stalled rays to the user's ray
  // queue
  demandLoadingGeometry::RayIndex numRayIndicesAddedToUserRayQueue = 0;
  demandLoadingGeometry::RayIndex numStalledRays = 0;
  int assetsUsed = 0;
  auto prevPriority = assetPriorities[0].second;
  for (size_t assetIndex = 0; assetIndex < m_assetCache->getAssets()->size(); assetIndex++) {
    const auto rayCount = assetCounts[assetIndex];
    if (rayCount == 0) {
      continue;
    }

    if (m_assetCache->isResident(assetIndex)) {
      // Copy stalled rays to user's ray queue
      CUDA_CHECK(cudaMemcpyAsync(d_endOfUserRayQueue,
                                 m_stalledRayQueues[assetIndex],
                                 rayCount * sizeof(*d_endOfUserRayQueue),
                                 cudaMemcpyDeviceToDevice,
                                 stream));
      d_endOfUserRayQueue += rayCount;
      assetCounts[assetIndex] = 0;
      numRayIndicesAddedToUserRayQueue += rayCount;
      assetsUsed++;
    }
    else {
      numStalledRays += rayCount;
    }
  }
  m_assetRayCounts->write(assetCounts.data(),
                          stream);  // Update asset counts for stalled ray queues

  std::cout << "Assets used: " << assetsUsed
            << ", Assets in cache: " << m_assetCache->getResidentAssetCount() << std::endl;

  // //
  // ##########################################################################################################################
  // m_assetCache->processAssetEntriesThreadSafe([&, this](const std::vector<AssetEntry>
  // &assetEntries) {
  //   for (size_t i = 0 /*skip top-level chunk AABB (-1), and material (0)*/; i <
  //   m_chunkAssets.size(); i++) {
  //     if (assetEntries[i].resident) {
  //       const auto wasLoaded = !previousIterAssetResidencies[i];
  //       glow::pipeline::render::AssetMetrics metrics = {
  //           (size_t)assetCounts->counts[i],
  //           1,
  //           (size_t)(wasLoaded ? 1 : 0),
  //           (size_t)(wasLoaded ? assetCounts->counts[i] : 0),
  //       };
  //       m_chunkAssets[i - 2]->addMetrics(metrics);
  //     }
  //   }
  // }); TODO(jberchtold)

  return LaunchData{m_deviceContext, numRayIndicesAddedToUserRayQueue, numStalledRays};
}

void GeometryDemandLoaderImpl::postLaunch(cudaStream_t stream)
{
  // Unlock mutex (then in second implementation, use cuda event to prevent stream sync on host
  // thread)
  CUDA_CHECK(cudaStreamSynchronize(stream));
  m_assetCache->getFreeingAssetsMutex().unlock();
}

void GeometryDemandLoaderImpl::updateAssets()
{
  m_chunkAssets.clear();
  m_assets.clear();

  for (const auto &chunk : m_chunks) {
    auto chunkAsset = std::make_shared<ChunkAsset>(&chunk, m_assets);
    m_chunkAssets.push_back(chunkAsset);
    m_assets.push_back(chunkAsset);
  }

  for (const auto &mesh : m_meshes) {
    auto meshAsset = std::make_shared<GeometryAsset>(&mesh, 0);
    m_assets.push_back(meshAsset);
  }
}

OptixTraversableHandle GeometryDemandLoaderImpl::createTopLevelTraversable(
    unsigned int baseDlgSbtOffset)
{
  const auto res = m_optixManager->createTopLevelTraversable(m_chunkAssets, baseDlgSbtOffset);
  if (res.has_error()) {
    std::cerr << "Error: updatedTopLevelAABBs\n";
    std::exit(1);
  }
  return res.value();
}

void GeometryDemandLoaderImpl::updateAssetCache()
{
  const auto memoryUsedByOtherComponentsLikeAssetIdBuffer = 0;  // TODO(jberchtold);
  const auto maxCacheMemory = m_options.maxMemory - memoryUsedByOtherComponentsLikeAssetIdBuffer;
  m_assetCache = std::make_shared<AssetCache>(
      maxCacheMemory, 0, m_chunks.size(), &m_assets, m_optixManager, m_cuContext);
}

}  // namespace demandLoadingGeometry