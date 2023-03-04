#include "AssetCache.hpp"

#include <cuda_check_macros.h>

namespace demandLoadingGeometry {
AssetCache::AssetCache(size_t maxMemoryUsage,
                       int chunkAssetIdStart,
                       int meshAssetIndexStart,
                       std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> *assets,
                       std::shared_ptr<glow::optix::OptixManager> geometryOptixManager,
                       CUcontext cuContext)
    : useCounts(assets->size()),
      maxMemoryUsage(maxMemoryUsage),
      chunkAssetIdStart(chunkAssetIdStart),
      m_meshAssetIndexStart(meshAssetIndexStart),
      assets(assets),
      geometryOptixManager(geometryOptixManager),
      m_assetEntries(assets->size()),
      m_cuContext(cuContext)
{
  memset(useCounts.data(), 0, sizeof(useCounts[0]) * useCounts.size());
}

void AssetCache::startThread()
{
  m_thread = std::thread([this]() {
    cuCtxPushCurrent(m_cuContext);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    while (m_threadActive) {
      AssetLoadRequest request{};
      {
        std::lock_guard guard(m_queuedAssetRequestMutex);
        request = m_queuedAssetRequest;
      }
      if (getAssetIndex(request.assetId) >= getAssets()->size()) {
        continue;
      }

      if (isResident(request.assetId)) {
        continue;  // Requested asset is already loaded, no need to update top level AS
      }

      const auto res = getAsset(
          request.assetId, request.minRayDifferential, request.assetPriorities, stream);
      if (res.has_error()) {
        std::cerr << "Error in AssetCache thread: " << res.error()->getMessage() << std::endl;
        std::cerr << "Exiting.\n";
        std::exit(1);
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      updateTopLevelAS(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    cuCtxPopCurrent(NULL);
  });
}

void AssetCache::stopThread()
{
  m_threadActive = false;
  m_thread.join();
}

void AssetCache::queueAsset(float minRayDifferential,
                            const std::vector<std::pair<int, double>> &assetPriorities)
{
  for (const auto &assetPriority : assetPriorities) {
    const auto assetId = assetPriority.first;
    if (!isResident(assetId)) {
      queueAsset(assetId, minRayDifferential, assetPriorities);
      return;
    }
  }
}

result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> AssetCache::getAsset(
    int assetId,
    float minRayDifferential,
    const std::vector<std::pair<int, double>> &assetPriorities,
    cudaStream_t stream)
{
  // SCOPED_NVTX_RANGE_FUNCTION_NAME(); // TODO(jberchtold)
  std::lock_guard guard(mutex);
  return getAssetNoLock(assetId, minRayDifferential, assetPriorities, stream);
}

result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> AssetCache::getAssetNoBuildNoUseCount(
    int assetId, cudaStream_t stream)
{
  auto asset = (*assets)[getAssetIndex(assetId)];
  if (asset->isResident()) {
    return asset;
  }
  return nullptr;
}

void AssetCache::clearChunkAssetUsages(cudaStream_t stream)
{
  // const auto mr =
  // (rmm::mr::statistics_resource_adaptor<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>*)
  // rmm::mr::get_current_device_resource(); const auto used = mr->get_bytes_counter(); std::cout
  // << "Memory usage: " << (memoryUsage/(1024*1024)) << ", " <<
  // (glow::memory::getMemoryUsage()/(1024*1024)) << ", " << (used.value/(1024*1024)) << std::endl;
  for (size_t i = 0; i < useCounts.size(); i++) {
    if ((*assets)[i]->getDependencies().size() > 0) {
      useCounts[i] = 0;
    }
  }
}

bool AssetCache::isResident(int assetId)
{
  if (assetId == -1) {
    return true;  // Currently top-level IAS is always resident
  }
  else if (assetId < chunkAssetIdStart) {
    return true;  // Mark textures as resident for now
  }
  return (*assets)[getAssetIndex(assetId)]->isResident();
}

const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> *AssetCache::getAssets() const
{
  return assets;
}

int AssetCache::getAssetIndex(int assetId)
{
  return assetId /* - chunkAssetIdStart*/;
}

result<size_t, Err> AssetCache::getSize(int assetId)
{
  const auto assetIndex = getAssetIndex(assetId);
  if (assetIndex < 0) {
    return 0;  // For top-level chunk AABB IAS and materials
  }
  return (*assets)[assetIndex]->getMemoryFootprintEstimate(geometryOptixManager.get());
}

int AssetCache::getResidentAssetCount()
{
  return residentAssetCount;
}

void AssetCache::processAssetEntriesThreadSafe(
    std::function<void(const std::vector<OptixTraversableHandle> &)> callback)
{
  std::lock_guard guard(m_assetEntriesMutex);
  callback(m_assetEntries);
}

std::mutex &AssetCache::getFreeingAssetsMutex()
{
  return m_freeingAssetsMutex;
}

void AssetCache::queueAsset(int assetId,
                            float minRayDifferential,
                            const std::vector<std::pair<int, double>> &assetPriorities)
{
  {
    std::lock_guard guard(m_queuedAssetRequestMutex);
    m_queuedAssetRequest = {
        assetId,
        minRayDifferential,
        assetPriorities};  // TODO(jberchtold) this will copy asset priorities. Which is required
                           // because the reference will be modified in the main thread. But is
                           // still a potential source of perf degradation (although it's likely
                           // negligible given the other work happening)
  }
  if (!m_threadActive) {
    m_threadActive = true;
    startThread();
  }
}

result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> AssetCache::getAssetNoLock(
    int assetId,
    float minRayDifferential,
    const std::vector<std::pair<int, double>> &assetPriorities,
    cudaStream_t stream)
{
  {
    std::cout << "Assets: " << assets->size() << ", " << assetId << std::endl;

    const auto asset = (*assets)[getAssetIndex(assetId)];
    if (asset->isResident()) {
      std::cout << "Asset already resident\n" << std::endl;
      useCounts[getAssetIndex(assetId)]++;
      return asset;
    }
  }

  useCounts[getAssetIndex(assetId)]++;

  std::unordered_map<int, std::shared_ptr<glow::pipeline::render::IAsset>> assetDependencies;
  for (const auto dependencyIndex : (*assets)[getAssetIndex(assetId)]->getDependencies()) {
    // std::cout << "A: " << dependencyIndex << std::endl;
    // std::cout << "Mem: " << (memoryUsage / (1024 * 1024)) << std::endl;
    UNWRAP(dependencyAsset,
           getAssetNoLock(
               dependencyIndex + chunkAssetIdStart, minRayDifferential, assetPriorities, stream));
    if (dependencyAsset == nullptr) {
      // Revert useCount changes
      for (const auto dependencyIndex2 : (*assets)[getAssetIndex(assetId)]->getDependencies()) {
        useCounts[dependencyIndex2]--;
        if (useCounts[dependencyIndex2] <= 0) {
          std::lock_guard guard(getFreeingAssetsMutex());
          UNWRAP_VOID(freeAssetAndUnusedDependencies(dependencyIndex2));
        }
        if (dependencyIndex2 == dependencyIndex) {
          break;
        }
      }
      useCounts[getAssetIndex(assetId)]--;
      return nullptr;
    }
    assetDependencies[dependencyIndex] = dependencyAsset;
  }

  size_t assetMemoryUsageEstimate = 0;
  {
    const auto asset = (*assets)[getAssetIndex(assetId)];
    // Update memory usage, and busy loop free if necessary
    UNWRAP_ASSIGN(assetMemoryUsageEstimate,
                  asset->getMemoryFootprintEstimate((void *)geometryOptixManager.get()));
    // glow::log::info(LogChannel::GEOMETRY_ASSET_CACHE, "Asset memory usage [{}]: {} MB", assetId,
    // (assetMemoryUsageEstimate / (1024 * 1024)));
    if (assetMemoryUsageEstimate + memoryUsage > maxMemoryUsage) {
      // std::cout << (assetMemoryUsageEstimate + memoryUsage) << ", " << maxMemoryUsage <<
      // std::endl;
      UNWRAP(success,
             tryFreeUpSpace(assetMemoryUsageEstimate - (maxMemoryUsage - memoryUsage),
                            getAssetIndex(assetId),
                            assetPriorities));
      // std::cout << (assetMemoryUsageEstimate + memoryUsage) << ", " << maxMemoryUsage <<
      // std::endl;
      if (!success) {
        // std::cout << getAssetIndex(assetId) << ": " << (assetMemoryUsageEstimate / (1024 *
        // 1024)) << "MB, " << (memoryUsage / (1024 * 1024)) << "MB, " << (maxMemoryUsage / (1024 *
        // 1024)) << "MB" << std::endl; for (int i = 0; i < useCounts.size(); i++) {
        //   if ((*assets)[i]->isResident()) {
        //     std::cout << i << ", 1, " << useCounts[i] << std::endl;
        //   }
        // }
        return nullptr;
      }
    }
    memoryUsage += assetMemoryUsageEstimate;
  }

  const auto assetRes = createAsset(assetId,
                                    assetMemoryUsageEstimate,
                                    assetDependencies,
                                    assetPriorities,
                                    minRayDifferential,
                                    stream);
  if (assetRes.has_error()) {
    memoryUsage -= assetMemoryUsageEstimate;
    // Revert useCount changes
    for (const auto dependencyIndex : (*assets)[getAssetIndex(assetId)]->getDependencies()) {
      useCounts[dependencyIndex]--;
      if (useCounts[dependencyIndex] <= 0) {
        std::lock_guard guard(getFreeingAssetsMutex());
        UNWRAP_VOID(freeAssetAndUnusedDependencies(dependencyIndex));
      }
    }
    useCounts[getAssetIndex(assetId)]--;
    return nullptr;
  }
  return assetRes.value();
}

result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> AssetCache::createAsset(
    int assetId,
    size_t originalAssetMemoryUsageEstimate,
    const std::unordered_map<int, std::shared_ptr<glow::pipeline::render::IAsset>>
        &assetDependencies,
    const std::vector<std::pair<int, double>> &assetPriorities,
    float minRayDifferential,
    cudaStream_t stream)
{
  const auto asset = (*assets)[getAssetIndex(assetId)];
  UNWRAP_VOID(
      asset->build(geometryOptixManager.get(), assetDependencies, minRayDifferential, stream));
  // glow::log::info(LogChannel::GEOMETRY_ASSET_CACHE, "Starting asset build: {}", assetId);
  // const auto build = [&]() {
  //   return asset->build(geometryOptixManager.get(), assetDependencies, minRayDifferential,
  //   stream);
  // };

  // auto res = build();
  // int i = assetPriorities.size() - 1;
  // while (res.has_error()) {
  //   while (true) {
  //     if (i < 0) {
  //       return failure(res.error());
  //     }
  //     const auto assetId = assetPriorities[i].first;
  //     const auto assetIndex = getAssetIndex(assetId);
  //     if (assetIndex < 0) {
  //       i--;
  //       continue;
  //     }

  //     const auto asset = (*assets)[assetIndex];
  //     i--;
  //     std::cout << i << std::endl;
  //     if (asset->isResident()) {
  //       std::cout << "Freed additional asset\n";
  //        std::lock_guard guard(getFreeingAssetsMutex());
  //       UNWRAP(size, freeAssetAndUnusedDependencies(assetIndex));
  //       std::cout << "Size: " << size << std::endl;
  //       break;
  //     }
  //   }
  //   res = build();
  // }
  residentAssetCount++;
  {
    UNWRAP(assetMemoryUsage,
           asset->getMemoryFootprintEstimate((void *)geometryOptixManager.get()));
    memoryUsage -= originalAssetMemoryUsageEstimate;
    memoryUsage += assetMemoryUsage;
  }

  {
    std::lock_guard guard(m_assetEntriesMutex);
    m_assetEntries[getAssetIndex(assetId)] = asset->getAS();
  }

  // glow::log::info(LogChannel::GEOMETRY_ASSET_CACHE, "New asset: {}", memoryUsage);
  return asset;
}

result<size_t, Err> AssetCache::freeAssetAndUnusedDependencies(int assetIndex)
{
  const auto i = assetIndex;
  size_t sizeFreed = 0;
  const auto referenceCount = useCounts[i];
  if ((*assets)[i]->isResident()) {
    // std::cout << "Tried to free: " << assetIndex << ", " << referenceCount << std::endl;
  }
  if (referenceCount == 0 && (*assets)[i]->isResident()) {
    {
      std::lock_guard guard(m_assetEntriesMutex);
      m_assetEntries[assetIndex] = NULL_TRAVERSABLE_HANDLE;
    }

    const auto asset = (*assets)[assetIndex];
    UNWRAP(size, asset->getMemoryFootprintEstimate((void *)geometryOptixManager.get()));
    asset->free();
    memoryUsage -= size;
    sizeFreed += size;
    residentAssetCount--;

    for (const auto dependency : asset->getDependencies()) {
      // std::cout << "Free: " << dependency << std::endl;
      useCounts[dependency]--;
      UNWRAP(size, freeAssetAndUnusedDependencies(dependency));
      sizeFreed += size;
    }
  }

  return sizeFreed;
}

result<bool, Err> AssetCache::tryFreeUpSpace(
    size_t requiredSpace,
    int requiredAssetIndex,
    const std::vector<std::pair<int, double>> &assetPriorities)
{
  size_t sizeFreed = 0;
  auto assetPriorityIter = assetPriorities.rbegin();
  while (sizeFreed < requiredSpace) {
    if (assetPriorityIter == assetPriorities.rend()) {
      return false;
    }
    const auto assetIndex = getAssetIndex(assetPriorityIter->first);
    const auto i = assetIndex;
    if (i < 0) {  // Nothing to free, top-level chunk AABB
      assetPriorityIter++;
      continue;
    }

    if (assetIndex == requiredAssetIndex) {  // Early return and don't try to free a higher
                                             // priority asset to load this one
      return false;
    }

    {
      std::lock_guard guard(getFreeingAssetsMutex());
      UNWRAP(size, freeAssetAndUnusedDependencies(i));
      sizeFreed += size;
      assetPriorityIter++;
    }
  }
  return true;
}

void AssetCache::updateTopLevelAS(cudaStream_t stream)
{
  std::lock_guard guard(m_freeingAssetsMutex);
  geometryOptixManager->updateTopLevelAS(
      std::span(m_assetEntries.begin(), m_assetEntries.begin() + m_meshAssetIndexStart), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace demandLoadingGeometry