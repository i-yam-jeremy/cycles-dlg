#pragma once

#include <memory>

#include "asset.h"
#include "optixmanager.h"
#include <DemandLoadingGeometry.hpp>
#include <atomic>
#include <thread>
#include <util/monad/error.h>
#include <util/monad/result.h>

namespace demandLoadingGeometry {
class AssetCache {
public:
  AssetCache(size_t maxMemoryUsage,
             int chunkAssetIdStart,
             int meshAssetIndexStart,
             std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> *assets,
             std::shared_ptr<glow::optix::OptixManager> geometryOptixManager);

  void startThread();
  void stopThread();

  void queueAsset(float minRayDifferential, const std::vector<std::pair<int, double>> &assetPriorities);

  result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> getAsset(int assetId, float minRayDifferential, const std::vector<std::pair<int, double>> &assetPriorities, cudaStream_t stream);

  result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> getAssetNoBuildNoUseCount(int assetId, cudaStream_t stream);

  void clearChunkAssetUsages(cudaStream_t stream);

  bool isResident(int assetId);

  const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> *getAssets() const;
  int getAssetIndex(int assetId);

  result<size_t, Err> getSize(int assetId);
  int getResidentAssetCount();

  void processAssetEntriesThreadSafe(std::function<void(const std::vector<OptixTraversableHandle> &)> callback);
  std::mutex &getFreeingAssetsMutex();

private:
  std::shared_ptr<glow::optix::OptixManager> geometryOptixManager;
  std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> *assets = nullptr;
  const int chunkAssetIdStart = 0;
  const int m_meshAssetIndexStart = 0;

  struct AssetLoadRequest {
    int assetId;
    float minRayDifferential;
    std::vector<std::pair<int, double>> assetPriorities;
  };

  std::mutex mutex;
  size_t memoryUsage = 0;
  std::vector<size_t> useCounts;
  const size_t maxMemoryUsage;
  int residentAssetCount = 0;
  std::vector<OptixTraversableHandle> m_assetEntries;
  std::thread m_thread;
  std::atomic<bool> m_threadActive = false;
  std::mutex m_assetEntriesMutex;
  AssetLoadRequest m_queuedAssetRequest;
  std::mutex m_queuedAssetRequestMutex;

  std::mutex m_freeingAssetsMutex; // Mutex for freeing assets (locked so asset builder thread can't free assets when main trace thread is running and using them)

  void queueAsset(int assetId, float minRayDifferential, const std::vector<std::pair<int, double>> &assetPriorities);

  void updateTopLevelAS(cudaStream_t stream);

  result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> getAssetNoLock(int assetId, float minRayDifferential, const std::vector<std::pair<int, double>> &assetPriorities, cudaStream_t stream);
  result<std::shared_ptr<glow::pipeline::render::IAsset>, Err> createAsset(int assetId, size_t originalAssetMemoryUsageEstimate, const std::unordered_map<int, std::shared_ptr<glow::pipeline::render::IAsset>> &assetDependencies, const std::vector<std::pair<int, double>> &assetPriorities, float minRayDifferential, cudaStream_t stream);
  result<size_t, Err> freeAssetAndUnusedDependencies(int assetIndex);
  result<bool, Err> tryFreeUpSpace(size_t requiredSpace, int requiredAssetIndex, const std::vector<std::pair<int, double>> &assetPriorities);
};
} // namespace demandLoadingGeometry