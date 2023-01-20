#include "device_memory_wrapper.h"

#include <mutex>
#include <nvtx3/nvToolsExt.h>
#include <rmm/mr/device/per_device_resource.hpp>

static std::mutex memoryUsageMutex;
static size_t memoryUsage = 0;

void *glow::memory::alloc(size_t size, cudaStream_t stream) {
  if (size == 0) {
    return nullptr;
  }
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
  if (mr == nullptr) {
    std::cerr << "MR null\n";
    std::exit(1);
  }
  try {
    nvtxRangePushA("Memory alloc");
    void *ptr = mr->allocate(size, stream);
    if (ptr != nullptr) {
      std::lock_guard guard(memoryUsageMutex);
      memoryUsage += size;
    }
    // if (ptr == nullptr) {
    //   std::cerr << "Allocation Failed\n"
    //             << std::endl;
    //   std::exit(1);
    // }
    nvtxRangePop();
    return ptr;
  } catch (...) {
    // std::cerr << "Alloc failed: " << size << " bytes. " << e.what() << std::endl;
    // while (1) {
    // }
    return nullptr;
  }
}

void glow::memory::dealloc(void *ptr, size_t size, cudaStream_t stream) {
  nvtxRangePushA("Memory free");
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
  mr->deallocate(ptr, size, stream);
  {
    std::lock_guard guard(memoryUsageMutex);
    memoryUsage -= size;
  }
  nvtxRangePop();
}

size_t glow::memory::getMemoryUsage() {
  std::lock_guard guard(memoryUsageMutex);
  return memoryUsage;
}