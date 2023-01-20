#pragma once

#include "chunk_data.h"
#include <cuda_runtime.h>
#include <device_memory_wrapper.h>
#include <memory>

namespace glow::pipeline::sceneloader::partition {
void selectAllInstancesForSubchunk_cuda(std::shared_ptr<Chunk> subChunk, const InstanceList &parentInstances, const glow::memory::DevicePtr<OptixAabb> &meshAabbs, glow::memory::DevicePtr<int> &d_num_selected_out, glow::memory::DevicePtr<char> &d_temp_storage_ptr, cudaStream_t stream);
} // namespace glow::pipeline::sceneloader::partition