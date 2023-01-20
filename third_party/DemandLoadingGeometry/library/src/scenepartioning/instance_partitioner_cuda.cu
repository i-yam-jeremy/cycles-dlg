#include "instance_partitioner_cuda.h"

#include <cub/cub.cuh>

#define CHECK_TEMP_STORAGE(actual_size, temp_buffer)                                                             \
  if (actual_size > temp_buffer.size()) {                                                                        \
    std::cerr << "Temp storage exceeded provided buffer: " << actual_size << ", " << temp_buffer.size() << "\n"; \
    std::cerr << "Looping indefinitely\n";                                                                       \
    while (1) {                                                                                                  \
    }                                                                                                            \
  }

namespace glow::pipeline::sceneloader::partition {

__device__ bool aabbsOverlap(const OptixAabb &a, const OptixAabb &b) {
  const auto x = fmin(a.maxX, b.maxX) - fmax(a.minX, b.minX);
  const auto y = fmin(a.maxY, b.maxY) - fmax(a.minY, b.minY);
  const auto z = fmin(a.maxZ, b.maxZ) - fmax(a.minZ, b.minZ);
  return (x > 0) && (y > 0) && (z > 0);
}

OptixAabb aabbIntersection(const OptixAabb &a, const OptixAabb &b) {
  OptixAabb out{};
  out.minX = fmax(a.minX, b.minX);
  out.minY = fmax(a.minY, b.minY);
  out.minZ = fmax(a.minZ, b.minZ);
  out.maxX = fmin(a.maxX, b.maxX);
  out.maxY = fmin(a.maxY, b.maxY);
  out.maxZ = fmin(a.maxZ, b.maxZ);
  return out;
}

class InstanceInAabb {
public:
  __host__ __device__ __forceinline__
  InstanceInAabb(const OptixAabb &aabb, const OptixAabb *meshAabbs) : m_aabb(aabb), m_meshAabbs(meshAabbs) {}

  __device__ __forceinline__ bool operator()(const Instance &instance) const {
    const auto aabb = m_meshAabbs[instance.meshId];
    const glm::vec3 aabbCorners[] = {
        glm::vec3(aabb.minX, aabb.minY, aabb.minZ),
        glm::vec3(aabb.minX, aabb.minY, aabb.maxZ),
        glm::vec3(aabb.minX, aabb.maxY, aabb.minZ),
        glm::vec3(aabb.minX, aabb.maxY, aabb.maxZ),
        glm::vec3(aabb.maxX, aabb.minY, aabb.minZ),
        glm::vec3(aabb.maxX, aabb.minY, aabb.maxZ),
        glm::vec3(aabb.maxX, aabb.maxY, aabb.minZ),
        glm::vec3(aabb.maxX, aabb.maxY, aabb.maxZ)};

    const auto instanceXform = instance.xform.toMat();
    // TODO optimization: just find chunk indices transformed AABB is in, not all chunks containing re-fit larger world space AABB
    OptixAabb worldSpaceInstanceAabb = {1e20, 1e20, 1e20, -1e20, -1e20, -1e20};
    for (const auto pLocal : aabbCorners) {
      const auto pWorld = glm::vec3(instanceXform * glm::vec4(pLocal, 1));
      worldSpaceInstanceAabb = {
          fmin(worldSpaceInstanceAabb.minX, pWorld.x),
          fmin(worldSpaceInstanceAabb.minY, pWorld.y),
          fmin(worldSpaceInstanceAabb.minZ, pWorld.z),
          fmax(worldSpaceInstanceAabb.maxX, pWorld.x),
          fmax(worldSpaceInstanceAabb.maxY, pWorld.y),
          fmax(worldSpaceInstanceAabb.maxZ, pWorld.z)};
    }

    return aabbsOverlap(m_aabb, worldSpaceInstanceAabb);
  }

private:
  OptixAabb m_aabb;
  const OptixAabb *m_meshAabbs;
};

void selectAllInstancesForSubchunk_cuda(std::shared_ptr<Chunk> subChunk, const InstanceList &parentInstances, const glow::memory::DevicePtr<OptixAabb> &meshAabbs, glow::memory::DevicePtr<int> &d_num_selected_out, glow::memory::DevicePtr<char> &d_temp_storage_ptr, cudaStream_t stream) {
  int num_items = parentInstances.size();
  auto d_in = parentInstances.rawPtr();
  auto d_out = subChunk->getInstances().rawPtr();

  InstanceInAabb select_op(subChunk->getAabb(), meshAabbs.rawPtr());
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::If(NULL, temp_storage_bytes, d_in, d_out, d_num_selected_out.rawPtr(), num_items, select_op, stream));
  CHECK_TEMP_STORAGE(temp_storage_bytes, d_temp_storage_ptr);

  // Run selection
  CUDA_CHECK(cub::DeviceSelect::If((void *)d_temp_storage_ptr.rawPtr(), temp_storage_bytes, d_in, d_out, d_num_selected_out.rawPtr(), num_items, select_op, stream));

  int num_selected = -1;
  d_num_selected_out.read(&num_selected);

  subChunk->getInstances().setSize(num_selected);

  std::cout << "Subchunk selectif: " << num_selected << std::endl;
}
} // namespace glow::pipeline::sceneloader::partition