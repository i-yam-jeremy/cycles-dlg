#pragma once

#include <cuda_check_macros.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <affinexform.h>

namespace glow::pipeline::sceneloader::partition {
struct Instance {
  int meshId; // TODO(jberchtold) maybe move this onto the instance list itself and store multiple instance lists, one per meshID? so meshId doesn't need to be duplicated across all instances
  uint32_t instanceId;
  demandLoadingGeometry::AffineXform xform;
};

class InstanceList {
public:
  InstanceList(size_t capacity /*in number of instances, not bytes*/) : m_capacity(capacity), m_size(0) {
    std::cout << "Allocating: " << (sizeof(Instance)*capacity) << std::endl;
    CUDA_CHECK(cudaMallocManaged(&m_instances, sizeof(Instance) * capacity));
  }

  ~InstanceList() {
    CUDA_CHECK(cudaFree(m_instances));
  }

  bool push_back_maybe(const Instance &instance) {
    if (m_size >= m_capacity) {
      std::cerr << "Warning: InstanceList capacity overflowed. Not adding instances\n";
      std::exit(1);
      return false;
    }

    (*this)[m_size++] = instance;
    return true;
  }

  __host__ Instance *rawPtr() {
    return m_instances;
  }

  __host__ const Instance *rawPtr() const {
    return m_instances;
  }

  __host__ __device__ size_t capacity() const {
    return m_capacity;
  }

  __host__ __device__ size_t size() const {
    return m_size;
  }

  __host__ void setSize(size_t newSize) {
    m_size = newSize;
  }

  __host__ __device__ Instance &operator[](size_t index) {
    return m_instances[index];
  }

  __host__ __device__ const Instance &operator[](size_t index) const {
    return m_instances[index];
  }

private:
  Instance *m_instances;
  size_t m_capacity;
  size_t m_size;
};

class Chunk {
public:
  Chunk(OptixAabb aabb, size_t instanceCapacity)
      : m_aabb(aabb), m_instances(instanceCapacity) {}

  const OptixAabb &getAabb() const {
    return m_aabb;
  }

  InstanceList &getInstances() {
    return m_instances;
  }

  const InstanceList &getInstances() const {
    return m_instances;
  }

private:
  OptixAabb m_aabb;
  InstanceList m_instances;
};
} // namespace glow::pipeline::sceneloader::partition