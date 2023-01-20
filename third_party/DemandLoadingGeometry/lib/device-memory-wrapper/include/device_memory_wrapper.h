#pragma once

#include <cuda.h>
#include <cuda_check_macros.h>
#include <cuda_runtime.h>
#include <optix_types.h>

namespace glow::memory {

void *alloc(size_t size, cudaStream_t stream);
void dealloc(void *ptr, size_t size, cudaStream_t stream);

size_t getMemoryUsage();

template <typename T>
class DevicePtr {
public:
  DevicePtr() = delete;
  DevicePtr(const DevicePtr<T> &other) = delete;
  DevicePtr(DevicePtr<T> &&other) = delete;
  DevicePtr(size_t sizeInBytes, cudaStream_t stream) : sizeInBytes(sizeInBytes), m_stream(stream) {
    rawDevicePtr = (T *)alloc(sizeInBytes, stream);
  }
  __host__ ~DevicePtr() {
    if (rawDevicePtr != nullptr) {
      dealloc(rawDevicePtr, size(), m_stream);
      rawDevicePtr = nullptr;
    }
  }

  __host__ T *rawPtr() {
    return rawDevicePtr;
  }

  __host__ const T *rawPtr() const {
    return rawDevicePtr;
  }

  __host__ CUdeviceptr rawOptixPtr() {
    return (CUdeviceptr)rawDevicePtr;
  }
  __host__ __device__ size_t size() const {
    return sizeInBytes;
  }
  __host__ void write(const T *hostPtrSrc) {
    CUDA_CHECK(cudaMemcpyAsync(rawDevicePtr, hostPtrSrc, size(), cudaMemcpyHostToDevice, m_stream));
  };
  __host__ void write(const T *hostPtrSrc, cudaStream_t stream) { // Write on a stream other than the stream this buffer was created on
    CUDA_CHECK(cudaMemcpyAsync(rawDevicePtr, hostPtrSrc, size(), cudaMemcpyHostToDevice, stream));
  };
  __host__ void read(T *hostPtrDst) const {
    CUDA_CHECK(cudaMemcpyAsync(hostPtrDst, rawDevicePtr, size(), cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
  };
  __host__ void read(T *hostPtrDst, cudaStream_t stream) const { // Write on a stream other than the stream this buffer was created on
    CUDA_CHECK(cudaMemcpyAsync(hostPtrDst, rawDevicePtr, size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  __device__ T *operator->() {
    return rawDevicePtr;
  };
  __device__ const T *operator->() const {
    return rawDevicePtr;
  };
  __device__ T &operator[](const size_t index) {
    return rawDevicePtr[index];
  };
  __device__ const T &operator[](const size_t index) const {
    return rawDevicePtr[index];
  };

private:
  T *rawDevicePtr = nullptr;
  size_t sizeInBytes = 0;
  cudaStream_t m_stream;
};

} // namespace glow::memory