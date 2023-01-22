#pragma once

#include <chrono>
#include <iostream>
#include <sstream>

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                        \
  do {                                                          \
    cudaError_t error = call;                                   \
    if (error != cudaSuccess) {                                 \
      std::cerr << "CUDA call (" << #call << " ) failed with error: '" \
         << cudaGetErrorString(error)                           \
         << "' (" __FILE__ << ":" << __LINE__ << ")\n";          \
      while (1) {                                               \
      }                                                         \
      std::exit(1);                                             \
    }                                                           \
  } while (0)

#define CUDA_SYNC_CHECK()                                      \
  do {                                                         \
    cudaDeviceSynchronize();                                   \
    cudaError_t error = cudaGetLastError();                    \
    if (error != cudaSuccess) {                                \
      std::cerr << "CUDA error on synchronize with error '"    \
                << cudaGetErrorString(error)                   \
                << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
    }                                                          \
  } while (0)

// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW(call)                                       \
  do {                                                                 \
    cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                        \
      std::cerr << "CUDA call (" << #call << " ) failed with error: '" \
                << cudaGetErrorString(error)                           \
                << "' (" __FILE__ << ":" << __LINE__ << ")\n";         \
      std::terminate();                                                \
    }                                                                  \
  } while (0)

// cuRAND
#define CURAND_CALL(x)                                \
  do {                                                \
    if ((x) != CURAND_STATUS_SUCCESS) {               \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while (0)
