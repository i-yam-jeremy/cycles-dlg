#include "optixmanager.h"

#include <iomanip>
#include <iostream>

#include "populate_optix_instances.h"
#include <ShaderNames.hpp>
#include <cuda_check_macros.h>
#include <cuda_runtime.h>
#include <device_memory_wrapper.h>
#include <nvtx3/nvToolsExt.h>
#include <string.h>
#include <util/monad/error.h>

using namespace glow::memory;

extern "C" const char EMBEDDED_PTX[];

glow::optix::OptixManager::OptixManager(const std::shared_ptr<glow::optix::OptixConnector> optix, OptixDeviceContext context)
    : optix(optix), context(context), log(std::vector<char>(2048)) {}

result<OptixProgramGroup, Err> glow::optix::OptixManager::createProgramGroup(const OptixPipelineCompileOptions &pipeline_compile_options, const OptixModuleCompileOptions &module_compile_options) {
  if (m_hitgroup_prog_group != nullptr) {
    return m_hitgroup_prog_group;
  }

  UNWRAP_VOID(optix->optixModuleCreateFromPTX_(
      context,
      &module_compile_options,
      &pipeline_compile_options,
      EMBEDDED_PTX,
      strlen(EMBEDDED_PTX),
      log,
      &m_module));

  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc hitgroup_prog_group_desc = {};
  hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_prog_group_desc.hitgroup.moduleCH = m_module;
  hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = DEMANDLOADINGGEOMETRY_CHUNK_CH_SHADER_NAME_STRING;
  UNWRAP_VOID(optix->optixProgramGroupCreate_(
      context,
      &hitgroup_prog_group_desc,
      1, // num program groups
      &program_group_options,
      log, &m_hitgroup_prog_group));

  return m_hitgroup_prog_group;
}

result<void, Err> glow::optix::OptixManager::sbtRecordPackHeader(void *sbtEntry) {
  UNWRAP_VOID(optix->optixSbtRecordPackHeader_(m_hitgroup_prog_group, sbtEntry));
  return {};
}

result<OptixTraversableHandle, Err> glow::optix::OptixManager::createTopLevelTraversable(const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &topLevelAssets) {
  std::vector<demandLoadingGeometry::Chunk::InstanceList> instanceXforms(topLevelAssets.size());
  for (size_t i = 0; i < topLevelAssets.size(); i++) {
    const auto asset = topLevelAssets[i];
    auto &instanceList = instanceXforms[i];
    instanceList.assetIndex = i;
    instanceList.instanceXforms.push_back(demandLoadingGeometry::AffineXform(asset->getChunkXform()));
  }
  UNWRAP(as, createTopLevelAS(instanceXforms, topLevelAssets, (cudaStream_t)0));
  this->topLevelAABBHandle = std::get<0>(as);
  this->topLevelAABBBuffer = std::get<1>(as);

  return this->topLevelAABBHandle;
}

void glow::optix::OptixManager::updateTopLevelAS(const std::span<OptixTraversableHandle> &topLevelAssets, cudaStream_t stream) {
  if (topLevelAssets.size() != m_topLevelChunkInstancesHost.size()) {
    std::cerr << "updateTopLevelAS: Incorrect number of input traversable handles passed in";
    std::exit(1);
  }

  nvtxRangePushA("updateTopLevelAS");

  for (size_t i = 0; i < topLevelAssets.size(); i++) {
    auto asHandle = topLevelAssets[i];
    if (asHandle == demandLoadingGeometry::NULL_TRAVERSABLE_HANDLE) {
      std::lock_guard guard(aabbASMutexes[i]);
      const auto entry = aabbASes.find(i);
      if (entry == aabbASes.end()) {
        std::cerr << "Cannot find aabb AS\n";
        std::exit(1);
      }
      asHandle = entry->second.first;
    }
    m_topLevelChunkInstancesHost[i].traversableHandle = asHandle;
  }

  m_topLevelChunkInstances->write(m_topLevelChunkInstancesHost.data());

  // Instance build input.
  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  buildInput.instanceArray.instances = m_topLevelChunkInstances->rawOptixPtr();
  buildInput.instanceArray.numInstances = static_cast<unsigned int>(m_topLevelChunkInstancesHost.size());

  const auto res = updateAS(topLevelAABBHandle, *topLevelAABBBuffer, buildInput, stream);
  if (res.has_error()) {
    std::cerr << "Failed to update top level AS\n";
    std::exit(1);
  }
  nvtxRangePop();
}

result<std::shared_ptr<glow::optix::ASData>, Err> glow::optix::OptixManager::buildAS(const demandLoadingGeometry::Mesh &mesh, int sbtOffset, cudaStream_t stream) {
  auto asData = std::make_shared<glow::optix::ASData>();

  std::vector<OptixBuildInput> inputs(mesh.buildInputs.size());
  memset(inputs.data(), 0, sizeof(OptixBuildInput) * inputs.size());
  std::vector<std::vector<CUdeviceptr>> pointBufferPointers;
  std::vector<std::shared_ptr<glow::memory::DevicePtr<float3>>> pointsBuffers;
  std::vector<std::shared_ptr<glow::memory::DevicePtr<uint3>>> indexBuffers;
  uint32_t triangleInputFlags[] = {OPTIX_GEOMETRY_FLAG_NONE};
  size_t primCount = 0;
  size_t i = 0;
  for (const auto &buildInput : mesh.buildInputs) {
    auto &input = inputs[i];

    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    auto pointsBuffer = std::make_shared<glow::memory::DevicePtr<float3>>(buildInput->positions.size() * sizeof(glm::vec3), stream);
    pointsBuffer->write(buildInput->positions.data());
    pointBufferPointers.push_back({pointsBuffer->rawOptixPtr()});

    auto indicesBuffer = std::make_shared<glow::memory::DevicePtr<uint3>>(buildInput->indices.size() * sizeof(glm::ivec3), stream);
    indicesBuffer->write(buildInput->indices.data());

    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    input.triangleArray.numVertices = (int)buildInput->positions.size();
    input.triangleArray.vertexBuffers = pointBufferPointers[pointBufferPointers.size() - 1].data();

    input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    input.triangleArray.numIndexTriplets = ((int)buildInput->indices.size());
    input.triangleArray.indexBuffer = indicesBuffer->rawOptixPtr();

    input.triangleArray.flags = triangleInputFlags;
    input.triangleArray.numSbtRecords = 1;
    input.triangleArray.primitiveIndexOffset = primCount;

    pointsBuffers.push_back(pointsBuffer);
    indexBuffers.push_back(indicesBuffer);
    primCount += buildInput->indices.size();
    i++;
  }

  UNWRAP(as, buildASandCompact(inputs, stream));
  asData->gasHandle = as.first;
  asData->gasBuffer = as.second;

  return std::move(asData);
}

namespace {
OptixAabb scaleAABB(const OptixAabb &aabb, float scale) {
  const glm::vec3 lower(aabb.minX, aabb.minY, aabb.minZ);
  const glm::vec3 upper(aabb.maxX, aabb.maxY, aabb.maxZ);
  const auto center = (lower + upper) / 2.f;
  const auto scaledLower = (lower - center) * scale + center;
  const auto scaledUpper = (upper - center) * scale + center;
  return {scaledLower.x, scaledLower.y, scaledLower.z, scaledUpper.x, scaledUpper.y, scaledUpper.z};
}
} // namespace

result<OptixTraversableHandle, Err> glow::optix::OptixManager::getAssetAabbAS(int assetId, const std::shared_ptr<glow::pipeline::render::IAsset> asset, cudaStream_t stream) {
  std::lock_guard guard(aabbASMutexes[assetId]);
  const auto entry = aabbASes.find(assetId);
  if (entry != aabbASes.end()) {
    return entry->second.first;
  }

  // Construct AABB made of triangles
  const auto aabb = scaleAABB(asset->getAABB(), 1.0);
  std::vector<float3> positions;
  std::vector<uint3> indices;

  positions.push_back({aabb.minX, aabb.minY, aabb.minZ});
  positions.push_back({aabb.maxX, aabb.minY, aabb.minZ});
  positions.push_back({aabb.minX, aabb.maxY, aabb.minZ});
  positions.push_back({aabb.maxX, aabb.maxY, aabb.minZ});
  positions.push_back({aabb.minX, aabb.minY, aabb.maxZ});
  positions.push_back({aabb.maxX, aabb.minY, aabb.maxZ});
  positions.push_back({aabb.minX, aabb.maxY, aabb.maxZ});
  positions.push_back({aabb.maxX, aabb.maxY, aabb.maxZ});

  indices.push_back({0, 1, 2});
  indices.push_back({3, 2, 1});
  indices.push_back({5, 4, 7});
  indices.push_back({6, 7, 4});
  indices.push_back({1, 5, 3});
  indices.push_back({7, 3, 5});
  indices.push_back({4, 0, 6});
  indices.push_back({2, 6, 0});
  indices.push_back({2, 3, 6});
  indices.push_back({7, 6, 3});
  indices.push_back({1, 0, 5});
  indices.push_back({4, 5, 0});

  // Create build input
  OptixBuildInput input = {};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  std::vector<CUdeviceptr> pointBufferPointers;
  auto pointsBuffer = std::make_shared<glow::memory::DevicePtr<float3>>(positions.size() * sizeof(float3), stream);
  pointsBuffer->write(positions.data());
  pointBufferPointers.push_back({pointsBuffer->rawOptixPtr()});

  auto indicesBuffer = std::make_shared<glow::memory::DevicePtr<uint3>>(indices.size() * sizeof(uint3), stream);
  indicesBuffer->write(indices.data());

  input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  input.triangleArray.vertexStrideInBytes = sizeof(float3);
  input.triangleArray.numVertices = (int)positions.size();
  input.triangleArray.vertexBuffers = pointBufferPointers.data();

  input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  input.triangleArray.indexStrideInBytes = sizeof(int) * 3;
  input.triangleArray.numIndexTriplets = ((int)indices.size());
  input.triangleArray.indexBuffer = indicesBuffer->rawOptixPtr();

  uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING | OPTIX_GEOMETRY_FLAG_NONE};
  input.triangleArray.flags = flags;
  input.triangleArray.numSbtRecords = 1;
  input.triangleArray.primitiveIndexOffset = 0;

  UNWRAP(as, buildASandCompact(input, stream));

  aabbASes[assetId] = as;

  return as.first;
}

namespace {
OptixInstance createInstanceForChunk(int assetIndex, OptixTraversableHandle asHandle, const demandLoadingGeometry::AffineXform &xform) {
  OptixInstance instance = {};
  instance.instanceId = assetIndex;
  instance.visibilityMask = 0xFF;
  instance.flags = OPTIX_INSTANCE_FLAG_NONE;
  memcpy(instance.transform, xform.data, sizeof(instance.transform));
  instance.sbtOffset = 0;
  instance.traversableHandle = asHandle;
  return instance;
}
} // namespace

result<std::tuple<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err> glow::optix::OptixManager::createTopLevelAS(const std::vector<demandLoadingGeometry::Chunk::InstanceList> &instanceXforms, const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &assets, cudaStream_t stream) {

  nvtxRangePushA("createTopLevelAS:mesh AABB AS and instance building");

  auto &instances = m_topLevelChunkInstancesHost;
  instances.clear();
  instances.reserve(instanceXforms.size());
  for (const auto &instanceList : instanceXforms) {
    const auto asset = assets[instanceList.assetIndex];
    UNWRAP(gasHandle, getAssetAabbAS(instanceList.assetIndex, asset, stream));

    for (const auto &xform : instanceList.instanceXforms) {
      instances.push_back(createInstanceForChunk(instanceList.assetIndex, gasHandle, xform));
    }
  }
  nvtxRangePop();

  nvtxRangePushA("createTopLevelAS:main AS build");
  m_topLevelChunkInstances = std::make_shared<DevicePtr<OptixInstance>>(sizeof(OptixInstance) * instances.size(), stream);
  m_topLevelChunkInstances->write(instances.data());

  // Instance build input.
  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  buildInput.instanceArray.instances = m_topLevelChunkInstances->rawOptixPtr();
  buildInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

  UNWRAP(as, buildASandCompact(buildInput, stream, true /* allowUpdates */));
  nvtxRangePop();
  return std::make_tuple(as.first, as.second);
}

result<std::tuple<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err> glow::optix::OptixManager::createChunkAS(const demandLoadingGeometry::Chunk &chunk, const std::vector<std::shared_ptr<glow::pipeline::render::IAsset>> &assets, const std::unordered_map<int, std::shared_ptr<glow::pipeline::render::IAsset>> &assetDependencies, float rayDifferential, int instanceCount, cudaStream_t stream) {

  nvtxRangePushA("createChunkAS: instance list building");

  DevicePtr<OptixInstance> deviceInstances(sizeof(OptixInstance) * instanceCount, stream);
  size_t instanceOffset = 0;
  for (const auto &instanceList : chunk.instanceLists) {
    const auto asset = assets[instanceList.assetIndex];
    const auto asHandle = assetDependencies.find(instanceList.assetIndex);
    if (asHandle == assetDependencies.end()) {
      std::cout << "Asset dependency not found: " << instanceList.assetIndex << "\n";
      // std::exit(1);
      continue;
    }

    glow::memory::DevicePtr<demandLoadingGeometry::AffineXform> xforms(sizeof(instanceList.instanceXforms[0]) * instanceList.instanceXforms.size(), stream);
    xforms.write(instanceList.instanceXforms.data());

    populateOptixInstances(deviceInstances.rawPtr() + instanceOffset, xforms.rawPtr(), asset->getAS(), asset->getSBTOffset(), instanceList.instanceXforms.size(), stream);
    instanceOffset += instanceList.instanceXforms.size();
  }
  nvtxRangePop();

  nvtxRangePushA("createChunkAS:main AS build");

  // Instance build input.
  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  buildInput.instanceArray.instances = deviceInstances.rawOptixPtr();
  buildInput.instanceArray.numInstances = static_cast<unsigned int>(instanceCount);

  UNWRAP(as, buildASandCompact(buildInput, stream));
  nvtxRangePop();
  return std::make_tuple(as.first, as.second);
}

result<std::pair<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err> glow::optix::OptixManager::buildASandCompact(const OptixBuildInput &buildInput, cudaStream_t stream, bool allowUpdates) {
  std::vector<OptixBuildInput> inputs = {buildInput};
  return buildASandCompact(inputs, stream, allowUpdates);
}

result<std::pair<OptixTraversableHandle, std::shared_ptr<glow::memory::DevicePtr<char>>>, Err> glow::optix::OptixManager::buildASandCompact(const std::vector<OptixBuildInput> &buildInputs, cudaStream_t stream, bool allowUpdates) {

  nvtxRangePushA("buildASandCompact:main build");

  OptixAccelBufferSizes bufferSizes;
  OptixAccelBuildOptions build_options = accel_options;
  if (allowUpdates) {
    build_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  }
  UNWRAP_VOID(optix->optixAccelComputeMemoryUsage_(context, &build_options, buildInputs.data(),
                                                   static_cast<unsigned int>(buildInputs.size()), // Number of build inputs
                                                   &bufferSizes));

  const auto uncompactedBuffer = std::make_shared<DevicePtr<char>>(bufferSizes.outputSizeInBytes, stream);
  if (uncompactedBuffer->rawPtr() == nullptr) {
    return failure(std::make_shared<glow::util::monad::Error>("Memory allocation"));
  }

  DevicePtr<uint64_t> compactedSizeBuffer(sizeof(uint64_t), stream);

  OptixAccelEmitDesc emitDesc;
  emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.rawOptixPtr();

  // std::cout << "Temp Size: " << bufferSizesIAS.tempSizeInBytes << std::endl;
  // std::cout << "Uncompacted Size: " << bufferSizesIAS.outputSizeInBytes << std::endl;

  OptixTraversableHandle asHandle;
  {
    DevicePtr<char> deviceTempBuffer(bufferSizes.tempSizeInBytes, stream); // Inner scope so this temp buffer is freed as soon as possible
    if (deviceTempBuffer.rawPtr() == nullptr) {
      return failure(std::make_shared<glow::util::monad::Error>("Memory allocation"));
    }
    UNWRAP_VOID(optix->optixAccelBuild_(context,
                                        stream, // CUDA stream
                                        &build_options,
                                        buildInputs.data(),
                                        static_cast<unsigned int>(buildInputs.size()), // num build inputs
                                        deviceTempBuffer.rawOptixPtr(),
                                        deviceTempBuffer.size(),
                                        uncompactedBuffer->rawOptixPtr(),
                                        uncompactedBuffer->size(),
                                        &asHandle,
                                        &emitDesc, // emitted property list
                                        1));       // num emitted properties
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  nvtxRangePop();

  nvtxRangePushA("buildASandCompact:compaction");
  uint64_t compactedSize;
  compactedSizeBuffer.read(&compactedSize);

  if (compactedSize >= uncompactedBuffer->size()) {
    nvtxRangePop();
    return std::make_pair(asHandle, uncompactedBuffer);
  }

  const auto compactedBuffer = std::make_shared<DevicePtr<char>>(compactedSize, stream);
  if (compactedBuffer->rawPtr() == nullptr) {
    return failure(std::make_shared<glow::util::monad::Error>("Memory allocation"));
  }
  // std::cout << "Compacted Size: " << compactedSize << std::endl;
  UNWRAP_VOID(optix->optixAccelCompact_(context,
                                        stream,
                                        asHandle,
                                        compactedBuffer->rawOptixPtr(),
                                        compactedSize,
                                        &asHandle));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  nvtxRangePop();
  return std::make_pair(asHandle, compactedBuffer);
}

result<void, Err> glow::optix::OptixManager::updateAS(OptixTraversableHandle handle, glow::memory::DevicePtr<char> &buffer, const OptixBuildInput &buildInput, cudaStream_t stream) {
  std::vector<OptixBuildInput> buildInputs(1);
  buildInputs[0] = buildInput;
  return updateAS(handle, buffer, buildInputs, stream);
}

result<void, Err> glow::optix::OptixManager::updateAS(OptixTraversableHandle asHandle, glow::memory::DevicePtr<char> &buffer, const std::vector<OptixBuildInput> &buildInputs, cudaStream_t stream) {
  nvtxRangePushA("updateAS");

  OptixAccelBufferSizes bufferSizes;
  OptixAccelBuildOptions build_options = accel_options;
  build_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  build_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
  UNWRAP_VOID(optix->optixAccelComputeMemoryUsage_(context, &build_options, buildInputs.data(),
                                                   static_cast<unsigned int>(buildInputs.size()), // Number of build inputs
                                                   &bufferSizes));

  {
    DevicePtr<char> deviceTempBuffer(bufferSizes.tempSizeInBytes, stream); // Inner scope so this temp buffer is freed as soon as possible
    if (deviceTempBuffer.rawPtr() == nullptr) {
      return failure(std::make_shared<glow::util::monad::Error>("Memory allocation"));
    }
    UNWRAP_VOID(optix->optixAccelBuild_(context,
                                        stream, // CUDA stream
                                        &build_options,
                                        buildInputs.data(),
                                        static_cast<unsigned int>(buildInputs.size()), // num build inputs
                                        deviceTempBuffer.rawOptixPtr(),
                                        deviceTempBuffer.size(),
                                        buffer.rawOptixPtr(),
                                        buffer.size(),
                                        &asHandle,
                                        nullptr, // emitted property list
                                        0));     // num emitted properties
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  nvtxRangePop();
  return {};
}

result<OptixAccelBufferSizes, Err> glow::optix::OptixManager::computeMemoryUsage(const OptixBuildInput &buildInput) {
  std::vector<OptixBuildInput> inputs = {buildInput};
  return computeMemoryUsage(inputs);
}

result<OptixAccelBufferSizes, Err> glow::optix::OptixManager::computeMemoryUsage(const std::vector<OptixBuildInput> &buildInputs) {
  OptixAccelBufferSizes bufferSizes;
  UNWRAP_VOID(optix->optixAccelComputeMemoryUsage_(context, &accel_options, buildInputs.data(),
                                                   buildInputs.size(), // Number of build inputs
                                                   &bufferSizes));
  return bufferSizes;
}