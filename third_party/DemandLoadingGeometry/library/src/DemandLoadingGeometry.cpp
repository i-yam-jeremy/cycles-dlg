#include <DemandLoadingGeometry.hpp>

#include <DemandLoadingGeometryImpl.hpp>
#include <memory>
#include <scenepartioning/instance_partitioner.h>

namespace {
std::unique_ptr<glow::pipeline::sceneloader::partition::InstancePartitioner> createInstancePartitioner(const OptixAabb &sceneBounds, InstancePartitionerType type) {
  switch (type) {
  case InstancePartitionerType::OCTREE:
    return std::move(std::make_unique<glow::pipeline::sceneloader::partition::OctreePartitioner>(sceneBounds));
  case InstancePartitionerType::KDTREE:
    return std::move(std::make_unique<glow::pipeline::sceneloader::partition::KdTreePartitioner>(sceneBounds));
  }
  return nullptr;
}
} // namespace
namespace demandLoadingGeometry {

SBTBuffer::~SBTBuffer() {
  delete[] reinterpret_cast<char *>(data);
}

GeometryDemandLoader *createDemandLoader(Options &options, OptixDeviceContext optixContext) {
  auto instancePartitioner = createInstancePartitioner(options.sceneBounds, options.instancePartitionerType);
  auto impl = new GeometryDemandLoaderImpl(std::move(instancePartitioner), options, optixContext);
  auto geoDemandLoader = new GeometryDemandLoader(impl);
  return geoDemandLoader;
}

GeometryDemandLoader::GeometryDemandLoader(GeometryDemandLoaderImpl *impl) : m_impl(impl) {
}

GeometryDemandLoader::~GeometryDemandLoader() {
  delete m_impl;
}

std::optional<OptixProgramGroup> GeometryDemandLoader::getOptixProgramGroup(const OptixPipelineCompileOptions &pipeline_compile_options, const OptixModuleCompileOptions &module_compile_options) {
  return m_impl->getOptixProgramGroup(pipeline_compile_options, module_compile_options);
}

MeshHandle GeometryDemandLoader::addMesh(const Mesh &mesh) {
  return m_impl->addMesh(mesh);
}

void GeometryDemandLoader::addInstance(MeshHandle meshHandle, const OptixAabb &meshAabb, const glm::mat4 &xform) {
  m_impl->addInstance(meshHandle, meshAabb, xform);
}

void GeometryDemandLoader::updateScene() {
  m_impl->updateScene();
}

std::unique_ptr<SBTBuffer> GeometryDemandLoader::getInternalApiHitgroupSbtEntries(size_t sizeOfUserSbtStruct) {
  return std::move(m_impl->getInternalApiHitgroupSbtEntries(sizeOfUserSbtStruct));
}

demandLoadingGeometry::LaunchData GeometryDemandLoader::preLaunch(demandLoadingGeometry::RayIndex *d_endOfUserRayQueue, cudaStream_t stream) {
  return m_impl->preLaunch(d_endOfUserRayQueue, stream);
}

void GeometryDemandLoader::postLaunch(cudaStream_t stream) {
  m_impl->postLaunch(stream);
}

} // namespace demandLoadingGeometry