#include <MeshHandle.hpp>

namespace demandLoadingGeometry {

MeshHandle::MeshHandle() : meshIndex(-1) {}

bool operator==(const MeshHandle &lhs, const MeshHandle &rhs) {
  return lhs.meshIndex == rhs.meshIndex;
}

} // namespace demandLoadingGeometry

std::size_t std::hash<demandLoadingGeometry::MeshHandle>::operator()(demandLoadingGeometry::MeshHandle const &meshHandle) const noexcept {
  return std::hash<int>{}(meshHandle.meshIndex);
}