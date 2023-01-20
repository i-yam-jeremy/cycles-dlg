#pragma once

#include <functional>

namespace demandLoadingGeometry {
class MeshHandle {
public:
  MeshHandle();

private:
  friend class GeometryDemandLoaderImpl;
  friend class std::hash<MeshHandle>;
  friend bool operator==(const MeshHandle &lhs, const MeshHandle &rhs);
  MeshHandle(int meshIndex) : meshIndex(meshIndex) {}
  int meshIndex;
};

bool operator==(const MeshHandle &lhs, const MeshHandle &rhs);

} // namespace demandLoadingGeometry

template <>
struct std::hash<demandLoadingGeometry::MeshHandle> {
  std::size_t operator()(demandLoadingGeometry::MeshHandle const &meshHandle) const noexcept;
};