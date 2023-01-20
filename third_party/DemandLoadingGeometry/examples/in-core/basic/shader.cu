// This API can be used for a non-wavefront approach to potentially reduce load times.
// However, if the scene is larger than memory, a wavefront approach is required for
// reasonable performance. A non-wavefront approach will require all assets along the
// full traced path to be resident at once, which will not be efficient and may not
// even be possible depending on the size of the scene and the VRAM of the device.

#include <DemandLoadingGeometry.h>
#include <optix.h>

struct RayPayload {
  size_t rayIndex;
  short2 pixel;
};

__device__ void __raygen__example() {
  const auto ray = TODO; // Generate ray from camera
  bool isResident = false;
  const auto asHandle = dlgContext.requestGeometry(ray.origin, ray.direction, rayIndex, &isResident);
  if (!isResident) {
    return;
  }

  PayloadConverter<RayPayload> payload(RayPayload{rayIndex, pixel});
  optixTrace(asHandle, ray.origin, ray.direction, payload, ...);

  // TODO figure out how to make this API nice and easy to use from the kernel, without tons of relaunches, and also work with the standard DemandLoading texture API
  // NOTE if sticking with the current accumT/t method, that handles overlapping AABBs, need to still store intersections info in a buffer, because the nearest intersection geo may not still be loaded if overlapping chunks
  // HOWEVER if only supporting overlapping chunks (which is possible if I chop up large meshes), then it's possible to not do as much retracing, and know for sure the closest intersection will be the first intersection hit, but how to store that info across kernel launches for demand loading, just recalculate after textures load?

  // const Intersection intersection = {
  //     assetEntry.materialId,
  //     interpolateAttribs(assetEntry.normals, payload.barycentricCoords),
  //     interpolateAttribs(assetEntry.texCoords, payload.barycentricCoords),
  // };
  // bool isResident = false;
  // materialEval(rayBuf[rayIndex], intersection, isResident);
  // if (!isResident) {
  //   return;
  // }
}

__device__ void __closesthit__example() {
  const auto payload = PayloadConverter<RayPayload>().get();

  const auto color = TODO; // Blend Red Green and Blue based on barycentric coordinates
  writeColor(outputImage, pixel, color);

  dlgContext.resetRayMetadata(payload.rayIndex);
}

__device__ void __miss__example() {
  const auto payload = PayloadConverter<RayPayload>().get();
  writeColor(outputImage, pixel, BACKGROUND_COLOR);
  dlgContext.resetRayMetadata(payload.rayIndex);
}