#pragma once

#include <cuda.h>
#include <optix_types.h>

#include <optional>
#include <util/monad/error.h>
#include <util/monad/result.h>

namespace glow::optix {
struct OptixError;

class OptixConnector {
 public:
  OptixConnector()
  {
  }
  virtual result<void, std::shared_ptr<OptixError>> optixUtilAccumulateStackSizes_(
      OptixProgramGroup programGroup, OptixStackSizes *stackSizes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixUtilComputeStackSizes_(
      const OptixStackSizes *stackSizes,
      unsigned int maxTraceDepth,
      unsigned int maxCCDepth,
      unsigned int maxDCDepth,
      unsigned int *directCallableStackSizeFromTraversal,
      unsigned int *directCallableStackSizeFromState,
      unsigned int *continuationStackSize) const;

  virtual result<void, std::shared_ptr<OptixError>> optixInit_(void) const;

  virtual const char *optixGetErrorName_(OptixResult result) const;

  virtual const char *optixGetErrorString_(OptixResult result) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextCreate_(
      CUcontext fromContext,
      const OptixDeviceContextOptions *options,
      OptixDeviceContext *context) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextDestroy_(
      OptixDeviceContext context) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextGetProperty_(
      OptixDeviceContext context,
      OptixDeviceProperty property,
      void *value,
      size_t sizeInBytes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextSetLogCallback_(
      OptixDeviceContext context,
      OptixLogCallback callbackFunction,
      void *callbackData,
      unsigned int callbackLevel) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextSetCacheEnabled_(
      OptixDeviceContext context, int enabled) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextSetCacheLocation_(
      OptixDeviceContext context, const char *location) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextSetCacheDatabaseSizes_(
      OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextGetCacheEnabled_(
      OptixDeviceContext context, int *enabled) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextGetCacheLocation_(
      OptixDeviceContext context, char *location, size_t locationSize) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDeviceContextGetCacheDatabaseSizes_(
      OptixDeviceContext context, size_t *lowWaterMark, size_t *highWaterMark) const;

  virtual result<void, std::shared_ptr<OptixError>> optixModuleCreateFromPTX_(
      OptixDeviceContext context,
      const OptixModuleCompileOptions *moduleCompileOptions,
      const OptixPipelineCompileOptions *pipelineCompileOptions,
      const char *PTX,
      size_t PTXsize,
      std::vector<char> &log,
      OptixModule *module) const;

  virtual result<void, std::shared_ptr<OptixError>> optixModuleCreateFromPTXWithTasks_(
      OptixDeviceContext context,
      const OptixModuleCompileOptions *moduleCompileOptions,
      const OptixPipelineCompileOptions *pipelineCompileOptions,
      const char *PTX,
      size_t PTXsize,
      std::vector<char> &log,
      OptixModule *module,
      OptixTask *firstTask) const;

  virtual result<void, std::shared_ptr<OptixError>> optixModuleGetCompilationState_(
      OptixModule module, OptixModuleCompileState *state) const;

  virtual result<void, std::shared_ptr<OptixError>> optixModuleDestroy_(OptixModule module) const;

  virtual result<void, std::shared_ptr<OptixError>> optixBuiltinISModuleGet_(
      OptixDeviceContext context,
      const OptixModuleCompileOptions *moduleCompileOptions,
      const OptixPipelineCompileOptions *pipelineCompileOptions,
      const OptixBuiltinISOptions *builtinISOptions,
      OptixModule *builtinModule) const;

  virtual result<void, std::shared_ptr<OptixError>> optixTaskExecute_(
      OptixTask task,
      OptixTask *additionalTasks,
      unsigned int maxNumAdditionalTasks,
      unsigned int *numAdditionalTasksCreated) const;

  virtual result<void, std::shared_ptr<OptixError>> optixProgramGroupCreate_(
      OptixDeviceContext context,
      const OptixProgramGroupDesc *programDescriptions,
      unsigned int numProgramGroups,
      const OptixProgramGroupOptions *options,
      std::vector<char> &log,
      OptixProgramGroup *programGroups) const;

  virtual result<void, std::shared_ptr<OptixError>> optixProgramGroupDestroy_(
      OptixProgramGroup programGroup) const;

  virtual result<void, std::shared_ptr<OptixError>> optixProgramGroupGetStackSize_(
      OptixProgramGroup programGroup, OptixStackSizes *stackSizes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixPipelineCreate_(
      OptixDeviceContext context,
      const OptixPipelineCompileOptions *pipelineCompileOptions,
      const OptixPipelineLinkOptions *pipelineLinkOptions,
      const OptixProgramGroup *programGroups,
      unsigned int numProgramGroups,
      std::vector<char> &log,
      OptixPipeline *pipeline) const;

  virtual result<void, std::shared_ptr<OptixError>> optixPipelineDestroy_(
      OptixPipeline pipeline) const;

  virtual result<void, std::shared_ptr<OptixError>> optixPipelineSetStackSize_(
      OptixPipeline pipeline,
      unsigned int directCallableStackSizeFromTraversal,
      unsigned int directCallableStackSizeFromState,
      unsigned int continuationStackSize,
      unsigned int maxTraversableGraphDepth) const;

  virtual result<void, std::shared_ptr<OptixError>> optixAccelComputeMemoryUsage_(
      OptixDeviceContext context,
      const OptixAccelBuildOptions *accelOptions,
      const OptixBuildInput *buildInputs,
      unsigned int numBuildInputs,
      OptixAccelBufferSizes *bufferSizes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixAccelBuild_(
      OptixDeviceContext context,
      CUstream stream,
      const OptixAccelBuildOptions *accelOptions,
      const OptixBuildInput *buildInputs,
      unsigned int numBuildInputs,
      CUdeviceptr tempBuffer,
      size_t tempBufferSizeInBytes,
      CUdeviceptr outputBuffer,
      size_t outputBufferSizeInBytes,
      OptixTraversableHandle *outputHandle,
      const OptixAccelEmitDesc *emittedProperties,
      unsigned int numEmittedProperties) const;

  virtual result<void, std::shared_ptr<OptixError>> optixAccelGetRelocationInfo_(
      OptixDeviceContext context,
      OptixTraversableHandle handle,
      OptixAccelRelocationInfo *info) const;

  virtual result<void, std::shared_ptr<OptixError>> optixAccelCheckRelocationCompatibility_(
      OptixDeviceContext context, const OptixAccelRelocationInfo *info, int *compatible) const;

  virtual result<void, std::shared_ptr<OptixError>> optixAccelRelocate_(
      OptixDeviceContext context,
      CUstream stream,
      const OptixAccelRelocationInfo *info,
      CUdeviceptr instanceTraversableHandles,
      size_t numInstanceTraversableHandles,
      CUdeviceptr targetAccel,
      size_t targetAccelSizeInBytes,
      OptixTraversableHandle *targetHandle) const;

  virtual result<void, std::shared_ptr<OptixError>> optixAccelCompact_(
      OptixDeviceContext context,
      CUstream stream,
      OptixTraversableHandle inputHandle,
      CUdeviceptr outputBuffer,
      size_t outputBufferSizeInBytes,
      OptixTraversableHandle *outputHandle) const;

  virtual result<void, std::shared_ptr<OptixError>> optixConvertPointerToTraversableHandle_(
      OptixDeviceContext onDevice,
      CUdeviceptr pointer,
      OptixTraversableType traversableType,
      OptixTraversableHandle *traversableHandle) const;

  virtual result<void, std::shared_ptr<OptixError>> optixSbtRecordPackHeader_(
      OptixProgramGroup programGroup, void *sbtRecordHeaderHostPointer) const;

  virtual result<void, std::shared_ptr<OptixError>> optixLaunch_(
      OptixPipeline pipeline,
      CUstream stream,
      CUdeviceptr pipelineParams,
      size_t pipelineParamsSize,
      const OptixShaderBindingTable *sbt,
      unsigned int width,
      unsigned int height,
      unsigned int depth) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserCreate_(
      OptixDeviceContext context,
      OptixDenoiserModelKind modelKind,
      const OptixDenoiserOptions *options,
      OptixDenoiser *returnHandle) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserCreateWithUserModel_(
      OptixDeviceContext context,
      const void *data,
      size_t dataSizeInBytes,
      OptixDenoiser *returnHandle) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserDestroy_(
      OptixDenoiser handle) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserComputeMemoryResources_(
      const OptixDenoiser handle,
      unsigned int maximumInputWidth,
      unsigned int maximumInputHeight,
      OptixDenoiserSizes *returnSizes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserSetup_(
      OptixDenoiser denoiser,
      CUstream stream,
      unsigned int inputWidth,
      unsigned int inputHeight,
      CUdeviceptr denoiserState,
      size_t denoiserStateSizeInBytes,
      CUdeviceptr scratch,
      size_t scratchSizeInBytes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserInvoke_(
      OptixDenoiser handle,
      CUstream stream,
      const OptixDenoiserParams *params,
      CUdeviceptr denoiserData,
      size_t denoiserDataSize,
      const OptixDenoiserGuideLayer *guideLayer,
      const OptixDenoiserLayer *layers,
      unsigned int numLayers,
      unsigned int inputOffsetX,
      unsigned int inputOffsetY,
      CUdeviceptr scratch,
      size_t scratchSizeInBytes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserComputeIntensity_(
      OptixDenoiser handle,
      CUstream stream,
      const OptixImage2D *inputImage,
      CUdeviceptr outputIntensity,
      CUdeviceptr scratch,
      size_t scratchSizeInBytes) const;

  virtual result<void, std::shared_ptr<OptixError>> optixDenoiserComputeAverageColor_(
      OptixDenoiser handle,
      CUstream stream,
      const OptixImage2D *inputImage,
      CUdeviceptr outputAverageColor,
      CUdeviceptr scratch,
      size_t scratchSizeInBytes) const;
};

struct OptixError : public glow::util::monad::Error {
  OptixError(OptixResult optixError,
             const OptixConnector &connector,
             std::optional<std::string> log = {})
      : optixError(optixError),
        glow::util::monad::Error("OptiX Error[" +
                                 std::string(connector.optixGetErrorName_(optixError)) +
                                 "]: " + std::string(connector.optixGetErrorString_(optixError)) +
                                 (log.has_value() ? ("\nLog: " + log.value()) : ""))
  {
    throw new std::runtime_error("Howdy\n");
  };
  OptixResult optixError;
};
}  // namespace glow::optix