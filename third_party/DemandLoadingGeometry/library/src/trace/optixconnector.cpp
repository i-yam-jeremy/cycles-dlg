#include "optixconnector.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <iostream>

#define OPTIX_CHECK(call)                                               \
  {                                                                     \
    OptixResult res = (call);                                           \
    if (res != OPTIX_SUCCESS) {                                         \
      auto err = std::make_shared<glow::optix::OptixError>(res, *this); \
      std::cout << err->getMessage() << std::endl;                      \
      while (1) {                                                       \
      }                                                                 \
      return failure(err);                                              \
    }                                                                   \
    return {};                                                          \
  }

#define OPTIX_CHECK_LOG(call, logValue)                                      \
  {                                                                          \
    std::vector<char> &__log = logValue;                                     \
    size_t sizeof_log = __log.size();                                        \
    auto logStringSize = &sizeof_log;                                        \
    char *logString = __log.data();                                          \
    OptixResult res = (call);                                                \
    if (res != OPTIX_SUCCESS) {                                              \
      const size_t sizeof_log_returned = sizeof_log;                         \
      const std::string log = std::string(logString, sizeof_log_returned);   \
      return failure(std::make_shared<glow::optix::OptixError>(res, *this)); \
    }                                                                        \
    return {};                                                               \
  }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixUtilAccumulateStackSizes_(OptixProgramGroup programGroup, OptixStackSizes *stackSizes) const { OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroup, stackSizes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixUtilComputeStackSizes_(const OptixStackSizes *stackSizes,
                                                                                                                unsigned int maxTraceDepth,
                                                                                                                unsigned int maxCCDepth,
                                                                                                                unsigned int maxDCDepth,
                                                                                                                unsigned int *directCallableStackSizeFromTraversal,
                                                                                                                unsigned int *directCallableStackSizeFromState,
                                                                                                                unsigned int *continuationStackSize) const { OPTIX_CHECK(optixUtilComputeStackSizes(stackSizes,
                                                                                                                                                                                                    maxTraceDepth,
                                                                                                                                                                                                    maxCCDepth,
                                                                                                                                                                                                    maxDCDepth,
                                                                                                                                                                                                    directCallableStackSizeFromTraversal,
                                                                                                                                                                                                    directCallableStackSizeFromState,
                                                                                                                                                                                                    continuationStackSize)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixInit_(void) const { OPTIX_CHECK(optixInit()); }

const char *glow::optix::OptixConnector::optixGetErrorName_(OptixResult result) const { return optixGetErrorName(result); }

const char *glow::optix::OptixConnector::optixGetErrorString_(OptixResult result) const { return optixGetErrorString(result); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextCreate_(CUcontext fromContext, const OptixDeviceContextOptions *options, OptixDeviceContext *context) const { OPTIX_CHECK(optixDeviceContextCreate(fromContext, options, context)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextDestroy_(OptixDeviceContext context) const { OPTIX_CHECK(optixDeviceContextDestroy(context)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextGetProperty_(OptixDeviceContext context, OptixDeviceProperty property, void *value, size_t sizeInBytes) const { OPTIX_CHECK(optixDeviceContextGetProperty(context, property, value, sizeInBytes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextSetLogCallback_(OptixDeviceContext context,
                                                                                                                      OptixLogCallback callbackFunction,
                                                                                                                      void *callbackData,
                                                                                                                      unsigned int callbackLevel) const { OPTIX_CHECK(optixDeviceContextSetLogCallback(context,
                                                                                                                                                                                                       callbackFunction,
                                                                                                                                                                                                       callbackData,
                                                                                                                                                                                                       callbackLevel)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextSetCacheEnabled_(OptixDeviceContext context, int enabled) const { OPTIX_CHECK(optixDeviceContextSetCacheEnabled(context, enabled)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextSetCacheLocation_(OptixDeviceContext context, const char *location) const { OPTIX_CHECK(optixDeviceContextSetCacheLocation(context, location)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextSetCacheDatabaseSizes_(OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark) const { OPTIX_CHECK(optixDeviceContextSetCacheDatabaseSizes(context, lowWaterMark, highWaterMark)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextGetCacheEnabled_(OptixDeviceContext context, int *enabled) const { OPTIX_CHECK(optixDeviceContextGetCacheEnabled(context, enabled)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextGetCacheLocation_(OptixDeviceContext context, char *location, size_t locationSize) const { OPTIX_CHECK(optixDeviceContextGetCacheLocation(context, location, locationSize)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDeviceContextGetCacheDatabaseSizes_(OptixDeviceContext context, size_t *lowWaterMark, size_t *highWaterMark) const { OPTIX_CHECK(optixDeviceContextGetCacheDatabaseSizes(context, lowWaterMark, highWaterMark)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixModuleCreateFromPTX_(OptixDeviceContext context,
                                                                                                              const OptixModuleCompileOptions *moduleCompileOptions,
                                                                                                              const OptixPipelineCompileOptions *pipelineCompileOptions,
                                                                                                              const char *PTX,
                                                                                                              size_t PTXsize,
                                                                                                              std::vector<char> &log,
                                                                                                              OptixModule *module) const { OPTIX_CHECK_LOG(optixModuleCreateFromPTX(context,
                                                                                                                                                                                    moduleCompileOptions,
                                                                                                                                                                                    pipelineCompileOptions,
                                                                                                                                                                                    PTX,
                                                                                                                                                                                    PTXsize,
                                                                                                                                                                                    logString,
                                                                                                                                                                                    logStringSize,
                                                                                                                                                                                    module),
                                                                                                                                                           log); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixModuleCreateFromPTXWithTasks_(OptixDeviceContext context,
                                                                                                                       const OptixModuleCompileOptions *moduleCompileOptions,
                                                                                                                       const OptixPipelineCompileOptions *pipelineCompileOptions,
                                                                                                                       const char *PTX,
                                                                                                                       size_t PTXsize,
                                                                                                                       std::vector<char> &log,
                                                                                                                       OptixModule *module,
                                                                                                                       OptixTask *firstTask) const { OPTIX_CHECK_LOG(optixModuleCreateFromPTXWithTasks(context,
                                                                                                                                                                                                       moduleCompileOptions,
                                                                                                                                                                                                       pipelineCompileOptions,
                                                                                                                                                                                                       PTX,
                                                                                                                                                                                                       PTXsize,
                                                                                                                                                                                                       logString,
                                                                                                                                                                                                       logStringSize,
                                                                                                                                                                                                       module,
                                                                                                                                                                                                       firstTask),
                                                                                                                                                                     log); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixModuleGetCompilationState_(OptixModule module, OptixModuleCompileState *state) const { OPTIX_CHECK(optixModuleGetCompilationState(module, state)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixModuleDestroy_(OptixModule module) const { OPTIX_CHECK(optixModuleDestroy(module)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixBuiltinISModuleGet_(OptixDeviceContext context,
                                                                                                             const OptixModuleCompileOptions *moduleCompileOptions,
                                                                                                             const OptixPipelineCompileOptions *pipelineCompileOptions,
                                                                                                             const OptixBuiltinISOptions *builtinISOptions,
                                                                                                             OptixModule *builtinModule) const { OPTIX_CHECK(optixBuiltinISModuleGet(context,
                                                                                                                                                                                     moduleCompileOptions,
                                                                                                                                                                                     pipelineCompileOptions,
                                                                                                                                                                                     builtinISOptions,
                                                                                                                                                                                     builtinModule)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixTaskExecute_(OptixTask task, OptixTask *additionalTasks, unsigned int maxNumAdditionalTasks, unsigned int *numAdditionalTasksCreated) const { OPTIX_CHECK(optixTaskExecute(task, additionalTasks, maxNumAdditionalTasks, numAdditionalTasksCreated)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixProgramGroupCreate_(OptixDeviceContext context,
                                                                                                             const OptixProgramGroupDesc *programDescriptions,
                                                                                                             unsigned int numProgramGroups,
                                                                                                             const OptixProgramGroupOptions *options,
                                                                                                             std::vector<char> &log,
                                                                                                             OptixProgramGroup *programGroups) const { OPTIX_CHECK_LOG(optixProgramGroupCreate(context,
                                                                                                                                                                                               programDescriptions,
                                                                                                                                                                                               numProgramGroups,
                                                                                                                                                                                               options,
                                                                                                                                                                                               logString,
                                                                                                                                                                                               logStringSize,
                                                                                                                                                                                               programGroups),
                                                                                                                                                                       log); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixProgramGroupDestroy_(OptixProgramGroup programGroup) const { OPTIX_CHECK(optixProgramGroupDestroy(programGroup)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixProgramGroupGetStackSize_(OptixProgramGroup programGroup, OptixStackSizes *stackSizes) const { OPTIX_CHECK(optixProgramGroupGetStackSize(programGroup, stackSizes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixPipelineCreate_(OptixDeviceContext context,
                                                                                                         const OptixPipelineCompileOptions *pipelineCompileOptions,
                                                                                                         const OptixPipelineLinkOptions *pipelineLinkOptions,
                                                                                                         const OptixProgramGroup *programGroups,
                                                                                                         unsigned int numProgramGroups,
                                                                                                         std::vector<char> &log,
                                                                                                         OptixPipeline *pipeline) const { OPTIX_CHECK_LOG(optixPipelineCreate(context,
                                                                                                                                                                              pipelineCompileOptions,
                                                                                                                                                                              pipelineLinkOptions,
                                                                                                                                                                              programGroups,
                                                                                                                                                                              numProgramGroups,
                                                                                                                                                                              logString,
                                                                                                                                                                              logStringSize,
                                                                                                                                                                              pipeline),
                                                                                                                                                          log); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixPipelineDestroy_(OptixPipeline pipeline) const { OPTIX_CHECK(optixPipelineDestroy(pipeline)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixPipelineSetStackSize_(OptixPipeline pipeline,
                                                                                                               unsigned int directCallableStackSizeFromTraversal,
                                                                                                               unsigned int directCallableStackSizeFromState,
                                                                                                               unsigned int continuationStackSize,
                                                                                                               unsigned int maxTraversableGraphDepth) const { OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
                                                                                                                                                                                                    directCallableStackSizeFromTraversal,
                                                                                                                                                                                                    directCallableStackSizeFromState,
                                                                                                                                                                                                    continuationStackSize,
                                                                                                                                                                                                    maxTraversableGraphDepth)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixAccelComputeMemoryUsage_(OptixDeviceContext context,
                                                                                                                  const OptixAccelBuildOptions *accelOptions,
                                                                                                                  const OptixBuildInput *buildInputs,
                                                                                                                  unsigned int numBuildInputs,
                                                                                                                  OptixAccelBufferSizes *bufferSizes) const { OPTIX_CHECK(optixAccelComputeMemoryUsage(context,
                                                                                                                                                                                                       accelOptions,
                                                                                                                                                                                                       buildInputs,
                                                                                                                                                                                                       numBuildInputs,
                                                                                                                                                                                                       bufferSizes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixAccelBuild_(OptixDeviceContext context,
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
                                                                                                     unsigned int numEmittedProperties) const { OPTIX_CHECK(optixAccelBuild(context,
                                                                                                                                                                            stream,
                                                                                                                                                                            accelOptions,
                                                                                                                                                                            buildInputs,
                                                                                                                                                                            numBuildInputs,
                                                                                                                                                                            tempBuffer,
                                                                                                                                                                            tempBufferSizeInBytes,
                                                                                                                                                                            outputBuffer,
                                                                                                                                                                            outputBufferSizeInBytes,
                                                                                                                                                                            outputHandle,
                                                                                                                                                                            emittedProperties,
                                                                                                                                                                            numEmittedProperties)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixAccelGetRelocationInfo_(OptixDeviceContext context, OptixTraversableHandle handle, OptixAccelRelocationInfo *info) const { OPTIX_CHECK(optixAccelGetRelocationInfo(context, handle, info)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixAccelCheckRelocationCompatibility_(OptixDeviceContext context, const OptixAccelRelocationInfo *info, int *compatible) const { OPTIX_CHECK(optixAccelCheckRelocationCompatibility(context, info, compatible)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixAccelRelocate_(OptixDeviceContext context,
                                                                                                        CUstream stream,
                                                                                                        const OptixAccelRelocationInfo *info,
                                                                                                        CUdeviceptr instanceTraversableHandles,
                                                                                                        size_t numInstanceTraversableHandles,
                                                                                                        CUdeviceptr targetAccel,
                                                                                                        size_t targetAccelSizeInBytes,
                                                                                                        OptixTraversableHandle *targetHandle) const { OPTIX_CHECK(optixAccelRelocate(context,
                                                                                                                                                                                     stream,
                                                                                                                                                                                     info,
                                                                                                                                                                                     instanceTraversableHandles,
                                                                                                                                                                                     numInstanceTraversableHandles,
                                                                                                                                                                                     targetAccel,
                                                                                                                                                                                     targetAccelSizeInBytes,
                                                                                                                                                                                     targetHandle)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixAccelCompact_(OptixDeviceContext context,
                                                                                                       CUstream stream,
                                                                                                       OptixTraversableHandle inputHandle,
                                                                                                       CUdeviceptr outputBuffer,
                                                                                                       size_t outputBufferSizeInBytes,
                                                                                                       OptixTraversableHandle *outputHandle) const { OPTIX_CHECK(optixAccelCompact(context,
                                                                                                                                                                                   stream,
                                                                                                                                                                                   inputHandle,
                                                                                                                                                                                   outputBuffer,
                                                                                                                                                                                   outputBufferSizeInBytes,
                                                                                                                                                                                   outputHandle)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixConvertPointerToTraversableHandle_(OptixDeviceContext onDevice,
                                                                                                                            CUdeviceptr pointer,
                                                                                                                            OptixTraversableType traversableType,
                                                                                                                            OptixTraversableHandle *traversableHandle) const { OPTIX_CHECK(optixConvertPointerToTraversableHandle(onDevice,
                                                                                                                                                                                                                                  pointer,
                                                                                                                                                                                                                                  traversableType,
                                                                                                                                                                                                                                  traversableHandle)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixSbtRecordPackHeader_(OptixProgramGroup programGroup, void *sbtRecordHeaderHostPointer) const { OPTIX_CHECK(optixSbtRecordPackHeader(programGroup, sbtRecordHeaderHostPointer)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixLaunch_(OptixPipeline pipeline,
                                                                                                 CUstream stream,
                                                                                                 CUdeviceptr pipelineParams,
                                                                                                 size_t pipelineParamsSize,
                                                                                                 const OptixShaderBindingTable *sbt,
                                                                                                 unsigned int width,
                                                                                                 unsigned int height,
                                                                                                 unsigned int depth) const { OPTIX_CHECK(optixLaunch(pipeline,
                                                                                                                                                     stream,
                                                                                                                                                     pipelineParams,
                                                                                                                                                     pipelineParamsSize,
                                                                                                                                                     sbt,
                                                                                                                                                     width,
                                                                                                                                                     height,
                                                                                                                                                     depth)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserCreate_(OptixDeviceContext context, OptixDenoiserModelKind modelKind, const OptixDenoiserOptions *options, OptixDenoiser *returnHandle) const { OPTIX_CHECK(optixDenoiserCreate(context, modelKind, options, returnHandle)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserCreateWithUserModel_(OptixDeviceContext context, const void *data, size_t dataSizeInBytes, OptixDenoiser *returnHandle) const { OPTIX_CHECK(optixDenoiserCreateWithUserModel(context, data, dataSizeInBytes, returnHandle)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserDestroy_(OptixDenoiser handle) const { OPTIX_CHECK(optixDenoiserDestroy(handle)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserComputeMemoryResources_(const OptixDenoiser handle,
                                                                                                                         unsigned int maximumInputWidth,
                                                                                                                         unsigned int maximumInputHeight,
                                                                                                                         OptixDenoiserSizes *returnSizes) const { OPTIX_CHECK(optixDenoiserComputeMemoryResources(handle,
                                                                                                                                                                                                                  maximumInputWidth,
                                                                                                                                                                                                                  maximumInputHeight,
                                                                                                                                                                                                                  returnSizes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserSetup_(OptixDenoiser denoiser,
                                                                                                        CUstream stream,
                                                                                                        unsigned int inputWidth,
                                                                                                        unsigned int inputHeight,
                                                                                                        CUdeviceptr denoiserState,
                                                                                                        size_t denoiserStateSizeInBytes,
                                                                                                        CUdeviceptr scratch,
                                                                                                        size_t scratchSizeInBytes) const { OPTIX_CHECK(optixDenoiserSetup(denoiser,
                                                                                                                                                                          stream,
                                                                                                                                                                          inputWidth,
                                                                                                                                                                          inputHeight,
                                                                                                                                                                          denoiserState,
                                                                                                                                                                          denoiserStateSizeInBytes,
                                                                                                                                                                          scratch,
                                                                                                                                                                          scratchSizeInBytes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserInvoke_(OptixDenoiser handle,
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
                                                                                                         size_t scratchSizeInBytes) const { OPTIX_CHECK(optixDenoiserInvoke(handle,
                                                                                                                                                                            stream,
                                                                                                                                                                            params,
                                                                                                                                                                            denoiserData,
                                                                                                                                                                            denoiserDataSize,
                                                                                                                                                                            guideLayer,
                                                                                                                                                                            layers,
                                                                                                                                                                            numLayers,
                                                                                                                                                                            inputOffsetX,
                                                                                                                                                                            inputOffsetY,
                                                                                                                                                                            scratch,
                                                                                                                                                                            scratchSizeInBytes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserComputeIntensity_(OptixDenoiser handle,
                                                                                                                   CUstream stream,
                                                                                                                   const OptixImage2D *inputImage,
                                                                                                                   CUdeviceptr outputIntensity,
                                                                                                                   CUdeviceptr scratch,
                                                                                                                   size_t scratchSizeInBytes) const { OPTIX_CHECK(optixDenoiserComputeIntensity(handle,
                                                                                                                                                                                                stream,
                                                                                                                                                                                                inputImage,
                                                                                                                                                                                                outputIntensity,
                                                                                                                                                                                                scratch,
                                                                                                                                                                                                scratchSizeInBytes)); }

result<void, std::shared_ptr<glow::optix::OptixError>> glow::optix::OptixConnector::optixDenoiserComputeAverageColor_(OptixDenoiser handle,
                                                                                                                      CUstream stream,
                                                                                                                      const OptixImage2D *inputImage,
                                                                                                                      CUdeviceptr outputAverageColor,
                                                                                                                      CUdeviceptr scratch,
                                                                                                                      size_t scratchSizeInBytes) const { OPTIX_CHECK(optixDenoiserComputeAverageColor(handle,
                                                                                                                                                                                                      stream,
                                                                                                                                                                                                      inputImage,
                                                                                                                                                                                                      outputAverageColor,
                                                                                                                                                                                                      scratch,
                                                                                                                                                                                                      scratchSizeInBytes)); }