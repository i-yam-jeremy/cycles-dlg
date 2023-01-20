project(DemandLoadingGeometry LANGUAGES CXX CUDA VERSION 1.0)

include(FindCUDAToolkit)
set(OPTIX_INCLUDE_DIR "/usr/local/optix/include")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake")

find_package(glm CONFIG REQUIRED)
find_package(rmm REQUIRED)

find_package(OptiX 7.5.0)

add_library(DemandLoadingGeomtry_ptxShaders_headerDependencies INTERFACE)
target_include_directories(DemandLoadingGeomtry_ptxShaders_headerDependencies INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${OPTIX_INCLUDE_DIR}")

file(GLOB_RECURSE PTX_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/trace/GeometryDeviceContext.cu")

embed_ptx(
    OUTPUT_TARGET
        DemandLoadingGeometry_ptxShaders 
    PTX_LINK_LIBRARIES
        DemandLoadingGeomtry_ptxShaders_headerDependencies
        ${CUDA_CUDA_LIBRARY}
        glm::glm
    SOURCES
        ${PTX_SOURCES})

file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
list(FILTER SOURCES EXCLUDE REGEX "GeometryDeviceContext\\.cu")
add_library(DemandLoadingGeometry "${SOURCES}")
target_compile_definitions(DemandLoadingGeometry PUBLIC GLM_FORCE_CUDA)
set_property(TARGET DemandLoadingGeometry PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET DemandLoadingGeometry PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET DemandLoadingGeometry PROPERTY CUDA_ARCHITECTURES OFF)
target_link_libraries(DemandLoadingGeometry PRIVATE 
    ${CUDA_CUDA_LIBRARY}
    CUDA::cuda_driver
    dl

    rmm::rmm

    DemandLoadingGeometry_ptxShaders

    glm::glm

    cuda_check_macros
    device_memory_wrapper
    result_type
)
target_link_libraries(DemandLoadingGeometry PUBLIC     glm)
target_include_directories(DemandLoadingGeometry PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_include_directories(DemandLoadingGeometry PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)