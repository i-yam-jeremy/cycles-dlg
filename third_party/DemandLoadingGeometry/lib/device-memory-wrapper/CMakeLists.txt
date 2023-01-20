file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
add_library(device_memory_wrapper "${SOURCES}")
target_link_libraries(device_memory_wrapper PUBLIC 
    CUDA::cudart
    CUDA::curand
    rmm::rmm
    cuda_check_macros
)

target_include_directories(device_memory_wrapper PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/lib"
    "${OPTIX_INCLUDE_DIR}"
)
target_include_directories(device_memory_wrapper PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)