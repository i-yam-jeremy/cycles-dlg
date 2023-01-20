add_executable(example_basic "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(example_basic PRIVATE DemandLoadingGeometry)
