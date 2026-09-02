cmake_minimum_required(VERSION 3.25)
project(loom_blas_research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "$ENV{ROCM_PATH}")
endif()
if(NOT ROCM_PATH)
  message(FATAL_ERROR "Set ROCM_PATH explicitly (for example, /opt/rocm)")
endif()

add_executable(blas-probe research/tools/blas_probe.cpp)
target_compile_definitions(blas-probe PRIVATE
  __HIP_PLATFORM_AMD__
  ROCBLAS_BETA_FEATURES_API)
target_include_directories(blas-probe PRIVATE "${ROCM_PATH}/include")
target_link_directories(blas-probe PRIVATE "${ROCM_PATH}/lib")
target_link_libraries(blas-probe PRIVATE amdhip64 hipblaslt rocblas)
target_compile_options(blas-probe PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(blas-probe PROPERTIES
  BUILD_RPATH "${ROCM_PATH}/lib"
  INSTALL_RPATH "${ROCM_PATH}/lib")

add_executable(hipblaslt-route-probe research/tools/hipblaslt_route_probe.cpp)
target_compile_definitions(hipblaslt-route-probe PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(hipblaslt-route-probe PRIVATE "${ROCM_PATH}/include")
target_link_directories(hipblaslt-route-probe PRIVATE "${ROCM_PATH}/lib")
target_link_libraries(hipblaslt-route-probe PRIVATE amdhip64 hipblaslt)
target_compile_options(hipblaslt-route-probe PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(hipblaslt-route-probe PROPERTIES
  BUILD_RPATH "${ROCM_PATH}/lib"
  INSTALL_RPATH "${ROCM_PATH}/lib")

add_executable(loom-tile-probe research/tools/loom_tile_probe.cpp)
target_compile_definitions(loom-tile-probe PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(loom-tile-probe PRIVATE "${ROCM_PATH}/include")
target_link_directories(loom-tile-probe PRIVATE "${ROCM_PATH}/lib")
target_link_libraries(loom-tile-probe PRIVATE amdhip64)
target_compile_options(loom-tile-probe PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(loom-tile-probe PROPERTIES
  BUILD_RPATH "${ROCM_PATH}/lib"
  INSTALL_RPATH "${ROCM_PATH}/lib")
