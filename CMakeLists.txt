cmake_minimum_required(VERSION 3.22)
project(qwen_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nlohmann_json)

find_package(CUDAToolkit REQUIRED)
find_package(BLAS REQUIRED)

add_library(infer_core
  src/archive.cpp
  src/buffer.cpp
  src/model_config.cpp
  src/tokenizer.cpp
  src/cpu_ops.cpp
  src/qwen2.cpp
  src/cuda/kernels.cu)

target_include_directories(infer_core PUBLIC include)
target_link_libraries(infer_core
  PUBLIC nlohmann_json::nlohmann_json CUDA::cudart CUDA::cublas ${BLAS_LIBRARIES})
target_compile_options(infer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
target_compile_definitions(infer_core PUBLIC INFER_WITH_CUDA=1)
set_target_properties(infer_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(TARGET CUDA::nvToolsExt)
  target_link_libraries(infer_core PUBLIC CUDA::nvToolsExt)
  target_compile_definitions(infer_core PUBLIC INFER_WITH_NVTX=1)
endif()

add_executable(llm_infer src/main.cpp)
target_link_libraries(llm_infer PRIVATE infer_core)


add_executable(bf16_linear_benchmark tools/benchmark_bf16_linear.cpp)
target_link_libraries(bf16_linear_benchmark PRIVATE infer_core)

option(INFER_BUILD_TESTS "Build unit tests" ON)
if(INFER_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_executable(infer_tests
    tests/test_core.cpp
    tests/test_ops.cpp
    tests/test_cuda.cpp
    tests/test_qwen2.cpp)
  target_link_libraries(infer_tests PRIVATE infer_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(infer_tests)

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME BenchmarkRunner.SchemaSmoke
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/run_benchmark.py --self-test)
  add_test(NAME ExportModel.BFloat16SelfTest
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/export_model.py --self-test)
endif()
