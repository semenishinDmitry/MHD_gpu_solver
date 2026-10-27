cmake_minimum_required(VERSION 3.20)
project(mhd_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MHDCompilerFlags)

# --- Build provenance --------------------------------------------------------
find_package(Git QUIET)
set(MHD_GIT_COMMIT "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MHD_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(MHD_NATIVE_ARCH "Enable -march=native" ON)
option(MHD_ENABLE_LTO "Enable thin LTO for Clang Release" OFF)
option(MHD_ENABLE_SANITIZERS "Enable Address+UB sanitizers (Debug-friendly)" OFF)

set(MHD_BUILDINFO_NATIVE_ARCH "OFF")
if(MHD_NATIVE_ARCH)
    set(MHD_BUILDINFO_NATIVE_ARCH "ON")
endif()
set(MHD_BUILDINFO_SANITIZERS "OFF")
if(MHD_ENABLE_SANITIZERS)
    set(MHD_BUILDINFO_SANITIZERS "ON")
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/include/mhd/build_info.hpp.in
    ${CMAKE_BINARY_DIR}/generated/mhd/build_info.hpp
    @ONLY
)

add_library(mhd_headers INTERFACE)
target_include_directories(mhd_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/generated
)

# --- GPU backends ------------------------------------------------------------
# Metal (Apple GPUs, fp32). Shaders are embedded into the binary and compiled at runtime,
# so only the Command Line Tools are required (no offline `metal` compiler).
if(APPLE)
    set(MHD_ENABLE_METAL_DEFAULT ON)
else()
    set(MHD_ENABLE_METAL_DEFAULT OFF)
endif()
option(MHD_ENABLE_METAL "Build the Metal GPU backend (macOS only)" ${MHD_ENABLE_METAL_DEFAULT})
option(MHD_ENABLE_CUDA "Build the CUDA GPU backend (requires the CUDA toolkit)" OFF)
set(MHD_CUDA_ARCHITECTURES "70;80;86;89;90" CACHE STRING "CUDA architectures for the CUDA backend")

if(MHD_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "MHD_ENABLE_METAL requires macOS")
    endif()
    enable_language(OBJCXX)
    set(MHD_METAL_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/mhd_kernels.metal)
    file(READ ${MHD_METAL_SHADER} MHD_METAL_SOURCE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mhd_metal_source.hpp.in
                   ${CMAKE_BINARY_DIR}/generated/mhd_metal_source.hpp @ONLY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MHD_METAL_SHADER})

    add_library(mhd_metal STATIC src/metal/metal_solver.mm)
    set_target_properties(mhd_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_features(mhd_metal PRIVATE cxx_std_20)
    target_include_directories(mhd_metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/generated)
    target_compile_options(mhd_metal PRIVATE -fobjc-arc -O2 -Wall -Wextra)
    target_compile_definitions(mhd_metal PUBLIC MHD_HAS_METAL=1)
    target_link_libraries(mhd_metal PUBLIC "-framework Metal" "-framework Foundation")
    target_link_libraries(mhd_headers INTERFACE mhd_metal)
endif()

if(MHD_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(mhd_cuda STATIC src/cuda/cuda_solver.cu)
    set_target_properties(mhd_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${MHD_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
    target_include_directories(mhd_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(mhd_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_definitions(mhd_cuda PUBLIC MHD_HAS_CUDA=1)
    target_link_libraries(mhd_cuda PUBLIC CUDA::cudart)
    target_link_libraries(mhd_headers INTERFACE mhd_cuda)
    if(TARGET mhd_metal)
        target_compile_definitions(mhd_metal PRIVATE MHD_HAS_CUDA=1)
        target_compile_definitions(mhd_cuda PRIVATE MHD_HAS_METAL=1)
    endif()
endif()

add_executable(mhd_solver_app main.cpp)
set_target_properties(mhd_solver_app PROPERTIES OUTPUT_NAME mhd_solver)
target_link_libraries(mhd_solver_app PRIVATE mhd_headers)
mhd_apply_flags(mhd_solver_app)

add_executable(mhd_verify tools/mhd_verify.cpp)
target_link_libraries(mhd_verify PRIVATE mhd_headers)
mhd_apply_flags(mhd_verify)

# Speed / accuracy comparison of CPU (scalar, SIMD, threads) vs CUDA / Metal backends.
add_executable(mhd_backend_compare tools/mhd_backend_compare.cpp)
target_link_libraries(mhd_backend_compare PRIVATE mhd_headers)
mhd_apply_flags(mhd_backend_compare)

option(MHD_BUILD_TESTS "Build unit tests" ON)
if(MHD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    # Accuracy part of the GPU gate (exit 0 / skipped when no GPU is present). The speed part
    # runs only in .github/workflows/gpu.yml on real GPU runners.
    add_test(NAME GpuGate.Accuracy COMMAND mhd_backend_compare --gate --no-speed)
    set_tests_properties(GpuGate.Accuracy PROPERTIES LABELS "gpu" TIMEOUT 600)
endif()

option(MHD_BUILD_PYTHON "Build pybind11 Python module" ON)
if(MHD_BUILD_PYTHON)
    add_subdirectory(python)
endif()

option(MHD_BUILD_BENCHMARKS "Build Google Benchmark targets" OFF)
if(MHD_BUILD_BENCHMARKS)
    enable_testing()
    add_subdirectory(benchmarks)
endif()

# --- Developer convenience targets ------------------------------------------
find_program(MHD_CLANG_FORMAT NAMES clang-format)
if(MHD_CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/format.sh format
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format"
    )
    add_custom_target(format-check
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/format.sh check
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Checking clang-format"
    )
endif()

find_program(MHD_CLANG_TIDY NAMES clang-tidy)
if(MHD_CLANG_TIDY)
    add_custom_target(lint
        COMMAND ${MHD_CLANG_TIDY} -p ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR}/main.cpp
            ${CMAKE_SOURCE_DIR}/tools/mhd_verify.cpp
            ${CMAKE_SOURCE_DIR}/python/mhd_pybind.cpp
            --quiet
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy on selected TUs"
    )
endif()

add_custom_target(verify-alfven
    COMMAND $<TARGET_FILE:mhd_verify> --convergence-alfven alfven_convergence.csv
    DEPENDS mhd_verify
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Alfvén wave convergence study"
)

message(STATUS "MHD solver: C++${CMAKE_CXX_STANDARD}, build=${CMAKE_BUILD_TYPE}, git=${MHD_GIT_COMMIT}")
message(STATUS "  MHD_NATIVE_ARCH=${MHD_BUILDINFO_NATIVE_ARCH} MHD_ENABLE_SANITIZERS=${MHD_BUILDINFO_SANITIZERS}")
message(STATUS "  MHD_ENABLE_METAL=${MHD_ENABLE_METAL} MHD_ENABLE_CUDA=${MHD_ENABLE_CUDA}")
