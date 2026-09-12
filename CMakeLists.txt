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

add_executable(mhd_solver_app main.cpp)
set_target_properties(mhd_solver_app PROPERTIES OUTPUT_NAME mhd_solver)
target_link_libraries(mhd_solver_app PRIVATE mhd_headers)
mhd_apply_flags(mhd_solver_app)

add_executable(mhd_verify tools/mhd_verify.cpp)
target_link_libraries(mhd_verify PRIVATE mhd_headers)
mhd_apply_flags(mhd_verify)

option(MHD_BUILD_TESTS "Build unit tests" ON)
if(MHD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
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
