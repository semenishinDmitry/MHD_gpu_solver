cmake_minimum_required(VERSION 3.20)
project(mhd_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prefer Clang/LLVM via: cmake -DCMAKE_CXX_COMPILER=clang++ ...
# or use scripts/setup_and_build.sh|.ps1

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MHDCompilerFlags)

add_library(mhd_headers INTERFACE)
target_include_directories(mhd_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(mhd_solver_app main.cpp)
set_target_properties(mhd_solver_app PROPERTIES OUTPUT_NAME mhd_solver)
target_link_libraries(mhd_solver_app PRIVATE mhd_headers)
mhd_apply_flags(mhd_solver_app)

option(MHD_BUILD_TESTS "Build unit tests" ON)
if(MHD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(MHD_BUILD_PYTHON "Build pybind11 Python module" ON)
if(MHD_BUILD_PYTHON)
    add_subdirectory(python)
endif()
