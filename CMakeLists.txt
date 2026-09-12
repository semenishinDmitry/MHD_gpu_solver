cmake_minimum_required(VERSION 3.20)
project(mhd_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mhd_headers INTERFACE)
target_include_directories(mhd_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(mhd_solver
    main.cpp
)
target_link_libraries(mhd_solver PRIVATE mhd_headers)
target_compile_options(mhd_solver PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Release>>:-O3>
)

option(MHD_BUILD_TESTS "Build unit tests" ON)
if(MHD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
