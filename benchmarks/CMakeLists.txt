include(FetchContent)

option(MHD_BUILD_BENCHMARKS "Build Google Benchmark targets" OFF)

if(NOT MHD_BUILD_BENCHMARKS)
    return()
endif()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(mhd_bench bench_mhd.cpp)
target_link_libraries(mhd_bench PRIVATE mhd_headers benchmark::benchmark)
mhd_apply_flags(mhd_bench)

# Optional ctest wrapper (not part of default unit-test gate; labeled "benchmark").
add_test(NAME Bench.OrszagTangQuick COMMAND mhd_bench --benchmark_filter=BM_OrszagTang_Run/32 --benchmark_min_time=0.05s)
set_tests_properties(Bench.OrszagTangQuick PROPERTIES LABELS "benchmark")
