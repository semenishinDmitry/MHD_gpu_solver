include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(mhd_tests
    test_grid.cpp
    test_state.cpp
    test_state_field.cpp
    test_boundary_conditions.cpp
    test_mhd_config.cpp
    test_initial_condition.cpp
    test_mhd_operator.cpp
    test_time_integrator.cpp
    test_muscl_glm.cpp
)

target_link_libraries(mhd_tests PRIVATE
    mhd_headers
    GTest::gtest_main
)

target_compile_options(mhd_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

include(GoogleTest)
gtest_discover_tests(mhd_tests)
