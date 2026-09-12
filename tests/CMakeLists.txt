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
    test_nonideal.cpp
    test_field_ops.cpp
    test_physics_extra.cpp
    test_solver_api.cpp
    test_performance_invariants.cpp
    test_regression_golden.cpp
)

target_link_libraries(mhd_tests PRIVATE
    mhd_headers
    GTest::gtest_main
)
mhd_apply_flags(mhd_tests)

target_compile_definitions(mhd_tests PRIVATE
    MHD_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/goldens"
)

include(GoogleTest)
gtest_discover_tests(mhd_tests)
