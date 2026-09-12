include(FetchContent)

option(MHD_FETCH_PYBIND11 "Fetch pybind11 if not found" ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if(MHD_FETCH_PYBIND11)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(mhd_solver mhd_pybind.cpp)
target_link_libraries(mhd_solver PRIVATE mhd_headers)
mhd_apply_flags(mhd_solver)

# Install into build/python for easy PYTHONPATH usage
set_target_properties(mhd_solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

add_custom_command(TARGET mhd_solver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python"
    COMMENT "Python module output: ${CMAKE_BINARY_DIR}/python"
)
