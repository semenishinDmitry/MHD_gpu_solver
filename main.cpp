#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "state/state.hpp"
#include "state/state_field.hpp"
#include "initial_condition/inititial_condition.hpp"
#include "physics_config/mhd_config.hpp"
#include <iostream>

int main() {
    // Define grid parameters
    int nx = 100;
    int ny = 100;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;

    // Create a grid
    Grid2D grid(nx, ny, x_min, x_max, y_min, y_max);

    // Create a state field
    StateField state_field(grid.get_size_x(), grid.get_size_y());

    // Initialize the state field with a uniform initial condition
    initialize_state_field(state_field, grid, InitialConditionType::Uniform);
    std::cout<<grid.get_dx()<<" "<<grid.get_dy()<<std::endl;

    // Further simulation code would go here...

    return 0;
}