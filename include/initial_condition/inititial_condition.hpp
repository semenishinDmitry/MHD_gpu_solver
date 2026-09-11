#pragma once

#include "state/state.hpp"
#include "grid/grid.hpp"
#include "state/state_field.hpp"
#include <cmath>
#include <stdexcept>
#include <numbers>
using InitialCondition2D = PrimitiveState2D (*)(const double x, const double y);

enum class InitialConditionType {
    Uniform,
    SineWave,
    BlastWave
};

PrimitiveState2D uniform_initial_condition(const double x, const double y){
    return PrimitiveState2D(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0); // rho = 1.0 and pressure = 1.0
}

PrimitiveState2D sine_wave_initial_condition(const double x, const double y){
    double rho = 1.0 + 0.1 * std::sin(2.0 * std::numbers::pi * x);
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    double E = 1.0;
    double Bx = 0.0;
    double By = 0.0;
    double Bz = 0.0;

    return PrimitiveState2D(rho, vx, vy, vz, E, Bx, By, Bz);
}

PrimitiveState2D blast_wave_initial_condition(const double x, const double y){
    double rho = 1.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    double E = 1.0;
    double Bx = 1.0;
    double By = 0.0;
    double Bz = 0.0;

    return PrimitiveState2D(rho, vx, vy, vz, E, Bx, By, Bz);
}

void initialize_state_field(StateField& state_field, const Grid2D& grid, InitialConditionType initial_condition_type) {
    InitialCondition2D initial_condition;

    switch (initial_condition_type) {
        case InitialConditionType::Uniform:
            initial_condition = uniform_initial_condition;
            break;
        case InitialConditionType::SineWave:
            initial_condition = sine_wave_initial_condition;
            break;
        case InitialConditionType::BlastWave:
            initial_condition = blast_wave_initial_condition;
            break;
        default:
            throw std::invalid_argument("Unknown initial condition type");
    }

    //TODO: optimze accesses
    for (int j = 0; j < grid.ny; ++j) {
        for (int i = 0; i < grid.nx; ++i) {
            double x = grid.get_x(i);
            double y = grid.get_y(j);
            PrimitiveState2D primitive_state = initial_condition(x, y);

            int idx = state_field.index(i, j);
            state_field.rho[idx] = primitive_state.rho; // Assuming get_vx() returns density
            state_field.mx[idx] = primitive_state.rho * primitive_state.get_vx(); // Assuming momentum in x-direction
            state_field.my[idx] = primitive_state.rho * primitive_state.get_vy(); // Assuming momentum in y-direction
            state_field.mz[idx] = primitive_state.rho * primitive_state.get_vz(); // Assuming momentum in z-direction
            state_field.energy[idx] = primitive_state.E; // Assuming get_vx() returns energy
            state_field.bx[idx] = primitive_state.Bx; // Assuming get_vx() returns magnetic field in x-direction
            state_field.by[idx] = primitive_state.By; // Assuming get_vy() returns magnetic field in y-direction
            state_field.bz[idx] = primitive_state.Bz; // Assuming get_vz() returns magnetic field in z-direction
        }
    }
}

