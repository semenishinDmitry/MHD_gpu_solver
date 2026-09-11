#pragma once

enum class BoundaryConditionType {
    Periodic,
    Reflective,
    Outflow
};

struct BoundaryCondition2D {
    BoundaryConditionType type_x;
    BoundaryConditionType type_y;

    BoundaryCondition2D(BoundaryConditionType type_x_, BoundaryConditionType type_y_)
        : type_x(type_x_), type_y(type_y_) {}
};
