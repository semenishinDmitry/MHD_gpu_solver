#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"

TEST(BoundaryCondition2D, StoresAxisTypes) {
    BoundaryCondition2D bc(
        BoundaryConditionType::Periodic,
        BoundaryConditionType::Reflective);

    EXPECT_EQ(bc.type_x, BoundaryConditionType::Periodic);
    EXPECT_EQ(bc.type_y, BoundaryConditionType::Reflective);
}

TEST(BoundaryCondition2D, SupportsOutflow) {
    BoundaryCondition2D bc(
        BoundaryConditionType::Outflow,
        BoundaryConditionType::Outflow);

    EXPECT_EQ(bc.type_x, BoundaryConditionType::Outflow);
    EXPECT_EQ(bc.type_y, BoundaryConditionType::Outflow);
}
