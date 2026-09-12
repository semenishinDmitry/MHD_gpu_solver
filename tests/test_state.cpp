#include <gtest/gtest.h>

#include "state/state.hpp"

TEST(State2D, DefaultConstructible)
{
    EXPECT_NO_THROW(State2D{});
}

TEST(State2D, VelocityFromMomentum)
{
    State2D state(2.0, 4.0, -6.0, 8.0, 1.0, 0.1, 0.2, 0.3);

    EXPECT_DOUBLE_EQ(state.get_vx(), 2.0);
    EXPECT_DOUBLE_EQ(state.get_vy(), -3.0);
    EXPECT_DOUBLE_EQ(state.get_vz(), 4.0);
}

TEST(PrimitiveState2D, DefaultIsZero)
{
    PrimitiveState2D state;
    EXPECT_DOUBLE_EQ(state.rho, 0.0);
    EXPECT_DOUBLE_EQ(state.vx, 0.0);
    EXPECT_DOUBLE_EQ(state.E, 0.0);
}

TEST(PrimitiveState2D, ConstructorAndGetters)
{
    PrimitiveState2D state(1.5, 0.1, 0.2, 0.3, 2.0, 0.4, 0.5, 0.6);

    EXPECT_DOUBLE_EQ(state.rho, 1.5);
    EXPECT_DOUBLE_EQ(state.get_vx(), 0.1);
    EXPECT_DOUBLE_EQ(state.get_vy(), 0.2);
    EXPECT_DOUBLE_EQ(state.get_vz(), 0.3);
}
