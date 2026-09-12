#include <gtest/gtest.h>

#include "mhd/mhd_types.hpp"
#include "state/state_field.hpp"

TEST(StateField, AllocatesAllArraysIncludingPsi)
{
    StateField field(3, 4);
    EXPECT_EQ(field.rho.size(), 12u);
    EXPECT_EQ(field.psi.size(), 12u);
}

TEST(StateField, GetSetStateRoundTrip)
{
    StateField field(4, 2);
    const MHDState U{1.1, 0.2, 0.3, 0.4, 5.5, 0.6, 0.7, 0.8, 0.9};
    field.set_state(3, 1, U);
    const MHDState V = field.get_state(3, 1);

    EXPECT_DOUBLE_EQ(V.rho, U.rho);
    EXPECT_DOUBLE_EQ(V.psi, U.psi);
    EXPECT_DOUBLE_EQ(V.energy, U.energy);
}

TEST(StateField, CopyCellIncludesPsi)
{
    StateField field(3, 3);
    field.set_state(1, 1, MHDState{2.0, 1.0, 0.0, 0.0, 3.0, 0.1, 0.0, 0.0, 0.25});
    field.copy_cell(0, 0, 1, 1);
    EXPECT_DOUBLE_EQ(field.get_state(0, 0).psi, 0.25);
}
