#include <gtest/gtest.h>

#include "state/state_field.hpp"

TEST(StateField, AllocatesAllArrays) {
    StateField field(3, 4);

    EXPECT_EQ(field.nx, 3);
    EXPECT_EQ(field.ny, 4);

    const std::size_t expected = 12;
    EXPECT_EQ(field.rho.size(), expected);
    EXPECT_EQ(field.mx.size(), expected);
    EXPECT_EQ(field.my.size(), expected);
    EXPECT_EQ(field.mz.size(), expected);
    EXPECT_EQ(field.energy.size(), expected);
    EXPECT_EQ(field.bx.size(), expected);
    EXPECT_EQ(field.by.size(), expected);
    EXPECT_EQ(field.bz.size(), expected);
}

TEST(StateField, ArraysStartAtZero) {
    StateField field(2, 2);

    for (double value : field.rho) {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
    for (double value : field.energy) {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
}

TEST(StateField, IndexIsRowMajorInY) {
    StateField field(5, 3);

    EXPECT_EQ(field.index(0, 0), 0);
    EXPECT_EQ(field.index(4, 0), 4);
    EXPECT_EQ(field.index(0, 1), 5);
    EXPECT_EQ(field.index(2, 2), 12);
}

TEST(StateField, IndexAllowsWriteAndRead) {
    StateField field(4, 2);
    const int idx = field.index(3, 1);

    field.rho[idx] = 1.25;
    field.mx[idx] = 2.5;

    EXPECT_DOUBLE_EQ(field.rho[idx], 1.25);
    EXPECT_DOUBLE_EQ(field.mx[idx], 2.5);
}
