#include <gtest/gtest.h>

#include "physics_config/mhd_config.hpp"

TEST(MHDConfig, DefaultValues) {
    MHDConfig config;

    EXPECT_DOUBLE_EQ(config.gamma, 5.0 / 3.0);
    EXPECT_DOUBLE_EQ(config.Omega, 0.0);
    EXPECT_DOUBLE_EQ(config.q, 0.0);
    EXPECT_DOUBLE_EQ(config.eta_ohm, 0.0);
    EXPECT_DOUBLE_EQ(config.eta_hall, 0.0);
    EXPECT_DOUBLE_EQ(config.eta_ambip, 0.0);
}

TEST(MHDConfig, ParameterizedConstructor) {
    MHDConfig config(1.4, 1.0, 1.5, 0.01, 0.02, 0.03);

    EXPECT_DOUBLE_EQ(config.gamma, 1.4);
    EXPECT_DOUBLE_EQ(config.Omega, 1.0);
    EXPECT_DOUBLE_EQ(config.q, 1.5);
    EXPECT_DOUBLE_EQ(config.eta_ohm, 0.01);
    EXPECT_DOUBLE_EQ(config.eta_hall, 0.02);
    EXPECT_DOUBLE_EQ(config.eta_ambip, 0.03);
}
