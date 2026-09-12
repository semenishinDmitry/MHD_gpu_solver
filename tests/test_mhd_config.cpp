#include <gtest/gtest.h>

#include "physics_config/mhd_config.hpp"

TEST(MHDConfig, DefaultValues)
{
    MHDConfig config;

    EXPECT_DOUBLE_EQ(config.gamma, 5.0 / 3.0);
    EXPECT_DOUBLE_EQ(config.Omega, 0.0);
    EXPECT_DOUBLE_EQ(config.q, 0.0);
    EXPECT_FALSE(config.nonideal.any());
    EXPECT_DOUBLE_EQ(config.eta_ohm(), 0.0);
    EXPECT_DOUBLE_EQ(config.eta_hall(), 0.0);
    EXPECT_DOUBLE_EQ(config.eta_ambip(), 0.0);
}

TEST(MHDConfig, ParameterizedConstructorEnablesNonIdeal)
{
    MHDConfig config(1.4, 1.0, 1.5, 0.01, 0.02, 0.03);

    EXPECT_DOUBLE_EQ(config.gamma, 1.4);
    EXPECT_TRUE(config.nonideal.ohmic);
    EXPECT_TRUE(config.nonideal.hall);
    EXPECT_TRUE(config.nonideal.ambipolar);
    EXPECT_DOUBLE_EQ(config.eta_ohm(), 0.01);
    EXPECT_DOUBLE_EQ(config.eta_hall(), 0.02);
    EXPECT_DOUBLE_EQ(config.eta_ambip(), 0.03);
}

TEST(NonIdealConfig, IndependentEnableDisable)
{
    NonIdealConfig ni = NonIdealConfig::ideal();
    EXPECT_FALSE(ni.any());

    ni.enable_ohmic(0.1);
    EXPECT_TRUE(ni.any());
    EXPECT_TRUE(ni.ohmic);
    EXPECT_FALSE(ni.hall);

    ni.enable_hall(0.01).enable_ambipolar(0.02);
    EXPECT_TRUE(ni.hall);
    EXPECT_TRUE(ni.ambipolar);

    ni.disable_ohmic();
    EXPECT_FALSE(ni.ohmic);
    EXPECT_TRUE(ni.any());

    EXPECT_DOUBLE_EQ(ni.max_diffusivity(), 0.02);
}
