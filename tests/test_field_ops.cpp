#include <gtest/gtest.h>

#include "mhd/field_ops.hpp"
#include "mhd/mhd_types.hpp"
#include "state/state_field.hpp"

TEST(FieldOps, XpayMatchesManual)
{
    StateField x(4, 3);
    StateField y(4, 3);
    StateField out(4, 3);

    for (std::size_t i = 0; i < x.rho.size(); ++i) {
        x.rho[i] = 1.0 + static_cast<double>(i);
        y.rho[i] = 0.5 * static_cast<double>(i);
        x.psi[i] = 2.0;
        y.psi[i] = 3.0;
    }

    field_xpay(out, x, 2.0, y);

    for (std::size_t i = 0; i < out.rho.size(); ++i) {
        EXPECT_NEAR(out.rho[i], x.rho[i] + 2.0 * y.rho[i], 1e-14);
        EXPECT_NEAR(out.psi[i], 2.0 + 2.0 * 3.0, 1e-14);
    }
}

TEST(FieldOps, SspRk2CombineFormula)
{
    StateField U(2, 2);
    StateField U_star(2, 2);
    StateField rhs(2, 2);

    U.rho[0] = 1.0;
    U_star.rho[0] = 3.0;
    rhs.rho[0] = 4.0;
    const double dt = 0.5;

    field_ssp_rk2_combine(U, U_star, rhs, dt);
    // 0.5*(1+3) + 0.5*0.5*4 = 2 + 1 = 3
    EXPECT_NEAR(U.rho[0], 3.0, 1e-14);
}

TEST(FieldOps, CopyPreservesPsi)
{
    StateField a(3, 3);
    StateField b(3, 3);
    a.psi[4] = 1.25;
    field_copy(b, a);
    EXPECT_DOUBLE_EQ(b.psi[4], 1.25);
}
