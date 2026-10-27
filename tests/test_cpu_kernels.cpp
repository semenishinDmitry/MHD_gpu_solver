#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "initial_condition/inititial_condition.hpp"
#include "mhd/cfl.hpp"
#include "mhd/hll.hpp"
#include "mhd/kernels/muscl_hll.hpp"
#include "mhd/mask.hpp"
#include "mhd/parallel.hpp"
#include "mhd/reconstruction.hpp"
#include "mhd/rhs.hpp"
#include "mhd/rhs_reference.hpp"
#include "mhd/simd.hpp"
#include "mhd/solver_api.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Optimized CPU path (row-fused, SIMD, multithreaded) vs the original scalar implementation.

namespace {

// Restores process-wide execution settings after each test.
class ExecGuard {
  public:
    ExecGuard() : simd_(mhd::simd_enabled()), min_cells_(mhd::parallel_min_cells()) {}
    ~ExecGuard()
    {
        mhd::set_simd_enabled(simd_);
        mhd::set_parallel_min_cells(min_cells_);
        mhd::set_num_threads(0);
    }

  private:
    bool simd_;
    long long min_cells_;
};

double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    double scale = 0.0;
    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        scale = std::max(scale, std::abs(b[i]));
        diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    return scale > 0.0 ? diff / scale : diff;
}

double max_rhs_diff(const StateField& a, const StateField& b)
{
    double d = 0.0;
    d = std::max(d, max_rel_diff(a.rho, b.rho));
    d = std::max(d, max_rel_diff(a.mx, b.mx));
    d = std::max(d, max_rel_diff(a.my, b.my));
    d = std::max(d, max_rel_diff(a.mz, b.mz));
    d = std::max(d, max_rel_diff(a.energy, b.energy));
    d = std::max(d, max_rel_diff(a.bx, b.bx));
    d = std::max(d, max_rel_diff(a.by, b.by));
    d = std::max(d, max_rel_diff(a.bz, b.bz));
    d = std::max(d, max_rel_diff(a.psi, b.psi));
    return d;
}

bool bitwise_equal(const StateField& a, const StateField& b)
{
    return a.rho == b.rho && a.mx == b.mx && a.my == b.my && a.mz == b.mz && a.energy == b.energy &&
           a.bx == b.bx && a.by == b.by && a.bz == b.bz && a.psi == b.psi;
}

struct Case {
    std::string ic;
    SlopeLimiter limiter;
    NonIdealConfig nonideal;
    bool with_mask;
};

// Grid with an odd interior width so every row has a scalar tail after the vector body.
void run_rhs_case(const Case& c)
{
    const Grid2D grid(37, 29, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    initialize_state_field(U, grid, parse_initial_condition_name(c.ic), 5.0 / 3.0);
    // Non-zero psi so the GLM terms are exercised.
    for (std::size_t k = 0; k < U.psi.size(); ++k) {
        U.psi[k] = 1e-2 * std::sin(0.37 * static_cast<double>(k));
    }
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);
    apply_boundary_conditions(U, grid, bc);

    FluidMask mask(grid.nx, grid.ny, true);
    if (c.with_mask) {
        for (int j = 10; j < 18; ++j) {
            for (int i = 12; i < 20; ++i) {
                mask.fluid[static_cast<std::size_t>(mask.index(i, j))] = 0;
            }
        }
        apply_solid_wall_fill(U, grid, mask, 5.0 / 3.0);
    }
    const FluidMask* mask_ptr = c.with_mask ? &mask : nullptr;

    const double c_h = compute_cfl_dt(U, grid, 5.0 / 3.0, 0.4, c.nonideal, mask_ptr).c_h;

    StateField ref(U.nx, U.ny);
    compute_rhs_reference(U, ref, grid, 5.0 / 3.0, c_h, 0.1, c.limiter, c.nonideal, mask_ptr);

    for (bool simd : {false, true}) {
        mhd::set_simd_enabled(simd);
        StateField out(U.nx, U.ny);
        RHSWorkspace work(U.nx, U.ny);
        compute_rhs(U, out, grid, work, 5.0 / 3.0, c_h, 0.1, c.limiter, c.nonideal, mask_ptr);
        EXPECT_LT(max_rhs_diff(out, ref), 1e-12)
            << c.ic << " simd=" << simd << " limiter=" << static_cast<int>(c.limiter);
    }
}

} // namespace

TEST(CpuKernels, BackendReportsWidth)
{
    EXPECT_GE(mhd::simd::kWidth, 1);
    EXPECT_NE(std::string(mhd::simd::kBackendName), "");
}

TEST(CpuKernels, LimitersMatchScalarReference)
{
    const double vals[] = {-2.0, -0.5, -1e-300, 0.0, 1e-300, 0.3, 1.0, 4.0};
    for (double a : vals) {
        for (double b : vals) {
            EXPECT_EQ(mhd::kernels::minmod(a, b), minmod(a, b)) << a << " " << b;
            EXPECT_EQ(mhd::kernels::mc_limiter(a, b), mc_limiter(a, b)) << a << " " << b;
        }
    }
}

TEST(CpuKernels, VectorLimiterMatchesScalarLanes)
{
    using V = mhd::simd::vdouble;
    constexpr int w = mhd::simd::kWidth;
    std::vector<double> a(w);
    std::vector<double> b(w);
    std::vector<double> out(w);
    const double vals[] = {-2.0, -0.5, 0.0, 0.3, 1.0, 4.0};
    for (double a0 : vals) {
        for (double b0 : vals) {
            for (int k = 0; k < w; ++k) {
                a[static_cast<std::size_t>(k)] = a0 * (1.0 + 0.1 * k);
                b[static_cast<std::size_t>(k)] = b0 * (1.0 - 0.07 * k);
            }
            const V r = mhd::kernels::mc_limiter(mhd::simd::load<V>(a.data()),
                                                 mhd::simd::load<V>(b.data()));
            mhd::simd::store(out.data(), r);
            for (int k = 0; k < w; ++k) {
                const auto s = static_cast<std::size_t>(k);
                EXPECT_EQ(out[s], mc_limiter(a[s], b[s]));
            }
        }
    }
}

TEST(CpuKernels, FaceFluxMatchesHllReferenceBothDirections)
{
    const double gamma = 5.0 / 3.0;
    const double c_h = 2.5;
    const MHDPrimitive Wmm{1.0, 0.3, -0.2, 0.1, 0.9, 0.7, -0.4, 0.2, 0.01};
    const MHDPrimitive Wm{1.1, 0.25, -0.1, 0.05, 1.0, 0.75, -0.35, 0.25, 0.02};
    const MHDPrimitive W0{0.8, 0.1, 0.05, -0.1, 0.7, 0.8, -0.3, 0.1, -0.01};
    const MHDPrimitive Wp{0.9, -0.1, 0.1, 0.0, 0.6, 0.6, -0.2, 0.3, 0.0};

    auto rot = [](const MHDPrimitive& W, bool y) {
        mhd::kernels::PrimN<double> P{W.rho,      y ? W.vy : W.vx, y ? W.vx : W.vy, W.vz,
                                      W.pressure, y ? W.by : W.bx, y ? W.bx : W.by, W.bz,
                                      W.psi};
        return P;
    };

    for (bool y : {false, true}) {
        MHDPrimitive WL{};
        MHDPrimitive WR{};
        muscl_interface_x(Wmm, Wm, W0, Wp, SlopeLimiter::MC, WL, WR);
        const MHDFlux ref = y ? hll_flux_y(WL, WR, gamma, c_h) : hll_flux_x(WL, WR, gamma, c_h);

        const auto F = mhd::kernels::muscl_hll_face<SlopeLimiter::MC>(
            rot(Wmm, y), rot(Wm, y), rot(W0, y), rot(Wp, y), gamma, 1.0 / (gamma - 1.0), c_h);
        const double tol = 1e-13;
        EXPECT_NEAR(F.rho, ref.rho, tol);
        EXPECT_NEAR(y ? F.mt : F.mn, ref.mx, tol);
        EXPECT_NEAR(y ? F.mn : F.mt, ref.my, tol);
        EXPECT_NEAR(F.mz, ref.mz, tol);
        EXPECT_NEAR(F.e, ref.energy, tol);
        EXPECT_NEAR(y ? F.bt : F.bn, ref.bx, tol);
        EXPECT_NEAR(y ? F.bn : F.bt, ref.by, tol);
        EXPECT_NEAR(F.bz, ref.bz, tol);
        EXPECT_NEAR(F.psi, ref.psi, tol);
    }
}

TEST(CpuKernels, RhsMatchesReferenceIdeal)
{
    ExecGuard guard;
    run_rhs_case({"orszag_tang", SlopeLimiter::MC, NonIdealConfig::ideal(), false});
    run_rhs_case({"orszag_tang", SlopeLimiter::Minmod, NonIdealConfig::ideal(), false});
    run_rhs_case({"rotor", SlopeLimiter::MC, NonIdealConfig::ideal(), false});
    run_rhs_case({"blast", SlopeLimiter::Minmod, NonIdealConfig::ideal(), false});
}

TEST(CpuKernels, RhsMatchesReferenceNonIdealAndMask)
{
    ExecGuard guard;
    NonIdealConfig all;
    all.enable_ohmic(1e-3).enable_hall(2e-3).enable_ambipolar(5e-4);
    run_rhs_case({"orszag_tang", SlopeLimiter::MC, all, false});
    run_rhs_case({"orszag_tang", SlopeLimiter::MC, NonIdealConfig::ideal(), true});
    run_rhs_case({"rotor", SlopeLimiter::Minmod, all, true});
}

TEST(CpuKernels, ThreadCountDoesNotChangeResultsBitwise)
{
    ExecGuard guard;
    mhd::set_parallel_min_cells(0); // force threading even on a small grid

    auto run = [](int threads) {
        mhd::set_num_threads(threads);
        MHDSolver s(45, 33);
        s.set_limiter_mc();
        s.enable_ohmic(1e-3);
        s.initialize("orszag_tang");
        s.run(0.05);
        return s.state();
    };

    const StateField one = run(1);
    const StateField three = run(3);
    const StateField many = run(8);
    EXPECT_TRUE(bitwise_equal(one, three));
    EXPECT_TRUE(bitwise_equal(one, many));
}

TEST(CpuKernels, SimdAndScalarSolvesAgreeToRoundoff)
{
    ExecGuard guard;
    auto run = [](bool simd) {
        mhd::set_simd_enabled(simd);
        MHDSolver s(40, 40);
        s.initialize("orszag_tang");
        s.run(0.1);
        return s.state();
    };
    const StateField scalar = run(false);
    const StateField vec = run(true);
    EXPECT_LT(max_rhs_diff(vec, scalar), 1e-10);
}

TEST(CpuKernels, ThreadPoolCoversRangeExactlyOnce)
{
    mhd::ThreadPool pool(4);
    std::vector<int> hits(1000, 0);
    pool.parallel_for(0, 1000, 7, [&](int b, int e, int tid) {
        EXPECT_GE(tid, 0);
        EXPECT_LT(tid, 4);
        for (int i = b; i < e; ++i) {
            ++hits[static_cast<std::size_t>(i)];
        }
    });
    for (int h : hits) {
        EXPECT_EQ(h, 1);
    }
}

TEST(CpuKernels, ThreadPoolPropagatesExceptions)
{
    mhd::ThreadPool pool(3);
    EXPECT_THROW(pool.parallel_for(0, 100, 1,
                                   [](int b, int, int) {
                                       if (b == 50) {
                                           throw std::runtime_error("boom");
                                       }
                                   }),
                 std::runtime_error);
    // Pool remains usable afterwards.
    int sum = 0;
    pool.parallel_for(0, 10, 100, [&](int b, int e, int) { sum += e - b; });
    EXPECT_EQ(sum, 10);
}
