// Speed and accuracy comparison of the execution backends.
//
//   CPU reference   original scalar implementation (compute_rhs_reference), 1 thread
//   CPU scalar      optimized kernels, SIMD off, 1 thread
//   CPU SIMD        optimized kernels, SIMD on, 1 thread
//   CPU SIMD+MT     optimized kernels, SIMD on, all threads
//   Metal fp32      Apple GPU (if built and present)
//   CUDA fp64/fp32  NVIDIA GPU (if built and present)
//
// Accuracy is measured against the fp64 CPU solution of the same problem and, for the circular
// Alfvén wave, against the exact solution (convergence order per backend).
//
// Usage:
//   mhd_backend_compare [--sizes 128,256,512] [--t-end 0.05] [--accuracy-n 256]
//                       [--accuracy-t 0.5] [--alfven-max 512] [--no-reference]
//                       [--markdown out.md] [--csv out.csv]
//
// CI gate mode (pass/fail with exit code, see docs/gpu.md#ci-gate):
//   mhd_backend_compare --gate [--min-speedup 1.5] [--no-speed] [--require-gpu]
//                       [--markdown summary.md]
//   exit 0 = all checks passed (or no GPU and --require-gpu not given), 1 = a check failed,
//   2 = usage error, 3 = no GPU backend although --require-gpu was given.

#include "mhd/build_info.hpp"
#include "mhd/gpu/device_solver.hpp"
#include "mhd/parallel.hpp"
#include "mhd/rhs_reference.hpp"
#include "mhd/simd.hpp"
#include "mhd/solver_api.hpp"
#include "verification/diagnostics.hpp"
#include "verification/problems.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Variant {
    std::string label;
    mhd::Backend backend = mhd::Backend::CPU;
    mhd::Precision precision = mhd::Precision::FP64;
    int threads = 1;
    bool simd = true;
    bool reference = false;
};

struct Options {
    std::vector<int> sizes = {128, 256, 512, 1024};
    double t_end = 0.05;
    int accuracy_n = 256;
    double accuracy_t = 0.5;
    int alfven_max = 512;
    bool include_reference = true;
    std::string markdown_path;
    std::string csv_path;
    bool gate = false;
    bool gate_speed = true;
    bool require_gpu = false;
    double min_speedup = 1.5;
};

std::vector<int> parse_list(const std::string& s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(std::stoi(item));
    }
    return out;
}

void apply_exec(const Variant& v)
{
    mhd::set_num_threads(v.threads);
    mhd::set_simd_enabled(v.simd);
}

// The pre-optimization solver: original RHS, serial, scalar.
SolveResult solve_reference(MHDSolver& s, double t_end)
{
    const Grid2D& grid = s.grid();
    const SolveParams& p = s.params();
    StateField& U = s.state();
    StateField U_star(U.nx, U.ny);
    StateField rhs(U.nx, U.ny);
    RHSReferenceWorkspace work(U.nx, U.ny);
    const BoundaryConditions& bc = s.boundary_conditions();

    SolveResult r{};
    while (r.t < t_end) {
        const CFLResult cfl = compute_cfl_dt(U, grid, p.gamma, p.cfl, p.nonideal);
        const double dt = std::min(cfl.dt, t_end - r.t);
        apply_boundary_conditions(U, grid, bc);
        compute_rhs_reference(U, rhs, grid, work, p.gamma, cfl.c_h, p.glm_alpha, p.limiter,
                              p.nonideal);
        field_xpay(U_star, U, dt, rhs);
        apply_boundary_conditions(U_star, grid, bc);
        compute_rhs_reference(U_star, rhs, grid, work, p.gamma, cfl.c_h, p.glm_alpha, p.limiter,
                              p.nonideal);
        field_ssp_rk2_combine(U, U_star, rhs, dt);
        r.t += dt;
        ++r.steps;
        r.c_h = cfl.c_h;
    }
    apply_boundary_conditions(U, grid, bc);
    return r;
}

using Setup = std::function<void(MHDSolver&)>;

struct RunOutput {
    double seconds = 0.0;
    int steps = 0;
    StateField state;
    double gamma = 5.0 / 3.0;
    double max_div_b = 0.0;
};

RunOutput run_case(const Variant& v, int nx, int ny, double x_max, double y_max,
                   const Setup& setup, double t_end)
{
    apply_exec(v);
    MHDSolver s(nx, ny, 0.0, x_max, 0.0, y_max);
    setup(s);
    if (v.backend != mhd::Backend::CPU) {
        s.set_backend(v.backend, v.precision);
        // Warm-up: device creation, shader/pipeline compilation, first-touch of buffers.
        MHDSolver warm(nx, ny, 0.0, x_max, 0.0, y_max);
        setup(warm);
        warm.set_backend(v.backend, v.precision);
        warm.run(std::min(t_end, 1e-4));
    }

    const auto t0 = std::chrono::steady_clock::now();
    SolveResult r{};
    if (v.reference) {
        r = solve_reference(s, t_end);
    } else {
        r = s.run(t_end);
    }
    const auto t1 = std::chrono::steady_clock::now();

    RunOutput out;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.steps = r.steps;
    out.state = s.state();
    out.gamma = s.params().gamma;
    out.max_div_b = s.max_div_b();
    return out;
}

struct ErrorNorms {
    double rho_l1 = 0.0;   // relative L1
    double p_l1 = 0.0;     // relative L1
    double b_l1 = 0.0;     // relative L1 of (bx, by)
    double rho_linf = 0.0; // relative Linf
};

ErrorNorms compare_states(const StateField& a, const StateField& ref, const Grid2D& grid,
                          double gamma)
{
    ErrorNorms e;
    double rho_n = 0.0;
    double p_n = 0.0;
    double b_n = 0.0;
    double rho_max = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDPrimitive A = to_primitive(a.get_state(i, j), gamma);
            const MHDPrimitive R = to_primitive(ref.get_state(i, j), gamma);
            e.rho_l1 += std::abs(A.rho - R.rho);
            e.p_l1 += std::abs(A.pressure - R.pressure);
            e.b_l1 += std::abs(A.bx - R.bx) + std::abs(A.by - R.by);
            e.rho_linf = std::max(e.rho_linf, std::abs(A.rho - R.rho));
            rho_n += std::abs(R.rho);
            p_n += std::abs(R.pressure);
            b_n += std::abs(R.bx) + std::abs(R.by);
            rho_max = std::max(rho_max, std::abs(R.rho));
        }
    }
    e.rho_l1 /= rho_n;
    e.p_l1 /= p_n;
    e.b_l1 /= b_n;
    e.rho_linf /= rho_max;
    return e;
}

double total(const std::vector<double>& f, const Grid2D& grid, int nx_tot)
{
    double s = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            s += f[static_cast<std::size_t>(j * nx_tot + i)];
        }
    }
    return s;
}

std::string fmt(const char* f, double v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

void setup_orszag(MHDSolver& s)
{
    s.set_cfl(0.4);
    s.set_glm_alpha(0.1);
    s.set_limiter_mc();
    s.set_periodic_bc();
    s.set_ideal();
    s.initialize("orszag_tang");
}

void setup_orszag_nonideal(MHDSolver& s)
{
    setup_orszag(s);
    s.enable_ohmic(1e-3);
    s.enable_hall(5e-4);
}

// ------------------------------------------------------------------------------- CI gate

struct GateThresholds {
    double rho_l1;        // relative L1 vs fp64 CPU
    double linf;          // relative Linf of rho vs fp64 CPU
    double conservation;  // |mass drift| and |energy drift|
    double alfven_ratio;  // GPU error / CPU error vs exact solution
};

GateThresholds thresholds_for(mhd::Precision p)
{
    if (p == mhd::Precision::FP64) {
        return {1e-11, 1e-9, 1e-11, 1.001};
    }
    // fp32: measured ~6e-7 (L1), ~2e-6 (Linf), ~1e-7 (drift), <= 1.1 (Alfvén N<=256) on Apple
    // M5; thresholds leave ~10x headroom for other GPUs / drivers.
    return {1e-5, 5e-5, 2e-6, 1.25};
}

class Gate {
  public:
    void check(bool ok, const std::string& what, const std::string& detail)
    {
        rows_.push_back({ok, what, detail});
        failed_ = failed_ || !ok;
        std::cerr << (ok ? "[PASS] " : "[FAIL] ") << what << ": " << detail << '\n';
    }
    void note(const std::string& what, const std::string& detail)
    {
        rows_.push_back({true, what, detail + " (skipped)"});
        std::cerr << "[SKIP] " << what << ": " << detail << '\n';
    }
    bool failed() const
    {
        return failed_;
    }
    std::string markdown(const std::string& header) const
    {
        std::ostringstream md;
        md << "## GPU gate: " << (failed_ ? "FAILED" : "passed") << "\n\n" << header << "\n\n";
        md << "| | Check | Result |\n|---|---|---|\n";
        for (const auto& r : rows_) {
            md << "| " << (r.ok ? "✅" : "❌") << " | " << r.what << " | " << r.detail << " |\n";
        }
        return md.str();
    }

  private:
    struct Row {
        bool ok;
        std::string what;
        std::string detail;
    };
    std::vector<Row> rows_;
    bool failed_ = false;
};

double median_rate(const Variant& v, int n, double t_end, int repeats)
{
    std::vector<double> rates;
    for (int k = 0; k < repeats; ++k) {
        const RunOutput out = run_case(v, n, n, 1.0, 1.0, setup_orszag, t_end);
        rates.push_back(static_cast<double>(n) * n * out.steps / out.seconds / 1e6);
    }
    std::sort(rates.begin(), rates.end());
    return rates[rates.size() / 2];
}

int run_gate(const Options& opt, const std::vector<Variant>& variants, const Variant& cpu_best)
{
    std::vector<Variant> gpus;
    for (const auto& v : variants) {
        if (v.backend != mhd::Backend::CPU) {
            gpus.push_back(v);
        }
    }
    std::ostringstream header;
    header << "CPU: " << mhd::simd::kBackendName << ", " << cpu_best.threads << " threads. ";
    Gate gate;
    if (gpus.empty()) {
        header << "No GPU backend available on this machine.";
        gate.note("GPU backend", "none available (build with Metal/CUDA and run on a GPU)");
        const std::string md = gate.markdown(header.str());
        std::cout << md;
        if (!opt.markdown_path.empty()) {
            std::ofstream(opt.markdown_path) << md;
        }
        return opt.require_gpu ? 3 : 0;
    }

    // Accuracy: nonlinear problems vs fp64 CPU (same steps, small differences, conservation).
    const int n = 128;
    const double t_acc = 0.25;
    const Grid2D grid(n, n, 0.0, 1.0, 0.0, 1.0, 2);
    struct Problem {
        const char* name;
        Setup setup;
    };
    const Problem problems[] = {{"Orszag-Tang ideal", setup_orszag},
                                {"Orszag-Tang Ohmic+Hall", setup_orszag_nonideal}};
    for (const auto& pr : problems) {
        const RunOutput ref = run_case(cpu_best, n, n, 1.0, 1.0, pr.setup, t_acc);
        MHDSolver init(n, n);
        pr.setup(init);
        const double mass0 = total(init.state().rho, grid, init.size_x());
        const double energy0 = total(init.state().energy, grid, init.size_x());
        for (const auto& v : gpus) {
            const GateThresholds th = thresholds_for(v.precision);
            const RunOutput out = run_case(v, n, n, 1.0, 1.0, pr.setup, t_acc);
            const ErrorNorms e = compare_states(out.state, ref.state, grid, out.gamma);
            const double dm = std::abs(total(out.state.rho, grid, out.state.nx) - mass0) / mass0;
            const double de =
                std::abs(total(out.state.energy, grid, out.state.nx) - energy0) / energy0;
            const std::string tag = v.label + ", " + pr.name + " " + std::to_string(n) + "²";
            gate.check(out.steps == ref.steps, tag + ": step count",
                       std::to_string(out.steps) + " vs CPU " + std::to_string(ref.steps));
            const double l1 = std::max({e.rho_l1, e.p_l1, e.b_l1});
            gate.check(l1 <= th.rho_l1, tag + ": rel. L1 (ρ, p, B) vs fp64",
                       fmt("%.2e", l1) + " ≤ " + fmt("%.0e", th.rho_l1));
            gate.check(e.rho_linf <= th.linf, tag + ": rel. L∞ ρ vs fp64",
                       fmt("%.2e", e.rho_linf) + " ≤ " + fmt("%.0e", th.linf));
            gate.check(std::max(dm, de) <= th.conservation, tag + ": mass/energy drift",
                       fmt("%.1e", dm) + " / " + fmt("%.1e", de) + " ≤ " +
                           fmt("%.0e", th.conservation));
        }
    }

    // Accuracy vs exact solution: Alfvén wave error and order, relative to the fp64 CPU.
    auto alfven_error = [](const Variant& v, int na) {
        VerificationProblem p = make_fast_alfven();
        apply_exec(v);
        MHDSolver s(na, 1, 0.0, 1.0, 0.0, 1.0 / static_cast<double>(na));
        s.set_cfl(p.cfl);
        s.set_gamma(p.gamma);
        s.set_glm_alpha(p.glm_alpha);
        s.set_bc(p.bc_x, p.bc_y);
        s.set_limiter_mc();
        s.set_ideal();
        s.initialize(p.ic);
        if (v.backend != mhd::Backend::CPU) {
            s.set_backend(v.backend, v.precision);
        }
        s.run(1.0);
        return alfven_wave_l1_error(s);
    };
    const double cpu_128 = alfven_error(cpu_best, 128);
    const double cpu_256 = alfven_error(cpu_best, 256);
    for (const auto& v : gpus) {
        const GateThresholds th = thresholds_for(v.precision);
        const double e128 = alfven_error(v, 128);
        const double e256 = alfven_error(v, 256);
        const double order = std::log(e128 / e256) / std::log(2.0);
        gate.check(order >= 1.8, v.label + ", Alfvén wave: order N=128→256",
                   fmt("%.2f", order) + " ≥ 1.80");
        const double ratio = e256 / cpu_256;
        gate.check(ratio <= th.alfven_ratio, v.label + ", Alfvén wave N=256: error / CPU error",
                   fmt("%.3f", ratio) + " ≤ " + fmt("%.3f", th.alfven_ratio) + " (CPU " +
                       fmt("%.2e", cpu_256) + ", N=128 CPU " + fmt("%.2e", cpu_128) + ")");
    }

    // Speed: GPU vs the best CPU configuration (all threads + SIMD), median of 3 runs.
    if (opt.gate_speed) {
        const int ns = 512;
        const double t_speed = 0.02;
        const double cpu_rate = median_rate(cpu_best, ns, t_speed, 3);
        for (const auto& v : gpus) {
            const double rate = median_rate(v, ns, t_speed, 3);
            const double speedup = rate / cpu_rate;
            gate.check(speedup >= opt.min_speedup,
                       v.label + ": speed vs CPU (" + std::to_string(ns) + "², median of 3)",
                       fmt("%.1f", rate) + " vs " + fmt("%.1f", cpu_rate) + " Mcell/s = ×" +
                           fmt("%.2f", speedup) + " ≥ ×" + fmt("%.2f", opt.min_speedup));
        }
    } else {
        gate.note("Speed vs CPU", "disabled with --no-speed");
    }

    const std::string md = gate.markdown(header.str());
    std::cout << md;
    if (!opt.markdown_path.empty()) {
        std::ofstream(opt.markdown_path) << md;
    }
    return gate.failed() ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options opt;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        auto next = [&]() -> std::string {
            if (a + 1 >= argc) {
                std::cerr << "missing value for " << arg << '\n';
                std::exit(2);
            }
            return argv[++a];
        };
        if (arg == "--sizes") {
            opt.sizes = parse_list(next());
        } else if (arg == "--t-end") {
            opt.t_end = std::stod(next());
        } else if (arg == "--accuracy-n") {
            opt.accuracy_n = std::stoi(next());
        } else if (arg == "--accuracy-t") {
            opt.accuracy_t = std::stod(next());
        } else if (arg == "--alfven-max") {
            opt.alfven_max = std::stoi(next());
        } else if (arg == "--no-reference") {
            opt.include_reference = false;
        } else if (arg == "--markdown") {
            opt.markdown_path = next();
        } else if (arg == "--csv") {
            opt.csv_path = next();
        } else if (arg == "--gate") {
            opt.gate = true;
        } else if (arg == "--no-speed") {
            opt.gate_speed = false;
        } else if (arg == "--require-gpu") {
            opt.require_gpu = true;
        } else if (arg == "--min-speedup") {
            opt.min_speedup = std::stod(next());
        } else {
            std::cerr << "unknown option " << arg << '\n';
            return 2;
        }
    }

    const int hw = mhd::default_thread_count();
    std::vector<Variant> variants;
    if (opt.include_reference) {
        variants.push_back({"CPU reference (original code), 1T", mhd::Backend::CPU,
                            mhd::Precision::FP64, 1, false, true});
    }
    variants.push_back({"CPU scalar, 1T", mhd::Backend::CPU, mhd::Precision::FP64, 1, false});
    variants.push_back({std::string("CPU ") + mhd::simd::kBackendName + ", 1T", mhd::Backend::CPU,
                        mhd::Precision::FP64, 1, true});
    variants.push_back({std::string("CPU ") + mhd::simd::kBackendName + ", " +
                            std::to_string(hw) + "T",
                        mhd::Backend::CPU, mhd::Precision::FP64, hw, true});
    if (mhd::metal_available()) {
        variants.push_back({"Metal fp32", mhd::Backend::Metal, mhd::Precision::FP32, hw, true});
    }
    if (mhd::cuda_available()) {
        variants.push_back({"CUDA fp64", mhd::Backend::CUDA, mhd::Precision::FP64, hw, true});
        variants.push_back({"CUDA fp32", mhd::Backend::CUDA, mhd::Precision::FP32, hw, true});
    }
    const Variant fp64_ref{"CPU fp64 (reference solution)", mhd::Backend::CPU,
                           mhd::Precision::FP64, hw, true};

    if (opt.gate) {
        return run_gate(opt, variants, fp64_ref);
    }

    std::ostringstream md;
    std::ostringstream csv;
    csv << "section,variant,n,steps,seconds,mcups,speedup,rho_l1,p_l1,b_l1,rho_linf,"
           "mass_drift,energy_drift,max_div_b,alfven_l1,order\n";

    md << "# Backend comparison\n\n";
    md << "- Build: " << MHD_BUILD_COMPILER_ID << ' ' << MHD_BUILD_COMPILER_VERSION << ", "
       << MHD_BUILD_TYPE << ", native=" << MHD_BUILD_NATIVE_ARCH << ", commit "
       << MHD_BUILD_GIT_COMMIT << '\n';
    md << "- CPU SIMD backend: " << mhd::simd::kBackendName << " (" << mhd::simd::kWidth
       << " doubles/vector), hardware threads: " << hw << '\n';
    for (const auto& v : variants) {
        if (v.backend != mhd::Backend::CPU) {
            MHDSolver probe(8, 8);
            probe.set_backend(v.backend, v.precision);
            md << "- " << v.label << ": " << probe.backend_device() << '\n';
        }
    }
    md << '\n';

    // ---------------------------------------------------------------- speed
    md << "## Speed: Orszag-Tang, MC limiter, t_end = " << opt.t_end << "\n\n";
    md << "Throughput in Mcell-updates/s (one update = one full SSP-RK2 step of one cell). "
          "GPU times include host<->device transfers of the state.\n\n";
    md << "| Variant |";
    for (int n : opt.sizes) {
        md << ' ' << n << "² |";
    }
    md << '\n' << "|---|";
    for (std::size_t k = 0; k < opt.sizes.size(); ++k) {
        md << "---|";
    }
    md << '\n';

    std::vector<std::vector<double>> mcups(variants.size(), std::vector<double>(opt.sizes.size()));
    for (std::size_t si = 0; si < opt.sizes.size(); ++si) {
        const int n = opt.sizes[si];
        for (std::size_t vi = 0; vi < variants.size(); ++vi) {
            const RunOutput out = run_case(variants[vi], n, n, 1.0, 1.0, setup_orszag, opt.t_end);
            const double rate = static_cast<double>(n) * n * out.steps / out.seconds / 1e6;
            mcups[vi][si] = rate;
            std::cerr << "[speed] " << variants[vi].label << " n=" << n << " steps=" << out.steps
                      << " " << fmt("%.3f", out.seconds) << " s, " << fmt("%.2f", rate)
                      << " Mcell/s\n";
            csv << "speed," << variants[vi].label << ',' << n << ',' << out.steps << ','
                << out.seconds << ',' << rate << ",,,,,,,,,,\n";
        }
    }
    for (std::size_t vi = 0; vi < variants.size(); ++vi) {
        md << "| " << variants[vi].label << " |";
        for (std::size_t si = 0; si < opt.sizes.size(); ++si) {
            md << ' ' << fmt("%.1f", mcups[vi][si]);
            md << " (×" << fmt("%.1f", mcups[vi][si] / mcups[0][si]) << ") |";
        }
        md << '\n';
    }
    md << "\n(×k) = speed-up relative to the first row.\n\n";

    // ---------------------------------------------------------------- accuracy vs fp64
    auto accuracy_section = [&](const std::string& title, const Setup& setup) {
        const int n = opt.accuracy_n;
        const Grid2D grid(n, n, 0.0, 1.0, 0.0, 1.0, 2);
        const RunOutput ref = run_case(fp64_ref, n, n, 1.0, 1.0, setup, opt.accuracy_t);

        MHDSolver init(n, n);
        setup(init);
        const double mass0 = total(init.state().rho, grid, init.size_x());
        const double energy0 = total(init.state().energy, grid, init.size_x());

        md << "## Accuracy: " << title << ", " << n << "², t = " << opt.accuracy_t << "\n\n";
        md << "Differences from the fp64 CPU solution (relative L1 / Linf norms). Mass and energy "
              "drift are |Q(t) - Q(0)| / Q(0) (both are conserved exactly by the scheme with "
              "periodic boundaries, so the drift is pure round-off).\n\n";
        md << "| Variant | steps | ρ L1 | p L1 | B L1 | ρ L∞ | mass drift | energy drift | "
              "max ∇·B |\n|---|---|---|---|---|---|---|---|---|\n";
        for (const auto& v : variants) {
            if (v.reference) {
                continue;
            }
            const RunOutput out = run_case(v, n, n, 1.0, 1.0, setup, opt.accuracy_t);
            const ErrorNorms e = compare_states(out.state, ref.state, grid, out.gamma);
            const double mass = total(out.state.rho, grid, out.state.nx);
            const double energy = total(out.state.energy, grid, out.state.nx);
            const double md_mass = std::abs(mass - mass0) / mass0;
            const double md_energy = std::abs(energy - energy0) / energy0;
            md << "| " << v.label << " | " << out.steps << " | " << fmt("%.2e", e.rho_l1) << " | "
               << fmt("%.2e", e.p_l1) << " | " << fmt("%.2e", e.b_l1) << " | "
               << fmt("%.2e", e.rho_linf) << " | " << fmt("%.1e", md_mass) << " | "
               << fmt("%.1e", md_energy) << " | " << fmt("%.2e", out.max_div_b) << " |\n";
            csv << "accuracy:" << title << ',' << v.label << ',' << n << ',' << out.steps << ','
                << out.seconds << ",,," << e.rho_l1 << ',' << e.p_l1 << ',' << e.b_l1 << ','
                << e.rho_linf << ',' << md_mass << ',' << md_energy << ',' << out.max_div_b
                << ",,\n";
            std::cerr << "[accuracy] " << title << ' ' << v.label << " rho_l1=" << e.rho_l1
                      << '\n';
        }
        md << '\n';
    };
    accuracy_section("Orszag-Tang (ideal)", setup_orszag);
    accuracy_section("Orszag-Tang (Ohmic η=1e-3 + Hall η_H=5e-4)", setup_orszag_nonideal);

    // ---------------------------------------------------------------- Alfvén convergence
    md << "## Accuracy vs exact solution: circular Alfvén wave, t = 1 (one period)\n\n";
    md << "Relative L1 error of (v⊥, B⊥) and observed order between successive resolutions. "
          "Second order is expected; an fp32 backend levels off once the truncation error "
          "approaches single-precision round-off.\n\n";
    std::vector<int> alfven_ns;
    for (int n = 32; n <= opt.alfven_max; n *= 2) {
        alfven_ns.push_back(n);
    }
    md << "| Variant |";
    for (int n : alfven_ns) {
        md << " N=" << n << " |";
    }
    md << "\n|---|";
    for (std::size_t k = 0; k < alfven_ns.size(); ++k) {
        md << "---|";
    }
    md << '\n';
    for (const auto& v : variants) {
        if (v.reference || (v.backend == mhd::Backend::CPU && v.threads == 1)) {
            continue; // CPU variants agree to round-off; keep one CPU row
        }
        md << "| " << v.label << " |";
        double prev = 0.0;
        for (int n : alfven_ns) {
            VerificationProblem p = make_fast_alfven();
            const double ymax = 1.0 / static_cast<double>(n);
            const Setup setup = [&](MHDSolver& s) {
                s.set_cfl(p.cfl);
                s.set_gamma(p.gamma);
                s.set_glm_alpha(p.glm_alpha);
                s.set_bc(p.bc_x, p.bc_y);
                s.set_limiter_mc();
                s.set_ideal();
                s.initialize(p.ic);
            };
            apply_exec(v);
            MHDSolver s(n, 1, 0.0, 1.0, 0.0, ymax);
            setup(s);
            if (v.backend != mhd::Backend::CPU) {
                s.set_backend(v.backend, v.precision);
            }
            s.run(1.0);
            const double err = alfven_wave_l1_error(s);
            md << ' ' << fmt("%.2e", err);
            double order = 0.0;
            if (prev > 0.0) {
                order = std::log(prev / err) / std::log(2.0);
                md << " (" << fmt("%.2f", order) << ")";
            }
            md << " |";
            csv << "alfven," << v.label << ',' << n << ",,,,,,,,,,,," << err << ',' << order
                << '\n';
            prev = err;
        }
        md << '\n';
    }
    md << '\n';

    mhd::set_num_threads(0);
    mhd::set_simd_enabled(true);

    std::cout << md.str();
    if (!opt.markdown_path.empty()) {
        std::ofstream(opt.markdown_path) << md.str();
    }
    if (!opt.csv_path.empty()) {
        std::ofstream(opt.csv_path) << csv.str();
    }
    return 0;
}
