#include "mhd/build_info.hpp"
#include "verification/diagnostics.hpp"
#include "verification/problems.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_build_info()
{
    std::cout << "Compiler: " << MHD_BUILD_COMPILER_ID << '\n'
              << "Compiler version: " << MHD_BUILD_COMPILER_VERSION << '\n'
              << "C++ standard: " << MHD_BUILD_CXX_STANDARD << '\n'
              << "CMake version: " << MHD_BUILD_CMAKE_VERSION << '\n'
              << "Build type: " << MHD_BUILD_TYPE << '\n'
              << "Git commit: " << MHD_BUILD_GIT_COMMIT << '\n'
              << "Platform: " << MHD_BUILD_SYSTEM_NAME << " / " << MHD_BUILD_SYSTEM_PROCESSOR
              << '\n'
              << "MHD_NATIVE_ARCH: " << MHD_BUILD_NATIVE_ARCH << '\n'
              << "Sanitizers: " << MHD_BUILD_SANITIZERS << '\n';
}

struct ConvRow {
    int n = 0;
    double l1 = 0.0;
    double l2 = 0.0;
    double order_l1 = 0.0;
    double order_l2 = 0.0;
};

void run_alfven_convergence(const std::string& out_csv)
{
    const std::vector<int> ns = {32, 64, 128, 256};
    std::vector<ConvRow> rows;
    rows.reserve(ns.size());

    for (int n : ns) {
        VerificationProblem p = make_fast_alfven();
        p.nx = n;
        p.ny = 1;
        p.y_max = 1.0 / static_cast<double>(n);
        p.t_end = 1.0;
        p.cfl = 0.4;
        const MHDSolver solver = run_verification_problem(p);
        ConvRow row;
        row.n = n;
        row.l1 = alfven_wave_l1_error(solver);
        row.l2 = alfven_wave_l2_error(solver);
        rows.push_back(row);
    }

    for (std::size_t i = 1; i < rows.size(); ++i) {
        rows[i].order_l1 = std::log(rows[i - 1].l1 / rows[i].l1) / std::log(2.0);
        rows[i].order_l2 = std::log(rows[i - 1].l2 / rows[i].l2) / std::log(2.0);
    }

    std::ofstream out(out_csv);
    out << "resolution,L1_error,L2_error,observed_order_L1,observed_order_L2\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& r : rows) {
        out << r.n << ',' << r.l1 << ',' << r.l2 << ',';
        if (r.order_l1 == 0.0 && r.n == rows.front().n) {
            out << "nan,nan\n";
        } else {
            out << r.order_l1 << ',' << r.order_l2 << '\n';
        }
    }

    std::cout << "Alfvén wave convergence (t=1, periodic, gamma=5/3)\n";
    std::cout << std::setw(8) << "N" << std::setw(16) << "L1" << std::setw(16) << "L2"
              << std::setw(14) << "p_L1" << std::setw(14) << "p_L2" << '\n';
    for (const auto& r : rows) {
        std::cout << std::setw(8) << r.n << std::scientific << std::setprecision(4) << std::setw(16)
                  << r.l1 << std::setw(16) << r.l2;
        if (r.n == rows.front().n) {
            std::cout << std::setw(14) << "-" << std::setw(14) << "-" << '\n';
        } else {
            std::cout << std::fixed << std::setprecision(3) << std::setw(14) << r.order_l1
                      << std::setw(14) << r.order_l2 << '\n';
        }
    }
    std::cout << "Wrote " << out_csv << '\n';
}

void dump_refs(const std::string& dir)
{
    const std::vector<VerificationProblem> problems = {
        make_fast_sod(),    make_fast_brio_wu(), make_fast_orszag_tang(),
        make_fast_alfven(),
    };
    for (const auto& p : problems) {
        const MHDSolver solver = run_verification_problem(p);
        const auto d = compute_diagnostics(solver);
        const std::string path = dir + "/" + p.name + ".ref";
        std::ofstream out(path);
        out << "MHD_REF 1\n";
        out << std::scientific << std::setprecision(17);
        out << "min_rho " << d.min_rho << '\n';
        out << "max_rho " << d.max_rho << '\n';
        out << "min_pressure " << d.min_pressure << '\n';
        out << "max_pressure " << d.max_pressure << '\n';
        out << "mass " << d.mass << '\n';
        out << "total_energy " << d.total_energy << '\n';
        out << "max_div_b " << d.max_div_b << '\n';
        out << "steps " << static_cast<double>(d.steps) << '\n';
        if (p.ic == "alfven_wave") {
            out << "l2_error " << alfven_wave_l2_error(solver) << '\n';
        }
        std::cout << "Wrote " << path << '\n';
    }
}

void run_named(const std::string& name)
{
    VerificationProblem p;
    if (name == "sod") {
        p = make_fast_sod();
    } else if (name == "brio_wu") {
        p = make_fast_brio_wu();
    } else if (name == "orszag_tang") {
        p = make_fast_orszag_tang();
    } else if (name == "rotor") {
        p = make_fast_rotor();
    } else if (name == "kelvin_helmholtz" || name == "kh") {
        p = make_fast_kelvin_helmholtz();
    } else if (name == "alfven") {
        p = make_fast_alfven();
    } else {
        throw std::invalid_argument("Unknown problem: " + name);
    }
    const MHDSolver solver = run_verification_problem(p);
    const auto d = compute_diagnostics(solver);
    std::cout << "problem=" << p.name << " t=" << d.t << " steps=" << d.steps
              << " min_rho=" << d.min_rho << " max_rho=" << d.max_rho
              << " min_p=" << d.min_pressure << " max_div_b=" << d.max_div_b << " mass=" << d.mass
              << '\n';
}

void usage()
{
    std::cout
        << "mhd_verify usage:\n"
        << "  mhd_verify --build-info\n"
        << "  mhd_verify --run <sod|brio_wu|orszag_tang|rotor|kh|alfven>\n"
        << "  mhd_verify --convergence-alfven [out.csv]\n"
        << "  mhd_verify --dump-refs <dir>\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            usage();
            return 1;
        }
        const std::string cmd = argv[1];
        if (cmd == "--build-info") {
            print_build_info();
            return 0;
        }
        if (cmd == "--run" && argc >= 3) {
            run_named(argv[2]);
            return 0;
        }
        if (cmd == "--convergence-alfven") {
            const std::string out = (argc >= 3) ? argv[2] : "alfven_convergence.csv";
            run_alfven_convergence(out);
            return 0;
        }
        if (cmd == "--dump-refs" && argc >= 3) {
            dump_refs(argv[2]);
            return 0;
        }
        usage();
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 2;
    }
}
