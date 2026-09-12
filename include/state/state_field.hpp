#pragma once

#include "mhd/mhd_types.hpp"

#include <vector>

// Structure-of-arrays storage for conserved GLM-MHD variables (9 fields).
// Dimensions (nx, ny) are the full array sizes, including ghosts.
struct StateField {
    int nx = 0;
    int ny = 0;

    std::vector<double> rho; // density

    std::vector<double> mx; // momentum in x-direction
    std::vector<double> my;
    std::vector<double> mz;

    std::vector<double> energy;

    std::vector<double> bx; // magnetic field in x-direction
    std::vector<double> by;
    std::vector<double> bz;

    std::vector<double> psi; // GLM scalar field

    StateField() = default;

    StateField(int nx_, int ny_)
        : nx(nx_),
          ny(ny_),
          rho(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          mx(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          my(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          mz(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          energy(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          bx(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          by(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          bz(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          psi(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0)
    {}

    int index(int i, int j) const { return j * nx + i; }

    MHDState get_state(int i, int j) const
    {
        const int idx = index(i, j);
        return MHDState{
            rho[idx], mx[idx], my[idx], mz[idx], energy[idx], bx[idx], by[idx], bz[idx], psi[idx],
        };
    }

    void set_state(int i, int j, const MHDState& U)
    {
        const int idx = index(i, j);
        rho[idx] = U.rho;
        mx[idx] = U.mx;
        my[idx] = U.my;
        mz[idx] = U.mz;
        energy[idx] = U.energy;
        bx[idx] = U.bx;
        by[idx] = U.by;
        bz[idx] = U.bz;
        psi[idx] = U.psi;
    }

    void copy_cell(int i_dst, int j_dst, int i_src, int j_src)
    {
        set_state(i_dst, j_dst, get_state(i_src, j_src));
    }
};

// Primitive SoA (same layout/indexing as StateField).
struct PrimitiveField {
    int nx = 0;
    int ny = 0;

    std::vector<double> rho;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> vz;
    std::vector<double> pressure;
    std::vector<double> bx;
    std::vector<double> by;
    std::vector<double> bz;
    std::vector<double> psi;

    PrimitiveField() = default;

    PrimitiveField(int nx_, int ny_)
        : nx(nx_),
          ny(ny_),
          rho(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          vx(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          vy(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          vz(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          pressure(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          bx(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          by(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          bz(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0),
          psi(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 0.0)
    {}

    int index(int i, int j) const { return j * nx + i; }

    MHDPrimitive get(int i, int j) const
    {
        const int idx = index(i, j);
        return MHDPrimitive{
            rho[idx], vx[idx], vy[idx], vz[idx], pressure[idx], bx[idx], by[idx], bz[idx], psi[idx],
        };
    }

    void set(int i, int j, const MHDPrimitive& W)
    {
        const int idx = index(i, j);
        rho[idx] = W.rho;
        vx[idx] = W.vx;
        vy[idx] = W.vy;
        vz[idx] = W.vz;
        pressure[idx] = W.pressure;
        bx[idx] = W.bx;
        by[idx] = W.by;
        bz[idx] = W.bz;
        psi[idx] = W.psi;
    }
};
