#pragma once 

#include <vector>
#include <cstddef>

//TODO: use normal data structure

struct StateField {

    int nx;
    int ny;
    std::vector<double> rho;

    std::vector<double> mx;
    std::vector<double> my;
    std::vector<double> mz;

    std::vector<double> energy;

    std::vector<double> bx;
    std::vector<double> by;
    std::vector<double> bz;

    StateField(int nx_, int ny_)

        : nx(nx_), ny(ny_),
          rho(nx * ny),
          mx(nx * ny),
          my(nx * ny),
          mz(nx * ny),
          energy(nx * ny),
          bx(nx * ny),
          by(nx * ny),
          bz(nx * ny)
    {}

    int index(int i, int j) const {
        return j * nx + i;
    }

};