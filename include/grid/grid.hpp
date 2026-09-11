#pragma once

// TODO: make 1D, 3D, annulus grids

// 2D Grid
struct Grid2D
{
    int nx = 0; // number of elements on x-axis
    int ny = 0; // number of elements on y-axis
    int ng = 0; // number of ghost cells

    double x_min = 0;
    double x_max = 0;
    double y_min = 0;
    double y_max = 0;

    double dx = 0;
    double dy = 0;

    Grid2D() = default;
    Grid2D(const int nx, const int ny, const double x_min, const double x_max, const double y_min, const double y_max)
        : nx(nx), ny(ny), x_min(x_min), x_max(x_max), y_min(y_min), y_max(y_max)
    {
        dx = (x_max - x_min) / (nx - 1);
        dy = (y_max - y_min) / (ny - 1);
    }
public:
    double get_dx() const
    {
        return (x_max - x_min) / nx;
    }

    double get_dy() const
    {
        return (y_max - y_min) / ny;
    }

    double get_x(int i) const
    {
        return x_min + (i - ng + 0.5) * dx;
    }

    double get_y(int j) const
    {
        return y_min + (j - ng + 0.5) * dy;
    }

    int get_size_x() const
    {
        return nx + 2 * ng;
    }

    int get_size_y() const
    {
        return ny + 2 * ng;
    }
};