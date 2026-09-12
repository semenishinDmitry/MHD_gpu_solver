#pragma once

// Cell-centered 2D finite-volume grid with optional ghost layers.
// Interior cells occupy [ng, ng + nx) x [ng, ng + ny) in global indices.
struct Grid2D {
    int nx = 0; // number of interior cells in x
    int ny = 0; // number of interior cells in y
    int ng = 0; // number of ghost cells on each side

    double x_min = 0.0;
    double x_max = 0.0;
    double y_min = 0.0;
    double y_max = 0.0;

    double dx = 0.0;
    double dy = 0.0;

    Grid2D() = default;

    Grid2D(const int nx_, const int ny_, const double x_min_, const double x_max_,
           const double y_min_, const double y_max_, const int ng_ = 2)
        : nx(nx_), ny(ny_), ng(ng_), x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_)
    {
        dx = (x_max - x_min) / static_cast<double>(nx);
        dy = (y_max - y_min) / static_cast<double>(ny);
    }

    double get_dx() const
    {
        return dx;
    }
    double get_dy() const
    {
        return dy;
    }

    // Cell-center coordinates for global index (including ghosts).
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

    int i_begin() const
    {
        return ng;
    }
    int i_end() const
    {
        return ng + nx;
    }
    int j_begin() const
    {
        return ng;
    }
    int j_end() const
    {
        return ng + ny;
    }
};
