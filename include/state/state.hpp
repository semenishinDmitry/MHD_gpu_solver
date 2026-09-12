#pragma once

// State of the system in each cell of the 2D grid
// d_z = 0; but we still have z-momentum and z-magnetic field components for 2.5D simulations
class State2D {
    double rho = 0.0; // density

    double mx = 0.0; // momentum in x-direction
    double my = 0.0; // momentum in y-direction
    double mz = 0.0; // momentum in z-direction

    double E = 0.0; // energy

    double Bx = 0.0; // magnetic field in x-direction
    double By = 0.0; // magnetic field in y-direction
    double Bz = 0.0; // magnetic field in z-direction

  public:
    State2D() = default;
    State2D(const double rho, const double mx, const double my, const double mz, const double E,
            const double Bx, const double By, const double Bz)
        : rho(rho), mx(mx), my(my), mz(mz), E(E), Bx(Bx), By(By), Bz(Bz) {};

    double get_vx() const
    {
        return mx / rho;
    }
    double get_vy() const
    {
        return my / rho;
    }
    double get_vz() const
    {
        return mz / rho;
    }
    double get_energy() const
    {
        return E;
    }
    double get_bx() const
    {
        return Bx;
    }
    double get_by() const
    {
        return By;
    }
    double get_bz() const
    {
        return Bz;
    }
};
// State of the system in each cell of the 2D grid
// d_z = 0; but we still have z-velocity and z-magnetic field components for 2.5D simulations
class PrimitiveState2D {
  public:
    double rho = 0.0; // density

    double vx = 0.0; // velocity in x-direction
    double vy = 0.0; // velocity in y-direction
    double vz = 0.0; // velocity in z-direction

    double E = 0.0; // energy

    double Bx = 0.0; // magnetic field in x-direction
    double By = 0.0; // magnetic field in y-direction
    double Bz = 0.0; // magnetic field in z-direction

    PrimitiveState2D() = default;
    PrimitiveState2D(const double rho, const double vx, const double vy, const double vz,
                     const double E, const double Bx, const double By, const double Bz)
        : rho(rho), vx(vx), vy(vy), vz(vz), E(E), Bx(Bx), By(By), Bz(Bz) {};

    double get_vx() const
    {
        return vx;
    }
    double get_vy() const
    {
        return vy;
    }
    double get_vz() const
    {
        return vz;
    }
};