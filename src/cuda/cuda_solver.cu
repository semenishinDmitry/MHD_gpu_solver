// CUDA backend: device-resident GLM-MHD solver, fp64 or fp32.
//
// The face flux and CFL signal speed come from include/mhd/kernels/muscl_hll.hpp, the same
// source the CPU scalar/SIMD paths are instantiated from (MHD_HD = __host__ __device__).
// Structure mirrors src/metal/metal_solver.mm: dt is computed on the device and the host
// synchronizes once per batch of steps.

#include "mhd/gpu/device_solver.hpp"
#include "mhd/kernels/muscl_hll.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define MHD_CUDA_CHECK(expr)                                                                       \
    do {                                                                                           \
        const cudaError_t mhd_err_ = (expr);                                                       \
        if (mhd_err_ != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(mhd_err_) +       \
                                     " (" #expr ")");                                              \
        }                                                                                          \
    } while (0)

constexpr int kReduceThreads = 256;

template <class Real> struct Params {
    int nx_tot, ny_tot, nx, ny, ng, n;
    int bc_x, bc_y;
    int has_mask, reflective_walls;
    int ohmic, hall, ambipolar, nonideal_any;
    int partial_count;
    Real inv_dx, inv_dy, gamma, inv_gm1, glm_alpha, h;
    Real eta_ohm, eta_hall, eta_ambipolar;
    double cfl, dt_diff, t_end;
};

// Time bookkeeping stays in fp64 even for the fp32 field arithmetic.
struct Ctrl {
    double t;
    double dt;
    double c_h;
    unsigned steps;
    unsigned finished;
};

__device__ __forceinline__ int periodic_source(int g, int n, int ng, bool low)
{
    const int offset = low ? (g - ng) : g;
    return ng + ((offset % n) + n) % n;
}

template <class Real> __device__ __forceinline__ Real dmax(Real a, Real b)
{
    return (a < b) ? b : a;
}

// ------------------------------------------------------------------------- boundaries

template <class Real> __global__ void bc_x(Real* __restrict__ A, Params<Real> P)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (g >= P.ng || j >= P.ny_tot) {
        return;
    }
    const int row = j * P.nx_tot;
    const int src_l = (P.bc_x == 0) ? periodic_source(g, P.nx, P.ng, true) : P.ng;
    const int src_r = (P.bc_x == 0) ? periodic_source(g, P.nx, P.ng, false) : (P.ng + P.nx - 1);
    for (int c = 0; c < 9; ++c) {
        Real* f = A + static_cast<std::size_t>(c) * P.n + row;
        f[g] = f[src_l];
        f[P.ng + P.nx + g] = f[src_r];
    }
}

template <class Real> __global__ void bc_y(Real* __restrict__ A, Params<Real> P)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= P.nx_tot || g >= P.ng) {
        return;
    }
    const int src_b = (P.bc_y == 0) ? periodic_source(g, P.ny, P.ng, true) : P.ng;
    const int src_t = (P.bc_y == 0) ? periodic_source(g, P.ny, P.ng, false) : (P.ng + P.ny - 1);
    for (int c = 0; c < 9; ++c) {
        Real* f = A + static_cast<std::size_t>(c) * P.n;
        f[g * P.nx_tot + i] = f[src_b * P.nx_tot + i];
        f[(P.ng + P.ny + g) * P.nx_tot + i] = f[src_t * P.nx_tot + i];
    }
}

template <class Real>
__global__ void solid_fill(Real* __restrict__ A, const std::uint8_t* __restrict__ mask,
                           Params<Real> P)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii >= P.nx || jj >= P.ny || mask[jj * P.nx + ii] != 0) {
        return;
    }
    const int n = P.n;
    const int dst = (jj + P.ng) * P.nx_tot + (ii + P.ng);
    const int di[4] = {1, -1, 0, 0};
    const int dj[4] = {0, 0, 1, -1};
    const Real half(0.5);
    for (int k = 0; k < 4; ++k) {
        const int ni = ii + di[k];
        const int nj = jj + dj[k];
        if (ni < 0 || nj < 0 || ni >= P.nx || nj >= P.ny || mask[nj * P.nx + ni] == 0) {
            continue;
        }
        const int src = (nj + P.ng) * P.nx_tot + (ni + P.ng);
        Real rho = A[src];
        Real vx = A[n + src] / rho;
        Real vy = A[2 * n + src] / rho;
        const Real vz = A[3 * n + src] / rho;
        const Real bx = A[5 * n + src];
        const Real by = A[6 * n + src];
        const Real bz = A[7 * n + src];
        const Real emag = half * (bx * bx + by * by + bz * bz);
        Real p = (P.gamma - Real(1.0)) *
                 (A[4 * n + src] - half * rho * (vx * vx + vy * vy + vz * vz) - emag);
        if (k < 2) {
            vx = -vx;
        } else {
            vy = -vy;
        }
        rho = dmax(rho, Real(1.0e-12));
        p = dmax(p, Real(1.0e-12));
        A[dst] = rho;
        A[n + dst] = rho * vx;
        A[2 * n + dst] = rho * vy;
        A[3 * n + dst] = rho * vz;
        A[4 * n + dst] = p * P.inv_gm1 + half * rho * (vx * vx + vy * vy + vz * vz) + emag;
        A[5 * n + dst] = bx;
        A[6 * n + dst] = by;
        A[7 * n + dst] = bz;
        A[8 * n + dst] = A[8 * n + src];
        return;
    }
    A[dst] = Real(1.0);
    for (int c = 1; c < 9; ++c) {
        A[c * n + dst] = Real(0.0);
    }
    A[4 * n + dst] = P.inv_gm1;
}

// ------------------------------------------------------------------------- CFL / time

template <class Real>
__global__ void cfl_partial(const Real* __restrict__ A, const std::uint8_t* __restrict__ mask,
                            Real* __restrict__ partial, Params<Real> P)
{
    __shared__ Real sh[kReduceThreads];
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    Real v(0.0);
    if (gid < P.nx * P.ny) {
        const int ii = gid % P.nx;
        const int jj = gid / P.nx;
        if (P.has_mask == 0 || mask[jj * P.nx + ii] != 0) {
            const int idx = (jj + P.ng) * P.nx_tot + (ii + P.ng);
            const int n = P.n;
            v = dmax(Real(0.0), mhd::kernels::cell_max_signal_speed<Real>(
                                    A[idx], A[n + idx], A[2 * n + idx], A[3 * n + idx],
                                    A[4 * n + idx], A[5 * n + idx], A[6 * n + idx],
                                    A[7 * n + idx], P.gamma));
        }
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = kReduceThreads / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) {
            sh[threadIdx.x] = dmax(sh[threadIdx.x], sh[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sh[0];
    }
}

template <class Real>
__global__ void cfl_finalize(const Real* __restrict__ partial, Ctrl* ctrl, Params<Real> P)
{
    __shared__ Real sh[kReduceThreads];
    Real v(0.0);
    for (int k = threadIdx.x; k < P.partial_count; k += kReduceThreads) {
        v = dmax(v, partial[k]);
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = kReduceThreads / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) {
            sh[threadIdx.x] = dmax(sh[threadIdx.x], sh[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) {
        return;
    }
    const double c_h = static_cast<double>(sh[0]);
    const double inv_dx = static_cast<double>(P.inv_dx);
    const double inv_dy = static_cast<double>(P.inv_dy);
    double dt = (c_h > 0.0) ? P.cfl / (c_h * inv_dx + c_h * inv_dy) : HUGE_VAL;
    if (P.dt_diff > 0.0) {
        dt = fmin(dt, P.dt_diff);
    }
    const double remaining = P.t_end - ctrl->t;
    if (ctrl->finished != 0 || remaining <= 0.0) {
        ctrl->finished = 1;
        ctrl->dt = 0.0;
        return;
    }
    ctrl->dt = fmin(dt, remaining);
    ctrl->c_h = c_h;
}

__global__ void advance_time(Ctrl* ctrl)
{
    if (ctrl->dt <= 0.0) {
        return;
    }
    ctrl->t += ctrl->dt;
    ctrl->steps += 1;
    // Same termination rule as the CPU loop (while t < t_end).
}

// ------------------------------------------------------------------------- spatial operator

template <class Real>
__global__ void primitives(const Real* __restrict__ A, Real* __restrict__ W, Params<Real> P)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = P.n;
    if (idx >= n) {
        return;
    }
    const Real half(0.5);
    const Real rho = A[idx];
    const Real mx = A[n + idx];
    const Real my = A[2 * n + idx];
    const Real mz = A[3 * n + idx];
    const Real bx = A[5 * n + idx];
    const Real by = A[6 * n + idx];
    const Real bz = A[7 * n + idx];
    const Real inv_rho = Real(1.0) / rho;
    const Real ekin = half * (mx * mx + my * my + mz * mz) * inv_rho;
    const Real emag = half * (bx * bx + by * by + bz * bz);
    W[idx] = rho;
    W[n + idx] = mx * inv_rho;
    W[2 * n + idx] = my * inv_rho;
    W[3 * n + idx] = mz * inv_rho;
    W[4 * n + idx] = (P.gamma - Real(1.0)) * (A[4 * n + idx] - ekin - emag);
    W[5 * n + idx] = bx;
    W[6 * n + idx] = by;
    W[7 * n + idx] = bz;
    W[8 * n + idx] = A[8 * n + idx];
}

template <class Real>
__global__ void current(const Real* __restrict__ W, Real* __restrict__ J, Params<Real> P)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= P.nx_tot - 1 || j >= P.ny_tot - 1) {
        return;
    }
    const int n = P.n;
    const int nx = P.nx_tot;
    const int idx = j * nx + i;
    const Real* bx = W + 5 * n;
    const Real* by = W + 6 * n;
    const Real* bz = W + 7 * n;
    const Real inv_2dx = Real(0.5) * P.inv_dx;
    const Real inv_2dy = Real(0.5) * P.inv_dy;
    J[idx] = (bz[idx + nx] - bz[idx - nx]) * inv_2dy;
    J[n + idx] = -(bz[idx + 1] - bz[idx - 1]) * inv_2dx;
    J[2 * n + idx] =
        (by[idx + 1] - by[idx - 1]) * inv_2dx - (bx[idx + nx] - bx[idx - nx]) * inv_2dy;
}

template <class Real> struct V3 {
    Real x, y, z;
};

template <class Real> __device__ __forceinline__ V3<Real> cross3(V3<Real> a, V3<Real> b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

template <class Real>
__device__ __forceinline__ V3<Real> face_avg(const Real* f, int n, int a, int b)
{
    const Real half(0.5);
    return {half * (f[a] + f[b]), half * (f[n + a] + f[n + b]), half * (f[2 * n + a] + f[2 * n + b])};
}

// Mirrors nonideal_electric_field() in include/mhd/nonideal.hpp.
template <class Real>
__device__ V3<Real> nonideal_e(V3<Real> J, V3<Real> B, const Params<Real>& P)
{
    V3<Real> E{Real(0.0), Real(0.0), Real(0.0)};
    if (P.ohmic != 0) {
        E.x += P.eta_ohm * J.x;
        E.y += P.eta_ohm * J.y;
        E.z += P.eta_ohm * J.z;
    }
    if (P.hall == 0 && P.ambipolar == 0) {
        return E;
    }
    const Real B2 = B.x * B.x + B.y * B.y + B.z * B.z;
    if (B2 < Real(1.0e-14)) {
        return E;
    }
    const V3<Real> JxB = cross3(J, B);
    if (P.hall != 0) {
        const Real invB = Real(1.0) / mhd::simd::vsqrt(B2);
        E.x += P.eta_hall * JxB.x * invB;
        E.y += P.eta_hall * JxB.y * invB;
        E.z += P.eta_hall * JxB.z * invB;
    }
    if (P.ambipolar != 0) {
        const V3<Real> JxBxB = cross3(JxB, B);
        const Real invB2 = Real(1.0) / B2;
        E.x += P.eta_ambipolar * JxBxB.x * invB2;
        E.y += P.eta_ambipolar * JxBxB.y * invB2;
        E.z += P.eta_ambipolar * JxBxB.z * invB2;
    }
    return E;
}

template <class Real, bool YDir>
__device__ __forceinline__ mhd::kernels::PrimN<Real> load_prim(const Real* W, int n, int idx)
{
    mhd::kernels::PrimN<Real> p;
    p.rho = W[idx];
    p.vn = W[(YDir ? 2 : 1) * n + idx];
    p.vt = W[(YDir ? 1 : 2) * n + idx];
    p.vz = W[3 * n + idx];
    p.p = W[4 * n + idx];
    p.bn = W[(YDir ? 6 : 5) * n + idx];
    p.bt = W[(YDir ? 5 : 6) * n + idx];
    p.bz = W[7 * n + idx];
    p.psi = W[8 * n + idx];
    return p;
}

template <SlopeLimiter L, class Real, bool YDir>
__device__ __forceinline__ void face_flux(const Real* __restrict__ W, Real* __restrict__ F,
                                          const Real* __restrict__ J, const Ctrl* ctrl,
                                          const Params<Real>& P, int idx, int stride)
{
    const int n = P.n;
    const Real c_h = static_cast<Real>(ctrl->c_h);
    const auto fl = mhd::kernels::muscl_hll_face<L>(
        load_prim<Real, YDir>(W, n, idx - 2 * stride), load_prim<Real, YDir>(W, n, idx - stride),
        load_prim<Real, YDir>(W, n, idx), load_prim<Real, YDir>(W, n, idx + stride), P.gamma,
        P.inv_gm1, c_h);

    Real out[9] = {fl.rho, YDir ? fl.mt : fl.mn, YDir ? fl.mn : fl.mt, fl.mz, fl.e,
                   YDir ? fl.bt : fl.bn, YDir ? fl.bn : fl.bt, fl.bz, fl.psi};
    if (P.nonideal_any != 0) {
        const int a = idx - stride;
        const V3<Real> Jf = face_avg(J, n, a, idx);
        const V3<Real> B = face_avg(W + 5 * n, n, a, idx);
        const V3<Real> E = nonideal_e(Jf, B, P);
        if (!YDir) {
            out[6] += -E.z;
            out[7] += E.y;
            out[4] += B.y * E.z - B.z * E.y;
        } else {
            out[5] += E.z;
            out[7] += -E.x;
            out[4] += B.z * E.x - B.x * E.z;
        }
    }
    for (int c = 0; c < 9; ++c) {
        F[static_cast<std::size_t>(c) * n + idx] = out[c];
    }
}

template <SlopeLimiter L, class Real>
__global__ void flux_x(const Real* __restrict__ W, Real* __restrict__ fx,
                       const Real* __restrict__ J, const Ctrl* ctrl, Params<Real> P)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (k > P.nx || jj >= P.ny) {
        return;
    }
    face_flux<L, Real, false>(W, fx, J, ctrl, P, (jj + P.ng) * P.nx_tot + (k + P.ng), 1);
}

template <SlopeLimiter L, class Real>
__global__ void flux_y(const Real* __restrict__ W, Real* __restrict__ fy,
                       const Real* __restrict__ J, const Ctrl* ctrl, Params<Real> P)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii >= P.nx || k > P.ny) {
        return;
    }
    face_flux<L, Real, true>(W, fy, J, ctrl, P, (k + P.ng) * P.nx_tot + (ii + P.ng), P.nx_tot);
}

template <class Real>
__global__ void assemble_rhs(const Real* __restrict__ A, const Real* __restrict__ fx,
                             const Real* __restrict__ fy, Real* __restrict__ rhs,
                             const std::uint8_t* __restrict__ mask, const Ctrl* ctrl,
                             Params<Real> P)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii >= P.nx || jj >= P.ny) {
        return;
    }
    const int n = P.n;
    const int idx = (jj + P.ng) * P.nx_tot + (ii + P.ng);
    const bool solid = P.has_mask != 0 && mask[jj * P.nx + ii] == 0;
    for (int c = 0; c < 9; ++c) {
        const std::size_t o = static_cast<std::size_t>(c) * n + idx;
        const Real L = -(fx[o + 1] - fx[o]) * P.inv_dx - (fy[o + P.nx_tot] - fy[o]) * P.inv_dy;
        rhs[o] = solid ? Real(0.0) : L;
    }
    const Real c_h = static_cast<Real>(ctrl->c_h);
    const Real psi_damp =
        (c_h > Real(0.0) && P.glm_alpha > Real(0.0)) ? (c_h * P.glm_alpha / P.h) : Real(0.0);
    if (!solid) {
        rhs[8 * static_cast<std::size_t>(n) + idx] -= psi_damp * A[8 * static_cast<std::size_t>(n) + idx];
    }
}

template <class Real>
__global__ void rk_stage1(const Real* __restrict__ A, Real* __restrict__ B,
                          const Real* __restrict__ rhs, const Ctrl* ctrl, std::size_t total)
{
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= total) {
        return;
    }
    const Real dt = static_cast<Real>(ctrl->dt);
    B[k] = A[k] + dt * rhs[k];
}

template <class Real>
__global__ void rk_stage2(Real* __restrict__ A, const Real* __restrict__ B,
                          const Real* __restrict__ rhs, const Ctrl* ctrl, std::size_t total)
{
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= total) {
        return;
    }
    const Real half(0.5);
    const Real half_dt = half * static_cast<Real>(ctrl->dt);
    A[k] = half * (A[k] + B[k]) + half_dt * rhs[k];
}

// ------------------------------------------------------------------------- host side

template <class T> struct DeviceArray {
    T* ptr = nullptr;
    std::size_t count = 0;

    DeviceArray() = default;
    explicit DeviceArray(std::size_t n) : count(n)
    {
        MHD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max<std::size_t>(n, 1) * sizeof(T)));
        MHD_CUDA_CHECK(cudaMemset(ptr, 0, std::max<std::size_t>(n, 1) * sizeof(T)));
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& o) noexcept : ptr(o.ptr), count(o.count)
    {
        o.ptr = nullptr;
    }
    DeviceArray& operator=(DeviceArray&& o) noexcept
    {
        std::swap(ptr, o.ptr);
        std::swap(count, o.count);
        return *this;
    }
    ~DeviceArray()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

inline dim3 grid2(int w, int h, dim3 block)
{
    return dim3(static_cast<unsigned>((w + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
                static_cast<unsigned>((h + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)));
}

template <class Real> class CudaSolver final : public mhd::DeviceSolver {
  public:
    explicit CudaSolver(const Grid2D& grid) : grid_(grid)
    {
        int dev = 0;
        MHD_CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        MHD_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        name_ = std::string("CUDA: ") + prop.name;
        MHD_CUDA_CHECK(cudaStreamCreate(&stream_));

        n_ = static_cast<std::size_t>(grid.get_size_x()) * static_cast<std::size_t>(grid.get_size_y());
        U_ = DeviceArray<Real>(9 * n_);
        Ustar_ = DeviceArray<Real>(9 * n_);
        rhs_ = DeviceArray<Real>(9 * n_);
        prim_ = DeviceArray<Real>(9 * n_);
        fx_ = DeviceArray<Real>(9 * n_);
        fy_ = DeviceArray<Real>(9 * n_);
        J_ = DeviceArray<Real>(3 * n_);
        const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny;
        partial_count_ = static_cast<int>((cells + kReduceThreads - 1) / kReduceThreads);
        partial_ = DeviceArray<Real>(static_cast<std::size_t>(partial_count_));
        mask_ = DeviceArray<std::uint8_t>(cells);
        ctrl_ = DeviceArray<Ctrl>(1);
        MHD_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ctrl_host_), sizeof(Ctrl)));
        staging_.resize(9 * n_);
    }

    ~CudaSolver() override
    {
        if (ctrl_host_ != nullptr) {
            cudaFreeHost(ctrl_host_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    std::string device_name() const override
    {
        return name_;
    }

    mhd::Precision precision() const override
    {
        return sizeof(Real) == 8 ? mhd::Precision::FP64 : mhd::Precision::FP32;
    }

    void configure(const BoundaryConditions& bc, const FluidMask* mask,
                   bool reflective_walls) override
    {
        if (bc.type_x == BoundaryConditionType::Reflective ||
            bc.type_y == BoundaryConditionType::Reflective) {
            throw std::invalid_argument("Reflective boundary conditions are not implemented yet");
        }
        bc_ = bc;
        reflective_walls_ = reflective_walls;
        has_mask_ = mask != nullptr && !mask->empty() && !mask->all_fluid();
        if (has_mask_) {
            if (mask->nx != grid_.nx || mask->ny != grid_.ny) {
                throw std::invalid_argument("FluidMask size does not match Grid2D interior");
            }
            MHD_CUDA_CHECK(cudaMemcpy(mask_.ptr, mask->fluid.data(), mask->fluid.size(),
                                      cudaMemcpyHostToDevice));
        }
    }

    void upload(const StateField& U) override
    {
        check_shape(U);
        const std::vector<double>* comps[9] = {&U.rho, &U.mx, &U.my, &U.mz, &U.energy,
                                               &U.bx,  &U.by, &U.bz, &U.psi};
        for (std::size_t c = 0; c < 9; ++c) {
            const double* s = comps[c]->data();
            for (std::size_t k = 0; k < n_; ++k) {
                staging_[c * n_ + k] = static_cast<Real>(s[k]);
            }
        }
        MHD_CUDA_CHECK(cudaMemcpy(U_.ptr, staging_.data(), 9 * n_ * sizeof(Real),
                                  cudaMemcpyHostToDevice));
    }

    void download(StateField& U) override
    {
        check_shape(U);
        MHD_CUDA_CHECK(cudaMemcpy(staging_.data(), U_.ptr, 9 * n_ * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        std::vector<double>* comps[9] = {&U.rho, &U.mx, &U.my, &U.mz, &U.energy,
                                         &U.bx,  &U.by, &U.bz, &U.psi};
        for (std::size_t c = 0; c < 9; ++c) {
            double* d = comps[c]->data();
            for (std::size_t k = 0; k < n_; ++k) {
                d[k] = static_cast<double>(staging_[c * n_ + k]);
            }
        }
    }

    SolveResult solve(const SolveParams& params) override
    {
        if (params.t_end < 0.0) {
            throw std::invalid_argument("t_end must be non-negative");
        }
        if (params.cfl <= 0.0) {
            throw std::invalid_argument("CFL number must be positive");
        }
        const Params<Real> P = make_params(params);

        *ctrl_host_ = Ctrl{0.0, 0.0, 0.0, 0u, params.t_end > 0.0 ? 0u : 1u};
        MHD_CUDA_CHECK(cudaMemcpyAsync(ctrl_.ptr, ctrl_host_, sizeof(Ctrl),
                                       cudaMemcpyHostToDevice, stream_));

        int batch = 4;
        double last_t = 0.0;
        unsigned last_steps = 0;
        for (;;) {
            batch = std::min(batch, std::max(1, params.max_steps - static_cast<int>(ctrl_host_->steps)));
            for (int s = 0; s < batch; ++s) {
                encode_step(P, params.limiter);
            }
            MHD_CUDA_CHECK(cudaGetLastError());
            MHD_CUDA_CHECK(cudaMemcpyAsync(ctrl_host_, ctrl_.ptr, sizeof(Ctrl),
                                           cudaMemcpyDeviceToHost, stream_));
            MHD_CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (ctrl_host_->finished != 0 || ctrl_host_->t >= params.t_end ||
                static_cast<int>(ctrl_host_->steps) >= params.max_steps) {
                break;
            }
            const double t = ctrl_host_->t;
            const unsigned steps = ctrl_host_->steps;
            const double dt_avg = steps > last_steps ? (t - last_t) / (steps - last_steps) : 0.0;
            last_t = t;
            last_steps = steps;
            int est = batch * 2;
            if (dt_avg > 0.0) {
                est = static_cast<int>(std::ceil((params.t_end - t) / dt_avg)) + 1;
            }
            batch = std::clamp(est, 1, 256);
        }

        SolveResult result{};
        result.t = ctrl_host_->t;
        result.steps = static_cast<int>(ctrl_host_->steps);
        result.c_h = ctrl_host_->c_h;
        if (result.t < params.t_end) {
            throw std::runtime_error("solve: reached max_steps before t_end");
        }

        encode_boundaries(P, U_.ptr);
        MHD_CUDA_CHECK(cudaGetLastError());
        MHD_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return result;
    }

  private:
    void check_shape(const StateField& U) const
    {
        if (U.nx != grid_.get_size_x() || U.ny != grid_.get_size_y()) {
            throw std::invalid_argument("StateField size does not match Grid2D");
        }
    }

    Params<Real> make_params(const SolveParams& sp) const
    {
        Params<Real> P{};
        P.nx_tot = grid_.get_size_x();
        P.ny_tot = grid_.get_size_y();
        P.nx = grid_.nx;
        P.ny = grid_.ny;
        P.ng = grid_.ng;
        P.n = static_cast<int>(n_);
        P.bc_x = static_cast<int>(bc_.type_x);
        P.bc_y = static_cast<int>(bc_.type_y);
        P.has_mask = has_mask_ ? 1 : 0;
        P.reflective_walls = reflective_walls_ ? 1 : 0;
        P.ohmic = sp.nonideal.ohmic ? 1 : 0;
        P.hall = sp.nonideal.hall ? 1 : 0;
        P.ambipolar = sp.nonideal.ambipolar ? 1 : 0;
        P.nonideal_any = sp.nonideal.any() ? 1 : 0;
        P.partial_count = partial_count_;
        P.inv_dx = static_cast<Real>(1.0 / grid_.dx);
        P.inv_dy = static_cast<Real>(1.0 / grid_.dy);
        P.gamma = static_cast<Real>(sp.gamma);
        P.inv_gm1 = static_cast<Real>(1.0 / (sp.gamma - 1.0));
        P.glm_alpha = static_cast<Real>(sp.glm_alpha);
        const double h = std::min(grid_.dx, grid_.dy);
        P.h = static_cast<Real>(h);
        P.eta_ohm = static_cast<Real>(sp.nonideal.eta_ohm);
        P.eta_hall = static_cast<Real>(sp.nonideal.eta_hall);
        P.eta_ambipolar = static_cast<Real>(sp.nonideal.eta_ambipolar);
        P.cfl = sp.cfl;
        const double eta_max = sp.nonideal.max_diffusivity();
        P.dt_diff = eta_max > 0.0 ? sp.cfl * h * h / (4.0 * eta_max) : 0.0;
        P.t_end = sp.t_end;
        return P;
    }

    void encode_boundaries(const Params<Real>& P, Real* A)
    {
        const dim3 bx_block(2, 128);
        bc_x<Real><<<grid2(grid_.ng, grid_.get_size_y(), bx_block), bx_block, 0, stream_>>>(A, P);
        const dim3 by_block(128, 2);
        bc_y<Real><<<grid2(grid_.get_size_x(), grid_.ng, by_block), by_block, 0, stream_>>>(A, P);
        if (has_mask_ && reflective_walls_) {
            const dim3 b(32, 8);
            solid_fill<Real><<<grid2(grid_.nx, grid_.ny, b), b, 0, stream_>>>(A, mask_.ptr, P);
        }
    }

    template <SlopeLimiter L> void encode_fluxes(const Params<Real>& P)
    {
        const dim3 b(32, 4);
        flux_x<L, Real><<<grid2(grid_.nx + 1, grid_.ny, b), b, 0, stream_>>>(prim_.ptr, fx_.ptr,
                                                                             J_.ptr, ctrl_.ptr, P);
        flux_y<L, Real><<<grid2(grid_.nx, grid_.ny + 1, b), b, 0, stream_>>>(prim_.ptr, fy_.ptr,
                                                                             J_.ptr, ctrl_.ptr, P);
    }

    void encode_rhs(const Params<Real>& P, SlopeLimiter limiter, const Real* A)
    {
        const unsigned threads = 256;
        primitives<Real><<<static_cast<unsigned>((n_ + threads - 1) / threads), threads, 0,
                           stream_>>>(A, prim_.ptr, P);
        if (P.nonideal_any != 0) {
            const dim3 b(32, 8);
            current<Real><<<grid2(grid_.get_size_x() - 2, grid_.get_size_y() - 2, b), b, 0,
                            stream_>>>(prim_.ptr, J_.ptr, P);
        }
        if (limiter == SlopeLimiter::Minmod) {
            encode_fluxes<SlopeLimiter::Minmod>(P);
        } else {
            encode_fluxes<SlopeLimiter::MC>(P);
        }
        const dim3 b(32, 8);
        assemble_rhs<Real><<<grid2(grid_.nx, grid_.ny, b), b, 0, stream_>>>(
            A, fx_.ptr, fy_.ptr, rhs_.ptr, mask_.ptr, ctrl_.ptr, P);
    }

    void encode_step(const Params<Real>& P, SlopeLimiter limiter)
    {
        const std::size_t total = 9 * n_;
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);

        encode_boundaries(P, U_.ptr);
        cfl_partial<Real><<<static_cast<unsigned>(partial_count_), kReduceThreads, 0, stream_>>>(
            U_.ptr, mask_.ptr, partial_.ptr, P);
        cfl_finalize<Real><<<1, kReduceThreads, 0, stream_>>>(partial_.ptr, ctrl_.ptr, P);

        encode_rhs(P, limiter, U_.ptr);
        rk_stage1<Real><<<blocks, threads, 0, stream_>>>(U_.ptr, Ustar_.ptr, rhs_.ptr, ctrl_.ptr,
                                                         total);

        encode_boundaries(P, Ustar_.ptr);
        encode_rhs(P, limiter, Ustar_.ptr);
        rk_stage2<Real><<<blocks, threads, 0, stream_>>>(U_.ptr, Ustar_.ptr, rhs_.ptr, ctrl_.ptr,
                                                         total);
        advance_time<<<1, 1, 0, stream_>>>(ctrl_.ptr);
    }

    Grid2D grid_;
    BoundaryConditions bc_{BoundaryConditionType::Periodic, BoundaryConditionType::Periodic};
    bool has_mask_ = false;
    bool reflective_walls_ = true;
    std::string name_;
    cudaStream_t stream_ = nullptr;
    std::size_t n_ = 0;
    int partial_count_ = 0;
    DeviceArray<Real> U_, Ustar_, rhs_, prim_, fx_, fy_, J_, partial_;
    DeviceArray<std::uint8_t> mask_;
    DeviceArray<Ctrl> ctrl_;
    Ctrl* ctrl_host_ = nullptr;
    std::vector<Real> staging_;
};

} // namespace

bool mhd::cuda_available()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<mhd::DeviceSolver> mhd::make_cuda_solver(const Grid2D& grid, Precision precision)
{
    if (grid.ng < 2) {
        throw std::invalid_argument("CUDA solver requires grid.ng >= 2");
    }
    if (!cuda_available()) {
        throw std::runtime_error("CUDA: no device available");
    }
    if (precision == Precision::FP32) {
        return std::make_unique<CudaSolver<float>>(grid);
    }
    return std::make_unique<CudaSolver<double>>(grid);
}
