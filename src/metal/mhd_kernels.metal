// GLM-MHD kernels for Apple GPUs (Metal Shading Language, fp32).
//
// Mirrors include/mhd/kernels/muscl_hll.hpp (MUSCL + HLL + Dedner GLM in a rotated frame),
// include/mhd/nonideal.hpp and the SSP-RK2 / CFL / boundary logic of the CPU solver.
// Apple GPUs have no fp64, so this backend is single precision; see docs/gpu.md for the
// accuracy study against the fp64 CPU solver.
//
// Field layout: SoA, component c of cell idx at [c * n + idx], n = nx_tot * ny_tot.
//   conserved: rho, mx, my, mz, E, bx, by, bz, psi
//   primitive: rho, vx, vy, vz, p, bx, by, bz, psi
// The host (src/metal/metal_solver.mm) compiles this file at runtime and binds buffers as:
//   0 Params (constant)  1 Ctrl  2 A (state being processed)  3 prim  4 fx  5 fy  6 rhs
//   7 J (3 comps)  8 mask (interior, uchar)  9 partial maxima  10 B (second state)

#include <metal_stdlib>
using namespace metal;

struct Params {
    int nx_tot, ny_tot, nx, ny, ng, n;
    int limiter;        // 0 = minmod, 1 = MC
    int bc_x, bc_y;     // 0 = periodic, 2 = outflow (BoundaryConditionType)
    int has_mask, reflective_walls;
    int ohmic, hall, ambipolar, nonideal_any;
    int partial_count;
    float inv_dx, inv_dy, gamma, inv_gm1, cfl, glm_alpha, h, dt_diff;
    float eta_ohm, eta_hall, eta_ambipolar;
    float t_end_hi, t_end_lo;
};

struct Ctrl {
    float t_hi;
    float t_lo;
    float dt;
    float c_h;
    uint steps;
    uint finished;
};

constant int ROT_X[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
constant int ROT_Y[9] = {0, 2, 1, 3, 4, 6, 5, 7, 8}; // swap (vx,vy) and (bx,by)

// ---------------------------------------------------------------------------------------------
// Point-wise physics (rotated frame: 0 rho, 1 vn, 2 vt, 3 vz, 4 p, 5 bn, 6 bt, 7 bz, 8 psi)

inline float minmod(float a, float b)
{
    return (a * b <= 0.0f) ? 0.0f : ((fabs(a) < fabs(b)) ? a : b);
}

inline float mc_limiter(float a, float b)
{
    return minmod(0.5f * (a + b), minmod(2.0f * a, 2.0f * b));
}

inline float limit_slope(float a, float b, int limiter)
{
    return limiter == 0 ? minmod(a, b) : mc_limiter(a, b);
}

inline float vmaxf(float a, float b) { return (a < b) ? b : a; }
inline float vminf(float a, float b) { return (b < a) ? b : a; }

inline float fast_speed_n(thread const float* W, float gamma)
{
    const float inv_rho = 1.0f / W[0];
    const float cs2 = gamma * W[4] * inv_rho;
    const float bn2 = W[5] * W[5];
    const float b2 = bn2 + W[6] * W[6] + W[7] * W[7];
    const float va2 = b2 * inv_rho;
    const float van2 = bn2 * inv_rho;
    const float sum = cs2 + va2;
    const float disc = vmaxf(sum * sum - 4.0f * cs2 * van2, 0.0f);
    const float cf2 = 0.5f * (sum + sqrt(disc));
    return sqrt(vmaxf(cf2, 0.0f));
}

inline void conserved_and_flux(thread const float* W, float inv_gm1, float ch2, thread float* U,
                               thread float* F)
{
    const float mn = W[0] * W[1];
    const float mt = W[0] * W[2];
    const float mz = W[0] * W[3];
    const float v2 = W[1] * W[1] + W[2] * W[2] + W[3] * W[3];
    const float b2 = W[5] * W[5] + W[6] * W[6] + W[7] * W[7];
    const float pmag = 0.5f * b2;
    const float energy = W[4] * inv_gm1 + 0.5f * W[0] * v2 + pmag;
    const float pstar = W[4] + pmag;
    const float vb = W[1] * W[5] + W[2] * W[6] + W[3] * W[7];

    U[0] = W[0];
    U[1] = mn;
    U[2] = mt;
    U[3] = mz;
    U[4] = energy;
    U[5] = W[5];
    U[6] = W[6];
    U[7] = W[7];
    U[8] = W[8];

    F[0] = mn;
    F[1] = mn * W[1] + pstar - W[5] * W[5];
    F[2] = mn * W[2] - W[5] * W[6];
    F[3] = mn * W[3] - W[5] * W[7];
    F[4] = (energy + pstar) * W[1] - W[5] * vb + W[8] * W[5];
    F[5] = W[8];
    F[6] = W[1] * W[6] - W[2] * W[5];
    F[7] = W[1] * W[7] - W[3] * W[5];
    F[8] = ch2 * W[5];
}

// Face flux between cells idx - stride and idx; result in physical component order.
inline void muscl_hll_face(device const float* prim, int n, int idx, int stride, bool ydir,
                           constant Params& P, float c_h, thread float* Fphys)
{
    float WL[9];
    float WR[9];
    float Wm[9];
    float W0[9];
    for (int c = 0; c < 9; ++c) {
        const int pc = ydir ? ROT_Y[c] : ROT_X[c];
        device const float* q = prim + pc * n + idx;
        const float wmm = q[-2 * stride];
        const float wm = q[-stride];
        const float w0 = q[0];
        const float wp = q[stride];
        const float dm = limit_slope(wm - wmm, w0 - wm, P.limiter);
        const float d0 = limit_slope(w0 - wm, wp - w0, P.limiter);
        WL[c] = wm + 0.5f * dm;
        WR[c] = w0 - 0.5f * d0;
        Wm[c] = wm;
        W0[c] = w0;
    }

    const bool ok = WL[0] > 0.0f && WL[4] > 0.0f && WL[0] < INFINITY && WL[4] < INFINITY &&
                    WR[0] > 0.0f && WR[4] > 0.0f && WR[0] < INFINITY && WR[4] < INFINITY;
    if (!ok) {
        for (int c = 0; c < 9; ++c) {
            WL[c] = Wm[c];
            WR[c] = W0[c];
        }
    }

    const float cfL = fast_speed_n(WL, P.gamma);
    const float cfR = fast_speed_n(WR, P.gamma);
    const float sL = vminf(vminf(WL[1] - cfL, WR[1] - cfR), -c_h);
    const float sR = vmaxf(vmaxf(WL[1] + cfL, WR[1] + cfR), c_h);

    float UL[9];
    float UR[9];
    float FL[9];
    float FR[9];
    const float ch2 = c_h * c_h;
    conserved_and_flux(WL, P.inv_gm1, ch2, UL, FL);
    conserved_and_flux(WR, P.inv_gm1, ch2, UR, FR);

    const float denom = sR - sL;
    const float inv_denom = 1.0f / denom;
    const float sLsR = sL * sR;
    for (int c = 0; c < 9; ++c) {
        float f;
        if (sL >= 0.0f) {
            f = FL[c];
        } else if (sR <= 0.0f) {
            f = FR[c];
        } else if (fabs(denom) < 1.0e-14f) {
            f = 0.5f * (FL[c] + FR[c]);
        } else {
            f = (sR * FL[c] - sL * FR[c] + sLsR * (UR[c] - UL[c])) * inv_denom;
        }
        Fphys[ydir ? ROT_Y[c] : ROT_X[c]] = f;
    }
}

inline float3 nonideal_electric_field(float3 J, float3 B, constant Params& P)
{
    float3 E = float3(0.0f);
    if (P.ohmic != 0) {
        E += P.eta_ohm * J;
    }
    if (P.hall == 0 && P.ambipolar == 0) {
        return E;
    }
    const float B2 = dot(B, B);
    if (B2 < 1.0e-14f) {
        return E;
    }
    const float3 JxB = cross(J, B);
    if (P.hall != 0) {
        E += P.eta_hall * JxB * (1.0f / sqrt(B2));
    }
    if (P.ambipolar != 0) {
        E += P.eta_ambipolar * cross(JxB, B) * (1.0f / B2);
    }
    return E;
}

inline float cell_max_signal_speed(device const float* U, int n, int idx, float gamma)
{
    const float rho = U[idx];
    const float mx = U[n + idx];
    const float my = U[2 * n + idx];
    const float mz = U[3 * n + idx];
    const float e = U[4 * n + idx];
    const float bx = U[5 * n + idx];
    const float by = U[6 * n + idx];
    const float bz = U[7 * n + idx];
    const float inv_rho = 1.0f / rho;
    const float ekin = 0.5f * (mx * mx + my * my + mz * mz) * inv_rho;
    const float emag = 0.5f * (bx * bx + by * by + bz * bz);
    float W[9] = {rho, mx * inv_rho, my * inv_rho, mz * inv_rho,
                  (gamma - 1.0f) * (e - ekin - emag), bx, by, bz, 0.0f};
    const float sx = fabs(W[1]) + fast_speed_n(W, gamma);
    float Wy[9] = {W[0], W[2], W[1], W[3], W[4], by, bx, bz, 0.0f};
    const float sy = fabs(Wy[1]) + fast_speed_n(Wy, gamma);
    return vmaxf(sx, sy);
}

// ---------------------------------------------------------------------------------------------
// Boundary conditions and solid walls

// Same as periodic_source() in boundary_conditions.hpp (wraps modulo n, valid for n < ng).
inline int periodic_source(int g, int n, int ng, bool low)
{
    const int offset = low ? (g - ng) : g;
    return ng + ((offset % n) + n) % n;
}

kernel void bc_x(device float* A [[buffer(2)]], constant Params& P [[buffer(0)]],
                 uint2 gid [[thread_position_in_grid]])
{
    const int g = int(gid.x);
    const int j = int(gid.y);
    if (g >= P.ng || j >= P.ny_tot) {
        return;
    }
    const int row = j * P.nx_tot;
    const int left = g;
    const int right = P.ng + P.nx + g;
    const int src_l = (P.bc_x == 0) ? periodic_source(g, P.nx, P.ng, true) : P.ng;
    const int src_r = (P.bc_x == 0) ? periodic_source(g, P.nx, P.ng, false) : (P.ng + P.nx - 1);
    for (int c = 0; c < 9; ++c) {
        device float* f = A + c * P.n + row;
        f[left] = f[src_l];
        f[right] = f[src_r];
    }
}

kernel void bc_y(device float* A [[buffer(2)]], constant Params& P [[buffer(0)]],
                 uint2 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x);
    const int g = int(gid.y);
    if (i >= P.nx_tot || g >= P.ng) {
        return;
    }
    const int bottom = g;
    const int top = P.ng + P.ny + g;
    const int src_b = (P.bc_y == 0) ? periodic_source(g, P.ny, P.ng, true) : P.ng;
    const int src_t = (P.bc_y == 0) ? periodic_source(g, P.ny, P.ng, false) : (P.ng + P.ny - 1);
    for (int c = 0; c < 9; ++c) {
        device float* f = A + c * P.n;
        f[bottom * P.nx_tot + i] = f[src_b * P.nx_tot + i];
        f[top * P.nx_tot + i] = f[src_t * P.nx_tot + i];
    }
}

// Reflective (slip) fill of solid cells from the first fluid neighbour (right, left, top, bottom).
kernel void solid_fill(device float* A [[buffer(2)]], device const uchar* mask [[buffer(8)]],
                       constant Params& P [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    const int ii = int(gid.x);
    const int jj = int(gid.y);
    if (ii >= P.nx || jj >= P.ny || mask[jj * P.nx + ii] != 0) {
        return;
    }
    const int n = P.n;
    const int dst = (jj + P.ng) * P.nx_tot + (ii + P.ng);
    const int di[4] = {1, -1, 0, 0};
    const int dj[4] = {0, 0, 1, -1};
    for (int k = 0; k < 4; ++k) {
        const int ni = ii + di[k];
        const int nj = jj + dj[k];
        if (ni < 0 || nj < 0 || ni >= P.nx || nj >= P.ny || mask[nj * P.nx + ni] == 0) {
            continue;
        }
        const int src = (nj + P.ng) * P.nx_tot + (ni + P.ng);
        float rho = A[src];
        float vx = A[n + src] / rho;
        float vy = A[2 * n + src] / rho;
        const float vz = A[3 * n + src] / rho;
        const float bx = A[5 * n + src];
        const float by = A[6 * n + src];
        const float bz = A[7 * n + src];
        const float emag = 0.5f * (bx * bx + by * by + bz * bz);
        float p = (P.gamma - 1.0f) * (A[4 * n + src] - 0.5f * rho * (vx * vx + vy * vy + vz * vz) -
                                      emag);
        if (k < 2) {
            vx = -vx;
        } else {
            vy = -vy;
        }
        rho = vmaxf(rho, 1.0e-12f);
        p = vmaxf(p, 1.0e-12f);
        A[dst] = rho;
        A[n + dst] = rho * vx;
        A[2 * n + dst] = rho * vy;
        A[3 * n + dst] = rho * vz;
        A[4 * n + dst] = p * P.inv_gm1 + 0.5f * rho * (vx * vx + vy * vy + vz * vz) + emag;
        A[5 * n + dst] = bx;
        A[6 * n + dst] = by;
        A[7 * n + dst] = bz;
        A[8 * n + dst] = A[8 * n + src];
        return;
    }
    A[dst] = 1.0f;
    for (int c = 1; c < 9; ++c) {
        A[c * n + dst] = 0.0f;
    }
    A[4 * n + dst] = P.inv_gm1;
}

// ---------------------------------------------------------------------------------------------
// CFL: per-threadgroup max signal speed, then a single-threadgroup finalize that sets dt.

constant uint REDUCE_TG = 256;

kernel void cfl_partial(device const float* A [[buffer(2)]], device const uchar* mask [[buffer(8)]],
                        device float* partial [[buffer(9)]], constant Params& P [[buffer(0)]],
                        uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint tg [[threadgroup_position_in_grid]])
{
    threadgroup float sh[REDUCE_TG];
    float v = 0.0f;
    const int cells = P.nx * P.ny;
    if (int(gid) < cells) {
        const int ii = int(gid) % P.nx;
        const int jj = int(gid) / P.nx;
        if (P.has_mask == 0 || mask[jj * P.nx + ii] != 0) {
            const int idx = (jj + P.ng) * P.nx_tot + (ii + P.ng);
            v = vmaxf(0.0f, cell_max_signal_speed(A, P.n, idx, P.gamma));
        }
    }
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_TG / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] = vmaxf(sh[tid], sh[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partial[tg] = sh[0];
    }
}

kernel void cfl_finalize(device const float* partial [[buffer(9)]], device Ctrl& ctrl [[buffer(1)]],
                         constant Params& P [[buffer(0)]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float sh[REDUCE_TG];
    float v = 0.0f;
    for (int k = int(tid); k < P.partial_count; k += int(REDUCE_TG)) {
        v = vmaxf(v, partial[k]);
    }
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_TG / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] = vmaxf(sh[tid], sh[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid != 0) {
        return;
    }
    const float c_h = sh[0];
    float dt = (c_h > 0.0f) ? P.cfl / (c_h * P.inv_dx + c_h * P.inv_dy) : INFINITY;
    if (P.dt_diff > 0.0f) {
        dt = vminf(dt, P.dt_diff);
    }
    // Remaining time in double-float arithmetic (t and t_end are stored as hi + lo pairs).
    const float remaining = (P.t_end_hi - ctrl.t_hi) + (P.t_end_lo - ctrl.t_lo);
    if (ctrl.finished != 0 || remaining <= 1.0e-5f * dt) {
        ctrl.finished = 1;
        ctrl.dt = 0.0f;
        return;
    }
    ctrl.dt = vminf(dt, remaining);
    ctrl.c_h = c_h;
}

// t += dt with a compensated (two-sum) accumulator; counts real steps.
kernel void advance_time(device Ctrl& ctrl [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
    if (gid != 0 || ctrl.dt <= 0.0f) {
        return;
    }
    const float a = ctrl.t_hi;
    const float b = ctrl.dt;
    const float s = a + b;
    const float bp = s - a;
    const float err = (a - (s - bp)) + (b - bp);
    const float lo = ctrl.t_lo + err;
    const float hi = s + lo;
    ctrl.t_lo = lo - (hi - s);
    ctrl.t_hi = hi;
    ctrl.steps += 1;
}

// ---------------------------------------------------------------------------------------------
// Spatial operator

kernel void primitives(device const float* A [[buffer(2)]], device float* W [[buffer(3)]],
                       constant Params& P [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
    const int idx = int(gid);
    const int n = P.n;
    if (idx >= n) {
        return;
    }
    const float rho = A[idx];
    const float mx = A[n + idx];
    const float my = A[2 * n + idx];
    const float mz = A[3 * n + idx];
    const float bx = A[5 * n + idx];
    const float by = A[6 * n + idx];
    const float bz = A[7 * n + idx];
    const float inv_rho = 1.0f / rho;
    const float ekin = 0.5f * (mx * mx + my * my + mz * mz) * inv_rho;
    const float emag = 0.5f * (bx * bx + by * by + bz * bz);
    W[idx] = rho;
    W[n + idx] = mx * inv_rho;
    W[2 * n + idx] = my * inv_rho;
    W[3 * n + idx] = mz * inv_rho;
    W[4 * n + idx] = (P.gamma - 1.0f) * (A[4 * n + idx] - ekin - emag);
    W[5 * n + idx] = bx;
    W[6 * n + idx] = by;
    W[7 * n + idx] = bz;
    W[8 * n + idx] = A[8 * n + idx];
}

// J = curl B (2.5D), centred differences on all cells with both neighbours.
kernel void current(device const float* W [[buffer(3)]], device float* J [[buffer(7)]],
                    constant Params& P [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x) + 1;
    const int j = int(gid.y) + 1;
    if (i >= P.nx_tot - 1 || j >= P.ny_tot - 1) {
        return;
    }
    const int n = P.n;
    const int nx = P.nx_tot;
    const int idx = j * nx + i;
    device const float* bx = W + 5 * n;
    device const float* by = W + 6 * n;
    device const float* bz = W + 7 * n;
    const float inv_2dx = 0.5f * P.inv_dx;
    const float inv_2dy = 0.5f * P.inv_dy;
    J[idx] = (bz[idx + nx] - bz[idx - nx]) * inv_2dy;
    J[n + idx] = -(bz[idx + 1] - bz[idx - 1]) * inv_2dx;
    J[2 * n + idx] = (by[idx + 1] - by[idx - 1]) * inv_2dx - (bx[idx + nx] - bx[idx - nx]) * inv_2dy;
}

inline float3 load3(device const float* f, int n, int idx)
{
    return float3(f[idx], f[n + idx], f[2 * n + idx]);
}

// x-face i-1/2 of interior row j, stored at cell (i, j): faces i = ng .. ng + nx.
kernel void flux_x(device const float* W [[buffer(3)]], device float* fx [[buffer(4)]],
                   device const float* J [[buffer(7)]], device const Ctrl& ctrl [[buffer(1)]],
                   constant Params& P [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    const int k = int(gid.x);
    const int jj = int(gid.y);
    if (k > P.nx || jj >= P.ny) {
        return;
    }
    const int n = P.n;
    const int idx = (jj + P.ng) * P.nx_tot + (k + P.ng);
    float F[9];
    muscl_hll_face(W, n, idx, 1, false, P, ctrl.c_h, F);
    if (P.nonideal_any != 0) {
        const float3 Jf = 0.5f * (load3(J, n, idx - 1) + load3(J, n, idx));
        const float3 B = 0.5f * (load3(W + 5 * n, n, idx - 1) + load3(W + 5 * n, n, idx));
        const float3 E = nonideal_electric_field(Jf, B, P);
        F[6] += -E.z;
        F[7] += E.y;
        F[4] += B.y * E.z - B.z * E.y;
    }
    for (int c = 0; c < 9; ++c) {
        fx[c * n + idx] = F[c];
    }
}

// y-face j-1/2 of interior column i, stored at cell (i, j): faces j = ng .. ng + ny.
kernel void flux_y(device const float* W [[buffer(3)]], device float* fy [[buffer(5)]],
                   device const float* J [[buffer(7)]], device const Ctrl& ctrl [[buffer(1)]],
                   constant Params& P [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    const int ii = int(gid.x);
    const int k = int(gid.y);
    if (ii >= P.nx || k > P.ny) {
        return;
    }
    const int n = P.n;
    const int idx = (k + P.ng) * P.nx_tot + (ii + P.ng);
    float F[9];
    muscl_hll_face(W, n, idx, P.nx_tot, true, P, ctrl.c_h, F);
    if (P.nonideal_any != 0) {
        const int ib = idx - P.nx_tot;
        const float3 Jf = 0.5f * (load3(J, n, ib) + load3(J, n, idx));
        const float3 B = 0.5f * (load3(W + 5 * n, n, ib) + load3(W + 5 * n, n, idx));
        const float3 E = nonideal_electric_field(Jf, B, P);
        F[5] += E.z;
        F[7] += -E.x;
        F[4] += B.z * E.x - B.x * E.z;
    }
    for (int c = 0; c < 9; ++c) {
        fy[c * n + idx] = F[c];
    }
}

kernel void assemble_rhs(device const float* A [[buffer(2)]], device const float* fx [[buffer(4)]],
                         device const float* fy [[buffer(5)]], device float* rhs [[buffer(6)]],
                         device const uchar* mask [[buffer(8)]], device const Ctrl& ctrl [[buffer(1)]],
                         constant Params& P [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    const int ii = int(gid.x);
    const int jj = int(gid.y);
    if (ii >= P.nx || jj >= P.ny) {
        return;
    }
    const int n = P.n;
    const int idx = (jj + P.ng) * P.nx_tot + (ii + P.ng);
    const bool solid = P.has_mask != 0 && mask[jj * P.nx + ii] == 0;
    for (int c = 0; c < 9; ++c) {
        const int o = c * n + idx;
        const float L = -(fx[o + 1] - fx[o]) * P.inv_dx - (fy[o + P.nx_tot] - fy[o]) * P.inv_dy;
        rhs[o] = solid ? 0.0f : L;
    }
    const float c_h = ctrl.c_h;
    const float psi_damp = (c_h > 0.0f && P.glm_alpha > 0.0f) ? (c_h * P.glm_alpha / P.h) : 0.0f;
    if (!solid) {
        rhs[8 * n + idx] -= psi_damp * A[8 * n + idx];
    }
}

// B = A + dt * rhs over every cell (stage 1 of SSP-RK2).
kernel void rk_stage1(device const float* A [[buffer(2)]], device float* B [[buffer(10)]],
                      device const float* rhs [[buffer(6)]], device const Ctrl& ctrl [[buffer(1)]],
                      constant Params& P [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
    const int k = int(gid);
    if (k >= 9 * P.n) {
        return;
    }
    B[k] = A[k] + ctrl.dt * rhs[k];
}

// A = 1/2 (A + B) + dt/2 * rhs (stage 2 of SSP-RK2).
kernel void rk_stage2(device float* A [[buffer(2)]], device const float* B [[buffer(10)]],
                      device const float* rhs [[buffer(6)]], device const Ctrl& ctrl [[buffer(1)]],
                      constant Params& P [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
    const int k = int(gid);
    if (k >= 9 * P.n) {
        return;
    }
    A[k] = 0.5f * (A[k] + B[k]) + 0.5f * ctrl.dt * rhs[k];
}
