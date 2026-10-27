// Metal backend host code (Objective-C++, ARC).
// Kernels: src/metal/mhd_kernels.metal, embedded at build time and compiled at runtime
// (no offline `metal` toolchain required, works with Command Line Tools only).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mhd/gpu/device_solver.hpp"
#include "mhd_metal_source.hpp" // generated: kMetalKernelSource

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Must match `struct Params` / `struct Ctrl` in mhd_kernels.metal.
struct Params {
    int nx_tot, ny_tot, nx, ny, ng, n;
    int limiter;
    int bc_x, bc_y;
    int has_mask, reflective_walls;
    int ohmic, hall, ambipolar, nonideal_any;
    int partial_count;
    float inv_dx, inv_dy, gamma, inv_gm1, cfl, glm_alpha, h, dt_diff;
    float eta_ohm, eta_hall, eta_ambipolar;
    float t_end_hi, t_end_lo;
};
static_assert(sizeof(Params) == 29 * 4, "Params layout must match the shader");

struct Ctrl {
    float t_hi;
    float t_lo;
    float dt;
    float c_h;
    unsigned steps;
    unsigned finished;
};

constexpr NSUInteger kReduceTG = 256;

enum Buf : NSUInteger {
    kParams = 0,
    kCtrl = 1,
    kA = 2,
    kPrim = 3,
    kFx = 4,
    kFy = 5,
    kRhs = 6,
    kJ = 7,
    kMask = 8,
    kPartial = 9,
    kB = 10,
};

class MetalSolver final : public mhd::DeviceSolver {
  public:
    explicit MetalSolver(const Grid2D& grid) : grid_(grid)
    {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (device_ == nil) {
                throw std::runtime_error("Metal: no GPU device available");
            }
            queue_ = [device_ newCommandQueue];

            NSError* err = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
            // IEEE-conforming math: the accuracy study compares against the fp64 CPU solver,
            // and the compensated time accumulator relies on exact two-sum.
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeSafe;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            NSString* src = [NSString stringWithUTF8String:kMetalKernelSource];
            id<MTLLibrary> lib = [device_ newLibraryWithSource:src options:opts error:&err];
            if (lib == nil) {
                throw std::runtime_error(std::string("Metal: shader compilation failed: ") +
                                         err.localizedDescription.UTF8String);
            }
            auto pso = [&](const char* name) {
                id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
                if (fn == nil) {
                    throw std::runtime_error(std::string("Metal: missing kernel ") + name);
                }
                NSError* e = nil;
                id<MTLComputePipelineState> p = [device_ newComputePipelineStateWithFunction:fn
                                                                                       error:&e];
                if (p == nil) {
                    throw std::runtime_error(std::string("Metal: pipeline failed for ") + name);
                }
                return p;
            };
            bc_x_ = pso("bc_x");
            bc_y_ = pso("bc_y");
            solid_fill_ = pso("solid_fill");
            cfl_partial_ = pso("cfl_partial");
            cfl_finalize_ = pso("cfl_finalize");
            advance_time_ = pso("advance_time");
            primitives_ = pso("primitives");
            current_ = pso("current");
            flux_x_ = pso("flux_x");
            flux_y_ = pso("flux_y");
            assemble_ = pso("assemble_rhs");
            stage1_ = pso("rk_stage1");
            stage2_ = pso("rk_stage2");

            n_ = static_cast<std::size_t>(grid.get_size_x()) *
                 static_cast<std::size_t>(grid.get_size_y());
            const std::size_t field_bytes = 9 * n_ * sizeof(float);
            U_ = make_buffer(field_bytes);
            Ustar_ = make_buffer(field_bytes);
            rhs_ = make_buffer(field_bytes);
            prim_ = make_buffer(field_bytes);
            fx_ = make_buffer(field_bytes);
            fy_ = make_buffer(field_bytes);
            J_ = make_buffer(3 * n_ * sizeof(float));
            const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny;
            partial_count_ = static_cast<int>((cells + kReduceTG - 1) / kReduceTG);
            partial_ = make_buffer(static_cast<std::size_t>(partial_count_) * sizeof(float));
            mask_ = make_buffer(std::max<std::size_t>(cells, 1));
            std::memset(mask_.contents, 1, std::max<std::size_t>(cells, 1));
            ctrl_ = make_buffer(sizeof(Ctrl));
        }
    }

    std::string device_name() const override
    {
        return std::string("Metal: ") + device_.name.UTF8String;
    }

    mhd::Precision precision() const override
    {
        return mhd::Precision::FP32;
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
            std::memcpy(mask_.contents, mask->fluid.data(), mask->fluid.size());
        }
    }

    void upload(const StateField& U) override
    {
        check_shape(U);
        auto* dst = static_cast<float*>(U_.contents);
        const std::vector<double>* comps[9] = {&U.rho, &U.mx, &U.my, &U.mz, &U.energy,
                                               &U.bx,  &U.by, &U.bz, &U.psi};
        for (std::size_t c = 0; c < 9; ++c) {
            const double* s = comps[c]->data();
            float* d = dst + c * n_;
            for (std::size_t k = 0; k < n_; ++k) {
                d[k] = static_cast<float>(s[k]);
            }
        }
    }

    void download(StateField& U) override
    {
        check_shape(U);
        const auto* src = static_cast<const float*>(U_.contents);
        std::vector<double>* comps[9] = {&U.rho, &U.mx, &U.my, &U.mz, &U.energy,
                                         &U.bx,  &U.by, &U.bz, &U.psi};
        for (std::size_t c = 0; c < 9; ++c) {
            const float* s = src + c * n_;
            double* d = comps[c]->data();
            for (std::size_t k = 0; k < n_; ++k) {
                d[k] = static_cast<double>(s[k]);
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
        Params P = make_params(params);

        auto* ctrl = static_cast<Ctrl*>(ctrl_.contents);
        *ctrl = Ctrl{0.0f, 0.0f, 0.0f, 0.0f, 0u, params.t_end > 0.0 ? 0u : 1u};

        SolveResult result{};
        int batch = 4;
        double last_t = 0.0;
        unsigned last_steps = 0;
        while (ctrl->finished == 0) {
            if (static_cast<int>(ctrl->steps) >= params.max_steps) {
                break;
            }
            batch = std::min(batch, std::max(1, params.max_steps - static_cast<int>(ctrl->steps)));
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                bind_common(enc, P);
                for (int s = 0; s < batch; ++s) {
                    encode_step(enc, P);
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status == MTLCommandBufferStatusError) {
                    throw std::runtime_error(std::string("Metal: command buffer failed: ") +
                                             cb.error.localizedDescription.UTF8String);
                }
            }
            // Grow the batch toward the expected number of remaining steps (sync rarely).
            const double t = static_cast<double>(ctrl->t_hi) + static_cast<double>(ctrl->t_lo);
            const unsigned steps = ctrl->steps;
            const double dt_avg = steps > last_steps ? (t - last_t) / (steps - last_steps) : 0.0;
            last_t = t;
            last_steps = steps;
            int est = batch * 2;
            if (dt_avg > 0.0) {
                est = static_cast<int>(std::ceil((params.t_end - t) / dt_avg)) + 1;
            }
            batch = std::clamp(est, 1, 256);
        }

        result.t = static_cast<double>(ctrl->t_hi) + static_cast<double>(ctrl->t_lo);
        result.steps = static_cast<int>(ctrl->steps);
        result.c_h = ctrl->c_h;
        if (ctrl->finished == 0 && params.t_end - result.t > 1e-6 * params.t_end) {
            throw std::runtime_error("solve: reached max_steps before t_end");
        }
        // Snap to the requested end time (difference is fp32 round-off of the accumulator).
        result.t = params.t_end;

        // Final BC / wall fill like ::solve().
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            bind_common(enc, P);
            encode_boundaries(enc, U_);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        return result;
    }

  private:
    id<MTLBuffer> make_buffer(std::size_t bytes)
    {
        id<MTLBuffer> b = [device_ newBufferWithLength:std::max<std::size_t>(bytes, 16)
                                               options:MTLResourceStorageModeShared];
        if (b == nil) {
            throw std::runtime_error("Metal: buffer allocation failed");
        }
        std::memset(b.contents, 0, b.length);
        return b;
    }

    void check_shape(const StateField& U) const
    {
        if (U.nx != grid_.get_size_x() || U.ny != grid_.get_size_y()) {
            throw std::invalid_argument("StateField size does not match Grid2D");
        }
    }

    Params make_params(const SolveParams& sp) const
    {
        Params P{};
        P.nx_tot = grid_.get_size_x();
        P.ny_tot = grid_.get_size_y();
        P.nx = grid_.nx;
        P.ny = grid_.ny;
        P.ng = grid_.ng;
        P.n = static_cast<int>(n_);
        P.limiter = sp.limiter == SlopeLimiter::Minmod ? 0 : 1;
        P.bc_x = static_cast<int>(bc_.type_x);
        P.bc_y = static_cast<int>(bc_.type_y);
        P.has_mask = has_mask_ ? 1 : 0;
        P.reflective_walls = reflective_walls_ ? 1 : 0;
        P.ohmic = sp.nonideal.ohmic ? 1 : 0;
        P.hall = sp.nonideal.hall ? 1 : 0;
        P.ambipolar = sp.nonideal.ambipolar ? 1 : 0;
        P.nonideal_any = sp.nonideal.any() ? 1 : 0;
        P.partial_count = partial_count_;
        P.inv_dx = static_cast<float>(1.0 / grid_.dx);
        P.inv_dy = static_cast<float>(1.0 / grid_.dy);
        P.gamma = static_cast<float>(sp.gamma);
        P.inv_gm1 = static_cast<float>(1.0 / (sp.gamma - 1.0));
        P.cfl = static_cast<float>(sp.cfl);
        P.glm_alpha = static_cast<float>(sp.glm_alpha);
        P.h = static_cast<float>(std::min(grid_.dx, grid_.dy));
        const double eta_max = sp.nonideal.max_diffusivity();
        const double h = std::min(grid_.dx, grid_.dy);
        P.dt_diff = eta_max > 0.0 ? static_cast<float>(sp.cfl * h * h / (4.0 * eta_max)) : 0.0f;
        P.eta_ohm = static_cast<float>(sp.nonideal.eta_ohm);
        P.eta_hall = static_cast<float>(sp.nonideal.eta_hall);
        P.eta_ambipolar = static_cast<float>(sp.nonideal.eta_ambipolar);
        P.t_end_hi = static_cast<float>(sp.t_end);
        P.t_end_lo = static_cast<float>(sp.t_end - static_cast<double>(P.t_end_hi));
        return P;
    }

    void bind_common(id<MTLComputeCommandEncoder> enc, const Params& P)
    {
        [enc setBytes:&P length:sizeof(Params) atIndex:kParams];
        [enc setBuffer:ctrl_ offset:0 atIndex:kCtrl];
        [enc setBuffer:prim_ offset:0 atIndex:kPrim];
        [enc setBuffer:fx_ offset:0 atIndex:kFx];
        [enc setBuffer:fy_ offset:0 atIndex:kFy];
        [enc setBuffer:rhs_ offset:0 atIndex:kRhs];
        [enc setBuffer:J_ offset:0 atIndex:kJ];
        [enc setBuffer:mask_ offset:0 atIndex:kMask];
        [enc setBuffer:partial_ offset:0 atIndex:kPartial];
    }

    static void dispatch1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                          NSUInteger n, NSUInteger tg = 256)
    {
        [enc setComputePipelineState:pso];
        tg = std::min<NSUInteger>(tg, pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    static void dispatch2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                          NSUInteger w, NSUInteger h)
    {
        [enc setComputePipelineState:pso];
        const NSUInteger tw = std::min<NSUInteger>(pso.threadExecutionWidth, std::max<NSUInteger>(w, 1));
        const NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(
                                                         pso.maxTotalThreadsPerThreadgroup / tw, 8));
        [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    }

    void encode_boundaries(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> A)
    {
        [enc setBuffer:A offset:0 atIndex:kA];
        dispatch2(enc, bc_x_, static_cast<NSUInteger>(grid_.ng),
                  static_cast<NSUInteger>(grid_.get_size_y()));
        dispatch2(enc, bc_y_, static_cast<NSUInteger>(grid_.get_size_x()),
                  static_cast<NSUInteger>(grid_.ng));
        if (has_mask_ && reflective_walls_) {
            dispatch2(enc, solid_fill_, static_cast<NSUInteger>(grid_.nx),
                      static_cast<NSUInteger>(grid_.ny));
        }
    }

    void encode_rhs(id<MTLComputeCommandEncoder> enc, const Params& P)
    {
        const auto nx = static_cast<NSUInteger>(grid_.nx);
        const auto ny = static_cast<NSUInteger>(grid_.ny);
        dispatch1(enc, primitives_, n_);
        if (P.nonideal_any != 0) {
            dispatch2(enc, current_, static_cast<NSUInteger>(grid_.get_size_x() - 2),
                      static_cast<NSUInteger>(grid_.get_size_y() - 2));
        }
        dispatch2(enc, flux_x_, nx + 1, ny);
        dispatch2(enc, flux_y_, nx, ny + 1);
        dispatch2(enc, assemble_, nx, ny);
    }

    // One SSP-RK2 step; dt comes from the device-side CFL reduction.
    void encode_step(id<MTLComputeCommandEncoder> enc, const Params& P)
    {
        encode_boundaries(enc, U_);
        // dt from the state at the beginning of the step.
        [enc setComputePipelineState:cfl_partial_];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(partial_count_), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kReduceTG, 1, 1)];
        [enc setComputePipelineState:cfl_finalize_];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kReduceTG, 1, 1)];

        // Stage 1: U* = U + dt L(U)
        encode_rhs(enc, P);
        [enc setBuffer:Ustar_ offset:0 atIndex:kB];
        dispatch1(enc, stage1_, 9 * n_);

        // Stage 2: U = 1/2 (U + U*) + dt/2 L(U*)
        encode_boundaries(enc, Ustar_);
        encode_rhs(enc, P);
        [enc setBuffer:U_ offset:0 atIndex:kA];
        [enc setBuffer:Ustar_ offset:0 atIndex:kB];
        dispatch1(enc, stage2_, 9 * n_);

        dispatch1(enc, advance_time_, 1, 1);
    }

    Grid2D grid_;
    BoundaryConditions bc_{BoundaryConditionType::Periodic, BoundaryConditionType::Periodic};
    bool has_mask_ = false;
    bool reflective_walls_ = true;
    std::size_t n_ = 0;
    int partial_count_ = 0;

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> bc_x_, bc_y_, solid_fill_, cfl_partial_, cfl_finalize_,
        advance_time_, primitives_, current_, flux_x_, flux_y_, assemble_, stage1_, stage2_;
    id<MTLBuffer> U_, Ustar_, rhs_, prim_, fx_, fy_, J_, partial_, mask_, ctrl_;
};

} // namespace

bool mhd::metal_available()
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil;
    }
}

std::unique_ptr<mhd::DeviceSolver> mhd::make_metal_solver(const Grid2D& grid)
{
    if (grid.ng < 2) {
        throw std::invalid_argument("Metal solver requires grid.ng >= 2");
    }
    return std::make_unique<MetalSolver>(grid);
}
