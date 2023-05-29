#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>
#include "utils.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}


// morton utils
inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t xx = __expand_bits(x);
    uint32_t yy = __expand_bits(y);
    uint32_t zz = __expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

__global__ void kernel_morton3D(
    const int * __restrict__ coords,
    const uint32_t N,
    int * indices
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;
    indices[n] = __morton3D(coords[0], coords[1], coords[2]);
}
void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D<<<div_round_up(N, N_THREAD), N_THREAD>>>(
        coords.data_ptr<int>(), N, 
        indices.data_ptr<int>()
    );
}


inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
    x = x & 0x49249249;
    x = (x | (x >> 2)) & 0xc30c30c3;
    x = (x | (x >> 4)) & 0x0f00f00f;
    x = (x | (x >> 8)) & 0xff0000ff;
    x = (x | (x >> 16)) & 0x0000ffff;
    return x;
}


// indices: int32, [N]
// coords: int32, [N, 3]
__global__ void kernel_morton3D_invert(
    const int * __restrict__ indices,
    const uint32_t N,
    int * coords
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;

    const int ind = indices[n];

    coords[0] = __morton3D_invert(ind >> 0);
    coords[1] = __morton3D_invert(ind >> 1);
    coords[2] = __morton3D_invert(ind >> 2);
}

void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D_invert<<<div_round_up(N, N_THREAD), N_THREAD>>>(indices.data_ptr<int>(), N, coords.data_ptr<int>());
}






// rays_o/d: [N, 3]
// nears/fars: [N]
// scalar_t should always be float in use.
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    scalar_t * nears, scalar_t * fars
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    if (near > far_z || near_z > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;

    if (near < min_near) near = min_near;

    nears[n] = near;
    fars[n] = far;
}


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "near_far_from_aabb", ([&] {
        kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), aabb.data_ptr<scalar_t>(), N, min_near, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>());
    }));
}


// rays_o/d: [N, 3]
// radius: float
// coords: [N, 2]
template <typename scalar_t>
__global__ void kernel_sph_from_ray(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const float radius,
    const uint32_t N,
    scalar_t * coords
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    coords += n * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    // const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // solve t from || o + td || = radius
    const float A = dx * dx + dy * dy + dz * dz;
    const float B = ox * dx + oy * dy + oz * dz; // in fact B / 2
    const float C = ox * ox + oy * oy + oz * oz - radius * radius;

    const float t = (- B + sqrtf(B * B - A * C)) / A; // always use the larger solution (positive)

    // solve theta, phi (assume y is the up axis)
    const float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
    const float theta = atan2(sqrtf(x * x + z * z), y); // [0, PI)
    const float phi = atan2(z, x); // [-PI, PI)

    // normalize to [-1, 1]
    coords[0] = 2 * theta * RPI() - 1;
    coords[1] = phi * RPI();
}


void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "sph_from_ray", ([&] {
        kernel_sph_from_ray<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), radius, N, coords.data_ptr<scalar_t>());
    }));
}






// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, ts: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps

template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const uint8_t * __restrict__ grid,
    const float bound, const bool contract,//通过bound确定块的尺度
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t N, const uint32_t C, const uint32_t H,
    const scalar_t* __restrict__ nears, 
    const scalar_t* __restrict__ fars,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * ts,//这里是需要记录下来的contraction点
    int * rays,//[N_rays, 2]
    int * counter,//[1]
    const scalar_t* __restrict__ noises
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // is first pass running.
    const bool first_pass = (xyzs == nullptr);

    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    rays += n * 2;

    uint32_t num_steps = max_steps;

    if (!first_pass) {
        uint32_t point_index = rays[0];

        num_steps = rays[1];

        xyzs += point_index * 3;
        dirs += point_index * 3;
        ts += point_index * 2;
    }

    // ray marching

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;
    
    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];

    // printf("far \n%f",far);
    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * bound / H;
    // const float dt_max = 1e10f;
    
    float t0 = near;
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;
    float t = t0;
    uint32_t step = 0;

    //if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);
    
    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);//bound default to 2
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]
        //C是cascade，
        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;

        // contraction
        float cx = x, cy = y, cz = z;
        const float mag = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
        if (contract && mag > 1) {
            // L-INF norm
            const float Linf_scale = (2 - 1 / mag) / mag;
            cx *= Linf_scale;
            cy *= Linf_scale;
            cz *= Linf_scale;
        }
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (cx * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (cy * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (cz * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        // if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, step);

        if (occ || (contract && mag > 1)) {
            step++;
            t += dt;
            if (!first_pass) {
                xyzs[0] = cx; // write contracted coordinates!
                xyzs[1] = cy;
                xyzs[2] = cz;
                dirs[0] = dx;
                dirs[1] = dy;
                dirs[2] = dz;
                ts[0] = t;
                ts[1] = dt;
                xyzs += 3;
                dirs += 3;
                ts += 2;
            }
        // contraction case: cannot apply voxel skipping.
        // } else if (contract && mag > 1) {
        //     t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - cx) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - cy) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - cz) * rdz;

            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                dt = clamp(t * dt_gamma, dt_min, dt_max);
                t += dt;
            } while (t < tt);
        }
    }

    // printf("[n=%d] step=%d, near=%f, far=%f, dt=%f, num_steps=%f\n", n, step, near, far, dt_min, (far - near) / dt_min);

    // write rays
    if (first_pass) {// id, offset, num_steps
        uint32_t point_index = atomicAdd(counter, step);
        rays[0] = point_index;
        rays[1] = step;
    }
}

void march_rays_train(
    const at::Tensor rays_o, 
    const at::Tensor rays_d, 
    const at::Tensor grid, 
    const float bound, 
    const bool contract, 
    const float dt_gamma, 
    const uint32_t max_steps, 
    const uint32_t N, const uint32_t C, 
    const uint32_t H, const at::Tensor nears, 
    const at::Tensor fars, 
    at::optional<at::Tensor> xyzs, 
    at::optional<at::Tensor> dirs, 
    at::optional<at::Tensor> ts, 
    at::Tensor rays, at::Tensor counter, at::Tensor noises
) {

    static constexpr uint32_t N_THREAD = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<uint8_t>(), bound, contract, dt_gamma, max_steps, N, C, H, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>(),
            xyzs.has_value() ? xyzs.value().data_ptr<scalar_t>() : nullptr,
            dirs.has_value() ? dirs.value().data_ptr<scalar_t>() : nullptr,
            ts.has_value() ? ts.value().data_ptr<scalar_t>() : nullptr,
            rays.data_ptr<int>(), counter.data_ptr<int>(), noises.data_ptr<scalar_t>());
    }));
}
