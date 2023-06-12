/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "utils.h"

__global__ void unpack_rays_kernel(
    // input
    const int n_rays,
    const int *rays,
    // output
    int64_t *ray_indices)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = rays[i * 2 + 0];  // point idx start.
    const int steps = rays[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    ray_indices += base;

    for (int j = 0; j < steps; ++j)
    {
        ray_indices[j] = i;
    }
}


torch::Tensor unpack_rays(const torch::Tensor rays, const int n_samples)
{
    // DEVICE_GUARD(rays);
    CHECK_INPUT(rays);

    const int n_rays = rays.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // int n_samples = rays[n_rays - 1].sum(0).item<int>();
    torch::Tensor ray_indices = torch::empty(
        {n_samples}, rays.options().dtype(torch::kLong));

    unpack_rays_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        rays.data_ptr<int>(),
        ray_indices.data_ptr<int64_t>());
    return ray_indices;
}

