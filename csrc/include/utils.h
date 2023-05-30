#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



//packbits_32

torch::Tensor packbits_u32(
    torch::Tensor idx_array,
    torch::Tensor bits_array
);

torch::Tensor un_packbits_u32(
    torch::Tensor idx_array,
    torch::Tensor bits_array
);

torch::Tensor distance_mask(
    torch::Tensor dirsMap,//[WxH , 3]
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor mask,//[WxH , C]
    const float threshould//重叠阈值
);
void mega_nerf_mask(
    torch::Tensor dirsMap,//[WxH , 3]
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor t_range,
    torch::Tensor mask,
    const int samples,//每条射线上采样点数
    const float threshould//重叠阈值
);


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);

void near_far_from_aabb(
    const at::Tensor rays_o, 
    const at::Tensor rays_d, 
    const at::Tensor aabb, 
    const uint32_t N, 
    const float min_near, 
    at::Tensor nears, at::Tensor fars
);

void sph_from_ray(
    const at::Tensor rays_o, 
    const at::Tensor rays_d, 
    const float radius, 
    const uint32_t N, 
    at::Tensor coords
); 
void march_rays_train(
    const at::Tensor rays_o, 
    const at::Tensor rays_d, 
    const at::Tensor grid, 
    const at::Tensor bound, 
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
);
void composite_rays_train_forward(
    const at::Tensor sigmas, 
    const at::Tensor rgbs, 
    const at::Tensor ts, 
    const at::Tensor rays, 
    const uint32_t M, const uint32_t N, const float T_thresh, 
    at::Tensor weights, at::Tensor weights_sum, at::Tensor depth, 
    at::Tensor image
);

void composite_rays_train_backward(
    const at::Tensor grad_weights, 
    const at::Tensor grad_weights_sum, 
    const at::Tensor grad_depth, 
    const at::Tensor grad_image, 
    const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor ts, 
    const at::Tensor rays, 
    const at::Tensor weights_sum, 
    const at::Tensor depth, 
    const at::Tensor image, 
    const uint32_t M, const uint32_t N, const float T_thresh, 
    at::Tensor grad_sigmas, at::Tensor grad_rgbs
);

void march_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const at::Tensor rays_alive, 
    const at::Tensor rays_t, 
    const at::Tensor rays_o, const at::Tensor rays_d, 
    const at::Tensor bound, const bool contract, 
    const float dt_gamma, const uint32_t max_steps, 
    const uint32_t C, const uint32_t H, const at::Tensor grid, 
    const at::Tensor nears, const at::Tensor fars, 
    at::Tensor xyzs, at::Tensor dirs, at::Tensor ts, 
    at::Tensor noises
);
void composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const float T_thresh, 
    at::Tensor rays_alive, at::Tensor rays_t, 
    at::Tensor sigmas, at::Tensor rgbs, at::Tensor ts, 
    at::Tensor weights_sum, at::Tensor depth, at::Tensor image
);