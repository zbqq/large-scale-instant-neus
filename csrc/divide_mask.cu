
#include <torch/extension.h>
#include <iostream>
#include <ATen/ATen.h>
#include <math.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "helper_math.h"
using namespace std;



__global__ void packbits_u32_kernel(
    torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits> idx_array,
    torch::PackedTensorAccessor64<int64_t,1,torch::RestrictPtrTraits> bits_array
){
    // const int32_t n = blockIdx.x * blockDim.x + threadIdx.x;//一维时
    const int32_t n = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;//二维时
    if(n > bits_array.size(0))
        return;
    int mask_size = 32;
    if (n == bits_array.size(0))
        mask_size = (idx_array.size(0) % 32) - 1;
    const int64_t flag = 1;
    for(int i = 0 ; i < mask_size ; i++){
        int32_t hit_pix = idx_array[n*32 + i];
        if (hit_pix > 0){
            bits_array[n] |= flag << i;
        }
    }
}


torch::Tensor packbits_u32(
    torch::Tensor idx_array,
    torch::Tensor bits_array
){
    // 每个线程处理32位长数据即32个像素
    const int num_pixs = std::ceil(idx_array.size(0)/32);
    // const int threads = 256, blocks = (num_pixs+threads-1)/threads;
    const int BLOCK_W = 64;
    const int BLOCK_H = 16;
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((num_pixs + BLOCK_W*BLOCK_H - 1)/(BLOCK_W * BLOCK_H),1,1);
    // const dim3 gridSize(8,1,1);

    // torch::Tensor bit_array = torch::zeros({bits_array.size(0)},bits_array.options());
    AT_DISPATCH_ALL_TYPES(idx_array.type(),"packbits_u32",
    // AT_DISPATCH_ALL_TYPES(idx_array.type(),"packbits_u64_cu",
    ([&] {
        packbits_u32_kernel<<<gridSize, blockSize>>>(
        // packbits_u64_kernel<<<4, 64>>>(
            idx_array.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
            bits_array.packed_accessor64<int64_t,1,torch::RestrictPtrTraits>()
        );
    }));
    return bits_array;
}

__global__ void un_packbits_u32_kernel(
    torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits> idx_array,
    torch::PackedTensorAccessor64<int64_t,1,torch::RestrictPtrTraits> bits_array
){
    // const int32_t n = blockIdx.x * blockDim.x + threadIdx.x;//一维时
    const int32_t n = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;//二维时

    if(n > bits_array.size(0))
        return;
    int mask_size = 32;
    if (n == bits_array.size(0))
        mask_size = (idx_array.size(0) % 32) - 1;
    const int64_t flag = 1;
    for(int i = 0 ; i < mask_size ; i++){
        if (bits_array[n] & (flag << i)){
            idx_array[n*32 + i]++;
        }
    }
}
torch::Tensor un_packbits_u32(
    torch::Tensor idx_array,
    torch::Tensor bits_array
){
    // 每个线程处理64位长数据即64个像素
    const int num_pixs = std::ceil(idx_array.size(0)/64);
    // const int threads = 256, blocks = (num_pixs+threads-1)/threads;
    const int BLOCK_W = 64;
    const int BLOCK_H = 16;
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((num_pixs + BLOCK_W*BLOCK_H - 1)/(BLOCK_W * BLOCK_H),1,1);

    AT_DISPATCH_ALL_TYPES(idx_array.type(),"un_packbits_u32",
    ([&] {
        un_packbits_u32_kernel<<<gridSize, blockSize>>>(
            idx_array.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
            bits_array.packed_accessor64<int64_t,1,torch::RestrictPtrTraits>()
        );
    }));
    return idx_array;
}




__global__ void distance_mask_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dirsMap,//视线方向 [N 3]
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> locMap,//相机光心 [3]
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> centroids,//分块质心 [C 3]
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> mask,//一张图像对于C个质心的mask值 [N , C]

    const float threshould//重叠阈值
){
    // const int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
    // const int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    // const int n = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;
    const int32_t n = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;//二维时
    if(n >= dirsMap.size(0))
        return;
    const float dx = dirsMap[n][0], dy = dirsMap[n][1], dz = dirsMap[n][2];
    const float ox = locMap[0], oy = locMap[1], oz = locMap[2];
    float3 dir = make_float3(dx,dy,dz);
    float3 loc = make_float3(ox,oy,oz);
    float* d = new float[centroids.size(0)];
    float d_min = 99999.9;
    
    dir /= length(dir);

    for(int i  = 0; i < centroids.size(0); i++){
        float3 centroid = make_float3(centroids[i][0],centroids[i][1],centroids[i][2]); 
        float3 l_vec = centroid - loc;
        float3 d_vec = cross(l_vec,dir);
        d[i] = length(d_vec);
        if (d_min >= d[i])
            d_min = d[i];

        // \  d   |
        //  *-----|
        //   \    |
        // l  \   | dir   d = |(l X dir)|/|dir|
        //     \  |
        //      \ |
        //       \|

    }
    
    for(int i  = 0; i < centroids.size(0); i++){
        if (d[i] <= (d_min * threshould))
            mask[n][i] = 1;
    }
    delete d;
    d = nullptr;
}


torch::Tensor distance_mask(
    torch::Tensor dirsMap,//[WxH , 3]
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor mask,//[WxH , C]
    const float threshould//重叠阈值
){
    const int num_pisxels = dirsMap.size(0);
    const int BLOCK_W = 64;
    const int BLOCK_H = 16;
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((num_pisxels + BLOCK_W*BLOCK_H - 1)/(BLOCK_W*BLOCK_H),1,1);
    // const dim3 gridSize(1,1,1);
    AT_DISPATCH_ALL_TYPES(dirsMap.type(),"distance_mask",
    ([&] {
        distance_mask_kernel<<<gridSize, blockSize>>>(
        // packbits_u64_kernel<<<4, 64>>>(
            dirsMap.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            locMap.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            centroids.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            mask.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
            threshould

        );
    }));
    return mask;
}




__global__ void mega_nerf_mask_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dirsMap,//视线方向 [N 3]
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> locMap,//相机光心 [3]
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> centroids,//分块质心 [C 3]
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> mask,//一张图像对于C个质心的mask值 [N , C]
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> t_range,//一张图像对于C个质心的mask值 [N , 2]
    
    const int samples,//每条射线上采样点数
    const float threshould//重叠阈值
){
    const int32_t n = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;//二维时
    if(n >= dirsMap.size(0))
        return;
    const float dx = dirsMap[n][0], dy = dirsMap[n][1], dz = dirsMap[n][2];
    const float ox = locMap[0], oy = locMap[1], oz = locMap[2];
    float3 dir = make_float3(dx,dy,dz);
    float3 loc = make_float3(ox,oy,oz);
    
    dir /= length(dir);



    float3 current_pt = loc + dir * t_range[n][0];
    const float dt = (t_range[n][1] - t_range[n][0]) / samples;
    for(int i = 0; i < samples; i++){//遍历射线上所有采样点
        float d_min = 999.9;
        for(int j = 0; j < centroids.size(0); j++){//先计算离采样点最新的距离值
            float3 centroid = make_float3(centroids[j][0],centroids[j][1],centroids[j][2]);
            float d_tmp = length(current_pt - centroid);
            if (d_min > d_tmp)
                d_min = d_tmp;
        }

        for(int j = 0; j < centroids.size(0); j++){
            float3 centroid = make_float3(centroids[j][0],centroids[j][1],centroids[j][2]);
            float d_ratio_tmp = length(current_pt - centroid)/(d_min+1e-8);
            if(threshould >= d_ratio_tmp){
                mask[n][j]=1;
            }
        }
        current_pt += dt * dir;
    }
}


torch::Tensor mega_nerf_mask(
    torch::Tensor dirsMap,//[WxH , 3]
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor t_range,
    const int samples,//每条射线上采样点数
    const float threshould//重叠阈值
){
    
    const int BLOCK_W = 64;
    const int BLOCK_H = 16;
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((dirsMap.size(0) + BLOCK_W*BLOCK_H - 1)/(BLOCK_W*BLOCK_H),1,1);
    auto mask = torch::zeros({dirsMap.size(0),centroids.size(0)}, 
                                        torch::dtype(torch::kInt32).device(dirsMap.device()));
    AT_DISPATCH_ALL_TYPES(dirsMap.type(),"mega_nerf_mask",
    ([&] {
        mega_nerf_mask_kernel<<<gridSize, blockSize>>>(
            dirsMap.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            locMap.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            centroids.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            mask.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
            t_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            samples,
            threshould
        );
    }));
    return mask;
}
