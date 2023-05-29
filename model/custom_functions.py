import torch
import studio
from torch.cuda.amp import custom_fwd, custom_bwd
from torch_scatter import segment_csr
from einops import rearrange


class _near_far_from_aabb(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        Args:
            rays_o: float, [N, 3]
            rays_d: float, [N, 3]
            aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [N]
            fars: float, [N]
        '''
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)

        studio.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)

        return nears, fars
near_far_from_aabb=_near_far_from_aabb.apply

class _sph_from_ray(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, radius):
        ''' sph_from_ray, CUDA implementation
        get spherical coordinate on the background sphere from rays.
        Assume rays_o are inside the Sphere(radius).
        Args:
            rays_o: [N, 3]
            rays_d: [N, 3]
            radius: scalar, float
        Return:
            coords: [N, 2], in [-1, 1], theta and phi on a sphere. (further-surface)
        '''
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # num rays

        coords = torch.empty(N, 2, dtype=rays_o.dtype, device=rays_o.device)

        studio.sph_from_ray(rays_o, rays_d, radius, N, coords)

        return coords
sph_from_ray=_sph_from_ray.apply

class _morton3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords):
        ''' morton3D, CUDA implementation
        Args:
            coords: [N, 3], int32, in [0, 128) (for some reason there is no uint32 tensor in torch...) 
            TODO: check if the coord range is valid! (current 128 is safe)
        Returns:
            indices: [N], int32, in [0, 128^3)
            
        '''
        if not coords.is_cuda: coords = coords.cuda()
        
        N = coords.shape[0]

        indices = torch.empty(N, dtype=torch.int32, device=coords.device)
        
        studio.morton3D(coords.int(), N, indices)

        return indices
morton3D=_morton3D.apply

class _morton3D_invert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices):
        ''' morton3D_invert, CUDA implementation
        Args:
            indices: [N], int32, in [0, 128^3)
        Returns:
            coords: [N, 3], int32, in [0, 128)
            
        '''
        if not indices.is_cuda: indices = indices.cuda()
        
        N = indices.shape[0]

        coords = torch.empty(N, 3, dtype=torch.int32, device=indices.device)
        
        studio.morton3D_invert(indices.int(), N, coords)

        return coords
morton3D_invert=_morton3D_invert.apply

class _march_rays_train(torch.autograd.Function):
    """
    March the rays to get sample point positions and directions.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        hits_t: (N_rays, 2) near and far bounds from aabb intersection
        density_bitfield: (C*G**3//8)
        cascades: int
        scale: float
        exp_step_factor: the exponential factor to scale the steps
        grid_size: int
        max_samples: int

    Outputs:
        rays_a: (N_rays) ray_idx, start_idx, N_samples
        xyzs: (N, 3) sample positions
        dirs: (N, 3) sample view directions
        deltas: (N) dt for integration
        ts: (N) sample ts
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound,contract,
                density_bitfield, C, H, nears, fars,
                perturb=False,dt_gamma=0,max_steps=1024):

        rays_o = rays_o.float().contiguous().view(-1, 3)
        rays_d = rays_d.float().contiguous().view(-1, 3)
        density_bitfield = density_bitfield.contiguous()
        
        N = rays_o.shape[0]
        
        step_counter = torch.zeros(1, dtype=torch.int32, device=rays_o.device) # point counter, ray counter
        
        if perturb:
            noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)
            
        rays = torch.empty(N, 2, dtype=torch.int32, device=rays_o.device) # id, offset, num_steps
        
        studio.march_rays_train(\
            rays_o,rays_d,density_bitfield,bound,contract,dt_gamma,
                max_steps,
                N, C, H,
                nears, fars, 
                None, None, None, 
                rays, step_counter, noises)
        
        M = step_counter.item()
        # M=50000
        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        ts = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)

        # second pass: write outputs
        studio.march_rays_train(rays_o, rays_d, density_bitfield, bound, contract, dt_gamma, max_steps, N, C, H, nears, fars, xyzs, dirs, ts, rays, step_counter, noises)

        return xyzs, dirs, ts, rays
    @staticmethod
    @custom_bwd
    def backward(ctx, dL_drays_a, dL_dxyzs, dL_ddirs,
                 dL_ddeltas, dL_dts, dL_dtotal_samples):
        rays_a, ts = ctx.saved_tensors
        segments = torch.cat([rays_a[:, 1], rays_a[-1:, 1]+rays_a[-1:, 2]])
        dL_drays_o = segment_csr(dL_dxyzs, segments)
        dL_drays_d = \
            segment_csr(dL_dxyzs*rearrange(ts, 'n -> n 1')+dL_ddirs, segments)

        return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None
march_rays_train=_march_rays_train.apply



class _composite_rays_train(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, ts, rays, T_thresh=1e-4):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            ts: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights: float, [M]
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        
        sigmas = sigmas.float().contiguous()
        rgbs = rgbs.float().contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights = torch.zeros(M, dtype=sigmas.dtype, device=sigmas.device) # may leave unmodified, so init with 0
        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)

        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        studio.composite_rays_train_forward(sigmas, rgbs, ts, rays, M, N, T_thresh, weights, weights_sum, depth, image)

        ctx.save_for_backward(sigmas, rgbs, ts, rays, weights_sum, depth, image)
        ctx.dims = [M, N, T_thresh]

        return weights, weights_sum, depth, image
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights, grad_weights_sum, grad_depth, grad_image):
        
        grad_weights = grad_weights.contiguous()
        grad_weights_sum = grad_weights_sum.contiguous()
        grad_depth = grad_depth.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, ts, rays, weights_sum, depth, image = ctx.saved_tensors
        M, N, T_thresh = ctx.dims
   
        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)

        studio.composite_rays_train_backward(grad_weights, grad_weights_sum, grad_depth, grad_image, sigmas, rgbs, ts, rays, weights_sum, depth, image, M, N, T_thresh, grad_sigmas, grad_rgbs)

        return grad_sigmas, grad_rgbs, None, None, None


composite_rays_train = _composite_rays_train.apply

class _packbits(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, thresh, bitfield=None):
        ''' packbits, CUDA implementation
        Pack up the density grid into a bit field to accelerate ray marching.
        Args:
            grid: float, [C, H * H * H], assume H % 2 == 0
            thresh: float, threshold
        Returns:
            bitfield: uint8, [C, H * H * H / 8]
        '''
        if not grid.is_cuda: grid = grid.cuda()
        grid = grid.contiguous()

        C = grid.shape[0]
        H3 = grid.shape[1]
        N = C * H3 // 8

        if bitfield is None:
            bitfield = torch.empty(N, dtype=torch.uint8, device=grid.device)

        studio.packbits(grid, N, thresh, bitfield)

        return bitfield

packbits = _packbits.apply

class TruncExp(torch.autograd.Function):
    @staticmethod
    # @torch.cuda.amp.autocast()
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))
    
    
    