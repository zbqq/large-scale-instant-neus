
import torch 
import torch.nn.functional as F
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect

import torch.nn.functional as F
import torch
from .tcnn_nerf import SDF,RenderingNet,VarianceNetwork
from .base import baseModule
NEAR_DISTANCE = 0.01

class NeuS(baseModule):
    def __init__(self,config):#config.model
        super().__init__(config)
        self.config = config
        
        # self.render_step_size = 1.732 * 2 * self.config.aabb.radius_z / self.config.num_samples_per_ray
        
        self.geometry_network = SDF(self.config.geometry_network)
        self.variance = VarianceNetwork(self.config.init_variance)
        self.color_network = RenderingNet(self.config.color_network)
        
        # self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
        
        self.cos_anneal_ratio = 1.0
        # self.loss = NeRFLoss(lambda_distortion=0)
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio
        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)
        if self.config.use_nerfacc:
            def occ_eval_fn(x):
                sdf = self.geometry_network(x, with_grad=False, with_fea=False)['sigma']
                inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
                inv_s = inv_s.expand(sdf.shape[0], 1)
                estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
                estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                p = prev_cdf - next_cdf
                c = prev_cdf
                alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
                return alpha
                # return torch.ones_like(alpha,dtype=alpha.dtype)
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn,ema_decay=0.95)
            
        else:
            #     return density.reshape(-1).detach()
            
            def occ_eval_fn(x): # neus2's occ_eval_fn
                sdf = self.geometry_network(x, with_fea=False, with_grad=False)["sigma"]

                inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)#可以做一点改动
                inv_s = inv_s.expand(sdf.shape[0], 1)
                density = inv_s*torch.sigmoid(inv_s*sdf[...,None])*torch.sigmoid(inv_s*sdf[...,None])*(1/torch.sigmoid(inv_s*sdf[...,None])-1)
                density = density.clip(0.0, 1.0)
        
            self.update_extra_state(occ_eval_fn=lambda pts: occ_eval_fn(x=pts))
        
        
    # def get_alpha(self, sdf, normal, dirs, dists):
    def get_alpha(self, geo, dists):
        sdf,normal,dirs=geo['sigma'],geo['normals'],geo['dirs']
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-6) / (c + 1e-6)).view(-1).clip(0.0, 1.0)
        return alpha.reshape(-1,1)
    def forward(self,rays_o,rays_d,split):
        if self.config.use_nerfacc:
            sdf_grad_samples = []


            def alpha_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                midpoints = (t_starts + t_ends) / 2.
                positions = t_origins + t_dirs * midpoints
                out = self.geometry_network(positions, with_fea=False, with_grad=True)
                sdf, sdf_grad=out["sigma"], out["grad"]
                dists = t_ends - t_starts
                normal = F.normalize(sdf_grad, p=2, dim=-1)
                alpha = self.get_alpha(sdf, normal, t_dirs, dists)#获取rho
                return alpha#[N_samples, 1]
            def rgb_alpha_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                midpoints = (t_starts + t_ends) / 2.
                positions = t_origins + t_dirs * midpoints
                out = self.geometry_network(positions, with_fea=True, with_grad=True)
                sdf, sdf_grad, feature=out["sigma"],out["grad"],out["fea"]
                sdf_grad_samples.append(sdf_grad)
                dists = t_ends - t_starts
                normal = F.normalize(sdf_grad, p=2, dim=-1)
                alpha = self.get_alpha(sdf, normal, t_dirs, dists)
                rgb = self.color_network(t_dirs,feature, normal)
                return rgb, alpha
            # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.scene_aabb)

            with torch.no_grad():#ray_marching过程中不累加nabla

                ray_indices, t_starts, t_ends = \
                    ray_marching(
                        rays_o,rays_d,
                        scene_aabb=self.scene_aabb,
                        grid = self.occupancy_grid,
                        alpha_fn = alpha_fn,
                        near_plane=None, far_plane=None,
                        render_step_size=self.render_step_size,
                        stratified=True,
                        cone_angle = 0.0,
                        alpha_thre = 0.0
                    )

                # del t_min,t_max
            rgb, opacity, depth = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                rgb_alpha_fn=rgb_alpha_fn,
                render_bkgd=self.background_color,
            )
            result = {
                'rgb': rgb,
                'opacity': opacity,
                'depth': depth,
                'rays_valid': opacity > 0,
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays_o.device),
                'grad': torch.concat(sdf_grad_samples,dim=0),
                'inv_s':self.variance.inv_s
            } 

            return result
        else:            
            cam_near = torch.ones([rays_o.shape[0],1],device=rays_o.device) * 0.
            # cam_far = torch.ones([rays_o.shape[0],1],device=rays_o.device) * torch.norm(self.scale)
            cam_far = torch.ones([rays_o.shape[0],1],device=rays_o.device) * 15
            cam_near_far = torch.cat([cam_near,cam_far],dim=-1)
            result = self.render(rays_o,rays_d,cam_near_far=cam_near_far,split=split,perturb=True)
            
            inv_s=self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            result['inv_s']=inv_s[0]
            return result
        
        
        
        

    
    
    