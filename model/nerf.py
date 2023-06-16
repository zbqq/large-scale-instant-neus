
import torch 
import torch.nn.functional as F
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect
from utils.render import render
import torch.nn.functional as F
import torch
from .tcnn_nerf import vanillaMLP,RenderingNet
from .base import baseModule
NEAR_DISTANCE = 0.01

class vanillaNeRF(baseModule):
    def __init__(self,config):#config.model
        super().__init__(config)
        self.config = config
    
        self.geometry_network = vanillaMLP(self.config.geometry_network)
        self.color_net = RenderingNet(self.config.color_net)
        # self.loss = NeRFLoss(lambda_distortion=0)
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio

        def occ_eval_fn(x):
            sigma = self.geometry_network(x, with_fea=False, with_grad=False)["sigma"]
            sigma = torch.sigmoid(sigma)[...,None]
            return sigma.reshape(-1).detach()
        if self.config.use_nerfacc:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn,ema_decay=0.98)
        else:
            self.update_extra_state(occ_eval_fn = lambda pts: occ_eval_fn(x=pts))

        
    def get_alpha(self, sigma, dists):#计算

        alphas = torch.ones_like(sigma) - torch.exp(- sigma * dists)

        return alphas.view(-1,1)
    
    def forward(self,rays_o,rays_d,split):
        if self.config.use_nerfacc:
            sigma_grad_samples=[]
            def alpha_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                midpoints = (t_starts + t_ends) / 2.
                positions = t_origins + t_dirs * midpoints
                out = self.geometry_network(positions, with_fea=False, with_grad=False)
                sigma = out["sigma"].view(-1,1)
                dists = t_ends - t_starts
                alpha = self.get_alpha(sigma, dists)#获取rho
                return alpha#[N_samples, 1]
            def rgb_alpha_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                midpoints = (t_starts + t_ends) / 2.
                positions = t_origins + t_dirs * midpoints
                out = self.geometry_network(positions, with_fea=True, with_grad=True)
                sigma, normal, fea = out["sigma"].view(-1,1),out["grad"],out["fea"]
                sigma_grad_samples.append(normal)
                dists = t_ends - t_starts
                normal = F.normalize(normal, p=2, dim=-1)
                alpha = self.get_alpha(sigma, dists)
                rgb = self.color_net(t_dirs,fea, normal)
                return rgb, alpha
            # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.scene_aabb)
            def sigma_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
                out = self.geometry_network(positions,with_fea=False, with_grad=False)
                density = out['sigma']
                return density[...,None]
            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
                out = self.geometry_network(positions,with_fea=True, with_grad=False) 
                density,feature = out['sigma'],out['fea']
                rgb = self.color_net(t_dirs,feature)
                return rgb.to(torch.float32), density[...,None].to(torch.float32)
            with torch.no_grad():#ray_marching过程中不累加nabla
                t_min,t_max = ray_aabb_intersect(rays_o,rays_d,self.scene_aabb)

                ray_indices, t_starts, t_ends = \
                    ray_marching(
                        rays_o,rays_d,
                        t_min=t_min,t_max=t_max,
                        scene_aabb=self.scene_aabb,
                        grid = self.occupancy_grid,
                        # alpha_fn = alpha_fn,
                        sigma_fn = sigma_fn,
                        near_plane=None, far_plane=None,
                        render_step_size=self.render_step_size,
                        cone_angle = 0.0,
                        alpha_thre = 0.0
                    )

                # del t_min,t_max
            rgb, opacity, depth = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                # rgb_alpha_fn=rgb_alpha_fn,
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=self.background_color,
            )
            opacity = torch.clamp(opacity,1e-8,1000)
            result = {
                'rgb': rgb,
                'opacity': opacity,
                'depth': depth,
                'rays_valid': opacity > 0,
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays_o.device),
                # 'gradients': torch.concat(sigma_grad_samples,dim=0),
            }

            return result
        else:  
            result = self.render(rays_o,rays_d,split=split,perturb=True)
            return result
            
            
            
        

    
    
    