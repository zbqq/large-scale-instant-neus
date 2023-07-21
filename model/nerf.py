
import torch 
import torch.nn.functional as F
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect
from load_tool import draw_poses
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
        self.color_network = RenderingNet(self.config.color_network)
        # self.loss = NeRFLoss(lambda_distortion=0)
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio

        def occ_eval_fn(x):
            sigma = self.geometry_network(x, with_fea=False, with_grad=False)["sigma"]
            # sigma = torch.sigmoid(sigma)[...,None]
            return sigma.reshape(-1).detach() * self.render_step_size
        if self.config.use_nerfacc:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn)
        else:
            self.update_extra_state(occ_eval_fn = lambda pts: occ_eval_fn(x=pts))

        
    def get_alpha(self, geo, dists):#计算
        
        sigmas = geo['sigma'].view(-1,1)
        assert sigmas.shape == dists.shape
        alphas = torch.ones_like(sigmas) - torch.exp(- sigmas * dists)

        return alphas.view(-1,1)
    
    def forward(self,rays_o,rays_d,split):
        if self.config.use_nerfacc:
            sigma_grad_samples=[]
            
            # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.scene_aabb[None,...])
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
                rgb = self.color_network(t_dirs,feature)
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
            
            # opacity = torch.clamp(opacity,1e-8,1000)
            
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
            cam_near = torch.ones([rays_o.shape[0],1],device=rays_o.device) * 0.
            # cam_far = torch.ones([rays_o.shape[0],1],device=rays_o.device) * torch.norm(self.scale)
            cam_far = torch.ones([rays_o.shape[0],1],device=rays_o.device) * 30
            cam_near_far = torch.cat([cam_near,cam_far],dim=-1)
            result = self.render(rays_o,rays_d,cam_near_far=cam_near_far,split=split,perturb=True)
            return result
            
            
            
        

    
    
    