
import torch 
import torch.nn.functional as F
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect
from utils.render import render
import torch.nn.functional as F
import torch
from .tcnn_nerf import vanillaMLP,RenderingNet
from .base import baseModule
from .render_utils import render_without_nerfacc
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
            return sigma
        
        self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn,ema_decay=0.98)
        
    def get_alpha(self, sigma, dists):#计算

        alpha = torch.ones_like(sigma) - torch.exp(- sigma * dists)

        return alpha    

    def forward(self,rays_o,rays_d):
        if self.config.use_nerfacc:
            sigma_grad_samples=[]
            
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
            result = render_with_nerfacc(rays_o,rays_d)
            return result
            
            
            
        

    
    
    