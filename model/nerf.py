import tinycudann as tcnn
import torch 
import json
from torch import Tensor
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect
import time
from utils.render import render
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
from kornia.utils.grid import create_meshgrid3d
from torch.cuda.amp import custom_fwd, custom_bwd
from .custom_functions import TruncExp
import torch
from torch import nn
from .tcnn_nerf import vanillaMLP,RenderingNet
from .loss import NeRFLoss
from .base import baseModule
import tinycudann as tcnn
import studio
from einops import rearrange
# from .custom_functions import TruncExp
import numpy as np
import tqdm
from load_tool import draw_poses
NEAR_DISTANCE = 0.01

class vanillaNeRF(baseModule):
    def __init__(self,config):#config.model
        super().__init__(config)
        self.config = config
        self.render_step_size = 1.732 * 2 * self.config.aabb.radius_z / self.config.num_samples_per_ray
        
        self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
        
        self.geometry_network = vanillaMLP(self.config.geometry_network)
        self.color_net = RenderingNet(self.config.color_net)
        # self.loss = NeRFLoss(lambda_distortion=0)
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio

        def occ_eval_fn(x):
            sigma = self.geometry_network(x, with_fea=False, with_grad=False)["sigma"]
            sigma = torch.sigmoid(sigma)[...,None]
            return sigma
        
        self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn,ema_decay=0.95)
        
        

    def forward(self,rays_o,rays_d):
        sigma_grad_samples = []
        
        
        def alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            out = self.geometry_network(positions, with_fea=False, with_grad=True)
            sdf, sdf_grad=out["sdf"], out["grad"]
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)#获取rho
            return alpha[...,None]#[N_samples, 1]
        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            out = self.geometry_network(positions, with_fea=True, with_grad=True)
            sdf, sdf_grad, feature=out["sdf"],out["grad"],out["fea"]
            sigma_grad_samples.append(sdf_grad)
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            rgb = self.color_net(t_dirs,feature, normal)
            return rgb, alpha[...,None]  
        # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.scene_aabb)
        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            out = self.geometry_network(positions, with_fea=False, with_grad=True)
            sigma = out["sigma"]
            return sigma[...,None]#[N_samples, 1]
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            out = self.geometry_network(positions, with_fea=True, with_grad=True)
            sigma, grad, feature=out["sigma"],out["grad"],out["fea"]
            
            sigma_grad_samples.append(grad)
            normal = F.normalize(grad, p=2, dim=-1)
            rgb = self.color_net(t_dirs,feature, normal)
            sigma = sigma + torch.zeros_like(sigma,dtype=torch.float32)
            return rgb, sigma[...,None]
        with torch.no_grad():#ray_marching过程中不累加nabla
            t_min,t_max = ray_aabb_intersect(rays_o,rays_d,self.scene_aabb)
            
            ray_indices, t_starts, t_ends = \
                ray_marching(
                    rays_o,rays_d,
                    t_min=t_min,t_max=t_max,
                    scene_aabb=self.scene_aabb,
                    grid = self.occupancy_grid,
                    # alpha_fn = alpha_fn,
                    sigma_fn=sigma_fn,
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
        result = {
            'rgb': rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays_o.device),
            'gradients': torch.concat(sigma_grad_samples,dim=0),
        }
        
        return result

    
    
    