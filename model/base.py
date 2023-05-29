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
from .custom_functions import march_rays_train,near_far_from_aabb
# import .custom_functions
import torch
from torch import nn

from .tcnn_nerf import SDF,RenderingNet,VarianceNetwork
from .loss import NeRFLoss
import tinycudann as tcnn
import studio
from einops import rearrange
import numpy as np
import tqdm
from load_tool import draw_poses
NEAR_DISTANCE = 0.01

class baseModule(nn.Module):
    def __init__(self,config):#config.model
        super().__init__()
        self.config = config
        # L = 16; F = 2; log2_T = 19; N_min = 16
        # b = np.exp(np.log(2048*self.scale/N_min)/(L-1))
        # self.render_step_size = 1.732 * 2 * self.config.radius_z / self.config.num_samples_per_ray
        
        self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
    
    def setup(self,center,scale):
        self.center = center
        self.scale = scale
        
        self.C = max(1+int(np.ceil(np.log2(2*self.scale))), 1)
        self.H = self.config.grid_resolution
        
        self.geometry_network.setup(center,scale)
        # self.render_step_size = 1.732 * 2.5 * max(scale)/ self.config.num_samples_per_ray
        self.render_step_size = 1.732 * 2 * scale[2]/ self.config.num_samples_per_ray
        # 无人机视角下不包含
        
        self.register_buffer('scene_aabb', \
            torch.cat((\
            center-scale,
            center+scale
            ))
        )
        if self.config.grid_prune:
            if self.config.use_nerfacc:
                self.occupancy_grid = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=self.config.grid_resolution,
                    contraction_type=ContractionType.AABB
                )
            else:
                self.register_buffer('density_bitfield',
                    torch.zeros(self.C*self.H**3//8, dtype=torch.uint8))
                self.register_buffer('grid_coords',
                    create_meshgrid3d(self.G, self.G, self.G, False, dtype=torch.int32).reshape(-1, 3))
                self.register_buffer('density_grid',
                    torch.zeros(self.C, self.G**3))
                 
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio
        raise NotImplementedError
    def forward(self,rays_o,rays_d,split):
        raise NotImplementedError
    def render_whole_image(self,rays_o:Tensor,rays_d:Tensor):
        final_out=[]
        depths = []
        pbar=tqdm.tqdm(total=len(rays_o))
        t1 = time.time()
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            out = self(rays_o_batch,rays_d_batch)
            final_out.append(out["rgb"])
            depths.append(out["depth"])
            pbar.update(1)
        print("the time of rendering an image is ",time.time()-t1)
        rgbs=torch.concat(final_out,dim=0)
        depths=torch.concat(depths,dim=0)
        
        return {
            "rgb": rgbs,
            "depth": depths
        }
    def render(self,rays_o,rays_d,bg_color=None,perturb=False,cam_near_far=None,shading='full'):
        rays_o=rays_o.contiguous()
        rays_d=rays_d.contiguous()
        
        N=rays_o.shape[0]
        nears,fars = near_far_from_aabb(
            rays_o,rays_d,self.scene_aabb,min_near=0.2
        )
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, 0])
            fars = torch.minimum(fars, cam_near_far[:, 1])
        
        # mix background color
        if bg_color is None:
            bg_color = 1
        results = {}

        if self.training:
            
            xyzs, dirs, ts, rays = \
                march_rays_train(rays_o, rays_d, self.scale, 
                                        True, self.density_bitfield, 
                                        self.C, self.H, 
                                        nears, fars, perturb, 
                                        self.opt.dt_gamma, self.opt.max_steps)

            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                geo_output = self.geometry_network(xyzs,with_fea=True)
                
                outputs = self(xyzs, dirs, shading=shading)
                sigmas = outputs['sigma']
                rgbs = outputs['color']

            weights, weights_sum, depth, image = \
                studio.composite_rays_train(sigmas, rgbs, ts, rays, self.opt.T_thresh)

            results['num_points'] = xyzs.shape[0]
            results['weights'] = weights
            results['weights_sum'] = weights_sum
        else:
            pass
        
        
        
        
        
        
        
        
    
    