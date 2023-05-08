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
from .tcnn_nerf import SDF,RenderingNet,VarianceNetwork
from .loss import NeRFLoss
import tinycudann as tcnn
import studio
from einops import rearrange
# from .custom_functions import TruncExp
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
        # self.cascades = max(1+int(np.ceil(np.log2(2*self.scale))), 1)
        # self.render_step_size = 1.732 * 2 * self.config.radius_z / self.config.num_samples_per_ray
        self.render_step_size = 1.732 * 2 * self.config.aabb.radius_z / self.config.num_samples_per_ray
        
        self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
    
    def setup(self,center,scale):
        self.geometry_network.setup(center,scale)
        self.register_buffer('scene_aabb', \
            torch.cat((\
            self.geometry_network.xyz_min,
            self.geometry_network.xyz_max
            ))
        )
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=self.config.grid_resolution,
                    contraction_type=ContractionType.AABB
                )
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio
        raise NotImplementedError
    def forward(self,rays_o,rays_d):
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

    
    
    
    
    