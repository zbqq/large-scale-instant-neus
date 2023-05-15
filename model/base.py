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
        
        self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
    
    def setup(self,center,scale):
        self.geometry_network.setup(center,scale)
        # self.render_step_size = 1.732 * 2 * max(scale)/ self.config.num_samples_per_ray
        self.render_step_size = 1.732 * 3 * scale[2]/ self.config.num_samples_per_ray
        # 无人机视角下不包含
        
        self.register_buffer('scene_aabb', \
            torch.cat((\
            self.geometry_network.xyz_min,
            self.geometry_network.xyz_max
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
                    torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))
                self.register_buffer('grid_coords',
                    create_meshgrid3d(self.G, self.G, self.G, False, dtype=torch.int32).reshape(-1, 3))
                self.register_buffer('density_grid',
                    torch.zeros(self.cascades, self.G**3))

    @torch.no_grad()
    def update_density_grid(self,model, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            # density_grid_tmp[c, indices] = model(xyzs_w)
            density_grid_tmp[c, indices] = model.density(xyzs_w,return_feat=False)
            # density_grid_tmp[c, indices] = model(xyzs_w,with_fea=False,with_grad=False)["sdf"]

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,#小于0的保留
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()
        # mean_density = 0.001
        # self.density_bitfield+=1#
        studio.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)           
    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = studio.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells          
                
                
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

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
                
                coords2 = vren.morton3D_invert(indices2.int())
                # torch.cuda.synchronize(self.device)
                # concatenate
                cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            cells += [(indices1, coords1)]

        return cells
    
    
    
    