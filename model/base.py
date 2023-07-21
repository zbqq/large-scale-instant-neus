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
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
from kornia.utils.grid import create_meshgrid3d
from torch.cuda.amp import custom_fwd, custom_bwd
from .custom_functions import \
    march_rays_train, near_far_from_aabb, composite_rays_train, \
        morton3D, morton3D_invert, packbits,march_rays,composite_rays,\
            rendering_with_alpha
from utils.render import render_from_cdf,render_from_raymarch
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
def in_aabb(position,aabb):
    if (position-aabb[:3]>=0).sum()==3 and (position-aabb[3:]<=0).sum()==3:
        return True
    else:
        return False
class baseModule(nn.Module):
    def __init__(self,config):#config.model
        super().__init__()
        self.config = config
        # L = 16; F = 2; log2_T = 19; N_min = 16
        # b = np.exp(np.log(2048*self.scale/N_min)/(L-1))
        # self.render_step_size = 1.732 * 2 * self.config.radius_z / self.config.num_samples_per_ray
        
        self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
    
    def setup(self,center,scale):#这里的center，scale和divide中的背景aabb一致
        # self.center = center
        # self.scale = scale
        # center=torch.tensor([0.0,0.0,0.0]).to(center.device)
        # scale = torch.tensor([1.5,1.5,1.5]).to(center.device)
        
        
        self.register_buffer('center',center)#这是已经scale to的尺度
        self.register_buffer('scale',scale*self.config.scale_zoom_up)
        self.register_buffer('fg_scale',scale*self.config.scale_zoom_up*self.config.fb_ratio)
        self.C = max(1+int(np.ceil(np.log2(2*max(self.scale)))), 1)
        self.H = self.config.grid_resolution
        
        self.geometry_network.setup(self.center,self.scale)
        # self.render_step_size = 1.732 * 2.5 * max(scale)/ self.config.num_samples_per_ray
        self.render_step_size = 1.732 * 2 * scale[2]/ self.config.num_samples_per_ray
        # 无人机视角下不包含
        
        self.register_buffer('scene_aabb', \
            torch.cat((\
            self.center - self.scale,
            self.center + self.scale
            ))
        )
        self.register_buffer('scene_fg_aabb', \
            torch.cat((\
            self.center - self.fg_scale,
            self.center + self.fg_scale
            ))
        )
        if self.config.grid_prune and self.config.use_raymarch:
            if self.config.use_nerfacc:
                self.occupancy_grid = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=self.config.grid_resolution,
                    contraction_type=ContractionType.AABB
                )
            else:
                self.register_buffer('density_bitfield',
                    torch.zeros(self.C*self.H**3//8, dtype=torch.uint8))
                # self.register_buffer('grid_coords',
                #     create_meshgrid3d(self.H, self.H, self.H, False, dtype=torch.int32).reshape(-1, 3))
                self.register_buffer('density_grid',
                    torch.zeros(self.C, self.H**3))
                self.iter_density=0
            pass
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio
        raise NotImplementedError
    def preprocess_data(self,batch,stage):
        
        
        
        
        pass
    def forward(self,rays_o,rays_d,split):
        raise NotImplementedError
    def render_whole_image(self,rays_o:Tensor,rays_d:Tensor):
        final_out=[]
        depths = []
        pbar=tqdm.tqdm(total=len(rays_o))
        t1 = time.time()
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            out = self(rays_o_batch,rays_d_batch,split='val')
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
    
    def render(self,rays_o,rays_d,bg_color=None,perturb=False,cam_near_far=None,shading='full',split='train'):
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        aabb = self.scene_aabb.clone()
        aabb[0:2] -= self.scale[:2] * 10
        aabb[3:5] += self.scale[:2] * 10#扩大一点使得far不会终止到aabb上
        
        # aabb[0:3] -= self.scale[:3] * 2
        # aabb[3:6] += self.scale[:3] * 2#扩大一点使得far不会终止到aabb上

        nears,fars = near_far_from_aabb( # 位移不改变nears，fars
            rays_o,rays_d,aabb,0.02#确定far时需要把射线打到地面上，而不是在边界
        )
        
        # fars *= 1.1
        # valid_idx = fars<20
        # rays_o = rays_o[valid_idx]
        # rays_d = rays_d[valid_idx]
        # nears = nears[valid_idx]
        # fars = fars[valid_idx]        
        if cam_near_far is not None:
            nears = torch.minimum(nears, cam_near_far[:, 0])
            fars = torch.minimum(fars, cam_near_far[:, 1])
        # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=aabb[None,...],t_min=nears,t_max=fars)
        
        # mix background color
        if bg_color is None:
            bg_color = 1
        
        if self.config.use_raymarch:
            results = render_from_raymarch(
                rays_o,rays_d,self.center,self.scale,self.density_bitfield,
                self.C,self.H,nears,fars,self.config,perturb,
                split=split,aabb=aabb,
                geometry_network=self.geometry_network,
                color_network=self.color_network,
                get_alphas=self.get_alpha
            )
        else:
            results = render_from_cdf(
                rays_o,rays_d,
                nears,fars,
                config=self.config,b_bg=aabb,
                up_sample_steps=self.config.up_sample_steps,
                geometry_network=self.geometry_network,
                color_network=self.color_network,
                get_alphas=self.get_alpha
            )    
        return results
    def update_extra_state(self, decay=0.95, S=128,occ_eval_fn=None):
        
        with torch.no_grad():
            tmp_grid = - torch.ones_like(self.density_grid)
            # full update.
            if self.iter_density < 16:
                X = torch.arange(self.H, dtype=torch.int32, device=self.scene_aabb.device).split(S)#真实尺度下坐标
                Y = torch.arange(self.H, dtype=torch.int32, device=self.scene_aabb.device).split(S)
                Z = torch.arange(self.H, dtype=torch.int32, device=self.scene_aabb.device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            
                            # construct points
                            xx, yy, zz = torch.meshgrid(xs, ys, zs)
                            # i,j,k = xx[i,j,k],yy[i,j,k],zz[i,j,k]
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                            indices = morton3D(coords).long() # [N]
                            xyzs = 2 * coords.float() / (self.H - 1) - 1 # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.C):
                                bound = torch.min(torch.ones_like(self.scale)*2 ** cas, self.scale).view(-1,3)
                                half_grid_size = bound / self.H
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size) + self.center.view(-1,3)
                                # add noise in [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # query density
                                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                                    sigmas = occ_eval_fn(cas_xyzs)#[N]
                                    # geo_output = self.geometry_network(cas_xyzs,with_fea=False,with_grad=False)
                                    # sigmas = geo_output['sigma'].reshape(-1).detach()
                                # assign 
                                tmp_grid[cas, indices] = sigmas.to(torch.float32)

            # partial update (half the computation)
            else:
                N = self.H ** 3 // 4 # H * H * H / 4
                for cas in range(self.C):
                    # random sample some positions
                    coords = torch.randint(0, self.H, (N, 3), device=self.scene_aabb.device) # [N, 3], in [0, 128)
                    indices = morton3D(coords).long() # [N]
                    # random sample occupied positions
                    occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.scene_aabb.device)
                    occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                    occ_coords = morton3D_invert(occ_indices) # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                    # same below
                    xyzs = 2 * coords.float() / (self.H - 1) - 1 # [N, 3] in [-1, 1]
                    # bound = min(2 ** cas, max(self.scale))
                    bound = torch.min(torch.ones_like(self.scale)*2 ** cas, self.scale)
                    half_grid_size = bound / self.H
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size).view(-1,3) + self.center.view(-1,3)
                    # cas_xyzs = xyzs * (bound - half_grid_size) 
                    # add noise in [-hgs, hgs]
                    # draw_poses(pts3d=cas_xyzs.to('cpu'),aabb_=self.scene_aabb[None,...])
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # query density
                    with torch.cuda.amp.autocast(enabled=self.config.fp16):
                        sigmas = occ_eval_fn(cas_xyzs)#[N]
                    # assign 
                    tmp_grid[cas, indices] = sigmas.to(torch.float32)

            # ema update
            valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
            
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])

        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        # self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density*0.99, self.config.density_thresh)
        self.density_bitfield = packbits(self.density_grid.detach(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')
        return None
    def forward_from_pts(self,pts,dirs=None,require_rgb=True):
        # 给定真实空间中的三维点
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            # geo_output = self.geometry_network(xyzs,with_fea=True,with_grad=False)
            # sigmas,feas = geo_output['sigma'],geo_output['fea']
            # rgbs = self.color_network(dirs,feas)
            geo_output = self.geometry_network(pts,with_fea=True,with_grad=False)
            
            if self.config.color_network.use_normal:
                geo_output = self.geometry_network(pts,with_fea=True,with_grad=False)
                sigmas,feas = geo_output['sigma'],geo_output['fea']
                # rgbs = self.color_network(dirs,feas)
                if require_rgb:
                    rgbs = self.color_network(dirs,feas,normals)
                    return sigmas,rgbs
                return sigmas
            else:
                geo_output = self.geometry_network(pts,with_fea=True,with_grad=False)
                sigmas,feas = geo_output['sigma'],geo_output['fea']
                
                if require_rgb:
                    rgbs = self.color_network(dirs,feas)
                    return sigmas,rgbs
                return sigmas

        
        
        
        
    
    