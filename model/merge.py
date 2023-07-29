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
from .nerf import vanillaNeRF
# utils.render import 
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
from kornia.utils.grid import create_meshgrid3d
from torch.cuda.amp import custom_fwd, custom_bwd
from .custom_functions import TruncExp
import torch
from torch import nn
from utils.render import up_sample,cat_z_vals,get_rays_indices
from .tcnn_nerf import SDF,RenderingNet,VarianceNetwork
from .loss import NeRFLoss
import tinycudann as tcnn
import studio
from einops import rearrange
# from .custom_functions import TruncExp
import numpy as np
import tqdm
from scripts.load_tool import draw_poses
from .base import baseModule
from datasets.ray_utils import sampled_pdf,pts_from_rays
from model.custom_functions import near_far_from_aabb,march_rays_train,\
    rendering_with_alpha,morton3D,packbits
def in_aabb(pts,aabb):
    aabb_=aabb.view(1,-1)
    eps = torch.tensor([1e-4,1e-4,1e-4]).view(1,-1).to(pts.device)
    inward_idx = torch.logical_and(pts>=aabb_[0:1,:3]-eps ,pts <= aabb_[0:1,3:]+eps).sum(dim=-1)==3
    return inward_idx.view(-1,1)

class mainModule(baseModule):
    def __init__(self, config,sub_modules):
        super().__init__(config)
        self.config = config
        self.sub_modules = nn.ModuleList(sub_modules)
        centroids=[]
        aabbs=[]
        for i,child in enumerate(self.sub_modules):
            centroids.append(child.center.view(1,-1))
            aabbs.append(child.scene_aabb.view(1,-1))
        centroids = torch.concat(centroids,dim=0)
        aabbs = torch.concat(aabbs,dim=0)
        self.register_buffer('centroids',centroids)#[C,3]
        self.register_buffer('aabbs',aabbs)#[C,3]
        self.register_buffer('scene_aabb',torch.concat(\
            [torch.min(self.aabbs[:,:3],dim=0)[0],
            torch.max(self.aabbs[:,3:],dim=0)[0],
            ]))
        # self.center = self.scene_aabb[:3]+self.scene_aabb[3:]
        # self.scale = self.center - self.scene_aabb[:3]
        self.register_buffer('center',(self.scene_aabb[:3]+self.scene_aabb[3:])/2)
        self.register_buffer('scale',self.center - self.scene_aabb[:3])
        
        if self.config.point_sample.use_raymarch:
            self.C = 1+max(1+int(np.ceil(np.log2(2*max(self.scale)))), 1)
            self.H = self.config.occ_grid.grid_resolution
            self.register_buffer('density_bitfield',
                    torch.zeros(self.C*self.H**3//8, dtype=torch.uint8))
            self.register_buffer('density_grid',
                torch.zeros(self.C, self.H**3))
        else:
            pass
    def geometry_network(self,xyzs,with_grad = False,with_fea = True):
        sigmas = self.sigma_rgb_from_pts(xyzs,weights_type='UW',require_rgb=False)
        return {
            'sigma':sigmas.view(-1)
        }
        
    def sigma_rgb_from_pts(self,xyzs,dirs=None,weights_type=None,require_rgb=True):#输入三维点，输出sigma_rgb
        groups_idx = []
        # z_zoom_up = torch.tensor([-1.5,-1.5,-3,1.5,1.5,3],device=xyzs.device)
        z_zoom_up = torch.tensor([0,0,-3,0,0,3],device=xyzs.device)
        # 相机在外侧观测bbox，因此需要将z轴拉长
        for i,child in enumerate(self.sub_modules):
            group_idx = in_aabb(xyzs,child.scene_aabb + z_zoom_up)# [N_samples 1]
            # group_idx = in_aabb(xyzs,child.scene_aabb)# [N_samples 1]
            groups_idx.append(group_idx)
            # draw_poses(pts3d=xyzs,aabb_=(child.scene_aabb + z_zoom_up)[None,...])
            pass
        
        groups_idx = torch.cat(groups_idx,dim=-1) # [N C]
        overlap_num = groups_idx.sum(dim=-1) # [N 1]
        assert overlap_num.all()
        overlap_idx = overlap_num > 1 # [N 1]
        
        
        weights = groups_idx.float().clone()
        # group_rgb_weights = groups_idx.float().clone() * torch.arange(1,groups_idx.shape[1]+1,device=groups_idx.device).view(1,-1) * 0.1
        if weights_type == None or weights_type == 'UW':
            weights[overlap_idx] = groups_idx.float()[overlap_idx]/overlap_num[overlap_idx].view(-1,1)
            
        elif weights_type == 'IDW':
            inverse_distance = groups_idx.float()[overlap_idx]/(torch.cdist(xyzs[overlap_idx],self.centroids)+1e-8)#[N_overlap,C]
            weights[overlap_idx] = inverse_distance / inverse_distance.sum(dim=-1).unsqueeze(-1)
        # 根据采样点是否在grid中进行前向得到sigma、rgb，并合成
        sigma_rgb=torch.empty(0)
        
        for i,child in enumerate(self.sub_modules):
            if groups_idx[:,i].sum() == 0: # 如果xyzs没有一个在第i个块中，则跳过
                continue
            if require_rgb:
                
                sigmas,rgbs = child.forward_from_pts(xyzs[groups_idx[:,i],:],dirs[groups_idx[:,i],:],require_rgb=True)
                sigmas = sigmas.view(-1,1)
                rgbs = rgbs.view(-1,3)    
            else:
                sigmas = child.forward_from_pts(xyzs[groups_idx[:,i],:],require_rgb=False)
                sigmas = sigmas.view(-1,1)
                # sigmas = sigmas/sigmas.max()
                rgbs = torch.empty(0).to(sigmas.device)
            
                sigmas = sigmas/sigmas.max()

            if sigma_rgb.shape[0] == 0: # [N_sample 4] or [N_sample 3]
                sigma_rgb = \
                    torch.zeros([xyzs.shape[0],rgbs.shape[-1]+sigmas.shape[-1]],device = sigmas.device,dtype=sigmas.dtype)
            
            sigma_rgb[groups_idx[:,i],:] += torch.concat([sigmas,rgbs],dim=-1) * weights[groups_idx[:,i],i:i+1]
            # sigma_rgb[groups_idx[:,i],:] += torch.concat([torch.zeros_like(sigmas,dtype=sigmas.dtype,device=sigmas.device)
            #                                               ,rgbs],dim=-1) * group_rgb_weights[groups_idx[:,i],i:i+1]
        
        return sigma_rgb
    
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio

        def occ_eval_fn(x):
            
            sigmas = self.sigma_rgb_from_pts(x,require_rgb=False)
            sigmas = torch.sigmoid(sigmas)
            return sigmas.reshape(-1).detach()
            
        self.update_extra_state(occ_eval_fn = lambda pts: occ_eval_fn(x=pts))   
        
        
    
    def get_alpha(self, geo, dists):#计算
        sigma = geo['sigma'].view(-1,1)
        alphas = torch.ones_like(sigma) - torch.exp(- sigma * dists)

        return alphas.view(-1,1)
    def forward(self,rays_o,rays_d,weights_type=None,perturb=True):# [N_rays 3] [N_rays,3]
        
        device = rays_o.device
        fb_ratio = torch.ones([1,1,1],dtype=torch.float32).to(device)*self.config.aabb.fb_ratio
        N = rays_o.shape[0]
        results = torch.empty(0)
        scene_aabb =self.scene_aabb.clone()
        
        aabb = [scene_aabb.clone()]
        for i,child in enumerate(self.sub_modules):
            aabb.append(child.scene_aabb)
        aabb = torch.stack(aabb)
        # rays_o = rays_o - self.center.view(-1,3)#需要平移到以center为原点坐标系
        
        # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=aabb)
        rays_o = rays_o.split(int(rays_o.shape[0]/500))
        rays_d = rays_d.split(int(rays_d.shape[0]/500))
        
        # scene_aabb =self.scene_aabb - self.center.repeat([2])

        
        # aabb[0:2] -= self.scale[:2]
        # aabb[3:5] += self.scale[:2]#扩大一点使得far不会终止到aabb上
     
        
        final_results=[]
        
        print('rendering rays batch')
        pbar=tqdm.tqdm(total=len(rays_o))
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            
            nears,fars = near_far_from_aabb(# [N_rays 1]
                    rays_o_batch,rays_d_batch,scene_aabb,0.02
            )
        
            # sample 
            if self.config.point_sample.use_raymarch: # raymarch 采样
                rays_o_batch_rectified = rays_o_batch - self.center.view(-1,3)
                # scene_aabb =self.scene_aabb - self.center.repeat([2])
                xyzs, dirs, ts, rays = \
                march_rays_train(rays_o_batch_rectified, rays_d_batch, self.scale, fb_ratio,
                                True, self.density_bitfield, 
                                self.C, self.H, 
                                nears, fars, perturb, 
                                self.config.point_sample.ray_march.dt_gamma, self.config.point_sample.num_samples_per_ray,)
                xyzs += self.center.view(-1,3)
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            
            else: # 逆CDF采样
                n_rays,n_samples = rays_o_batch.shape[0],self.config.point_sample.num_samples_per_ray
                z_vals = torch.linspace(0.0, 1.0, n_samples,device=device).view(1,-1)
                z_vals = nears.view(-1,1) + (fars - nears).view(-1,1) * z_vals # [N_samples]
                with torch.no_grad(): 
                    sample_dist = (fars - nears).view(-1,1) / n_samples

                    for _ in range(0,self.config.point_sample.inv_cdf.up_sample_steps):

                        z_vals_sample = up_sample(rays_o_batch,rays_d_batch,z_vals,\
                            self.config.point_sample.inv_cdf.n_importance,sample_dist,model_name=self.config.name,
                            geometry_network=self.geometry_network,get_alphas=self.get_alpha)
                        z_vals = cat_z_vals(rays_o_batch,rays_d_batch,z_vals,z_vals_sample)

                xyzs = rays_o_batch[:, None, :] + rays_d_batch[:, None, :] * z_vals[..., :, None] # [N_rays, N_samples, 3]
                xyzs = xyzs.view(-1,3)
                dists = z_vals[...,1:] - z_vals[...,:-1] # [N_rays,N_samples - 1]    
                dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1).view(-1)
    
    
                dirs = rays_d_batch[:,None,:].expand(n_rays,z_vals.shape[1],3).reshape(-1,3)
                rays = get_rays_indices(z_vals)
                z_vals = z_vals.view(-1,1)
                dists = dists.view(-1,1)

                ts = torch.cat([z_vals,dists],dim=-1)
                
                
                
                pass
            # volume rendering
            # 计算各采样点是否在grid中或者在重叠区域，并以此计算权重
            sigma_rgb=self.sigma_rgb_from_pts(xyzs,dirs,weights_type=weights_type,require_rgb=True)
            sigmas,rgbs = sigma_rgb[:,:1],sigma_rgb[:,1:4]
            
            
            
            if self.config.rendering_from_alpha:
                if self.config.color_network.use_normal:
                    alphas = self.get_alpha({'sigma':sigmas,'normal':normals,'dir':dirs}
                                            ,ts[:,1])
                else:
                    alphas = self.get_alpha({'sigma':sigmas},ts[:,1:2])
                image,opacities,depth = rendering_with_alpha(alphas,rgbs,ts,rays,rays_o_batch.shape[0])
                results = torch.cat([image,depth],dim=-1)
            
            final_results.append(results)
            pbar.update(1)
        final_results = torch.cat(final_results,dim=0)
        rgb_depth = {
            'rgb':final_results[:,:3],
            'depth':final_results[:,3:4]
        }
        return rgb_depth      
    def update_extra_state(self, decay=0.95, S=128,occ_eval_fn=None):
        with torch.no_grad():
            tmp_grid = - torch.ones_like(self.density_grid)
            # full update.
            X = torch.arange(self.H, dtype=torch.int32, device=self.scene_aabb.device).detach().split(S)#真实尺度下坐标
            Y = torch.arange(self.H, dtype=torch.int32, device=self.scene_aabb.device).detach().split(S)
            Z = torch.arange(self.H, dtype=torch.int32, device=self.scene_aabb.device).detach().split(S)
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
            valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])

        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        # self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.config.occ_grid.density_thresh)
        self.density_bitfield = packbits(self.density_grid.detach(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')
        return None        

                


            # 这一部分是已经采样到点了
            
            # pts_fine,dirs_fine=sampled_pdf(rays_o_batch,rays_d_batch,self.aabbs)# [N_rays,N_samples,3]
            # for i,child in enumerate(self.sub_modules):
            #     group_idx = in_aabb(pts,child.scene_aabb)
            #     groups_idx.append(group_idx)
                
            #     else:#inverse CDF sample
                    

            #     if weights_type == None or weights_type == 'UW':
            #         weights[overlap_idx] = 1/groups_idx.sum(dim=-1).repeat([1,weights.shape[-1]])

            #     elif weights_type == 'IDW':
            #         inverse_distance = 1/(torch.cdist(pts[overlap_idx],self.centroids)+1e-8)#[N_o,C]
            #         weights[overlap_idx] = inverse_distance / inverse_distance.sum(dim=-1).unsqueeze(-1)

            #     for i,child in enumerate(self.sub_modules):
            #         rgbs,sigmas = child.forward_from_pts(pts[groups_idx[:,i],:],dirs[groups_idx[:,i],:])
            #         if results.shape[0] == 0:
            #             results = \
            #                 torch.zeros([pts.shape[0],rgbs.shape[1]+sigmas.shape[1]],device = sigmas.device,dtype=sigmas.dtype)
            #         results[groups_idx[:,i],:] += torch.concat([sigmas,rgbs],dim=-1) * weights[groups_idx[:,i],:]

            #     pts_fine,dirs_fine=sampled_pdf(rays_o_batch,rays_d_batch,self.aabbs)# [N_rays,N_samples,3]


        
        
        
        
        
        
        





























