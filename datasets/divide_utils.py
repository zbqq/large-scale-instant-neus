#
import torch
import os
# import matplotlib
from torch.utils.data import IterableDataset
from torch import nn
from torch import Tensor
from tqdm import tqdm
from typing import List
from nerfacc import OccupancyGrid,ray_aabb_intersect,ContractionType
from .ray_utils import *
# from datasets.datasets import revise
from load_tool import draw_poses
import math
import studio
#用于分割子场景
def get_aabb(center:Tensor,scale:Tensor):
    xyz_min = center-scale
    xyz_max = center+scale
    
    scene_aabb = torch.cat([
        xyz_min,
        xyz_max
    ],dim=0).squeeze(1)
    return scene_aabb
def in_aabb(position,aabb):
    if (position-aabb[:3]>0).sum()==3 and (position-aabb[3:]<0).sum()==3:
        return True
    else:
        return False
    
class divideTool():
    def __init__(self,config,split='divide'):
        self.config = config
        self.split = split   
        self.Rc_path = os.path.join(self.config.root_dir,'R_correct.txt')#需要输入的
        self.new_pts3d_path = os.path.join(self.config.root_dir,'new_pts3d.txt')#需要输入的
        self.centers_and_scales_path = os.path.join(self.config.root_dir,f'CaS_{self.model_num}.txt')#输出的
        self.mask_save_path = os.path.join(self.config.mask_dir,"model")
        self.model_num = self.config.grid_X * self.config.grid_Y
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda:1")  
        # else:
        #     raise RuntimeError("divide tool must use GPU!")
        if self.model_num > 1:
            for i in range(0,self.model_num):
                #存放每一个model对poses的需求索引，以及poses对该model的intersect_pix_idx
                os.makedirs(os.path.join(self.mask_save_path,'{}'.format(i)),exist_ok=True)
            pass
        else:
            os.makedirs(os.path.join(self.mask_save_path,'0'),exist_ok=True)
        self.aabbs = []
        self.grids=[]
        self.centers=[]
    def gen_centers_from_pts(self,grid_dim:Tensor):
        pts = torch.tensor(np.loadtxt(self.new_pts3d_path,usecols=(0,1,2)),dtype=torch.float32)
        cameras_position = self.poses[:,:,3]#[M,3]

        max_position = torch.max(torch.cat([pts,cameras_position],dim=0),dim=0)[0]
        min_position = torch.min(torch.cat([pts,cameras_position],dim=0),dim=0)[0]
        # max_position = torch.max(cameras_position,dim=0)[0]
        # min_position = torch.min(cameras_position,dim=0)[0]
        max_position[2] = pts[:,2].max()
        
        
        radius = (max_position-min_position) / grid_dim / 2 # 前景radius
        radius[2] /= self.config.fb_ratio # 背景radius
        ranges = max_position - min_position
        offsets = [torch.arange(s) * ranges[i] / s + ranges[i] / (s * 2) for i, s in enumerate(grid_dim)]#每个方格的中心world coordinate
        
        # 根据稀疏点云划分grid,meshgrid格式
        centroids = torch.stack((\
                             torch.ones(grid_dim[0],grid_dim[1])*min_position[0],
                             torch.ones(grid_dim[0],grid_dim[1])*min_position[1],
                             torch.ones(grid_dim[0],grid_dim[1])*min_position[2],
                             )).permute(1,2,0)  #X,Y,Z
        
        
        centroids[:,:,0] += offsets[0].unsqueeze(1) # X x Y 个区域的x坐标
        centroids[:,:,1] += offsets[1] # X x Y个区域的y坐标
        centroids[:,:,2] += offsets[2] # X x Y个区域的y坐标
        # draw_poses(aabb_=torch.concat([min_position,max_position])[None,...])
        centroids = centroids.view(-1,3)#n个长方体的几何中心
        center_and_scale=[]
        for i in range(0,centroids.shape[0]):
            center = centroids[i,:].reshape(3,-1)
            self.centers.append(center)
            # scene_aabb = get_aabb(center=center,scale=torch.cat([radius[:2],torch.tensor([0.2])]).view(3,1))#divide时需要将scale_z变得很小
            scene_aabb = get_aabb(center=center,scale=radius[:3].view(3,1))

            self.aabbs.append(scene_aabb)
            center_and_scale.append(torch.concat([center.view(3),radius.view(3)]))
            
            if self.split == 'divide':
                grid = OccupancyGrid(
                roi_aabb = scene_aabb,
                resolution = 128,
                contraction_type = ContractionType.AABB
                )       
                self.grids.append(grid)
        center_and_scale = torch.stack(center_and_scale)
        self.aabbs = torch.stack(self.aabbs)
        np.savetxt(self.centers_and_scales_path,center_and_scale.numpy())
        # 原始SFM尺度与坐标
 
    def load_centers(self):
        temp = torch.tensor(np.loadtxt(self.centers_and_scales_path),dtype=torch.float32)
        self.centers = temp[...,:3].view(-1,3)
        self.scales = temp[...,3:].view(-1,3)
        self.aabbs = torch.concat([self.centers-self.scales,self.centers+self.scales],dim=-1)
    def divide(self,grid_dim,mask_type='aabb_intersect'):#从已经得到的sparse pts进行区域的分割，
        """
            
        """
        self.gen_centers_from_pts(grid_dim)#存真实的scale
        self.load_centers()#读取
        self.scale_to(scale=self.config.scale_to,current_model_idx=self.current_model_num)
        self.scales *= self.config.scale_zoom_up # 让前景有重叠
        self.fg_scales = self.scales * self.config.fb_ratio
        self.aabbs = torch.concat([self.centers-self.scales,self.centers+self.scales],dim=-1)
        self.fg_aabbs = torch.concat([self.centers-self.fg_scales,self.centers+self.fg_scales],dim=-1)
        # 放缩到2,4,8,...
        # draw_poses(poses_=self.poses,aabb_=self.aabbs)
        bits_array_len = math.floor(self.img_wh[0]*self.img_wh[1]/32)
        
        if mask_type == 'mega_nerf_mask':
            for i in tqdm(range(0,self.poses.shape[0])):
                rays_o,rays_d = get_rays(directions = self.directions,c2w = self.poses[i])
                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)
                mask = torch.zeros([rays_o.shape[0],self.centers.shape[0]],device=self.device).to(torch.int32)
                t_range = torch.tensor([0.,5.]).repeat([rays_o.shape[0],1]).to(self.device)
                studio.mega_nerf_mask(rays_d,rays_o[0],self.centers.to(self.device),t_range,mask,int(1024),1.2)
                torch.cuda.empty_cache()
                for j in range(0,self.centers.shape[0]):
                    
                    bits_array = torch.zeros([bits_array_len],dtype=torch.int64).to(self.device)

                    studio.packbits_u32(mask[:,j],bits_array)
                    bits_array = bits_array.to('cpu')
                    if (mask[:,j] > 0).sum() > self.img_wh[0]*self.img_wh[1]/10:
                    # if (mask > 0).sum() > 4000:
                        torch.save({
                                    "bits_array":bits_array,
                                    "pose_idx":i,
                                    "rays_nums":mask[:,j].sum()
                                },os.path.join(self.mask_save_path,"{}".format(j),'metadata_{}.pt'.format(i)))  
                
                
        elif mask_type=='aabb_intersect':
            # for i in range(self.config.model_start_num,len(self.grids)):
            for i in range(0,len(self.grids)):
                for j in tqdm(range(0,self.poses.shape[0])):
                    rays_o,rays_d = get_rays(directions = self.directions,c2w = self.poses[j])
                    rays_o = rays_o.to(self.device)
                    rays_d = rays_d.to(self.device)
                    aabb = self.aabbs[i].to(self.device)
                    t_min,t_max = ray_aabb_intersect(rays_o,rays_d,aabb)

                    # intersect_pix_idx = torch.where(t_min<100)[0]#存放
                    intersect_pix_idx = (t_min < 100).to(torch.int32).to(self.device) # [w*h, ],dtype = bool
                    bits_array = torch.zeros([bits_array_len],dtype=torch.int64).to(self.device)
                    studio.packbits_u32(intersect_pix_idx,bits_array)
                    bits_array = bits_array.to('cpu')
                    # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.aabbs,aabb_idx = 10,img_wh=self.img_wh)
                    if (t_min<100).sum() > 4000:

                        # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.aabbs,aabb_idx = 10,img_wh=self.img_wh)
                        torch.save({
                            "bits_array":bits_array,
                            "pose_idx":j,
                            "rays_nums":intersect_pix_idx.sum()
                        },os.path.join(self.mask_save_path,"{}".format(i),'metadata_{}.pt'.format(j)))  
        elif mask_type == 'ray_distance_mask':
            for i in tqdm(range(0,self.poses.shape[0])):
                rays_o,rays_d = get_rays(directions = self.directions,c2w = self.poses[i])
                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)
                mask = torch.zeros([rays_o.shape[0],self.centers.shape[0]],device=self.device).to(torch.int32)
                t_range = torch.tensor([0.,5.]).repeat([rays_o.shape[0],1]).to(self.device)
                studio.distance_mask(rays_d,rays_o[0],self.centers.to(self.device),mask,1.15)
                torch.cuda.empty_cache()
                for j in range(0,self.centers.shape[0]):
                    
                    bits_array = torch.zeros([bits_array_len],dtype=torch.int64).to(self.device)

                    studio.packbits_u32(mask[:,j],bits_array)
                    bits_array = bits_array.to('cpu')
                    if (mask > 0).sum() > 4000:
                        torch.save({
                                    "bits_array":bits_array,
                                    "pose_idx":i,
                                    "rays_nums":mask[:,j].sum()
                                },os.path.join(self.mask_save_path,"{}".format(j),'metadata_{}.pt'.format(i)))  
        elif mask_type == 'camera_position_mask':
            for i in tqdm(range(0,self.poses.shape[0])):
                rays_o,rays_d = get_rays(directions = self.directions,c2w = self.poses[i])
                rays_o = rays_o.to(self.device)
                

                for j in range(0,self.centers.shape[0]):
                    aabb=self.fg_aabbs[j,:].to(self.device)
                    # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.fg_aabbs,aabb_idx = 10,img_wh=self.img_wh)
                    if in_aabb(position=rays_o[0,:],aabb=aabb):
                        mask=torch.ones(rays_o.shape[0],device=self.device).to(torch.int32)
                    else:
                        mask=torch.zeros(rays_o.shape[0],device=self.device).to(torch.int32)
                    bits_array = torch.zeros([bits_array_len],dtype=torch.int64).to(self.device)
                    studio.packbits_u32(mask,bits_array)
                    bits_array = bits_array.to('cpu')
                    if (mask > 0).sum() > 4000:
                        torch.save({
                                    "bits_array":bits_array,
                                    "pose_idx":i,
                                    "rays_nums":mask.sum()
                                },os.path.join(self.mask_save_path,"{}".format(j),'metadata_{}.pt'.format(i)))  

    def get_R_correct(self):
        with open(self.Rc_path,'r') as f:
            lines = f.readlines()
        matrices = []
        matrix = []
        for line in lines:
            if not line.strip():
                if matrix:
                    matrices.append(np.array(matrix,dtype=np.float32))
                    matrix = []
            else:
                row = [float(x) for x in line.split()]
                matrix.append(row)
        R_correct = torch.eye(4)
        for i in range(0,len(matrices)):
            R_correct = torch.tensor(matrices[i]) @ R_correct
        return R_correct
    def scale_to(self,scale,current_model_idx):
        max_scale = self.scales[0].max()
        factor = scale/max_scale
        self.centers *= factor
        self.poses[...,3] *= factor
        self.scales *= factor
        try:
            self.aabbs *= factor#for divide tool
        except:
            pass
        # self.poses[...,3] -= self.centers[current_model_idx].view(-1,3)
        
        
        
        
        
        
        