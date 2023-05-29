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
class divideTool():
    def __init__(self,config,split='divide'):
        self.config = config
        self.split = split   
        self.Rc_path = os.path.join(self.config.root_dir,'R_correct.txt')#需要输入的
        self.new_pts3d_path = os.path.join(self.config.root_dir,'new_pts3d.txt')#需要输入的
        self.centers_and_scales_path = os.path.join(self.config.root_dir,'CaS.txt')#输出的
        self.mask_save_path = os.path.join(self.config.mask_dir,"model")
        self.model_num = self.config.grid_X * self.config.grid_Y
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  
        else:
            raise RuntimeError("divide tool must use GPU!")
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
        min_position = pts.min(dim=0)[0]#是否需要扩大一点?
        max_position = pts.max(dim=0)[0]
        
        radius = (max_position-min_position)/grid_dim/2 * 1.5# grid的半径，扩大1.5倍以实现有重叠
        ranges = max_position - min_position
        offsets = [torch.arange(s) * ranges[i] / s + ranges[i] / (s * 2) for i, s in enumerate(grid_dim)]#每个方格的中心world
        
        # 根据稀疏点云划分grid,meshgrid格式
        centroids = torch.stack((\
                             torch.ones(grid_dim[0], grid_dim[1])*min_position[0],
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
            scene_aabb = get_aabb(center=center,scale=torch.cat([radius[:2],torch.tensor([0.2])]).view(3,1))#divide时需要将scale_z变得很小
            # scene_aabb = get_aabb(center=center,scale=radius[:3].view(3,1))#divide时需要将scale_z变得很小

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
 
    def load_centers(self):
        temp = torch.tensor(np.loadtxt(self.centers_and_scales_path),dtype=torch.float32)
        self.centers = temp[...,:3].view(-1,3)
        self.scales = temp[...,3:].view(-1,3)+torch.tensor([4,4,3]).view(1,3)
    
    def divide(self,grid_dim):#从已经得到的sparse pts进行区域的分割，
        """
            
        """
        self.gen_centers_from_pts(grid_dim)

        bits_array_len = math.floor(self.img_wh[0]*self.img_wh[1]/32)
        for i in range(self.config.model_start_num,len(self.grids)):
            grid = self.grids[i]
            for j in tqdm(range(0,self.poses.shape[0])):
                rays_o,rays_d = get_rays(directions = self.directions,c2w = self.poses[j])
                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)
                aabb = grid._roi_aabb.to(self.device)
                t_min,t_max = ray_aabb_intersect(rays_o,rays_d,aabb)
                # intersect_pix_idx = torch.where(t_min<100)[0]#存放
                intersect_pix_idx = (t_min < 100).to(torch.int32).to(self.device) # [w*h, ],dtype = bool
                bits_array = torch.zeros([bits_array_len],dtype=torch.int64).to(self.device)
                studio.packbits_u32(intersect_pix_idx,bits_array)
                # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.aabbs,aabb_idx = 10,img_wh=self.img_wh)
                if (t_min<100).sum() > 4000:

                    # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.aabbs,aabb_idx = 10,img_wh=self.img_wh)
                    torch.save({
                        "bits_array":bits_array,
                        "pose_idx":j,
                        "rays_nums":intersect_pix_idx.sum()
                    },os.path.join(self.mask_save_path,"{}".format(i),'metadata_{}.pt'.format(j)))  
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
        self.poses[...,3] -= self.centers[current_model_idx].view(-1,3)
        
        
        
        
        
        
        