import torch
import math
import numpy as np
from PIL import Image
import os
import torchvision.transforms.functional as TF
import json
from .ray_utils import *
from datasets.datasets import BaseDataset
from load_tool import draw_poses
from .divide_utils import divideTool
def get_ray_directionsRUB(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

class BlenderDataset(BaseDataset,divideTool):
    def __init__(self, config, split='train', downsample=1.0):
        super().__init__(config, split, downsample)
        super(BaseDataset,self).__init__(config,split)
        self.read_data()
        
        
    def read_data(self):
        # with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
        path = os.path.join(self.config.root_dir, f"{meta['frames'][0]['file_path']}.png")
        img_0 = Image.open(path)
        w,h = int(img_0.size[0]*self.downsample), int(img_0.size[1]*self.downsample)
        self.img_wh=(w,h)

        # if 'w' in meta and 'h' in meta:
        #     W, H = int(meta['w']), int(meta['h'])
        # else:
        #     W, H = 800, 800

        # w, h = self.config.img_wh
        # assert round(W / w * h) == H
        
        # self.w, self.h = w, h
        # self.img_wh=(w,h)
        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * self.img_wh[0] / math.tan(0.5 * meta['camera_angle_x']) * self.downsample # scaled focal length
        fx,fy = self.focal,self.focal
        cx,cy = self.img_wh[0] * self.downsample, self.img_wh[1] * self.downsample
        
        self.K = torch.FloatTensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]])
        self.K_inv = torch.linalg.inv(self.K)
        # self.directions = get_ray_directions(h, w, self.K)#相机坐标系下的dirs，也即归一化坐标|1
        self.directions = get_ray_directionsRUB(w, h, self.focal, self.focal, w//2, h//2, True).reshape(-1,3)           
        

        all_c2w = []
        self.img_paths = []
        for _, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            # c2w[:,1:3] *= -1. 
            all_c2w.append(c2w)
            self.img_paths.append(os.path.join(self.config.root_dir, f"{frame['file_path']}.png"))

            # img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            # img = Image.open(img_path)
            # img = img.resize(self.config.img_wh, Image.BICUBIC)
            # img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            # self.all_fg_masks.append(img[..., -1]>0) # (h, w)
            # img = img[...,:3] * img[...,-1:] + (1 - img[...,-1:]) # white background
            # self.all_images.append(img)

        self.poses = torch.stack(all_c2w, dim=0).float()
        # draw_poses(poses_=self.poses)
        if self.split == 'train':
            # pass
            self.load_centers()
            self.load_mask()
            # self.scale_to(scale=self.config.scale_to,current_model_idx=self.current_model_num)
        if self.split == 'test':
            self.load_centers()
            # self.load_mask()
            # self.scale_to(scale=self.config.scale_to,current_model_idx=self.current_model_num)
  
        
        
        
        
        