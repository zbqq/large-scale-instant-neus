import torch
import numpy as np
import os
from tqdm import tqdm
import trimesh
from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from datasets.datasets import BaseDataset
from load_tool import draw_poses
from .divide_utils import divideTool
# RDF2BRU = np.array([[1,0,0],[0,0,-1],[0,1,0]]) @ np.array([[0,0,1],[0,-1,0],[1,0,0]])
class ColmapDataset(BaseDataset,divideTool):
    def __init__(self, config,split='train', downsample=1.0, *kwargs):
        super().__init__(config, split, downsample)
        super(BaseDataset,self).__init__(config,split)
        self.read_intrinsics()
        self.read_meta()
        # self.load_center([self.config.grid_X,self.config.grid_Y])
    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)
        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model == 'PINHOLE':
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        elif camdata[1].model == 'OPENCV':
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
            self.camera_params = np.array([camdata[1].params[4],
                                          camdata[1].params[5],
                                          camdata[1].params[6],
                                          camdata[1].params[7]
                                          ])
            
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.camera_model = camdata[1].model
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        
        # self.K_inv = torch.linalg.inv(self.K).to(self.device)
        # self.directions = get_ray_directions(h, w, self.K).to(self.device)#相机坐标系下的dirs，也即归一化坐标|1
        self.K_inv = torch.linalg.inv(self.K)
        self.directions = get_ray_directions(h, w, self.K)#相机坐标系下的dirs，也即归一化坐标|1
    
    def read_meta(self):
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        folder = 'images'
        self.img_paths = [os.path.join(self.root_dir, folder, name)
                         for name in sorted(img_names)]
        
        
        w2c_mats = []
        bottom = torch.tensor([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = torch.from_numpy(im.qvec2rotmat())
            t = torch.from_numpy(im.tvec.reshape(3, 1))
            w2c_mats += [torch.concat([torch.concat([R, t], 1), bottom], 0)]
        w2c_mats = torch.stack(w2c_mats, 0).to(torch.float32)
        self.R_correct = self.get_R_correct()
        #校正poses，这里的poses是所有的poses
        self.poses = (self.R_correct[None,...] @ torch.linalg.inv(w2c_mats))[perm, :3] # (N_images, 3, 4) cam2world matrices
        self.grid_dim = torch.tensor([self.config.grid_X,
                             self.config.grid_Y,
                             1])
        # if self.split == 'divide': # 首先需要用cloudCopmare根据预处理后的稀疏点云分割，得到scale
        #     # grid_dim = torch.tensor([])
        #     pass
        #     # self.gen_centers_from_pts(grid_dim)
        #     # pts3d = self.load_center()
        #     # pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        #     # pts3d = torch.tensor([np.concatenate([pts3d[k].xyz,[1]]) for k in pts3d],dtype=torch.float32) # (N, 3)
        #     # pts3d = (self.R_correct @ pts3d[:,:,None]).reshape(-1,4)[:,:3]
        #     # draw_poses(pts3d=pts3d)
        self.rays = []
        if self.split == 'train':
            self.load_centers()#得到centers和scale
            self.load_mask()
            # self.idxs = [self.idxs[i] for i in range(0,len(self.idxs)) if i%8!=0]
            # self.img_paths = [self.img_paths[self.idxs[i]] for i in range(0,len(self.idxs)) if i%8!=0]
            # self.poses = [self.poses[self.idxs[i]] for i in range(0,len(self.idxs)) if i%8!=0]
            # self.poses = torch.stack(self.poses)
        if self.split == 'test':
            self.load_centers()
            
            self.load_mask()
            del self.poses
            # self.idxs = [self.idxs[i] for i in range(0,len(self.idxs)) if i%8==0]
            # # self.img_paths = [self.img_paths[self.idxs[i]] for i in range(0,len(self.idxs)) if i%8==0]
            # self.poses = [self.poses[self.idxs[i]] for i in range(0,len(self.idxs)) if i%8==0]
            # self.poses = torch.stack(self.poses)
        pass