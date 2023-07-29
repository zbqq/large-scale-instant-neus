import pytorch_lightning as pl
import torch
import torch.nn as nn
from kornia.utils import create_meshgrid3d
from utils.utils import parse_optimizer
from utils.utils import load_ckpt_path
import numpy as np
import cv2
import os
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect
from utils.ray_utils import draw_aabb_mask
from skimage.metrics import peak_signal_noise_ratio as psnr
from datasets.colmap import ColmapDataset
from model.loss import NeRFLoss
from tqdm import tqdm
from model.nerf import vanillaNeRF
from model.neus import NeuS
from model.merge import mainModule

from systems.base import BaseSystem

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from torch.utils.data import DataLoader
from utils.ray_utils import get_rays
DATASETS={
    'colmap':ColmapDataset
}
MODELS={
    'nerf':vanillaNeRF,
    'neus':NeuS
} 
def load_checkpoint(config,ckpt_path=None):
    assert ckpt_path is not None
    model = MODELS[config.model.name](config)
    system_dict = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict['model']
    return model
class mainSystem(BaseSystem):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.model_dir = config.save_dir
        self.model_nums = config.grid_X * config.grid_Y
        self.model_idxs=[int(idx) for idx in config.merge_modules.split(',')]
        
        
    def save_model(self,ckpt_name):
        ckpt = {
            'model':self.model.state_dict()
        }
        torch.save(ckpt,ckpt_name)
    
    def setup(self,stage):
        models=[]
        centers=[]
        aabbs=[]
        for i in self.model_idxs: # 加载模型
            ckpt_paths = os.listdir(os.path.join(self.model_dir,f'{i}/{self.config.model.name}/ckpts'))
            model_path = os.path.join(self.model_dir,f'{i}/{self.config.model.name}/ckpts',ckpt_paths[-1])
            term = torch.load(model_path)        
            model = term['model'] # 这里是直接取得ckpt的aabb，因此是scale后的，
            # del model['density_bitfield']
            # del model['density_grid']
            x = MODELS[self.config.model.name](config=self.config.model)
            x.setup(model['center'].cpu(),model['scale'].cpu())
            x.load_state_dict(model,strict=False)
            models.append(x)
            centers.append(x.center)
            aabbs.append(x.scene_aabb)
            del term
            del model
            pass
        centers = torch.stack(centers)
        aabbs = torch.stack(aabbs)
        
        self.model = mainModule(config=self.config.model,sub_modules=models)
        
        self.test_dataset = DATASETS[self.config.dataset.name](self.config.dataset,split='merge_test',downsample=self.config.dataset.test_downsample)
        self.test_dataset.gen_traj(centers,aabbs)
        # self.test_dataset = DATASETS[self.config.dataset.name](self.config.dataset,split='test',downsample=self.config.dataset.test_downsample)
        # self.register_buffer('poses',self.test_dataset.poses)
        self.register_buffer('test_directions', self.test_dataset.directions)
        self.register_buffer('aabbs',self.test_dataset.aabbs)
        # self.register_buffer()
    def forward(self, pose,split):
        # poses = batch['pose']
        
        assert split == 'merge_test'
        
        rays_o, rays_d = get_rays(self.test_directions,pose[None,...])
        return self.model(rays_o,rays_d,weights_type="UW")

    def test_step(self, batch,batch_idx):
        poses = self.test_dataset.poses_traj.to(self.device)
        self.model = self.model.to(self.device)
        if self.config.model.point_sample.use_raymarch:
            self.model.update_step(5,self.global_step)
        pbar = tqdm(total=poses.shape[0])
        prefix = self.config.save_dir + f"/merge/{self.config.model.name}"
        os.makedirs(prefix,exist_ok=True)
        aabbs = self.aabbs[self.model_idxs]
        for idx in range(0,poses.shape[0]):
            out = self(poses[idx],split='merge_test')

            

            W, H = self.test_dataset.img_wh
            rgbs_val = out["rgb"].view(H, W, 3)
            img_with_aabbmask = draw_aabb_mask(rgbs_val,self.test_dataset.w2cs[idx],self.test_dataset.K,aabbs)
            depth = out['depth'].view(H, W)

            # psnr_ = psnr(rgbs_true.to("cpu").numpy(),rgbs_val.to("cpu").numpy())
            img_name = os.path.join(prefix,"images",\
                                    f"merge_{self.config.merge_modules}_{idx}.png"
                                    )
            self.save_image_grid(img_name, [
                # {'type': 'rgb', 'img': rgbs_true, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': img_with_aabbmask, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgbs_val, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': depth, 'kwargs': {}},

                # {'type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}

            ])
            pbar.update(1)
        return None
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=0,
                          persistent_workers=False,
                          batch_size=None,
                          pin_memory=False)
        
        
        
        