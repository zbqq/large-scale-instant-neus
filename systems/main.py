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
        for i in self.model_idxs:
            ckpt_paths = os.listdir(os.path.join(self.model_dir,f'{i}/{self.config.model.name}/ckpts'))
            model_path = os.path.join(self.model_dir,f'{i}/{self.config.model.name}/ckpts',ckpt_paths[-1])
            term = torch.load(model_path)        
            model = term['model']
            # del model['density_bitfield']
            # del model['density_grid']
            x = MODELS[self.config.model.name](config=self.config.model)
            x.setup(model['center'].cpu(),model['scale'].cpu())
            x.load_state_dict(model)
            del x.density_bitfield
            del x.density_grid
            models.append(x)
            del term
            del model
            pass
        self.model = mainModule(config=self.config.model,sub_modules=models)
        
        self.test_dataset = DATASETS[self.config.dataset.name](self.config.dataset,split='merge_test',downsample=0.2)
        # self.test_dataset = DATASETS[self.config.dataset.name](self.config.dataset,split='test',downsample=self.config.dataset.test_downsample)
        self.register_buffer('poses',self.test_dataset.poses)
        self.register_buffer('test_directions', self.test_dataset.directions)
        
    def forward(self, pose,split):
        # poses = batch['pose']
        
        assert split == 'merge_test'
        
        # poses = self.poses[batch['pose_idx']]
        # pose = batch['pose']
        # dirs = self.directions
        rays_o, rays_d = get_rays(self.test_directions,pose)
        return self.model(rays_o,rays_d,weights_type="UW")

    def test_step(self, batch,batch_idx):
        
        self.model = self.model.to(self.device)
        self.model.update_step(5,self.global_step)
        pbar = tqdm(total=batch['poses'].shape[0])
        for idx in range(0,batch['poses'].shape[0]):
            out = self(batch['poses'][idx],split='merge_test')

            prefix = self.config.save_dir + f"/{self.current_model_num}/{self.config.model.name}"

            W, H = self.test_dataset.img_wh
            rgbs_val = out["rgb"].view(H, W, 3)
            depth = out['depth'].view(H, W)

            # psnr_ = psnr(rgbs_true.to("cpu").numpy(),rgbs_val.to("cpu").numpy())
            img_name = os.path.join(prefix,"images",\
                                    f"merge_{self.config.merge_modules}_{idx}.png"
                                    )
            self.save_image_grid(img_name, [
                # {'type': 'rgb', 'img': rgbs_true, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgbs_val, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': depth, 'kwargs': {}},

                # {'type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}

            ])
            pbar.update(1)
        return None
        
        
        
        
        