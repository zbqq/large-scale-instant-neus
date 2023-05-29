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

from datasets.colmap import ColmapDataset
from model.loss import NeRFLoss

from model.nerf import vanillaNeRF
from model.neus import NeuS

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from torch.utils.data import DataLoader
from base import ImageProcess
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
class mainSystem(pl.LightningModule,ImageProcess):
    def __init__(self,config):
        super().__init__(config)#最初的config
        #init不能将所有的模型都实例化
        self.model_num = self.config.dataset.grid_X * self.config.dataset.grid_Y
        self.model_paths=[]
        if self.model_num> 1:
            for i in range(0,self.model_num):
                self.model_paths.append(os.path.join(self.config.save_dir,'{}'.format(i),'{}'.format(self.config.model.name)))
                # 先实例化不setup不用占多少显存
            pass
        else:
            self.model_paths.append(os.path.join(self.config.save_dir,'0','{}'.format(self.config.model.name)))

        self.current_model_num = self.config.model_start_num # 训练到的第几个模型
        self.current_model_num_tmp = self.config.model_start_num # 训练到的第几个模型
    def setup(self,stage):

        self.test_dataset = DATASETS[self.config.dataset.name](self.config.dataset,split='test',downsample=1.0)
        models = []
        for i in range(self.model_paths):
            _,ckpt_path = load_ckpt_path(os.path.join(self.config.ckpt_dir,
                            '{}'.format(i),
                            self.config.model.name)
                            ) 
            model = load_checkpoint(self.config,ckpt_path=ckpt_path)
            models.append(model)
        self.models = nn.ModuleList(models)
    def on_train_start(self):
        # self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
        #                                 self.poses,
        #                                 self.train_dataset.img_wh)
        pass

    def configure_optimizers(self):
        
        self.train_dataset.device = self.device
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        self.register_buffer('test_directions', self.test_dataset.directions.to(self.device))
        # opts=[]
        # for n, p in self.model.named_parameters():
        #     if n not in ['dR', 'dT']: net_params += [p]
        
        # self.net_opt=torch.optim.Adam(net_params, self.config.system.optimizer.args.lr,eps=self.config.system.optimizer.args.eps)
        # self.net_opt = FusedAdam(net_params, lr=self.config.system.optimizer.lr, eps=self.config.system.optimizer.eps)
        
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.net_opt,step_size=self.config.system.scheduler.args.step_size,gamma=self.config.system.scheduler.args.gamma)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.net_opt,gamma=self.config.system.scheduler.args.gamma)
        
        # return [self.net_opt],[lr_scheduler]
        return {
            "optimizer":self.net_opt,
            "lr_scheduler":{
                "scheduler":lr_scheduler,
                "interval":"step",
                "frequency": 1,
                "strict": True,
                "name": None,
            }
        }
    def forward(self, batch,split):
        # poses = batch['pose']
        poses = self.poses[batch['pose_idx']]
        dirs = batch['directions']
        # dirs = self.directions
        rays_o, rays_d = get_rays(dirs,poses)
        for i in range(0,len(self.models)):
            t_min,t_max = ray_aabb_intersect(
                
            )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def training_step(self, batch, batch_idx):
        pass
    

    def validation_step(self, batch,batch_idx):
        output = self(batch)
        
        
        
        
        