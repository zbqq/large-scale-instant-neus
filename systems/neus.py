import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import os
import cv2
from torch.utils.data import DataLoader
from utils.ray_utils import get_rays
from apex.optimizers import FusedAdam
from systems.base import BaseSystem
from model.loss import NeRFLoss
from datasets.colmap import ColmapDataset
from model.neus import NeuS
from load_tool import draw_poses
from utils.utils import parse_optimizer
from utils.utils import load_ckpt_path
class NeuSSystem(BaseSystem):
    def __init__(self,config):
        super().__init__(config)#最初的config
        #init不能将所有的模型都实例化
        self.model_num = self.config.dataset.grid_X * self.config.dataset.grid_Y
        if self.model_num> 1:
            for i in range(0,self.model_num):
                os.makedirs(os.path.join(self.config.save_dir,'{}'.format(i),'{}'.format(self.config.model.name)),exist_ok=True)
                # 先实例化不setup不用占多少显存
            pass
        else:
            os.makedirs(os.path.join(self.config.save_dir,'0','{}'.format(self.config.model.name)),exist_ok=True)

        self.current_model_num = self.config.model_start_num # 训练到的第几个模型
        self.current_model_num_tmp = self.config.model_start_num # 训练到的第几个模型
        
            
    def setup(self,stage):
        # if not self.config.is_continue:
        # dataset = ColmapDataset(self.config.dataset)
        self.train_dataset = ColmapDataset(self.config.dataset,split='train',downsample=1.0)
        self.train_dataset.batch_size = self.config.dataset.batch_size
        # self.train_dataset.ray_sampling_strategy = self.config.dataset.ray_sampling_strategy
        self.test_dataset = ColmapDataset(self.config.dataset,split='test',downsample=0.2)
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        self.register_buffer('test_directions', self.test_dataset.directions.to(self.device))
        setattr(self,"model{}".format(self.current_model_num),NeuS(self.config.model)) # 需要浅拷贝
        self.model = getattr(self,"model{}".format(self.current_model_num),NeuS(self.config.model)) # 需要浅拷贝
        self.model.setup(self.train_dataset.centers[self.current_model_num,:],
                         self.train_dataset.scale[self.current_model_num,:])
        self.net_opt = parse_optimizer(self.config.system.optimizer,self.model)
        self.loss = NeRFLoss(config=self.config.system.loss,lambda_distortion=0)
        if self.config.is_continue:
            _,ckpt_path = load_ckpt_path(os.path.join(self.config.ckpt_dir,
                                        '{}'.format(self.current_model_num),
                                        self.config.model.name)
                                       ) 
            self.load_checkpoint(ckpt_path)
    def train_dataloader(self):
        if self.config.use_DDP:
            return DataLoader(self.train_dataset,
                              num_workers=0,
                              persistent_workers=False,
                              batch_size=None,
                              pin_memory=False)
        else:
            return DataLoader(self.train_dataset,
                              num_workers=0,
                              persistent_workers=False,
                              batch_size=None,
                              pin_memory=False)
    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=0,
                          batch_size=None,
                          pin_memory=False)
    
    def on_train_start(self):
        # self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
        #                                 self.poses,
        #                                 self.train_dataset.img_wh)
        pass

    def configure_optimizers(self):
        
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
        if split == 'train':
            # poses = batch['pose']
            poses = self.poses[batch['pose_idx']]
            dirs = batch['directions']
            # dirs = self.directions
            rays_o, rays_d = get_rays(dirs,poses)
            del dirs,poses
            # draw_poses(rays_o_=rays_o,rays_d_=rays_d,poses_=poses[None,...],aabb_=self.model.scene_aabb[None,...],img_wh=self.train_dataset.img_wh)
            # draw_poses(poses_=poses[None,...],aabb_=self.model.scene_aabb[None,...],img_wh=self.train_dataset.img_wh,pts3d=self.train_dataset.pts3d)
            
            return self.model(rays_o, rays_d)#返回render结果
        else:
            poses = self.poses[batch['pose_idx']]
            dirs = self.test_directions# 一副图像
            rays_o, rays_d = get_rays(dirs,poses)
            del dirs
            rays_o = rays_o.split(self.config.dataset.split_num)
            rays_d = rays_d.split(self.config.dataset.split_num)
            return self.model.render_whole_image(rays_o,rays_d)
            
    def training_step(self, batch, batch_idx):
        """
        batch:{
            "rays":rgbs, [N_rays 3]
            "directions":directions, [N_rays 3]
            "pose":pose [3 4]
        }
        """
        if self.global_step%self.config.model.grid_update_freq == 0:
            self.model.update_step(5,self.global_step)
        render_out = self(batch,split='train')
        loss_d = self.loss(render_out, batch)
        
        loss = sum(lo.mean() for lo in loss_d.values())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('sdf_lr', torch.tensor(self.net_opt.param_groups[0]['lr']), prog_bar=True)
        self.log('tex_lr', torch.tensor(self.net_opt.param_groups[1]['lr']), prog_bar=True)
        self.log('var_lr', torch.tensor(self.net_opt.param_groups[2]['lr']), prog_bar=True)
        if 'inv_s' in render_out.keys():
            self.log('train/inv_s', render_out['inv_s'], prog_bar=True)
        return {
            'loss': loss
        }
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    def training_step_end(self, step_output):
        if (self.global_step+1) % self.config.validate_freq == 0:
            with torch.no_grad():
                for idx,item in enumerate(self.val_dataloader()):
                    self.validation_step(item,0)
                    break
    def training_epoch_end(self,loss):#是否进入到此处是datasets len决定的
        #保存现模型
        lr = self.net_opt.param_groups[0]['lr']
        print('learning_rate',lr)
        self.current_model_num_tmp += 1
        #更新优化器在optimizer.step中执行
        if self.current_model_num != self.current_model_num_tmp:
            
            # 注册新模型
            del self.model#浅拷贝，等价于删掉modeli
            setattr(self,"model{}".format(self.current_model_num),NeuS(self.config.model))
            self.model = getattr(self,"model{}".format(self.current_model_num),NeuS(self.config.model))
            self.model.setup(self.train_dataset.centers[self.current_model_num,:],
                             self.train_dataset.scale[self.current_model_num,:])
            self.configure_optimizers()
        
    

    
    
    
    
    
    def validation_epoch_end(self, out):
    #     out = self.all_gather(out)
    #     if self.trainer.is_global_zero:
    #         out_set = {}
    #         for step_out in out:
    #             # DP
    #             if step_out['index'].ndim == 1:
    #                 out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
    #             # DDP
    #             else:
    #                 for oi, index in enumerate(step_out['index']):
    #                     out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
    #         psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
    #         self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    # def test_step(self, batch, batch_idx):
    #     out = self(batch)
    #     psnr = self.criterions['psnr'](out['comp_rgb'], batch['rgb'])
    #     W, H = self.config.dataset.img_wh
    #     img = out['comp_rgb'].view(H, W, 3)
    #     depth = out['depth'].view(H, W)
    #     opacity = out['opacity'].view(H, W)
    #     self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
    #         {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
    #         {'type': 'rgb', 'img': img, 'kwargs': {'data_format': 'HWC'}},
    #         {'type': 'grayscale', 'img': depth, 'kwargs': {}},
    #         {'type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
    #     ])
    #     return {
    #         'psnr': psnr,
    #         'index': batch['index']
    #     }      
        pass
    def test_epoch_end(self, out):
        # """
        # Synchronize devices.
        # Generate image sequence using test outputs.
        # """
        # out = self.all_gather(out)
        # if self.trainer.is_global_zero:
        #     out_set = {}
        #     for step_out in out:
        #         # DP
        #         if step_out['index'].ndim == 1:
        #             out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
        #         # DDP
        #         else:
        #             for oi, index in enumerate(step_out['index']):
        #                 out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
        #     psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
        #     self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

        #     self.save_img_sequence(
        #         f"it{self.global_step}-test",
        #         f"it{self.global_step}-test",
        #         '(\d+)\.png',
        #         save_format='mp4',
        #         fps=30
        #     )
            
        #     mesh = self.model.isosurface()
        #     self.save_mesh(
        #         f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
        #         mesh['v_pos'],
        #         mesh['t_pos_idx'],
        #     )
        pass
