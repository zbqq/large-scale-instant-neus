import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import os
import cv2
from torch.utils.data import DataLoader
# from utils.ray_utils import get_rays
# from apex.optimizers import FusedAdam
from systems.base import BaseSystem
from scripts.load_tool import draw_poses
from utils.utils import load_ckpt_path
from skimage.metrics import peak_signal_noise_ratio as psnr


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

class NeRFSystem(BaseSystem):
    def __init__(self,config):
        super().__init__(config)#最初的config
        pass
    def on_train_start(self):
        # self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
        #                                 self.poses,
        #                                 self.train_dataset.img_wh)
        pass
    # def preprocess_data(self, batch, stage):
    #     if 'index' in batch: # validation / testing
    #         index = batch['index']
    #     else:
    #         if self.config.model.batch_image_sampling:
    #             index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
    #         else:
    #             index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        
    #     if stage in ['train']:
    #         c2w 
        

    
    def forward(self, batch,split):
        if split == 'train':
            # poses = batch['pose']
            poses = self.poses[batch['pose_idx']]
            dirs = batch['directions']
            # dirs = self.directions
            rays_o, rays_d = get_rays(dirs,poses[None,...])
            # draw_poses(poses_=self.poses[...],aabb_=self.model.scene_aabb[None,...],img_wh=self.train_dataset.img_wh)
            
            # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.model.scene_aabb[None,...],img_wh=self.train_dataset.img_wh)
            del dirs,poses
            
            return self.model(rays_o, rays_d,split)#返回render结果
        else:
            poses = self.test_poses[batch['pose_idx']]
            dirs = self.test_directions# 一副图像
            rays_o, rays_d = get_rays(dirs,poses[None,...])
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
        
        if self.global_step%self.config.model.grid_update_freq == 0 and self.config.model.use_raymarch :
            self.model.update_step(5,self.global_step)
        render_out = self(batch,split='train')
        loss_d = self.loss(render_out, batch)
        
        loss = sum(lo.mean() for lo in loss_d.values())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("depth", render_out['depth'].mean(),prog_bar=True,sync_dist=True)
        
        self.log('geo_lr', torch.tensor(self.net_opt.param_groups[0]['lr']), prog_bar=True,sync_dist=True)
        # self.log('color_lr', torch.tensor(self.net_opt.param_groups[1]['lr']), prog_bar=True,sync_dist=True)
        with torch.no_grad():
            pred = render_out['rgb'].to("cpu").numpy()
            gt = batch['rays'].to("cpu").numpy()
        
        self.log('psnr',torch.tensor(psnr(gt,pred)),prog_bar=True,sync_dist=True)
        return {
            'loss': loss
        }
    
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
