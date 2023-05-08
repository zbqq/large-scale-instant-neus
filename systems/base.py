import pytorch_lightning as pl
import torch
from kornia.utils import create_meshgrid3d
import numpy as np
import cv2
import os

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm



class ImageProcess():#存储图像
    @property
    def save_dir(self):
        return self.config.save_dir
    def convert_data(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))
    
    DEFAULT_RGB_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1)}
    DEFAULT_UV_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1), 'cmap': 'checkerboard'}
    DEFAULT_GRAYSCALE_KWARGS = {'data_range': None, 'cmap': 'jet'}
    def ConcatImg(self,img_val,img_true):#均为numpy格式
        Img = np.concatenate(img_val,img_true)
        return Img
    def get_rgb_image_(self, img, data_format, data_range):
        img = self.convert_data(img)#numpy格式
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = ((img - data_range[0]) / (data_range[1] - data_range[0]) * 255.).astype(np.uint8)
        imgs = [img[...,start:start+3] for start in range(0, img.shape[-1], 3)]
        imgs = [img_ if img_.shape[-1] == 3 else np.concatenate([img_, np.zeros((img_.shape[0], img_.shape[1], 3 - img_.shape[2]), dtype=img_.dtype)], axis=-1) for img_ in imgs]
        img = np.concatenate(imgs, axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    def get_save_path(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    def get_grayscale_image_(self, img, data_range, cmap):
        img = self.convert_data(img)
        img = np.nan_to_num(img)
        if data_range is None:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.clip(data_range[0], data_range[1])
            img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in [None, 'jet', 'magma']
        if cmap == None:
            img = (img * 255.).astype(np.uint8)
            img = np.repeat(img[...,None], 3, axis=2)
        elif cmap == 'jet':
            img = (img * 255.).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif cmap == 'magma':
            img = 1. - img
            base = cm.get_cmap('magma')
            num_bins = 256
            colormap = LinearSegmentedColormap.from_list(
                f"{base.name}{num_bins}",
                base(np.linspace(0, 1, num_bins)),
                num_bins
            )(np.linspace(0, 1, num_bins))[:,:3]
            a = np.floor(img * 255.)
            b = (a + 1).clip(max=255.)
            f = img * 255. - a
            a = a.astype(np.uint16).clip(0, 255)
            b = b.astype(np.uint16).clip(0, 255)
            img = colormap[a] + (colormap[b] - colormap[a]) * f[...,None]
            img = (img * 255.).astype(np.uint8)
        return img
    def get_uv_image_(self, img, data_format, data_range, cmap):
        img = self.convert_data(img)
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in ['checkerboard', 'color']
        if cmap == 'checkerboard':
            n_grid = 64
            mask = (img * n_grid).astype(int)
            mask = (mask[...,0] + mask[...,1]) % 2 == 0
            img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img[mask] = np.array([255, 0, 255], dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cmap == 'color':
            img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_[..., 0] = (img[..., 0] * 255).astype(np.uint8)
            img_[..., 1] = (img[..., 1] * 255).astype(np.uint8)
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            img = img_
        return img
    def get_image_grid_(self, imgs):
        if isinstance(imgs[0], list):
            return np.concatenate([self.get_image_grid_(row) for row in imgs], axis=0)
        cols = []
        for col in imgs:
            assert col['type'] in ['rgb', 'uv', 'grayscale']
            if col['type'] == 'rgb':
                rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
                rgb_kwargs.update(col['kwargs'])
                cols.append(self.get_rgb_image_(col['img'], **rgb_kwargs))
            elif col['type'] == 'uv':
                uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
                uv_kwargs.update(col['kwargs'])
                cols.append(self.get_uv_image_(col['img'], **uv_kwargs))
            elif col['type'] == 'grayscale':
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col['kwargs'])
                cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))
        return np.concatenate(cols, axis=1)
    
    def save_rgb_image(self, filename, img, data_format=DEFAULT_RGB_KWARGS['data_format'], data_range=DEFAULT_RGB_KWARGS['data_range']):
        img = self.get_rgb_image_(img, data_format, data_range)
        cv2.imwrite(self.get_save_path(filename), img)
    def save_image_grid(self, filename, imgs):
        img = self.get_image_grid_(imgs)
        cv2.imwrite(self.get_save_path(filename), img)
    
class BaseSystem(pl.LightningModule,ImageProcess):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.scale=0.5
        self.cascades = max(1+int(np.ceil(np.log2(2*self.scale))), 1)
        
        
        self.grid_size = 128
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*self.scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        
        self.G = self.grid_size
        self.max_hits = 1
        # self.grid_coords = self.grid_coords.to("cuda")
        # self.density_grid = self.density_grid.to("cuda")
    def setup(self,stage):
        pass

    def forward(self, batch):
        raise NotImplementedError
    def preprocess_data(self, batch, stage):
        pass
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def extract_mesh(self):
        self.model.extract_mesh()
        
        
        pass
    def validation_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):        
        raise NotImplementedError
    
    def test_epoch_end(self, out):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError
    
    def save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.net_opt.state_dict(), #
            'epoch_step': self.global_step,
            'current_model_num': self.current_model_num,
        }
        torch.save(
            checkpoint,
                os.path.join(self.config.save_dir,"{}".format(self.current_model_num),
                                '{}'.format(self.config.model.name),
                                '{}-'.format(self.config.model.name)+
                                '{}-'.format(self.config.case_name)+
                                'ckpt_{:0>6d}.pt'.format(self.global_step)
                                ))
    def load_checkpoint(self,ckpt_path=None):
        if ckpt_path == None: return
        system_dict = torch.load(ckpt_path,map_location='cpu')
        self.current_modle_num = system_dict['current_model_num']
        # self.global_step = system_dict['epoch_step']
        self.model.load_state_dict(system_dict['model'])
        self.net_opt.load_state_dict(system_dict['optimizer'])
        # pass
    
    def validation_step(self, batch, batch_idx):#在traing_epoch_end之后
        """
            batch:{
                pose : [3 4]
                img_idxs : [1]
                rgb : [w*h 3]
            }
        """
        self.save_checkpoint()
        out = self(batch,split='val')# rays:[W*H,3]
        
        # psnr = self.criterions['psnr'](out['comp_rgb'], batch['rgb'])
        W, H = self.test_dataset.img_wh
        
        rgbs_true = batch["rays"].reshape(H,W,3)
        rgbs_val = out["rgb"].view(H, W, 3)
        depth = out['depth'].view(H, W)
        # opacity = out['opacity'].view(H, W)
        self.save_image_grid(f"model_{self.current_model_num}_it{self.global_step}_{batch['pose_idx']}.png", [
            {'type': 'rgb', 'img': rgbs_true, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': rgbs_val, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': depth, 'kwargs': {}},
            
            # {'type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
            
        ])
        return {
            # 'psnr': psnr,
            'index': batch['pose_idx']
        }
