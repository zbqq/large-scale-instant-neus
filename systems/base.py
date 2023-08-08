import pytorch_lightning as pl
import torch
from kornia.utils import create_meshgrid3d
from utils.utils import parse_optimizer
from utils.utils import load_ckpt_path
import numpy as np
import cv2
import os
import mcubes
import trimesh
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils.utils import parse_scheduler
from utils.ray_utils import draw_aabb_mask
from datasets.colmap import ColmapDataset
from datasets.blender import BlenderDataset
from model.loss import NeRFLoss

from model.nerf import vanillaNeRF
from model.neus import NeuS

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from utils.ray_utils import get_rays
from torch.utils.data import DataLoader

from model.val_utils import extract_geometry
DATASETS={
    'colmap':ColmapDataset,
    'blender':BlenderDataset
}
MODELS={
    'nerf':vanillaNeRF,
    'neus':NeuS
} 
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
    # DEFAULT_GRAYSCALE_KWARGS = {'data_range': None, 'cmap': 'jet'}
    DEFAULT_GRAYSCALE_KWARGS = {'data_range': [0,1.6], 'cmap': 'jet'}
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
                # cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))
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
         
        # self.scale=0.5
        # self.cascades = max(1+int(np.ceil(np.log2(2*self.scale))), 1)
        
        # self.grid_size = 128
        # L = 16; F = 2; log2_T = 19; N_min = 16
        # b = np.exp(np.log(2048*self.scale/N_min)/(L-1))
        # print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        
        # self.G = self.grid_size
        # self.max_hits = 1
        # self.grid_coords = self.grid_coords.to("cuda")
        # self.density_grid = self.density_grid.to("cuda")
        self.model_num = self.config.dataset.grid_X * self.config.dataset.grid_Y
        if self.model_num> 1:
            for i in range(0,self.model_num):
                prefix = os.path.join(self.config.save_dir,'{}'.format(i),'{}'.format(self.config.model.name))
                os.makedirs(prefix,exist_ok=True)
                os.makedirs(prefix+"/ckpts",exist_ok=True)
                os.makedirs(prefix+"/meshes",exist_ok=True)
                os.makedirs(prefix+"/images",exist_ok=True)
                
                # 先实例化不setup不用占多少显存
            pass
        else:
            os.makedirs(os.path.join(self.config.save_dir,'0','{}'.format(self.config.model.name)),exist_ok=True)
            prefix = os.path.join(self.config.save_dir,'0','{}'.format(self.config.model.name))
            os.makedirs(prefix,exist_ok=True)
            os.makedirs(prefix+"/ckpts",exist_ok=True)
            os.makedirs(prefix+"/meshes",exist_ok=True)
            os.makedirs(prefix+"/images",exist_ok=True)
            
        
        self.current_model_num = self.config.model_start_num # 训练到的第几个模型
        self.current_model_num_tmp = self.config.model_start_num # 训练到的第几个模型
        if self.config.model.ray_sample.use_dynamic_sample:
            self.train_num_samples = self.config.model.ray_sample.train_num_rays * \
                (self.config.model.point_sample.num_samples_per_ray+self.config.model.point_sample.num_samples_per_ray_bg)
        
    def setup(self,stage):
        self.discard_step = 0

        self.train_dataset = DATASETS[self.config.dataset.name](self.config.dataset,split='train',downsample=self.config.dataset.downsample)
        self.train_dataset.batch_size = self.config.dataset.ray_sample.batch_size
        
        self.test_dataset = DATASETS[self.config.dataset.name](self.config.dataset,split='test',downsample=self.config.dataset.test_downsample)
        
        self.model = MODELS[self.config.model.name](self.config.model)
        self.net_opt = parse_optimizer(self.config.system.optimizer,self.model)
        
        self.loss = NeRFLoss(config=self.config.system.loss,lambda_distortion=0)
        self.model.setup(self.train_dataset.centers[self.current_model_num,:],
                             self.train_dataset.scales[self.current_model_num,:])
        if self.config.is_continue:
            _,ckpt_path = load_ckpt_path(os.path.join(self.config.ckpt_dir,
                                        '{}'.format(self.current_model_num),
                                        self.config.model.name,'ckpts')
                                       ) 
            self.load_checkpoint(ckpt_path)

        
    def configure_optimizers(self):
        
        self.train_dataset.device = self.device
        self.model.to(self.device)
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('test_poses', self.test_dataset.poses.to(self.device))
        self.register_buffer('test_directions', self.test_dataset.directions.to(self.device))
        # opts=[]
        # for n, p in self.model.named_parameters():
        #     if n not in ['dR', 'dT']: net_params += [p]
        
        # self.net_opt=torch.optim.Adam(net_params, self.config.system.optimizer.args.lr,eps=self.config.system.optimizer.args.eps)
        # self.net_opt = FusedAdam(net_params, lr=self.config.system.optimizer.lr, eps=self.config.system.optimizer.eps)
        
        
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.net_opt,gamma=self.config.system.scheduler.args.gamma)
        
        # return [self.net_opt],[lr_scheduler]
        # optim = parse_optimizer(self.config.system.optimizer, self.model)
        # ret = {
        #     'optimizer': optim,
        # }
        # if 'scheduler' in self.config.system:
        #     ret.update({
        #         'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
        #     })
        # return ret
    
        lr_scheduler = torch.optim.lr_scheduler.StepLR(\
            self.net_opt,
            step_size=self.config.system.scheduler.args.step_size,
            gamma=self.config.system.scheduler.args.gamma
            )
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
        
    def forward(self, batch):
        raise NotImplementedError
    def preprocess_data(self,batch,stage):
        # if 'index' in batch: # validation / testing
        #     index = batch['index']
        # else:
        #     if self.config.model.batch_image_sampling:
        #         index = torch.randint(0, len(self.train_dataset.all_images), size=(self.train_num_rays,), device=self.train_dataset.all_images.device)
        #     else:
        #         index = torch.randint(0, len(self.train_dataset.all_images), size=(1,), device=self.train_dataset.all_images.device)
        # if stage in ['train']:
        #     c2w = self.poses[index]
        #     x = torch.randint(
        #         0, self.train_dataset.img_wh[0], size=(self.train_num_rays,), device=self.train_dataset.all_images.device
        #     )
        #     y = torch.randint(
        #         0, self.train_dataset.img_wh[1], size=(self.train_num_rays,), device=self.train_dataset.all_images.device
        #     )
        #     directions = self.train_dataset.directions[y, x]
        #     rays_o, rays_d = get_rays(directions, c2w)
        #     rgb = self.train_dataset.all_images[index, y, x].view(-1, self.train_dataset.all_images.shape[-1])
        #     fg_mask = self.train_dataset.all_fg_masks[index, y, x].view(-1)
        # else:
        #     # c2w = self.dataset.all_c2w[index][0]
        #     c2w = self.test_poses[index]
        #     directions = self.test_dataset.directions
        #     rays_o, rays_d = get_rays(directions, c2w)
        #     rgb = self.test_dataset.all_images[index].view(-1, self.test_dataset.all_images.shape[-1])
        #     fg_mask = self.test_dataset.all_fg_masks[index].view(-1)
        
        # rays = torch.cat([rays_o, rays_d], dim=-1)
        
        # batch.update({
        #     'rays': rays,
        #     'rgb': rgb,
        #     'fg_mask': fg_mask
        # })
        
        
        
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
    
    # def test_epoch_end(self, out):
    #     """
    #     Gather metrics from all devices, compute mean.
    #     Purge repeated results using data index.
    #     """
    #     raise NotImplementedError
    
    def save_checkpoint(self,ckpt_name):
        key_to_save=self.model.state_dict().keys()
        try:
            del key_to_save['density_bitfield']
        except:
            pass
        model = {key: self.model.state_dict()[key] for key in key_to_save}
        checkpoint = {
            'model': model,
            'optimizer': self.net_opt.state_dict(), #
            'epoch_step': self.global_step + self.discard_step,
            'current_model_num': self.current_model_num,
        }
        torch.save(checkpoint,ckpt_name)
    def load_checkpoint(self,ckpt_path=None):
        if ckpt_path == None: return
        system_dict = torch.load(ckpt_path,map_location='cpu')
        
        self.current_modle_num = system_dict['current_model_num']
        # self.global_step = system_dict['epoch_step']
        # del system_dict['model']['density_bitfield']
        self.model.load_state_dict(system_dict['model'],strict=False)
        # self.net_opt.load_state_dict(system_dict['optimizer'])
        self.discard_step = system_dict['epoch_step']
        
        # pass
    def validation_step(self, batch, batch_idx):#在traing_epoch_end之后
        """
            batch:{
                rays : [w*h 3]
                pose_idxs : [1]
                fg_mask : [w*h 1]
            }
        """
        
        # if self.global_step % self.config.save_ckpt_freq == 0:
        prefix = self.config.save_dir + f"/{self.current_model_num}/{self.config.model.name}"
        
        out = self(batch,split='val')# rays:[W*H,3]
        # psnr = self.criterions['psnr'](out['comp_rgb'], batch['rgb'])
        W, H = self.test_dataset.img_wh
        
        rgbs_true = batch["rays"].reshape(H,W,3)
        rgbs_val = out["rgb"].view(H, W, 3)
        
        psnr_ = psnr(rgbs_true.to("cpu").numpy(),rgbs_val.to("cpu").numpy())
        w2c = torch.linalg.inv(torch.cat([self.test_poses[batch['pose_idx']].cpu(),torch.tensor([[0.,0.,0.,1.]])],dim=0))[:3,...]
        rgbs_true = draw_aabb_mask(rgbs_true,w2c,self.test_dataset.K,self.model.scene_aabb[None,...])
        depth = out['depth'].view(H, W)
        # opacity = out['opacity'].view(H, W)
        img_name = os.path.join(prefix,"images",\
                                f"model_{self.current_model_num}_"+
                                f"it{self.global_step+self.discard_step}_"+
                                f"{batch['pose_idx']}_"+
                                f"psnr:{psnr_}.png"
                                )
        self.save_image_grid(img_name, [
            {'type': 'rgb', 'img': rgbs_true, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': rgbs_val, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': depth, 'kwargs': {}},
            
            # {'type': 'grayscale', 'img': opacity, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
            
        ])
        if (self.global_step+1) % self.config.val_mesh_freq == 0:
            os.makedirs(os.path.join(self.config.save_dir,'mesh'),exist_ok=True)
            mesh_name = os.path.join(prefix,"meshes",\
                                f"model_{self.current_model_num}_"+
                                f"it{self.global_step+self.discard_step}.ply"
                                )
            self.validate_mesh(mesh_name)
            
        if (self.global_step+1) % self.config.val_ckpt_freq == 0:
            ckpt_name = os.path.join(prefix,"ckpts",
                                f'{self.config.model.name}_'+
                                f'{self.config.case_name}_'+
                                'ckpt_{:0>6d}.pt'.format(self.global_step+self.discard_step)
                                )
            self.save_checkpoint(ckpt_name)
        
        
        
        return {
            # 'psnr': psnr,
            'index': batch['pose_idx']
        }
    def train_dataloader(self):
        if self.config.use_DDP:
            return DataLoader(self.train_dataset,
                              num_workers=1,
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
        if self.config.use_DDP:
            return DataLoader(self.test_dataset,
                              num_workers=1,
                              persistent_workers=False,
                              batch_size=None,
                              pin_memory=False)
        else:
            return DataLoader(self.test_dataset,
                              num_workers=0,
                              persistent_workers=False,
                              batch_size=None,
                              pin_memory=False)
            
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
            setattr(self,"model{}".format(self.current_model_num),MODELS[self.config.model.name](self.config.model))
            self.model = getattr(self,"model{}".format(self.current_model_num),MODELS[self.config.model.name](self.config.model))
            self.model.setup(self.train_dataset.centers[self.current_model_num,:],
                             self.train_dataset.scale[self.current_model_num,:])

            self.configure_optimizers()
    
    def validate_mesh(self,mesh_name):
        aabb = self.model.scene_aabb
        vertices, triangles =\
            extract_geometry(aabb, 
                             resolution=self.config.model.geometry_network.isosurface.resolution, 
                             threshold=self.config.model.geometry_network.isosurface.threshold,
                             query_func=lambda pts: self.model.geometry_network(pts,with_fea=False,with_grad=False)['sigma'])
        
        mesh = trimesh.Trimesh(vertices,triangles)
        mesh.export(mesh_name)
            
            
            