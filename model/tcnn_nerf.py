import tinycudann as tcnn
import torch 
import json
from torch import Tensor
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
from kornia.utils.grid import create_meshgrid3d
from torch.cuda.amp import custom_fwd, custom_bwd
from .custom_functions import TruncExp
import torch
from torch import nn
import tinycudann as tcnn
import studio
from einops import rearrange
# from .custom_functions import TruncExp
import numpy as np

NEAR_DISTANCE = 0.01

class RenderingNet(nn.Module):
    def __init__(self,config
                 ) -> None:
        super(RenderingNet, self).__init__() 
        self.config = OmegaConf.to_container(config)
        self.dir_encoding_config = self.config["dir_encoding_config"]
        self.color_config = self.config["mlp_network_config"]
        
        self.dir_encoder = tcnn.Encoding(#位置编码
            n_input_dims=3,
            encoding_config=self.dir_encoding_config
        )
        if self.config["use_normal"]:
            n_input_dims = self.dir_encoder.n_output_dims + 16 + 3
        else:
            n_input_dims = self.dir_encoder.n_output_dims + 16
            
        self.decoder = tcnn.Network(
            n_input_dims = n_input_dims,
            n_output_dims=3,
            network_config=self.color_config
        )
    def forward(self,d,fea,normal=None):
        """
            d:[N , 3]
            fea:[N, fea_name]
        """
        d = (d + 1.0) / 2.0
        d_embd = self.dir_encoder(d)
        # h = torch.cat([pts, d,gradients, geo_fea], dim=-1)
        if self.config["use_normal"]:
            h = torch.cat([d_embd,fea,normal], dim=-1)
        else:
            h = torch.cat([d_embd,fea], dim=-1)
        h = self.decoder(h)
        rgbs = h
        return rgbs
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
    
class VarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(VarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    @property
    def inv_s(self):
        return torch.exp(self.variance * 10.0)

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
class baseImplicitRep(nn.Module):#都采用grid或plane的方式
    
    def __init__(self, config):
        super().__init__()
        #这里出来的是sdf值
        # scale = 1.5
        self.config = OmegaConf.to_container(config)
    def setup(self,center,scale):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)
                # m.bias.data.zero_()
        self.register_buffer('center', center)
        self.register_buffer('scale', scale)
        self.register_buffer('xyz_min', -torch.ones(3)*self.scale + self.center)
        self.register_buffer('xyz_max', torch.ones(3)*self.scale + self.center)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)
        
        
        
class SDF(baseImplicitRep):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = OmegaConf.to_container(config)

        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config=self.config["xyz_encoding_config"])
        # self.network = \
        #     tcnn.Network(
        #         n_input_dims=self.xyz_encoder.n_output_dims,
        #         n_output_dims=16,
        #         network_config=self.config["mlp_network_config"]
        #     )
        self.activation = nn.Softplus(beta=100,threshold=20)
        self.network = nn.Sequential(
			nn.Linear(self.xyz_encoder.n_output_dims + 3, 64,bias=False),
			# nn.ReLU(True),
			self.activation,
			# nn.Linear(64, 64),
			# nn.ReLU(True),
			# self.activation,
			nn.Linear(64, 16,bias=False)
		)
        pass
    def forward(self, pts:Tensor,with_fea=True,with_grad=False):
        # pts = (pts-self.xyz_min.T)/(self.xyz_max.T-self.xyz_min.T).max()
        pts = (pts-self.xyz_min.T)/(self.xyz_max.T-self.xyz_min.T)
        # with torch.set_grad_enabled(with_grad):#罪魁祸首
        with torch.enable_grad():
            if with_grad:
                pts = pts.requires_grad_(True)
            h = self.xyz_encoder(pts).float()
            h = self.network(torch.cat([pts*(self.xyz_max.T-self.xyz_min.T)+self.xyz_min.T,h],dim=-1))
            sdf = h[:, 0]
            fea = h
            # sdf = sdf * (self.xyz_max.T-self.xyz_min.T).max()
            result={'sigma':sdf}
            if with_fea==True:
                result["fea"]=fea
            if with_grad:
                grad = torch.autograd.grad(
                            sdf, pts, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, 
                            retain_graph=True, 
                            only_inputs=True
                        )[0]
                normals = grad/torch.norm(grad,dim=-1).reshape(-1,1)
                result["normals"]=normals
                result["grad"]=grad
        return result
    
    
class UDF(baseImplicitRep):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = OmegaConf.to_container(config)
        
        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config=self.config["xyz_encoding_config"])
        self.network = nn.Sequential(
			nn.Linear(self.xyz_encoder.n_output_dims, 64),
			nn.ReLU(True),
			# nn.Linear(64, 64),
			# nn.ReLU(True),
			nn.Linear(64, 16)
		) 
    def forward(self, pts:Tensor,with_fea=True,with_grad=False):
        
        pts = (pts-self.xyz_min.T)/(self.xyz_max.T-self.xyz_min.T)
        # with torch.set_grad_enabled(with_grad):#罪魁祸首
        with torch.enable_grad():
            if with_grad:
                pts = pts.requires_grad_(True)
            h = self.xyz_encoder(pts).float()
            h = self.network(h)
            udf = h[:, 0]
            fea = h
            result={'udf':udf}
            if with_fea==True:
                result["fea"]=fea
            if with_grad:
                grad = torch.autograd.grad(
                            udf, pts, grad_outputs=torch.ones_like(udf),
                            create_graph=True, 
                            retain_graph=True, 
                            only_inputs=True
                        )[0]
                result["grad"]=grad
        return result
    
class vanillaMLP(baseImplicitRep):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = OmegaConf.to_container(config)
        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config=self.config["xyz_encoding_config"])
        self.activation = nn.Softplus(beta=100,threshold=20)
        self.network = nn.Sequential(
			nn.Linear(self.xyz_encoder.n_output_dims, 128, bias=True),
			# nn.ReLU(True),
			self.activation,
			# nn.Linear(64, 64),
			# nn.ReLU(True),
			# self.activation,
			nn.Linear(128, 16, bias=True)
		)

    def forward(self, pts:Tensor,with_fea=True,with_grad=False):
        
        pts = (pts-self.xyz_min.T)/(self.xyz_max.T-self.xyz_min.T)
        # with torch.set_grad_enabled(with_grad):#罪魁祸首
        
        with torch.enable_grad():
            if with_grad:
                pts = pts.requires_grad_(True)
            h = self.xyz_encoder(pts).float()
            # h = self.network(torch.cat([pts*(self.xyz_max.T-self.xyz_min.T)+self.xyz_min.T,h],dim=-1))
            
            h = self.network(h)
            sigma = h[:, 0]
            fea = h
            result={'sigma':sigma}
            if with_fea==True:
                result["fea"]=fea
            if with_grad:
                grad = torch.autograd.grad(
                            sigma, pts, grad_outputs=torch.ones_like(sigma),
                            create_graph=True, 
                            retain_graph=True, 
                            only_inputs=True
                        )[0]
                normals = grad/torch.norm(grad,dim=-1).reshape(-1,1)
                result["normals"]=normals
                result["grad"]=grad
        return result
     
    
    
    