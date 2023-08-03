import tinycudann as tcnn
import torch 
import json
from torch import Tensor
import sys
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
import numpy as np
import trimesh
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
from kornia.utils.grid import create_meshgrid3d
from torch.cuda.amp import custom_fwd, custom_bwd
from model.custom_functions import progressive_mask
import torch
from torch import nn
import tinycudann as tcnn
import studio
from einops import rearrange
# from .custom_functions import TruncExp
import numpy as np

NEAR_DISTANCE = 0.01

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        # return g * torch.exp(torch.clamp(x, max=15))
        return g * torch.exp(torch.clamp(x, max=10))
trunc_exp = _TruncExp.apply
    
    
    
    
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
        feature_dim = self.config["input_feature_dim"]
        if self.config["use_normal"]:
            n_input_dims = self.dir_encoder.n_output_dims + feature_dim + 3
        else:
            n_input_dims = self.dir_encoder.n_output_dims + feature_dim
            
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
            h = torch.cat([fea,d_embd,normal], dim=-1)
        else:
            h = torch.cat([fea,d_embd], dim=-1)
        h = self.decoder(h).float()
        rgbs = h
        rgbs = torch.sigmoid(h)
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
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val),requires_grad=True))

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
        self.include_xyzs = self.config['include_xyzs']
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
class progressive_encoding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.n_levels = self.config["xyz_encoding_config"]["n_levels"]
        self.n_features_per_level = self.config["xyz_encoding_config"]["n_features_per_level"]
        self.mask_type = self.config["progressive_mask"]["progresive_mask_type"]
        self.register_buffer("mask",torch.zeros([self.n_levels*self.n_features_per_level], dtype=torch.float32))
        if self.config["use_progressive_mask"]:
            self.progressive_mask = progressive_mask
        self.encoding = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config=self.config["xyz_encoding_config"])
    
    def forward(self,x,level):
        feature = self.encoding(x)
        self.mask[:level*self.n_features_per_level] = 1
        if self.config["use_progressive_mask"]:
            feature = self.progressive_mask(feature,self.mask.clone(),level,self.n_features_per_level,self.mask_type)
            
        return feature
        
    
    
class Plane_v7(nn.Module):
    def __init__(self,config,
                 desired_resolution=1024,
                 base_solution=128,
                 n_levels=4,
                 ):
        super(Plane_v7, self).__init__()

        per_level_scale = np.exp2(np.log2(desired_resolution / base_solution) / (int(n_levels) - 1))
        encoding_2d_config = {
            "otype": "Grid",
            "type": "Dense",
            "n_levels": n_levels,
            "n_features_per_level": 2,
            "base_resolution": base_solution,
            "per_level_scale":per_level_scale,
        }
        self.xy = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
        self.yz = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
        self.xz = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
        self.feat_dim = n_levels * 2 *3

    def forward(self, x, bound): # x已经归一化
        # x = (x + bound) / (2 * bound)  # zyq: map to [0, 1]
        xy_feat = self.xy(x[:, [0, 1]])
        yz_feat = self.yz(x[:, [0, 2]])
        xz_feat = self.xz(x[:, [1, 2]])
        return torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)

def get_Plane_encoder(config):
    plane_encoder = Plane_v7(config)
    plane_feat_dim = plane_encoder.feat_dim
    return plane_encoder, plane_feat_dim

        
        
        
class SDF(baseImplicitRep):
    def __init__(self, config):
        super().__init__(config=config)
        # self.config = OmegaConf.to_container(config)

        # self.xyz_encoder = \
        #     tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config=self.config["xyz_encoding_config"])
        self.xyz_encoder = progressive_encoding(self.config)
        network_input_dim = self.xyz_encoder.encoding.n_output_dims + 3 * int(self.include_xyzs)
        # if self.config.use_plane:
        #     self.plane_encoder, self.plane_dim = get_Plane_encoder(self.config.plane)
        #     network_input_dim += self.plane_dim
        self.current_level = 4
        self.sphere_init_radius = float(self.config["sphere_init_radius"])
        self.sphere_init=self.config["mlp_network_config"]['sphere_init']
        self.weight_norm=self.config["mlp_network_config"]['weight_norm']
        self.inside_out=False # 室内 or 室外
        
        
        n_neurons = self.config["mlp_network_config"]['n_neurons']
        self.activation = nn.Softplus(beta=100,threshold=20)
        # self.activation = nn.ReLU(inplace=True)
        self.network = nn.Sequential(
			self.make_linear(network_input_dim, n_neurons, is_first=True, is_last=False),
			# nn.Linear(network_input_dim, n_neurons,bias=False),
			self.activation,
			
			# self.activation,
			self.make_linear(n_neurons, self.config["feature_dim"], is_first=False, is_last=True)
			# nn.Linear(n_neurons, self.config["feature_dim"],bias=False)
        )
        pass
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)
        # layer = nn.Linear(dim_in, dim_out, bias=False)
        if self.sphere_init:
            if is_last:
                if self.inside_out:
                    torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                    torch.nn.init.normal_(layer.weight, mean=-math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
                else:
                    torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                    torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer  
    def forward(self, pts:Tensor,with_fea=True,with_grad=False):
        
        
        pts = (pts-self.xyz_min.T)/(self.xyz_max.T-self.xyz_min.T)
        with torch.enable_grad():
            if with_grad:
                pts = pts.requires_grad_(True)
            h = self.xyz_encoder(pts,self.current_level).float()
            # if self.config.use_plane:
            #     plane_fea = self.plane_encoder(pts)
            # h = self.network(torch.cat([pts*(self.xyz_max.T-self.xyz_min.T)+self.xyz_min.T,h],dim=-1))
            #     h = torch.cat([h,plane_fea],dim=-1)
            network_input = torch.cat([pts*(self.xyz_max.T-self.xyz_min.T)+self.xyz_min.T,h],dim=-1)
            h = self.network(network_input)
            # h = self.network(torch.cat([pts,h],dim=-1))
            # h = self.network(h)
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
                # with torch.no_grad():
                normals = F.normalize(grad, p=2, dim=-1)
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
        self.xyz_encoder = progressive_encoding(self.config)
        # self.xyz_encoder = \
        #     tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config=self.config["xyz_encoding_config"])
        network_input_dim=self.xyz_encoder.encoding.n_output_dims
        # if self.config.use_plane:
        #     self.plane_encoder, self.plane_dim = get_Plane_encoder(self.config.plane)
        #     network_input_dim += self.plane_dim
        self.current_level = 4
        # self.activation = nn.Softplus(beta=100,threshold=20)
        bias=False
        n_neurons = self.config["mlp_network_config"]["n_neurons"]
        self.network = nn.Sequential(
			nn.Linear(network_input_dim, n_neurons, bias=bias),
			nn.ReLU(True),
			# self.activation,
			# nn.Linear(64, 64),
			# nn.ReLU(True),
			# self.activation,
			nn.Linear(n_neurons, self.config["feature_dim"], bias=bias),
   
			# self.activation
		)
        # self.f=tcnn.NetworkWithInputEncoding(
        #     n_input_dims=3,
        #     n_output_dims=16,
        #     encoding_config=self.config["xyz_encoding_config"],
        #     network_config=self.config["mlp_network_config"]
        # )
        
    def forward(self, pts:Tensor,with_fea=True,with_grad=False):
        
        pts = (pts-self.xyz_min.T)/(self.xyz_max.T-self.xyz_min.T)
        # with torch.set_grad_enabled(with_grad):#罪魁祸首
        
        with torch.enable_grad():
            if with_grad:
                pts = pts.requires_grad_(True)
                
            h = self.xyz_encoder(pts,self.current_level).float()
            # # # if self.config.use_plane:
            # # #     plane_fea = self.plane_encoder(pts)
            # # # h = self.network(torch.cat([pts*(self.xyz_max.T-self.xyz_min.T)+self.xyz_min.T,h],dim=-1))
            # #     # h = torch.cat([h,plane_fea],dim=-1)
            h = self.network(h)
            
            # h = self.f(pts).float()
            sigma = h[:, 0]
            sigma = trunc_exp(sigma + float(self.config["density_bias"]))
            
            result={'sigma':sigma}
            if with_fea==True:
                result["fea"]=h
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
     
   