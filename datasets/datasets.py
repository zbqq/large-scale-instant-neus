#
import torch
import cv2 as cv
import torch.nn.functional as F
import numpy as np
import os
from einops import rearrange
from omegaconf import OmegaConf
# import matplotlib
from glob import glob
import imageio
# from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from torch import Tensor
import argparse
import cv2
from pathlib import Path
from math import cos,sin
from torch.utils.data import Dataset,IterableDataset
import studio
import re
# from divide_utils import divideTool
# from pyhocon import ConfigFactory
# import pyecharts
# from render.render import mainModule
# This function is borrowed from IDR: https://github.com/lioryariv/idr
# 从文件中获取或直接传入P[n_images,3,4]=[R|t]

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    #将投影矩阵分解为内参矩阵、旋转矩阵、位移向量
    K = out[0]
    R = out[1]
    t = out[2]
    K = K / K[2, 2]
    
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose
def get_anti_symetric_matrix(k:Tensor)->Tensor:
    R=torch.zeros([3,3])
    R[0,1],R[0,2]=-k[2],k[1]
    R[1,0],R[1,2]=k[2],-k[0]
    R[2,0],R[2,1]=-k[1],k[0]
    return R
def get_Ry(theta):
    R = torch.eye(3)
    R[0,0] = cos(theta)
    R[2,2] = cos(theta)
    R[0,2] = -sin(theta)
    R[2,0] = sin(theta)
    return R
def get_Rx(theta):
    R = torch.eye(3)
    R[1,1] = cos(theta)
    R[2,2] = cos(theta)
    R[1,2] = -sin(theta)
    R[2,1] = sin(theta)
    return R
def get_Rz(theta):
    R = torch.eye(3)
    R[0,0] = cos(theta)
    R[1,1] = cos(theta)
    R[0,1] = -sin(theta)
    R[1,0] = sin(theta)
    return R
def get_trueIdx(idx_array,batch_size):
    idx_array = idx_array.nonzero().view(-1)
    # print(idx_array.shape[0])
    batch_num = min(idx_array.shape[0],batch_size)
    return idx_array[torch.randperm(idx_array.shape[0])[0:batch_num]]

def revise(poses,theta,pts3d=None,axis='y'):
        if axis == 'x':
            R_revise = get_Rx(theta)
        elif axis == 'y':
            R_revise = get_Ry(theta)
        elif axis == 'z':
            R_revise = get_Rz(theta)
        else:
            R_revise = get_Rx(0)
        poses = R_revise @ poses
        if pts3d is not None:
            pts3d = (R_revise @ pts3d[:,:,None]).reshape(-1,3)
            center = pts3d.mean(dim=0).to(torch.float32)# [3]
            return poses,pts3d,center
        else:
            return poses
        # self.center = torch.zeros([1,3])# [3]
RDF_BRU = np.array([[0,1,0],[0,0,-1],[-1,0,0]],dtype=np.float32)#BRU = RDF * P
# RDF_BRU = np.array([[1,0,0],[0,1,0],[0,0,1]])#BRU = RDF * P
#包含：
# 图片、位姿(K|R|t)、尺度因子、图像参数(H|W)
# 对于大型场景，存储gridX、gridY
# 需要先preprocess生成npz，npz里的R已经为[x,y,z]

class BaseDataset(IterableDataset):
# class BaseDataset(Dataset,divideTool):
    def __init__(self,config,split='train',downsample=1.0):
        self.config = config
        self.split = split
        self.root_dir = config.root_dir
        self.downsample = downsample
        self.ray_sampling_strategy = config.ray_sampling_strategy
        self.batch_size = config.batch_size
        self.model_num = self.config.grid_X * self.config.grid_Y
        # self.transform_matrix = RDF_BRU
        self.current_model_num = self.config.model_start_num #记住更新！
        # if split == 'train' or split == 'test':#已经分割完毕
            # self.load_mask()#初始化迭代列表
    def read_intrinsics(self):#不同数据集的W,H,K不一样
        raise NotImplementedError
    def read_meta(self):
        raise NotImplementedError
    def load_mask(self):
        self.idxs=[]
        self.mask_name=[]
        self.load_path = os.path.join(self.config.mask_dir,"model/{}".format(self.current_model_num))
        for file_name in os.listdir(self.load_path):
            file_path = os.path.join(self.load_path,file_name)
            self.idxs.append(int(re.findall(r'\d+',file_name)[0])) 
            self.mask_name.append(file_path)
        self.idxs.sort()#对应到mask_name中
        self.mask_name.sort()
        if self.split == 'train':
            # self.idxs = [self.idxs[i] for i in range(0,len(self.idxs)) if i%8!=0]
            self.idxs = [i for i in range(0,len(self.idxs)) if i%8!=0]
        elif self.split == 'test':
            # self.idxs = [self.idxs[i] for i in range(0,len(self.idxs)) if i%8==0]
            self.idxs = [i for i in range(0,len(self.idxs)) if i%8==0]
    def revise(self,theta,axis='y'):
        self.poses,self.pts3d,self.center = \
            revise(poses=self.poses,pts3d=self.pts3d,theta=theta,axis=axis)
    def read_img(self,img_path:str,img_wh,blend_a=True):
        img = imageio.imread(img_path).astype(np.float32)/255.0
        if img.shape[2] == 4: # blend A to RGB
            if blend_a:
                img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
            else:
                img = img[..., :3]*img[..., -1:]
        img = cv2.resize(img, img_wh)
        img = rearrange(img, 'h w c -> (h w) c')
        return torch.tensor(img)
    def __len__(self):#一个epoch是一个len
        if self.split.startswith('train'):
            return self.config.batch_num#训练一个grid图片数的上限
        return len(self.idxs)
    def __iter__(self):#iterable datasets本身就是一个迭代器，__iter__返回本身，这个
        self.load_mask()
        if self.config.use_random:
            self.idx_list = np.random.choice(self.idxs,self.config.batch_num)#绝对坐标
        else:
            self.idx_list = []
            for _ in range(0,int(np.ceil(self.config.batch_num/len(self.idxs)))):
                self.idx_list.extend(self.idxs)
        self.idx_tmp = 0
        
        
        if self.split == 'train':
            """
                一个epoch(model)已经训练结束,加载下一个grid对应的pose
            """
            # if self.current_model_num >= self.model_num:
            #     return
            # self.current_model_num += 1
            
            while True: # batch_num由__len__确定
                item = torch.load(self.mask_name[self.idx_list[self.idx_tmp]])
                idx_array = torch.zeros([self.img_wh[0]*self.img_wh[1]],dtype=torch.int32).to(self.device)
                bits_array = item['bits_array'].to(self.device)
                studio.un_packbits_u32(idx_array,bits_array)
                idx_array = idx_array.to(torch.bool).to("cpu")
                true_idx = get_trueIdx(idx_array,self.batch_size)
                if true_idx.shape[0] < 4000:
                    self.idx_tmp += 1
                    del item,idx_array,bits_array
                    continue
                dirs=self.directions[true_idx]
                pose_idx = item['pose_idx']
                img = self.read_img(self.img_paths[pose_idx],self.img_wh,blend_a=False)# w*h 3
                rays = img[true_idx.to("cpu")]
                del idx_array,true_idx
                #加载一副图片的interset_pix_idxs            
                yield {
                    "rays":rays,
                    "directions":dirs,
                    "pose_idx": pose_idx,
                }
                self.idx_tmp += 1
                self.idx_tmp %= len(self.idx_list)
        else:
            
            self.pose_idx = self.idx_list[self.idx_tmp]
            while True:
                Idx = torch.load(self.mask_name[self.pose_idx])
                pose_idx = Idx['pose_idx']
                # pose = self.poses[pose_idx]
                img = self.read_img(self.img_paths[pose_idx],self.img_wh,blend_a=False)# w*h 3
                yield {
                    "rays": img,
                    "pose_idx": pose_idx
                }
                self.idx_tmp += 1
                self.idx_tmp %= len(self.idx_list)

    
    
class LoadPoseNeeded(IterableDataset):
    def __init__(self,config):
        super().__init__()
        self.config = config
        
    def len():
        return 1000
    def __iter__(self):
        for name in self.load_name:
            Idx = torch.load(os.path.join(self.config.mask_dir),name)
            #加载一副图片的interset_pix_idxs            
            yield Idx      
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--export', type=str, default='./exp')
    parser.add_argument('--case', type=str, default='test')
    
    args = parser.parse_args()#字典
    
    # torch.cuda.set_device(args.gpu)
    f = open(args.conf)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', args.case)
    f.close()
    conf = ConfigFactory.parse_string(conf_text)
    conf['dataset.data_dir'] = conf['dataset.data_dir'].replace('CASE_NAME', args.case)
    
    ddd = Dataset(conf['dataset'])
    rays_o,rays_d = ddd.gen_new_rays(1,torch.tensor([1,1,1]))
    
    # output=ddd.gen_sample(0,'train',100)
    # cv.imwrite("/home/will/data/hhh.jpg",output["color_gt"].detach().cpu().numpy())
    print('data end')
    
        
        
        
        

