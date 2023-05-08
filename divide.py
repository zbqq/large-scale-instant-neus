   
import torch
import os
import sys
from utils.config_util import load_config
import numpy as np
import argparse
from datasets.colmap import ColmapDataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path',default='./config/neus-colmap.yaml')
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--num_epochs',type=int,default=1)
    args, extras = parser.parse_known_args()
    
    config = load_config(args.conf_path)
    
    dataset = ColmapDataset(config.dataset,split='divide')
    grid_dim = torch.tensor([config.dataset.grid_X,
                             config.dataset.grid_Y,
                             1])
    
    dataset.divide(grid_dim)
    
    # with open(config.dataset.root_dir+'/R_correct.txt','r') as f:
    #     lines = f.readlines()
    # matrices = []
    # matrix = []
    # for line in lines:
    #     if not line.strip():
    #         if matrix:
    #             matrices.append(np.array(matrix))
    #             matrix = []
    #     else:
    #         row = [float(x) for x in line.split()]
    #         matrix.append(row)
    
    # print(matrices)
    
    # iterableF()
    # iterableF()
    
    # 在上面的代码中，super().__init__(a)会调用B类的__init__()吗
    
    # class A:
    #     def __init__(self, a):
    #         self.a = a
    #         print('A')

    # class B:
    #     def __init__(self, b):
    #         self.b = b
    #         print('B')

    # class C(A, B):
    #     def __init__(self, a, b, c):
    #         # super().__init__(a)
    #         super(C, self).__init__(a)
    #         self.c = c
    # # 那么super(A,self).__init__会调用哪个父类的__init__呢
    # print(C.mro())
    # c = C("a", "b", "c")
    # print(c.a, c.b, c.c)
    
    
    
    
    
    
    # root_path = '/home/will/data/data_hangpai/sfm/DJI951'
    # cameras, images, points3D=read_model(root_path,'.bin')
    # h = int(cameras[1].height)
    # w = int(cameras[1].width)
    # if cameras[1].model == 'SIMPLE_RADIAL':
    #         fx = fy = cameras[1].params[0]
    #         cx = cameras[1].params[1]
    #         cy = cameras[1].params[2]
    # elif cameras[1].model in ['PINHOLE', 'OPENCV']:
    #         fx = cameras[1].params[0]
    #         fy = cameras[1].params[1]
    #         cx = cameras[1].params[2]
    #         cy = cameras[1].params[3]
    # K = torch.FloatTensor([[fx, 0, cx],
    #                         [0, fy, cy],
    #                         [0,  0,  1]])
    # directions = get_ray_directions(h, w, K)
    
    # RDF_BRU = np.array([[0,1,0],[0,0,-1],[-1,0,0]])#BRU = RDF * P
    # w2c_mats=[]
    # c2w_mats=[]
    # bottom = np.array([0,0,0,1.]).reshape([1,4])
    # R_correct = Ry(torch.pi/4)
    # directions = get_ray_directions(h, w, K)
    # for i in images:
        
    #     R=images[i].qvec2rotmat()#RDF坐标系
    #     t=images[i].tvec.reshape(-1,1)
        
    #     m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    #     # print(poses)
    #     w2c_mats.append(m)
    #     c2w_m = np.linalg.inv(m)#此时c2w是将RDF相机坐标系变为RDF世界坐标系的线性变换，右乘一个单位阵相当于RDF世界坐标系的坐标
        
    #     c2w_m = torch.concat((
    #     torch.tensor(R_correct@np.linalg.inv(RDF_BRU) @ c2w_m[:3, :3]),#将RDF坐标变为BRU坐标
    #     torch.tensor(R_correct@np.linalg.inv(RDF_BRU) @ c2w_m[:3, 3:])
    #     ),
    #     dim=1
    #     ).numpy()
    #     c2w_mats.append(torch.tensor(c2w_m).to(torch.float32))
    # c2w_mats = torch.stack(c2w_mats)
    # c2w_mats=c2w_mats.reshape(len(images),3,-1)
    
    
    # divide(poses=c2w_mats,grid_dim=[3,3])
    
    
    
    
    
    
    
    