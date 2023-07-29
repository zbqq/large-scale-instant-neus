import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from typing import Union
# from preprocess.colmap_read_model import read_model
from math import cos,sin
import os
import sys
from datasets.colmap_utils import read_cameras_text,read_images_text,read_points3D_text,read_cameras_binary,read_images_binary,read_points3d_binary
camera_scale=0
def normalize(x):
    return x / torch.linalg.norm(x)
# 需要输入为[num,3,4]的tensor/ndarray poses
def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D

def Rx(theta):
    R = torch.eye(3)
    R[1,1] = cos(theta)
    R[2,2] = cos(theta)
    R[1,2] = -sin(theta)
    R[2,1] = sin(theta)
    return R
def Ry(theta):
    R = torch.eye(3)
    R[0,0] = cos(theta)
    R[2,2] = cos(theta)
    R[0,2] = -sin(theta)
    R[2,0] = sin(theta)
    return R
def draw_poses(poses_:Union[Tensor,ndarray]=None,
               rays_o_=None,
               rays_d_=None,
               pts3d:ndarray = None,
               aabb_=None,
               aabb_idx=None,
               img_wh=None,
               t_min=None,
               t_max=None
               )->None:
    if isinstance(poses_,Tensor):         
        poses=poses_[None,:,:].to("cpu").numpy()
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    if poses_ is not None:
        poses=poses_
        
        try:
            poses = poses.to("cpu")
        except:
            pass
        if len(poses_.shape)<=2:
            poses = poses[None,:,:]
        for i in range(0,poses.shape[0]):
        # for i in range(0,poses.shape[0],int(poses.shape[0]/10)):
            center_camera=poses[i,:3,3:4]
            xyz_camera=center_camera+poses[i,:3,:3]*2
            ax.scatter(center_camera[0,0],center_camera[1,0],center_camera[2,0],cmap="Reds")
            ax.plot([center_camera[0,0],xyz_camera[0,0]],[center_camera[1,0],xyz_camera[1,0]],[center_camera[2,0],xyz_camera[2,0]],color='r')
            ax.plot([center_camera[0,0],xyz_camera[0,1]],[center_camera[1,0],xyz_camera[1,1]],[center_camera[2,0],xyz_camera[2,1]],color='g')
            ax.plot([center_camera[0,0],xyz_camera[0,2]],[center_camera[1,0],xyz_camera[1,2]],[center_camera[2,0],xyz_camera[2,2]],color='b')

            ax.plot([xyz_camera[0,1],xyz_camera[0,0]],[xyz_camera[1,1],xyz_camera[1,0]],[xyz_camera[2,1],xyz_camera[2,0]],color='m')
            ax.plot([xyz_camera[0,2],xyz_camera[0,1]],[xyz_camera[1,2],xyz_camera[1,1]],[xyz_camera[2,2],xyz_camera[2,1]],color='m')
            ax.plot([xyz_camera[0,0],xyz_camera[0,2]],[xyz_camera[1,0],xyz_camera[1,2]],[xyz_camera[2,0],xyz_camera[2,2]],color='m')

            ax.scatter([center_camera[0,0]],[center_camera[1,0]],[center_camera[2,0]],color='m')
    if pts3d is not None:
        try:
            pts3d = pts3d.cpu()
        except:
            pass
        for i in range(0,pts3d.shape[0],int(pts3d.shape[0]/300)):
            ax.scatter([pts3d[i,0]],[pts3d[i,1]],[pts3d[i,2]],color='b')
    if rays_o_ is not None:
        rays_o = rays_o_.to("cpu")
        rays_d = rays_d_.to("cpu")
        
        if t_min is not None:
            t_max_ = t_max.to("cpu")
            t_min_ = t_min.to("cpu")
            rays = rays_o+rays_d*t_max_.view(-1,1)
            # rays = rays_o+rays_d*t_min_.view(-1,1)
        else:
            rays = rays_o+rays_d*3
        ax.scatter([rays_o[0,0]],[rays_o[0,1]],[rays_o[0,2]],color='r')
        
        # for i in range(0,rays_d.shape[0],int(rays_d.shape[0]/200)):
        for i in range(0,rays_d.shape[0]):
            ax.plot([rays_o[0,0],rays[i,0]],[rays_o[0,1],rays[i,1]],[rays_o[0,2],rays[i,2]],color='b')
        
        
        # ax.plot([rays_o[0,0],rays[0,0]],[rays_o[0,1],rays[0,1]],[rays_o[0,2],rays[0,2]],color='b')
        # ax.plot([rays_o[0,0],rays[img_wh[0]-1,0]],[rays_o[0,1],rays[img_wh[0]-1,1]],[rays_o[0,2],rays[img_wh[0]-1,2]],color='b')
        # ax.plot([rays_o[0,0],rays[img_wh[0]*(img_wh[1]-1),0]],[rays_o[0,1],rays[img_wh[0]*(img_wh[1]-1),1]],[rays_o[0,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        # ax.plot([rays_o[0,0],rays[-1,0]],[rays_o[0,1],rays[-1,1]],[rays_o[0,2],rays[-1,2]],color='b')
            
        # ax.plot([rays[0,0],rays[img_wh[0]-1,0]],[rays[0,1],rays[img_wh[0]-1,1]],[rays[0,2],rays[img_wh[0]-1,2]],color='b')
        # ax.plot([rays[0,0],rays[img_wh[0]*(img_wh[1]-1),0]], [rays[0,1],rays[img_wh[0]*(img_wh[1]-1),1]], [rays[0,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        # ax.plot([rays[-1,0],rays[img_wh[0]-1,0]],[rays[-1,1],rays[img_wh[0]-1,1]],[rays[-1,2],rays[img_wh[0]-1,2]],color='b')
        
        # ax.plot([rays[-1,0],rays[img_wh[0]*(img_wh[1]-1),0]],[rays[-1,1],rays[img_wh[0]*(img_wh[1]-1),1]],[rays[-1,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        
    if aabb_ is not None:
        for i in range(0,aabb_.shape[0]):
            color = 'r'  
            zorder = 0   
            if aabb_idx is not None:    
                if i in aabb_idx:
                    color = 'b'
                    zorder = 1        
            aabb = aabb_[i].to("cpu")
            ax.plot([aabb[0],aabb[3]],[aabb[1],aabb[1]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[1],aabb[4]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[1],aabb[1]],[aabb[2],aabb[5]],color=color,zorder=zorder)
            
            ax.plot([aabb[3],aabb[0]],[aabb[1],aabb[1]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[1],aabb[4]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[1],aabb[1]],[aabb[5],aabb[2]],color=color,zorder=zorder)
            
            ax.plot([aabb[0],aabb[3]],[aabb[4],aabb[4]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[4],aabb[1]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[4],aabb[4]],[aabb[5],aabb[2]],color=color,zorder=zorder)
            
            ax.plot([aabb[3],aabb[0]],[aabb[4],aabb[4]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[4],aabb[1]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[4],aabb[4]],[aabb[2],aabb[5]],color=color,zorder=zorder)
        
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([-5, 5])
    
    # plt.xticks(np.arange(-5, 5, 1))
    # plt.yticks(np.arange(-5, 5, 1))
    plt.autoscale(True)
    # plt.show()
    for i in range(0,20):
        ax.view_init(elev=10*i-100, azim=i*4)
        plt.savefig(f'./test{i}.png')
R_1 = torch.tensor([
[1.000000000000, 0.000000000000, 0.000000000000 ,0.000000000000],
[0.000000000000, 0.738635659218, -0.674104869366, 27.879625320435],
[0.000000000000, 0.674104869366, 0.738635659218 ,6.405389785767],
[0.000000000000, 0.000000000000, 0.000000000000 ,1.000000000000]
])
R_2 = torch.tensor([
[1.000000000000, 0.000000000000, 0.000000000000, 56.128658294678],
[0.000000000000, 1.000000000000, 0.000000000000, -20.065317153931],
[0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000],
[0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
])
if __name__ == '__main__':
    # root_path = '/home/will/data/Clutural_2/colmap'
    # root_path = '/home/will/data/public_data/magicSqure/sparse/0'
    root_path = '/home/will/data/data_hangpai/sfm/DJI951'
    # root_path = '/home/will/data/public_data/gerrard-hall/sparse/0'
    # root_path = '/home/will/data/public_data/scan65/sparse/0'
    # root_path = '/home/will/data/data_lab_reduce_frame/evelatorwithdynamic_290/images/sparse/0'
    cameras, images, points3D=read_model(root_path,'.bin')
    w2c_mats=[]
    c2w_mats=[]
    RDF_BRU = np.array([[0,1,0],[0,0,-1],[-1,0,0]])#BRU = RDF
    # ppp = np.array([0,0,1]).reshape([1,3])
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    # R_h = np.concatenate([np.concatenate([RDF_BRU, np.zeros([3,1])], 1), bottom], 0)
    R_correct = R_2 @ R_1
    # R_correct = Ry(0)
    for i in images:
        
        R=images[i].qvec2rotmat()#RDF坐标系
        t=images[i].tvec.reshape(-1,1)
        
        m = torch.tensor(np.concatenate([np.concatenate([R, t], 1), bottom], 0))
        # print(poses)
        w2c_mats.append(m)
        c2w_m = torch.linalg.inv(m).float()#此时c2w是将RDF相机坐标系变为RDF世界坐标系的线性变换，右乘一个单位阵相当于RDF世界坐标系的坐标
        c2w_m = torch.concat((
        # torch.tensor(R_correct@np.linalg.inv(RDF_BRU) @ c2w_m[:3, :3]),#将RDF坐标变为BRU坐标
        # torch.tensor(R_correct@np.linalg.inv(RDF_BRU) @ c2w_m[:3, 3:])
        torch.tensor(R_correct @ c2w_m),#将RDF坐标变为BRU坐标
        # torch.tensor(R_correct @ c2w_m)
        ),
        dim=1
        )
        c2w_mats.append(c2w_m)
    pts3d=[]
    for point in points3D:#RDF下的三维点世界坐标(以第一帧相机坐标为世界坐标)，左乘P^-1得到BRU世界坐标
        pt3d = torch.concat([torch.tensor(points3D[point].xyz),torch.tensor([1])],dim=-1).view(4,1).float()
        # pts3d.append((torch.linalg.inv(R_correct) @ pt3d).reshape(1,4)[:,:3])
        pts3d.append((R_correct @ pt3d).reshape(1,4)[:,:3])
    pts3d = torch.tensor(np.concatenate(pts3d,0))
    c2w_mats = torch.stack(c2w_mats)
    
    # np.linalg.inv(R_h) @ c2w_mats @ R_h
    # ppp = RDF2BRU @ ppp
    # c2w_mats=c2w_mats.reshape(len(images),3,-1)
    # w2c_mats=w2c_mats.reshape(len(images),4,-1)
    # c2w_mats=np.concatenate(c2w_mats,axis=0)
    # np.savetxt("/home/will/data/data_lab_reduce_frame/poses.txt",c2w_mats)
    # draw_poses(pts3d=pts3d)
    import trimesh
    pcd = trimesh.PointCloud(pts3d.numpy())
    pcd.export('./check.ply')
    # draw_poses(c2w_mats)
    # print(len(poses))
    pass