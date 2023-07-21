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
# from datasets.colmap_utils import read_cameras_text,read_images_text,read_points3D_text,read_cameras_binary,read_images_binary,read_points3d_binary
camera_scale=0
def normalize(x):
    return x / torch.linalg.norm(x)
# def read_model(path, ext):
#     if ext == ".txt":
#         cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
#         images = read_images_text(os.path.join(path, "images" + ext))
#         points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
#     else:
#         cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
#         images = read_images_binary(os.path.join(path, "images" + ext))
#         points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
#     return cameras, images, points3D


def draw_poses(poses_:Union[Tensor,ndarray]=None,
               rays_o_=None,
               rays_d_=None,
               pts3d:ndarray = None,
               aabb_=None,
               aabb_idx=None,
               img_wh=None)->None:
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
        # for i in range(0,poses.shape[0],int(poses.shape[0]/200)):
            center_camera=poses[i,:3,3:4]
            xyz_camera=center_camera+poses[i,:3,:3]*0.5
            ax.scatter(center_camera[0,0],center_camera[1,0],center_camera[2,0],cmap="Reds")
            ax.plot([center_camera[0,0],xyz_camera[0,0]],[center_camera[1,0],xyz_camera[1,0]],[center_camera[2,0],xyz_camera[2,0]],color='r')
            ax.plot([center_camera[0,0],xyz_camera[0,1]],[center_camera[1,0],xyz_camera[1,1]],[center_camera[2,0],xyz_camera[2,1]],color='g')
            ax.plot([center_camera[0,0],xyz_camera[0,2]],[center_camera[1,0],xyz_camera[1,2]],[center_camera[2,0],xyz_camera[2,2]],color='b')

            ax.plot([xyz_camera[0,1],xyz_camera[0,0]],[xyz_camera[1,1],xyz_camera[1,0]],[xyz_camera[2,1],xyz_camera[2,0]],color='m')
            ax.plot([xyz_camera[0,2],xyz_camera[0,1]],[xyz_camera[1,2],xyz_camera[1,1]],[xyz_camera[2,2],xyz_camera[2,1]],color='m')
            ax.plot([xyz_camera[0,0],xyz_camera[0,2]],[xyz_camera[1,0],xyz_camera[1,2]],[xyz_camera[2,0],xyz_camera[2,2]],color='m')

            ax.scatter([center_camera[0,0]],[center_camera[1,0]],[center_camera[2,0]],color='m')
    if pts3d is not None:
        for i in range(0,pts3d.shape[0],int(pts3d.shape[0]/300)):
            ax.scatter([pts3d[i,0]],[pts3d[i,1]],[pts3d[i,2]],color='b')
    if rays_o_ is not None:
        # rays_o = rays_o_.to("cpu")
        # rays_d = rays_d_.to("cpu")
        ax.scatter([rays_o[0,0]],[rays_o[0,1]],[rays_o[0,2]],color='r')
        rays = rays_o + rays_d * 5
        # for i in range(0,rays_d.shape[0],int(rays_d.shape[0]/500)):
        # for i in range(0,rays_d.shape[0]):
            # ax.plot([rays_o[0,0],rays[i,0]],[rays_o[0,1],rays[i,1]],[rays_o[0,2],rays[i,2]],color='b')
        
        
        ax.plot([rays_o[0,0],rays[0,0]],[rays_o[0,1],rays[0,1]],[rays_o[0,2],rays[0,2]],color='b')
        ax.plot([rays_o[0,0],rays[img_wh[0]-1,0]],[rays_o[0,1],rays[img_wh[0]-1,1]],[rays_o[0,2],rays[img_wh[0]-1,2]],color='b')
        ax.plot([rays_o[0,0],rays[img_wh[0]*(img_wh[1]-1),0]],[rays_o[0,1],rays[img_wh[0]*(img_wh[1]-1),1]],[rays_o[0,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        ax.plot([rays_o[0,0],rays[-1,0]],[rays_o[0,1],rays[-1,1]],[rays_o[0,2],rays[-1,2]],color='b')
            
        ax.plot([rays[0,0],rays[img_wh[0]-1,0]],[rays[0,1],rays[img_wh[0]-1,1]],[rays[0,2],rays[img_wh[0]-1,2]],color='b')
        ax.plot([rays[0,0],rays[img_wh[0]*(img_wh[1]-1),0]], [rays[0,1],rays[img_wh[0]*(img_wh[1]-1),1]], [rays[0,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        ax.plot([rays[-1,0],rays[img_wh[0]-1,0]],[rays[-1,1],rays[img_wh[0]-1,1]],[rays[-1,2],rays[img_wh[0]-1,2]],color='b')
        
        ax.plot([rays[-1,0],rays[img_wh[0]*(img_wh[1]-1),0]],[rays[-1,1],rays[img_wh[0]*(img_wh[1]-1),1]],[rays[-1,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        
    if aabb_ is not None:
        for i in range(0,aabb_.shape[0]):
            if i==aabb_idx:
                color = 'b'
                zorder = 1
            else:
                color = 'r'  
                zorder = 0          
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
        
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    
    plt.xticks(np.arange(-5, 5, 1))
    plt.yticks(np.arange(-5, 5, 1))
    plt.autoscale(True)
    plt.show()
# from mega_nerf.ray_utils import get_ray_directions
# from mega_nerf.ray_utils import get_rays


def read_from_mega_nerf(meta_data_dir,mask_dir,img_idx):
    meta_paths = os.listdir(meta_data_dir)# H,W,c2w,intrinsic,distortion
    params_path = os.path.join(mask_dir,'params.pt')
    params = torch.load(params_path)
    
    ## load params.pt
    centroids = params['centroids']
    range = params['ray_altitude_range']
    aabb = torch.concat([params['min_position'],params['max_position']])
    pose_scale_factor = params['pose_scale_factor']
    ray_altitude_range = params['ray_altitude_range']
    # ray_altitude_range=[torch.tensor(0.1),torch.tensor(5.0)]/10
    near = params['near']
    far = params['far']
    
    ## load meta_data.pt
    meta_camera = torch.load(os.path.join(meta_data_dir,meta_paths[img_idx]))
    c2w = meta_camera['c2w']
    img_wh=[meta_camera['W'],meta_camera['H']]
    directions = get_ray_directions(
        meta_camera['W'],
        meta_camera['H'],
        meta_camera['intrinsics'][0],
        meta_camera['intrinsics'][1],
        meta_camera['intrinsics'][2],
        meta_camera['intrinsics'][3],
        False,
        torch.device('cpu')
    )
    
    rays = get_rays(directions,c2w,near=near,far=far,ray_altitude_range=ray_altitude_range)#h,w,8 0.1和2是默认值
    rays_o = rays[...,:3]# [WH,3]
    rays_d = rays[...,3:6]# [WH,3]
    rays_near = rays[...,6]# [WH]
    rays_far = rays[...,7]# [WH]
    
    return rays_o.view(-1,3),rays_d.view(-1,3),centroids,img_wh,aabb
    
    
if __name__ == '__main__':
    meta_data_dir=''
    mask_dir=''
    img_idx=0
    
    rays_o,rays_d,centroids,img_wh,aabb = read_from_mega_nerf(meta_data_dir,mask_dir,img_idx)
    
    
    draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=aabb[None,...])
    pass
