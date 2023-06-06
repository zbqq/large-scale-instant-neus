
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from torch import Tensor
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect
import numpy as np
from einops import rearrange
import mcubes
from tqdm import tqdm
# from model.custom_functions import RayAABBIntersector,VolumeRenderer,RayMarcher
MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
def create_spheric_poses(radius, mean_h, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
        mean_h: mean camera height
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,2*mean_h],
            [0,0,1,-t]
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0],
            [0,np.cos(phi),-np.sin(phi)],
            [0,np.sin(phi), np.cos(phi)]
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th)],
            [0,1,0],
            [np.sin(th),0, np.cos(th)]
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0],[0,0,1],[0,1,0]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/12, radius)]
    return np.stack(spheric_poses, 0)

def extract_fields(scene_aabb, resolution, query_func):
    N = 64
    X = torch.linspace(scene_aabb[0], scene_aabb[3], resolution).split(N)
    Y = torch.linspace(scene_aabb[1], scene_aabb[4], resolution).split(N)
    Z = torch.linspace(scene_aabb[2], scene_aabb[5], resolution).split(N)
    pbar=tqdm(total=len(X)*len(Y)*len(Z))
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(scene_aabb.device)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                    pbar.update(1)
    return u

def extract_geometry(scene_aabb,resolution=128,threshold = 0,query_func=None):
    assert query_func is not None 
    u = extract_fields(scene_aabb, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = scene_aabb[3:].detach().cpu().numpy()
    b_min_np = scene_aabb[:3].detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles