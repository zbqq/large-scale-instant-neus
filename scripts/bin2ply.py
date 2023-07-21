import torch
import trimesh
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import os
import argparse
from utils.config_util import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path',default='./config/neus-colmap.yaml')
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--num_epochs',type=int,default=1)
    args, extras = parser.parse_known_args()
    config = load_config(args.conf_path)
    root_dir = config.dataset.root_dir
    pts3d = read_points3d_binary(os.path.join(root_dir, 'sparse/0/points3D.bin'))
    pts3d = torch.tensor([pts3d[k].xyz[:3] for k in pts3d])
    pcd = trimesh.PointCloud(pts3d)
    pcd.export(os.path.join(root_dir, 'sparse_points.ply'))
    