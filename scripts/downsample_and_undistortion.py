   
import torch
import cv2
from utils.config_util import load_config
import numpy as np
import argparse
from datasets.colmap import ColmapDataset
from systems.base import ImageProcess
import os
import glob
from tqdm import tqdm
from datasets.colmap_utils import read_cameras_binary
def params_from_models(camera_model:np.ndarray,downsample:float=1.0):
    ## Radial-Tangential distortion model
    fx = camera_model[0]*downsample
    fy = camera_model[1]*downsample
    cx = camera_model[2]*downsample
    cy = camera_model[3]*downsample
    K = torch.FloatTensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]])
    # distortion = np.array([
    #     camera_model[4],
    #     camera_model[5],
    #     camera_model[6],
    #     camera_model[7]
    # ])
    return K,distortion



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path',default='./config/neus-colmap.yaml')
    parser.add_argument('--gpu',type=str,default='0')
    require_undis=False
    
    args, extras = parser.parse_known_args()
    
    config = load_config(args.conf_path)
    
    dataset = ColmapDataset(config.dataset,split='divide',downsample=config.dataset.downsample)
    # assert dataset.camera_model == "OPENCV"
    # camera_models = np.loadtxt(os.path.join(config.root_dir,"sparse/0","cameras.txt"))
    camera_models = read_cameras_binary(os.path.join(config.root_dir,"sparse/0","cameras.bin"))
    # K,distortion=params_from_models(camera_models[1].params)
    input_path = os.path.join(config.root_dir,'images')
    prefix = "images_undistorted_{}".format(config.dataset.downsample)
    output_path = os.path.join(config.root_dir,prefix)
    os.makedirs(output_path,exist_ok=True)
    file_patterns = os.path.join(input_path, '**/*')
    file_paths = glob.glob(file_patterns, recursive=True)
    pbar = tqdm(total=len(file_paths))
    for file in file_paths:
        file_name = file.replace("images",prefix)
        # os.makedirs(file_name,exist_ok=True)
        if os.path.isdir(file):
            os.makedirs(file_name,exist_ok=True)
            continue
        distorted = cv2.imread(file)
        if require_undis:
            undistorted = cv2.undistort(distorted,
                            K.numpy(),
                            distortion
                            )
        else:
            undistorted = distorted
        undistorted = cv2.resize(undistorted,(dataset.img_wh[0],dataset.img_wh[1]))
        cv2.imwrite(file_name,undistorted)
        pbar.update(1)
        