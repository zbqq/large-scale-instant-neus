   
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
def params_from_models(camera_model:np.ndarray,downsample:float=1.0):
    ## Radial-Tangential distortion model
    fx = camera_model[1]*downsample
    fy = camera_model[2]*downsample
    cx = camera_model[3]*downsample
    cy = camera_model[4]*downsample
    K = torch.FloatTensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]])
    distortion = np.array([
        camera_model[5],
        camera_model[6],
        camera_model[7],
        camera_model[8],
        camera_model[9]
    ])
    return K,distortion



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path',default='./config/neus-colmap.yaml')
    parser.add_argument('--gpu',type=str,default='0')
    
    args, extras = parser.parse_known_args()
    
    config = load_config(args.conf_path)
    
    dataset = ColmapDataset(config.dataset,split='divide',downsample=config.dataset.downsample)
    # assert dataset.camera_model == "OPENCV"
    camera_models = np.loadtxt(os.path.join(config.root_dir,"cameras.txt"))
    K,distortion=params_from_models(camera_models[0,:])
    input_path = os.path.join(config.root_dir,'images')
    file_patterns = os.path.join(input_path, '**/*')
    file_paths = glob.glob(file_patterns, recursive=True)
    pbar = tqdm(total=len(file_paths))
    for file in file_paths:
        file_name = file.replace("images","images_undistorted_{}".format(config.dataset.downsample))
        if os.path.isdir(file):
            os.makedirs(file_name,exist_ok=True)
            continue
        distorted = cv2.imread(file)
        undistorted = cv2.undistort(distorted,
                        K.numpy(),
                        distortion
                        )
        undistorted = cv2.resize(undistorted,(dataset.img_wh[0],dataset.img_wh[1]))
        cv2.imwrite(file_name,undistorted)
        pbar.update(1)
        