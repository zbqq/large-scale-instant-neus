   
import torch
import cv2
from utils.config_util import load_config
import numpy as np
import argparse
from datasets.colmap import ColmapDataset
from systems.base import ImageProcess
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path',default='./config/neus-colmap.yaml')
    parser.add_argument('--gpu',type=str,default='0')
    
    args, extras = parser.parse_known_args()
    
    config = load_config(args.conf_path)
    
    dataset = ColmapDataset(config.dataset,split='divide',downsample=config.dataset.downsample)
    assert dataset.camera_model == "OPENCV"
    
    input_path = os.path.join(config.root_dir,'images')
    output_path = os.path.join(config.root_dir,'images_undistorted')
    for root, dirs, files in os.walk(input_path):
        if len(dirs) != 0:
            for dir in dirs:
                for file in files:
                    img_path = os.path.join(root,dir,file)
                    distorted = cv2.imread(img_path)
                    undistorted = cv2.undistort(distorted,
                                  dataset.K.numpy(),
                                  dataset.camera_params
                                  )
                    undistorted=cv2.resize(undistorted,(dataset.img_wh[0],dataset.img_wh[1]))
                    file_name = os.path.join(output_path,dir,file)
                    cv2.imwrite(file_name,undistorted)
        else:
            for file in files:
                img_path = os.path.join(root,file)
                distorted = cv2.imread(img_path)
                undistorted = cv2.undistort(distorted,
                              dataset.K.numpy(),
                              dataset.camera_params
                              )
                file_name = os.path.join(output_path,file)
                cv2.imwrite(file_name,undistorted)
    