import torch
import numpy as np
import os
from tqdm import tqdm
import trimesh
from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from datasets.datasets import BaseDataset
from scripts.load_tool import draw_poses
from .divide_utils import divideTool
# RDF2BRU = np.array([[1,0,0],[0,0,-1],[0,1,0]]) @ np.array([[0,0,1],[0,-1,0],[1,0,0]])
class MegaNeRFDataset(BaseDataset,divideTool):
    def __init__(self, config,split='train', downsample=1.0):
        super().__init__(config, split, downsample)
        super(BaseDataset,self).__init__(config,split)
        
        
    def read_data(self):
        if self.split == "train":
            prefix = os.path.join(self.root_dir,"train")
        else:
            prefix = os.path.join(self.root_dir,"val")
        images_paths = sorted(os.listdir(os.path.join(prefix,"rgbs")))
        meta_paths = sorted(os.listdir(os.path.join(prefix,"metadata")))

        