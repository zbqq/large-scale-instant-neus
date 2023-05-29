
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from torch import Tensor
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,ray_aabb_intersect

from einops import rearrange
# from model.custom_functions import RayAABBIntersector,VolumeRenderer,RayMarcher
MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01