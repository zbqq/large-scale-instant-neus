import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn

class HashPlane(nn.Module):
    def __init__(self):
        super(HashPlane).__init__()
        
        
        
        