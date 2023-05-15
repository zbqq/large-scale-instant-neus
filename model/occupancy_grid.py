import torch
import torch.nn as nn
from kornia.utils.grid import create_meshgrid3d



class occupancy_grid(nn.Module):
    def __init__(self,
                 aabb,
                 resolution,
                 
                 ):
        self.roi_aabb = aabb
        self.resolution = resolution
        
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))
        self.register_buffer('grid_coords',
            create_meshgrid3d(self.G, self.G, self.G, False, dtype=torch.int32).reshape(-1, 3))
        self.register_buffer('density_grid',
            torch.zeros(self.cascades, self.G**3))

