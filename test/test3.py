import torch
import os
from torch.utils.data import IterableDataset,DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import Trainer
import torch.distributed as dist
import studio
import numpy as np
import cv2

import test

@test.register('bb')
class b:
    def __init__(self) -> None:
        # self.b = 2
        super().__init__()
        print("bbb")
