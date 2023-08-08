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



if __name__ == '__main__':
    
    # image = cv2.imread('/home/will/1.png')
    # # mask = np.zeros_like(image)
    # mask = image.copy()
    # # pts = np.array(torch.abs(torch.rand([4,2])*torch.tensor([4,1])*500),dtype=np.int32)
    # pts = np.array((torch.rand([4,2])-0.5)*torch.tensor([4,1])*500,dtype=np.int32)
    # # pts = torch.abs(torch.rand([5,2])*torch.tensor([4,1])*500)
    # # for pt in pts:
    # #     cv2.circle(image,((int(pt[0])),(int(pt[1]))),radius=50,color=(0, 0, 125), thickness=-1)
    # color = (0, 0, 255, 0.5)  # BGR 格式，最后一个值表示透明度 (0-255)

    # mask = cv2.fillPoly(mask, [pts], color)
    # image=cv2.addWeighted(image,1,mask,0.3,0)
    # cv2.imwrite('/home/will/1_mask.png',image)
    
    
    import test
    b=test.make('bb')
    pass
    # cv2.imshow('Projection', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()