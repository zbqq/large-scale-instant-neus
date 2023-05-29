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
test={}
def run_time(name):
    def define(cls):
       test[name]=cls
    #    return cls 
    return define

@run_time('asd')
class T:
    def __init__(self):
        super().__init__
        

class b:
    def __init__(self) -> None:
        # self.b = 2
        super().__init__()
        
        
class te:
    def __init__(self):
        # self.a = b()
        pass

class ColmapDatasetBase(Dataset):
    # def setup(self,config,split):
    #     self.config = config
    #     self.slpit = split
    def __init__(self,config,split):
        super().__init__()
        self.config = config
    def __len__(self):
        return 100
    def __getitem__(self,idx):
        return {"value":idx}
# class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
#     def __init__(self, config, split):
#         self.setup(config, split)

#     def __iter__(self):
#         while True:
#             yield {}
        

class testDatasets(pl.LightningDataModule):
    def __init__(self,config,split):
        super().__init__()
        self.config = config
        self.split = split
    def setup(self):
        self.train_dataset = ColmapDatasetBase(self.config,'train')
    
    def training_step(self,batch,idx):
        pass
    
        
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)
    

class BaseDataset(IterableDataset):
# class BaseDataset(Dataset,divideTool):
    def __init__(self):
        self.start = 0
        self.end = 100
        self.data = torch.rand([5,5])
        for i in range(0,10):
            setattr(self,'data{}'.format(i),torch.rand([5,5]))
        self.current_num = 0
    def __len__(self):
        return self.data0.shape[0]
    def __iter__(self):
        data_idx = 0
        
        while True:
            data = getattr(self,'data{}'.format(self.current_num))[data_idx,:]
            data_idx += 1
            yield {
                'data':data,
                'model_idx':self.current_num
            }
    def test(self):
        setattr(self,'model0',ColmapDatasetBase(1,1))
        print(self.model0)
        self.model = getattr(self,'model0')
        del self.model
        print(self.mode0)

class testSystem(pl.LightningModule):
    def __init__(self,config,split):
        super().__init__()
        self.config = config
        self.split = split
        self.nn = torch.nn.Linear(5,3)
        self.act = torch.nn.ReLU(True)
        self.net = torch.nn.Sequential(self.nn,self.act)
        self.parameters_to_train=[]
        self.parameters_to_train+=list(self.net.parameters())
        # self.save_hyperparameters()#init的每一个参数都会被认为是超参
        # self.save_hyperparameters(ignore=["loss_fx", "generator_network"])
    def setup(self,stage):
        self.train_dataset = BaseDataset()
    def configure_optimizers(self):
        opt = []
        
        self.net_opt = torch.optim.Adam(self.parameters_to_train, lr=0.1, eps=1e-15)
        # opt += self.net_opt
        return self.net_opt
    def forward(self,value):
        out = self.net(value)
        
        return out
        
    def training_step(self,batch,idx):
        data = self(batch['data'])
        loss = ((torch.ones_like(data)-data)**2).mean()
        
        return {"loss":loss}
    def training_epoch_end(self,training_step_outputs):
        # 对所有epoch(__len__长度)的traing_step数据处理，但是已经backward了再到这一步
        # all_preds = torch.stack(training_step_outputs)
        pass
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1,num_workers=0)

def run(rank,world_size):
    # 调用任何其他方法之前，需要使用该函数初始化包。这会阻塞，直到所有进程都加入。
    dist.init_process_group("nccl",rank=rank,world_size=world_size)
    
    dataset = BaseDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    DDP()
    
if __name__ == '__main__':
    pass
    # system = testSystem(config =1 ,split=0)
    # trainer = Trainer(max_epochs=10)
    # trainer.fit(system)
    
    # test = BaseDataset()
    # test.test()
    # print(studio.packbits_u32)
    
    # from zipfile import ZipFile
    # with ZipFile('/home/zzy/lib/mega-nerf/mask/rubble-pixsfm_mask/0/000000.pt') as zf:
    #     with zf.open('000000.pt') as f:
    #         a = torch.load(f)
    
    # add_pts = torch.rand([36,2], requires_grad=True)# or initialized with aabb pts
    # w = torch.rand([2,2])
    # add_pts_ = add_pts @ w
    # loss = ((add_pts_-torch.ones_like(add_pts_))**2).mean()
    # # opt = torch.optim.Adam()
    # # add_pts.retain_grad()
    # loss.backward()
    # print(w)
    # print(add_pts.grad)
    
    # dist.init_process_group(backend='nccl', init_method='env://')
    print(os.path.dirname(os.path.abspath(__file__)))
    # dirsMap = torch.rand([10000,3]).to("cuda")
    # locMap = torch.zeros([3]).to("cuda")
    # centroids = torch.zeros([8,3]).to("cuda")
    # mask = torch.zeros([10000,8],dtype = torch.int32).to("cuda")
    # img_w = np.array([500],dtype = np.int32)[0]
    # threshould = 1.2
    # mask2 = studio.distance_mask(dirsMap,locMap,centroids,mask,threshould)
    # print(mask.sum())
    
    
    # pass
    
    
    