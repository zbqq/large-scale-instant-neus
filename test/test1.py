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
    # 一维
    # import matplotlib.pyplot as plt
    # N = 500
    # scale = 50
    # b_bg = 20
    # b_fg = 15
    # y = np.arange(N)/N * scale
    # x = y.copy()
    # def warp(x,b_bg,b_fg):
    #     for i in range(0,x.shape[0]):
    #         if x[i]>b_fg:
    #             Linf = (b_bg - b_fg/x[i]*(b_bg-b_fg))/np.abs(x[i])
    #             # Linf = (b_bg - b_fg/x[i])/np.abs(x[i])
    #             x[i] *= Linf
    #     return x
    # y = warp(y,b_bg,b_fg)
    # plt.plot(x,y)
    # plt.show()
    
    # 二维
    # import numpy as np
    # import matplotlib.pyplot as plt
    # N = 50
    # scale = [60,120]
    # b_bg = np.array([20,40])
    # # b_fg = [15,22.5]
    # b_fg = b_bg * 0.7
    # x = np.arange(N)/N * scale[0] - scale[0]/2
    # y = np.arange(N)/N * scale[1] - scale[1]/2
    # X,Y=np.meshgrid(x,y)
    # pts = np.concatenate([X[...,None],Y[...,None]],-1).reshape(-1,2)
    # def warp(pts,b_bg,b_fg):
    #     factor = b_fg[0]/b_fg[1]
    #     for i in range(0,pts.shape[0]):
    #         mag = np.abs((pts[i,:]))
    #         Linf_x=1
    #         Linf_y=1
    #         if mag[1] <= mag[0]/factor and mag[0] > b_fg[0]:
    #             Linf_x = (b_bg[0] - b_fg[0]/mag[0]*(b_bg[0]-b_fg[0]))/mag[0]
    #             Linf_y = (b_bg[1] - b_fg[1]/(mag[0]/factor)*(b_bg[1]-b_fg[1]))/(mag[0]/factor)
    #             pts[i,0] *= Linf_x
    #             pts[i,1] *= Linf_y
    #         elif mag[1] >= b_fg[1] and mag[0]/factor < mag[1]:
    #             Linf_x = (b_bg[0] - b_fg[0]/(mag[1]*factor)*(b_bg[0]-b_fg[0]))/(mag[1]*factor)
    #             Linf_y = (b_bg[1] - b_fg[1]/mag[1]*(b_bg[1]-b_fg[1]))/mag[1]
    #             pts[i,0] *= Linf_x
    #             pts[i,1] *= Linf_y
    #     return pts
    # pts_ = warp(pts,b_bg,b_fg)
    # plt.scatter(pts[:,0],pts[:,1])
    # plt.show()

    # 三维
    import numpy as np
    import matplotlib.pyplot as plt
    N = 20
    scale = [4,2,3]
    b_bg = torch.tensor([4,2,3],dtype=torch.float32).cuda()
    fb_ratio = torch.ones([3],dtype=torch.float32).cuda()*0.1
    b_fg = b_bg * fb_ratio
    factor=torch.tensor([b_fg[0]/b_fg[1],b_fg[1]/b_fg[2],b_fg[0]/b_fg[2]]).cuda()
    # b_fg = b_bg * 0.7
    # x = np.arange(N)/N * scale[0] - scale[0]/2
    # y = np.arange(N)/N * scale[1] - scale[1]/2
    # z = np.arange(N)/N * scale[2] - scale[2]/2
    
    x = np.arange(N)/N * scale[0]/2
    y = np.arange(N)/N * scale[1]/2
    z = np.arange(N)/N * scale[2]/2
    X,Y,Z=np.meshgrid(x,y,z)
    pts = torch.tensor(np.concatenate([X[...,None],Y[...,None],Z[...,None]],-1).reshape(-1,3),dtype=torch.float32).cuda()
    # pts = pts + torch.rand_like(pts).cuda()
    # pts=np.array([[32,33,30]])
    # def warp(pts,b_bg,b_fg):
    #     factor=[b_fg[0]/b_fg[1],b_fg[1]/b_fg[2],b_fg[0]/b_fg[2]]
        
    #     for i in range(0,pts.shape[0]):
    #         mag = np.abs((pts[i,:]))
    #         Linf_x,Linf_y,Linf_z = 1,1,1
    #         if mag[0] >= b_fg[0] and \
    #            mag[0]/factor[0] >= mag[1] and\
    #            mag[0]/factor[2] >= mag[2]:
    #             Linf_x = (b_bg[0] - b_fg[0]/mag[0]*(b_bg[0]-b_fg[0])) / mag[0]
    #             Linf_y = (b_bg[1] - b_fg[1]/(mag[0]/factor[0])*(b_bg[1]-b_fg[1])) / (mag[0]/factor[0])
    #             Linf_z = (b_bg[2] - b_fg[2]/(mag[0]/factor[2])*(b_bg[2]-b_fg[2])) / (mag[0]/factor[2])
                
    #         elif mag[1] >= b_fg[1] and \
    #              mag[1] >= mag[0]/factor[0] and \
    #              mag[1]/factor[1] >= mag[2]:
    #             Linf_x = (b_bg[0] - b_fg[0]/(mag[1]*factor[0])*(b_bg[0]-b_fg[0]))/(mag[1]*factor[0])
    #             Linf_y = (b_bg[1] - b_fg[1]/(mag[1])*(b_bg[1]-b_fg[1]))/(mag[1])
    #             Linf_z = (b_bg[2] - b_fg[2]/(mag[1]/factor[1])*(b_bg[2]-b_fg[2]))/(mag[1]/factor[1])
                
    #         elif mag[2] >= b_fg[2] and \
    #              mag[2] >= mag[0]/factor[2] and \
    #              mag[2] >= mag[1]/factor[1]:
    #             Linf_x = (b_bg[0] - b_fg[0]/(mag[2]*factor[2])*(b_bg[0]-b_fg[0]))/(mag[2]*factor[2])
    #             Linf_y = (b_bg[1] - b_fg[1]/(mag[2]*factor[1])*(b_bg[1]-b_fg[1]))/(mag[2]*factor[1])
    #             Linf_z = (b_bg[2] - b_fg[2]/(mag[2])*(b_bg[2]-b_fg[2]))/(mag[2])
                
    #         pts[i,0]*=Linf_x
    #         pts[i,1]*=Linf_y
    #         pts[i,2]*=Linf_z

    #     return pts
    # pts_ = warp(pts,b_bg,b_fg)
    studio.contract_rect(pts,b_bg,fb_ratio,int(pts.shape[0]))
    pts = pts.cpu()
    # from scripts.load_tool import draw_poses
    # draw_poses(pts3d=pts)

    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.scatter(pts[:,0],pts[:,1],pts[:,2])
    # plt.show()
    for i in range(0,20):
        ax.view_init(elev=10*i-100, azim=i*4)
        plt.savefig(f'./test{i}.png')
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
    
    
    