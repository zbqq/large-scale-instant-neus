import torch
import os
from omegaconf import OmegaConf
# from datasets
import numpy as np
import torch.multiprocessing as mp
import argparse
import logging
from systems.nerf import NeRFSystem
from systems.neus import NeuSSystem
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from typing import Optional
from numpy import ndarray
from utils.config_util import load_config
from utils.utils import load_ckpt_path
from model.tcnn_nerf import SDF
from datasets.colmap import ColmapDataset

# pytorch-lighning
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path',default='./config/neus-colmap.yaml')
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--num_epochs',type=int,default=1)
    systems = {}
    systems['nerf-system'] = NeRFSystem
    systems['neus-system'] = NeuSSystem
    
    args, extras = parser.parse_known_args()
    config = load_config(args.conf_path,cli_args=extras)
    mp.set_start_method('spawn')
    # torch.cuda.set_device(int(args.gpu))
    # n_gpus = len(args.gpu.split(','))
    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    
    
    
    callbacks=[
        # pl.callbacks.LambdaCallback(on_train_start=update_global_step)
        ]
    logger = TensorBoardLogger(save_dir=config.log_dir,
                               name=config.case_name,
                               default_hp_metric=False)
    # strategy = 'ddp_find_unused_parameters_false'
    strategy = 'ddp'
    # if config.is_continue==True:
    #     step = load_ckpt_path(config.save_dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    system = systems[config.system.name](config) 
    trainer = Trainer(
        max_epochs = args.num_epochs,
        check_val_every_n_epoch = args.num_epochs,
        devices = gpus,
        accelerator = 'gpu',
        callbacks = callbacks,
        logger = logger,
        # strategy = strategy,
        **config.trainer
    )
    # print("rank:",trainer.local_rank)
    trainer.fit(system,)
    
    
    
    