import pytorch_lightning as pl
import torch
import os
from utils.config_util import load_config
from model.nerf import vanillaNeRF
from model.neus import NeuS
from systems.main import mainSystem
import argparse

from pytorch_lightning import Trainer

from torch.utils.data import DataLoader
from datasets.colmap import ColmapDataset
DATASETS={
    'colmap':ColmapDataset
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path',default='./config/neus-colmap.yaml')
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--num_epochs',type=int,default=1)
    system = {}
    model_classes = {'nerf':vanillaNeRF,'neus':NeuS}
    
    args, extras = parser.parse_known_args()
    
    config = load_config(args.conf_path,cli_args=extras)
    system = mainSystem(config=config)
    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    test_dataset = DATASETS[config.dataset.name](config.dataset,split='merge_test',downsample=config.dataset.test_downsample)
    # test_dataset = DATASETS[config.dataset.name](config.dataset,split='merge_test',downsample=config.dataset.test_downsample)
    data_loader = DataLoader(test_dataset,
                              num_workers=0,
                              persistent_workers=False,
                              batch_size=None,
                              pin_memory=False)

    trainer = Trainer(
        accelerator = 'gpu',
        devices = gpus,
    )
    trainer.test(system,data_loader)

    
        
    
    