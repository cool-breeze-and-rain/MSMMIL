import argparse
import os
from pathlib import Path
import numpy as np
import glob
import shutil
import ast
import torch


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer




# --->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test', type=str)
    parser.add_argument('--config', default='config/Camelyon/MKMIL_abltion.yaml', type=str)
    # parser.add_argument('--config', default='config/TCGA-Lung/AFWMambaMIL_TCGA-Lung.yaml', type=str)
    parser.add_argument('--gpus', default=[3])
    parser.add_argument('--fold', default=0)
    parser.add_argument('--heatmap', default=True)
    args = parser.parse_args()
    return args


# ---->main
def main(cfg):
    # ---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    # ---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    # ---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    if type(cfg.General.gpus) is str:
        cfg.General.gpus = ast.literal_eval(cfg.General.gpus)
    # print(cfg.General.gpus, type(cfg.General.gpus))

    # ---->Define Data
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                          'train_num_workers': cfg.Data.train_dataloader.num_workers,
                          'test_batch_size': cfg.Data.test_dataloader.batch_size,
                          'test_num_workers': cfg.Data.test_dataloader.num_workers,
                          'dataset_name': cfg.Data.dataset_name,
                          'dataset_cfg': cfg.Data, }
    dm = DataInterface(**DataInterface_dict)

    # ---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                           'loss': cfg.Loss,
                           'optimizer': cfg.Optimizer,
                           'data': cfg.Data,
                           'log': cfg.log_path,
                           'heatmap': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)
    # ---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        # gpus=cfg.General.gpus,  # need modify
        accelerator='gpu',  # add
        devices=cfg.General.gpus,  # add
        amp_backend='apex',  # add
        amp_level=cfg.General.amp_level,
        precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    # ---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model=model, datamodule=dm)
    else:
        # path = "logs/config/Camelyon/AFWMambaMIL_512"
        # model_paths = list(glob.glob(path + '/*/*.ckpt'))
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            # print(os.path.dirname(path))
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)
            # shutil.move("logs/config/Camelyon/AFWMambaMIL/fold0/result.csv", os.path.dirname(path)+"/result.csv")

if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)

    # ---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.General.heatmap = args.heatmap
    cfg.Model.device = 'cuda' if len(args.gpus) != 0 else 'cpu'


    # ---->main
    main(cfg)
