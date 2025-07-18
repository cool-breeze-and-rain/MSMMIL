import random
import torch
import pandas as pd
import os
import h5py
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader


class Mambammil2dData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        # ---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        # ---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        # ---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        # full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        # features = torch.load(full_path)
        coords_path = Path(self.feature_dir.replace('pt_files', 'h5_files')) / f'{slide_id}.h5'
        file = h5py.File(coords_path, 'r')
        features = torch.tensor(file['features'][:])
        coords = file['coords'][:]

        # ----> shuffle
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]
            coords = coords[index]

        return features, label, coords, slide_id
