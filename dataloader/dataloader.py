#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:11:10 2020

@author: michal1
"""
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import os
from PIL import Image


class ImageFromCSVLoader(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                 
        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        x = Image.open(img_name) 
        y = np.array(self.data.iloc[idx, 1].astype(np.float32))
#        y = np.expand_dims(y, axis=0)
        y = torch.from_numpy(y).long()
        
        if self.transform:   
            
            x = self.transform(x)
                       
        return {'image': x, 'label': y}
