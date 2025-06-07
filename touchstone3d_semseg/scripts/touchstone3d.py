""" Touchstone3D dataset

Author: Umamaheswaran Raman Kumar & Abdur R. Fayjie, 2023
"""
import os
import random
import math
import glob
import pickle
import numpy as np
import h5py as h5
#import transforms3d
from itertools import combinations
import torch
from torch.utils.data import Dataset
from touchstone3d_semseg.scripts.utils import *


class Touchstone3DDataset(Dataset):
    def __init__(self, map_file, npts=2048, mode='train', data_path=None, 
                    pc_attribs='xyz', pc_augm=False, pc_augm_config=None):

        super(Touchstone3DDataset).__init__()
        with open(map_file, 'rb') as f:
            loaded_data = pickle.load(f)
            print(loaded_data)
            if len(loaded_data) == 2:
                self.CLASS_LABELS, self.LABEL2CLASS = loaded_data
            else:
                # Handle additional data
                self.CLASS_LABELS, self.LABEL2CLASS, *extra_data = loaded_data
                print("Extra data in map.pkl:", extra_data)
        
        self.npts = npts
        self.mode = mode
        self.data_path = data_path
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config
        self.file_names = [os.path.basename(data_file).split('/')[-1] for data_file in glob.glob(self.data_path + '/*.npy')]    

    def __len__(self):
        return len(self.file_names)
       
    def __getitem__(self, index):
        ptcloud, label = sample_pointcloud(
            self.data_path, self.npts, self.pc_attribs, 
            self.pc_augm, self.pc_augm_config, self.file_names[index]
        )

        if self.mode == 'train':
            return ptcloud.astype(np.float32), label.astype(np.int32)
        else:
            xyz_min = np.min(ptcloud, axis=0)
            file_name = self.file_names[index]
            return (
                ptcloud.astype(np.float32), 
                label.astype(np.int32), 
                xyz_min.astype(np.float32), 
                file_name
            )

