import torch
from torchvision import datasets, transforms, utils
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
class sampleDataset(Dataset):
    def __init__(self, csv_file, root_dir, sample_size=16):
        self.csv = csv_file
        self.root_dir = root_dir
        self.sample_size = sample_size
    def __len__(self):
        return self.sample_size
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.csv.iloc[idx, 'name'])
