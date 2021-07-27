import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


class Dataset_Base(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, anno_dict):
        self.anno_dict = anno_dict
        self.len = len(self.anno_dict)

    def __getitem__(self, index):
        self.annotation = torch.from_numpy(self.anno_dict)
        return self.annotation[index]

    def __len__(self):
        return self.len

#
# dataset = Dataset_Base()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=32,
#                           shuffle=True,
#                           num_workers=2)

