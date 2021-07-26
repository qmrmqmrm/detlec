import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


class Dataset_Base(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self, img_dict, anno_dict):
        self.img_dict = img_dict
        self.anno_dict = anno_dict
        self.len = len(img_dict)

    def __getitem__(self, index):
        self.image = torch.from_numpy(self.img_dict)
        self.annotation = torch.from_numpy(self.anno_dict)
        return self.image[index], self.annotation[index]

    def __len__(self):
        return self.len

#
# dataset = Dataset_Base()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=32,
#                           shuffle=True,
#                           num_workers=2)

