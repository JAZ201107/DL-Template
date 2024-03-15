import os

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import train_transform, test_transform


class MyDataset(Dataset):
    def __init__(self, data_dir, transform):
        """
        the data loader init function
        """

    def ___getitem__(self, idx):
        """
        Standard Implement
        """
        raise NotImplementedError

    def __len__(self):
        """
        The length of dataloader
        """
        raise NotImplementedError


def fetch_dataloader(types, data_dir, params):
    dataloaders = {}
    for split in ["train", "test", "val"]:
        if split in types:
            # path to files
            path = os.path.join(data_dir, "")

            if split == "train":
                dl = DataLoader(MyDataset(path, train_transform))
            else:
                df = DataLoader(MyDataset(path, test_transform))
