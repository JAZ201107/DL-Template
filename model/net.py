# Define model

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()

        raise NotImplementedError

    def forward(
        self,
    ):
        raise NotImplementedError
