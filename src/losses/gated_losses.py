import torch
import torch.nn as nn


class GatedL1Loss(nn.Module):
    def forward(self, gates):
        batch_size = gates.shape[0]
        return gates.sum() / batch_size
