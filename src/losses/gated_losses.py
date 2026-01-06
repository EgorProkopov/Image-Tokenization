import torch
import torch.nn as nn


class GatedL1Loss(nn.Module):
    """
    Auxiliary loss that minimizes the l1 norm of the tokens score gates
    """
    def forward(self, gates):
        batch_size = gates.shape[0]
        return gates.mean() / batch_size
