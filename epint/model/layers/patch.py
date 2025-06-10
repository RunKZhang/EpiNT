import torch.nn as nn
import warnings

class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            warnings.warn(
                "Stride and patch length are not equal. \
                          This may lead to unexpected behavior."
            )

    def forward(self, x):
        x = x.unfold(dimension=-1, 
                     size=self.patch_len, 
                     step=self.stride) # x : [batch_size x n_channels x num_patch x patch_len]
        return x