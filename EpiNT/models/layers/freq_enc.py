import torch
import torch.nn as nn
from .ffn import PositionwiseFeedForward
from timm.models.layers import DropPath
    
class FreqFilter(nn.Module):
    def __init__(self, L):
        super(FreqFilter, self).__init__()
        self.L = L
        self.scale = 0.02
        self.w = nn.Parameter(self.scale * torch.randn(1, self.L))

    def forward(self, x):
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        w = torch.fft.rfft(self.w, dim=-1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.L, dim=-1, norm="ortho")
        return out

class FreqEnocder(nn.Module):
    def __init__(self, patch_num, L_model, ffn_hidden, drop_prob) -> None:
        super().__init__()
        self.freqfilter = FreqFilter(L_model)
        self.norm1 = nn.LayerNorm([patch_num+1, L_model])
        # self.dropout1 = DropPath(drop_prob)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=L_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm([patch_num+1, L_model])

    def forward(self, x):
        _x = x
        x = self.norm1(x)
        x = self.freqfilter(x)
        x = self.dropout1(x)
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + _x
        return x