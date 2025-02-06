import torch
import torch.nn as nn
import numpy as np
    
class Quantize(nn.Module):
    def __init__(self, input_dim: int, vq_dim: int, num_embed: int, **kwargs):
        super(Quantize, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.vq_dim = vq_dim

        self.projector = torch.nn.init.normal_(torch.empty(input_dim))
        self.projector = nn.Parameter(self.projector, requires_grad=False)

        codebook = torch.nn.init.normal_(torch.empty(num_embed, vq_dim))
        # codebook = torch.nn.init.uniform_(torch.empty(num_embed, vq_dim))
        # codebook = init_codebook_sine_cosine(num_embed, vq_dim, frequency_range=(0.5, 500), phase_range=(0, 2 * torch.pi))
        # codebook = linear_combination_codebook(num_embed, vq_dim, self.projector.dtype)
        self.codebook = nn.Parameter(codebook, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # global filter in frequency 
            x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
            proj_fft = torch.fft.rfft(self.projector, dim=-1, norm='ortho')
            x_feature_fft = x_fft * proj_fft
            x_feature = torch.fft.irfft(x_feature_fft, n=self.vq_dim, dim=-1, norm='ortho')

            # quantizr 
            x_feature_norm = x_feature.norm(dim=-1, keepdim=True) 
            x_feature = x_feature / x_feature_norm # [B, C=1, patch_num, vq_dim]
            codebook_norm = self.codebook.norm(dim=-1, keepdim=True) 
            codebook = self.codebook / codebook_norm # [num_embed, vq_dim]
            similarity = torch.einsum('bcnd,md->bcnm', x_feature, codebook) # [B, C=1, patch_num, num_embed]
            idx = torch.argmax(similarity, dim=-1) 
            return idx
        
class VanillaQuantize(nn.Module):
    def __init__(self, input_dim: int, vq_dim: int, num_embed: int, **kwargs):
        super(VanillaQuantize, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.vq_dim = vq_dim

        self.projector = torch.nn.init.xavier_normal_(torch.empty(input_dim, vq_dim))
        self.projector = nn.Parameter(self.projector, requires_grad=False)

        codebook = torch.nn.init.normal_(torch.empty(num_embed, vq_dim))
        self.codebook = nn.Parameter(codebook, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # linear projection
            x_feature = x.matmul(self.projector)

            # quantizr 
            x_feature_norm = x_feature.norm(dim=-1, keepdim=True) 
            x_feature = x_feature / x_feature_norm # [B, C=1, patch_num, vq_dim]
            codebook_norm = self.codebook.norm(dim=-1, keepdim=True) 
            codebook = self.codebook / codebook_norm # [num_embed, vq_dim]
            similarity = torch.einsum('bcnd,md->bcnm', x_feature, codebook) # [B, C=1, patch_num, num_embed]
            idx = torch.argmax(similarity, dim=-1) 
            return idx
        
