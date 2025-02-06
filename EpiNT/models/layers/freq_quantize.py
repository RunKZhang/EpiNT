import torch
import torch.nn as nn

class FreqQuantize(nn.Module):
    def __init__(self, input_dim: int, vq_dim: int, num_embed: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.vq_dim = vq_dim

        # self.projector = torch.nn.init.normal_(torch.empty(input_dim))
        # self.projector = nn.Parameter(self.projector, requires_grad=False)
        self.proj_amp = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim//2+1, vq_dim)), requires_grad=False)
        self.proj_phase = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim//2+1, vq_dim)), requires_grad=False)

        # codebook_amp = torch.nn.init.normal_(torch.empty(num_embed, vq_dim))
        codebook_amp = torch.nn.init.uniform_(torch.empty(num_embed, vq_dim))
        self.codebook_amp = nn.Parameter(codebook_amp, requires_grad=False)
        # codebook_phase = torch.nn.init.normal_(torch.empty(num_embed, vq_dim))
        codebook_phase = torch.nn.init.uniform_(torch.empty(num_embed, vq_dim))
        self.codebook_phase = nn.Parameter(codebook_phase, requires_grad=False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # global filter in frequency 
            # x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
            # proj_fft = torch.fft.rfft(self.projector, dim=-1, norm='ortho')
            # x_feature_fft = x_fft*proj_fft
            # x_feature = torch.fft.irfft(x_feature_fft, n=self.vq_dim, dim=-1, norm='ortho')
            x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
            amp = torch.abs(x_fft)
            phase = torch.angle(x_fft)
            x_feature_amp = amp.matmul(self.proj_amp) # [B, C=1, patch_num, vq_dim]
            x_feature_phase = phase.matmul(self.proj_phase)

            # quantizr 
            # x_feature_norm = x_feature.norm(dim=-1, keepdim=True) 
            # x_feature = x_feature / x_feature_norm # [B, C=1, patch_num, vq_dim]
            x_feature_amp_norm = x_feature_amp.norm(dim=-1, keepdim=True)
            x_feature_phase_norm = x_feature_phase.norm(dim=-1, keepdim=True)
            x_feature_amp = x_feature_amp / x_feature_amp_norm
            x_feature_phase = x_feature_phase / x_feature_phase_norm
            codebook_amp_norm = self.codebook_amp.norm(dim=-1, keepdim=True) 
            codebook_phase_norm = self.codebook_phase.norm(dim=-1, keepdim=True)
            codebook_amp = self.codebook_amp / codebook_amp_norm # [num_embed, vq_dim]
            codebook_phase = self.codebook_phase / codebook_phase_norm
            similarity_amp = torch.einsum('bcnd,md->bcnm', x_feature_amp, codebook_amp) # [B, C=1, patch_num, num_embed]
            similarity_phase = torch.einsum('bcnd,md->bcnm', x_feature_phase, codebook_phase)
            idx_amp = torch.argmax(similarity_amp, dim=-1) 
            idx_phase = torch.argmax(similarity_phase, dim=-1)
            return idx_amp, idx_phase