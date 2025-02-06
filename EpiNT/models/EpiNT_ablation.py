import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.patch import Patching
from .layers.quantize import Quantize, VanillaQuantize
# from .layers.freq_enc import FreqEnocder
from .layers.pos_encoding import positional_encoding
from ..data.constants import DOWNSTREAM_NUM_CLASS
from .layers.rotary_tran_enc import TransformerEncoder
from .layers.tran_enc import TransformerEncoder as VanillaEncoder
from .layers.revin import RevIN


class Embedding(nn.Module):
    def __init__(self, patch_len, d_model, 
                 pos_emb, learn_pe, sequence_len,
                 cls_token=True,
                 patch_mask=False, mask_ratio=0.3,
                 mask_type="random" # added for ablation study
                 ):
        super().__init__()

        self.cls_token = cls_token
        self.patch_mask = patch_mask
        self.mask_ratio = mask_ratio
        self.d_model = d_model
        self.mask_type = mask_type

        self.proj = nn.Linear(patch_len, d_model)

        # [cls] token
        self.cls_embed = nn.Parameter(torch.randn(d_model), requires_grad=True)
        torch.nn.init.normal_(self.cls_embed, mean=0.0, std=0.1)
        
        # positional encoding
        self.W_pos = positional_encoding(pos_emb, 
                                         learn_pe,
                                         sequence_len // patch_len + 1,
                                         d_model)
        self.pos_flag = True if pos_emb is not None else False
        
        # learnable mask
        if mask_type == "learnable":
            self.mask_encoding = nn.Parameter(torch.randn(d_model), requires_grad=True)
        elif mask_type == "all_zero":
            self.mask_encoding = nn.Parameter(torch.zeros(d_model), requires_grad=False)
        # else:
        #     self.mask_encoding = None


    def random_masking(self, shape, mask_ratio, device):
        N, L = shape  # batch, length
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # create mask
        mask = torch.ones([N, L], dtype=bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :len_keep], 0)

        return mask
    
    def forward(self, x):
        # stft and projection
        batch_size, n_channels, patch_num, patch_len = x.shape

        x = self.proj(x)

        mask = None
        if self.patch_mask:
            mask = self.random_masking(
                shape=(batch_size*n_channels, patch_num),
                mask_ratio=self.mask_ratio,
                device=x.device,
            )
            masked_num = mask.sum() #[batch_size * patch_num * mask_ratio]
            
            # before mask
            x = x.reshape(batch_size*n_channels, patch_num, self.d_model) # (B*C, patch_num, d_model)

            # VQ_MTM's method, use random sample
            if self.mask_type == "random":
                random_sample = torch.normal(mean=0, std=0.02, size=(masked_num, self.d_model)).to(x.device) #[masked_num, d_model]
                x[mask] = random_sample.to(x.dtype) # method from VQ MTM, use random sample to replace the masked data

            # My method, use learnable mask
            else:
            # x[mask] = self.mask_encoding.to(x.dtype) # directly replace, may iccur unmatched precision
            # x = torch.where(mask.unsqueeze(-1), self.mask_encoding.unsqueeze(0).unsqueeze(0), x) # use where
                x = x * (~mask.unsqueeze(-1)) + self.mask_encoding * mask.unsqueeze(-1) # weighted sum method to apply mask
            
            # after mask
            x = x.reshape(batch_size, n_channels, patch_num, self.d_model) # (B, C, patch_num, d_model)
        
        # add [cls] token
        cls = self.cls_embed.expand(x.shape[0], x.shape[1], 1, x.shape[-1]).to(x.device)  # (B, C, 1, d_model)
        x = torch.cat([cls, x], dim=2)

        if self.pos_flag:
            x = x + self.W_pos # (B, C, patch_num+1, d_model)

        return x, mask

class EpiNT_ablation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.task = args.task
        self.patch_len = args.patch_len
        self.d_model = args.d_model
        self.patch_mask = True if self.task == 'pretrain' else False
        self.codebook_size = args.codebook_size

        if self.args.ablation == "Model 2":
            pos_emb = "uniform"
            learn_pe = True
        elif self.args.ablation == "Model 3":
            pos_emb = "sincos"
            learn_pe = False
        else:
            pos_emb = None
            learn_pe = False

        if self.args.ablation == "Model 4":
            mask_type = "learnable"
        elif self.args.ablation == "Model 5":
            mask_type = "all_zero"
        else:
            mask_type = "random"

        self.norm = nn.InstanceNorm1d(1, affine=True)
        self.patch = Patching(args.patch_len, args.stride_len)
        self.embed = Embedding(args.patch_len, args.d_model,
                               sequence_len=args.sequence_len,
                               pos_emb=pos_emb, learn_pe=learn_pe,
                               cls_token=self.args.cls_token,
                               patch_mask=self.patch_mask,
                               mask_ratio=self.args.mask_ratio,
                               mask_type = mask_type)

        if self.args.ablation in ['Model 1', 'Model 4', 'Model 5', 'Model 6', 'Model 7']:
            self.transformer_encoder = nn.Sequential(*[TransformerEncoder(patch_num = args.sequence_len // args.patch_len +1, 
                                                                        d_model=self.d_model,
                                                                        n_head=args.num_heads,
                                                                        ffn_hidden=args.dim_feedforward,
                                                                        drop_prob=args.dropout
                                                        ) for _ in range(args.num_layers)]) if args.num_layers > 0 else nn.Identity()
        elif self.args.ablation in ['Model 2', 'Model 3']:
            self.transformer_encoder = nn.Sequential(*[VanillaEncoder(patch_num = args.sequence_len // args.patch_len +1, 
                                                                      d_model=self.d_model,
                                                                      n_head=args.num_heads,
                                                                      ffn_hidden=args.dim_feedforward,
                                                                      drop_prob=args.dropout
                                                    ) for _ in range(args.num_layers)]) if args.num_layers > 0 else nn.Identity()

        self.get_decoder()
        self.get_loss_fn()
        self.get_head()

    def get_loss_fn(self):
        if self.task == 'pretrain':
            if self.args.ablation == "Model 6":
                self.loss = nn.MSELoss()
            else:
                self.loss_list = nn.ModuleList([nn.CrossEntropyLoss() for _ in range(self.args.num_quantizer)])
    
    def get_head(self):
        if self.task == 'classification':
            self.drop = nn.Dropout(self.args.head_dropout)
            self.head = nn.Linear(self.d_model, DOWNSTREAM_NUM_CLASS[self.args.seizure_task])        
            nn.init.kaiming_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        
    def get_decoder(self):
        if self.task == 'pretrain':
            if self.args.ablation == "Model 6":
                self.final_proj = nn.Linear(self.d_model, self.patch_len)
            elif self.args.ablation == "Model 7":
                self.quantizer_list = nn.ModuleList([
                    VanillaQuantize(
                        input_dim=self.patch_len,
                        vq_dim=self.args.codebook_dim,
                        num_embed=self.codebook_size,
                    ) for _ in range(self.args.num_quantizer)
                ])
                
                self.final_proj_list = nn.ModuleList([
                    nn.Linear(self.d_model, self.codebook_size) for _ in range(self.args.num_quantizer)
                ])

            else:
                self.quantizer_list = nn.ModuleList([
                    Quantize(
                        input_dim=self.patch_len,
                        vq_dim=self.args.codebook_dim,
                        num_embed=self.codebook_size,
                    ) for _ in range(self.args.num_quantizer)
                ])
                
                self.final_proj_list = nn.ModuleList([
                    nn.Linear(self.d_model, self.codebook_size) for _ in range(self.args.num_quantizer)
                ])
            

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1]) # make it to have one channel
        B, C, T  = x.shape

        x = self.norm(x)
        x = self.patch(x) # patching it to have (B, C, T // patch_len, patch_len)
        y, mask = self.embed(x) # embedding
        patch_num = y.shape[2] # T // patch_len (+ 1)

        y = y.reshape(B,  C * patch_num, self.d_model) # (B, C * patch_num, d_model)
        y = self.transformer_encoder(y) # transformer encoder
        y = y.reshape(B, C, patch_num, self.d_model)

        if self.task == 'pretrain':            
            # Get the idx
            masked_num = mask.sum() # (masked_num)

            # Get the prediction
            y = y[:, :, 1:, :] # (B, C, T // patch_len, d_model)
            y = y.permute(0, 2, 1, 3) 
            y = y[mask] # (masked_num, C, d_model)

            if self.args.ablation == "Model 6":
                y = self.final_proj(y)
                x = x.permute(0, 2, 1, 3)
                x = x[mask]
                loss = self.loss(y, x)

                return y, loss
            
            else:
                loss = 0
                coef = 1.0 / self.args.num_quantizer

                for i in range(self.args.num_quantizer):
                    pred = self.final_proj_list[i](y) # (masked_num, C, codebook_size)
                    pred = pred.reshape(masked_num, self.codebook_size)
                    idx = self.quantizer_list[i](x)
                    idx = idx.permute(0, 2, 1)
                    idx = idx[mask]
                    idx = idx.reshape(masked_num)
                    loss += coef * self.loss_list[i](pred, idx)

                return y, loss
        
        elif self.task == 'classification':
            y = y[:, :, 0, :]
            y = y.reshape(B, -1)
            y = self.drop(y)
            y = self.head(y)
            y = F.softmax(y, dim=-1)
            return y