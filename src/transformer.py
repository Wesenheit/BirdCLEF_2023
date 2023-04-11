import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

def get_positional_encoding(seqlen, hiddendim):
    encoding=torch.zeros(seqlen, hiddendim)
    range=torch.arange(0, seqlen).reshape(seqlen,1)
    temp=torch.pow(10000,-torch.arange(0, hiddendim, 2)/hiddendim)
    encoding[:,0::2]=torch.sin(range*temp)
    encoding[:,1::2]=torch.cos(range*temp)
    return encoding


class Block(nn.Module):
    def __init__(self, num_head, d_model, d_hidden,dropout=0) -> None:
        """
        One block of decoder layer of transformers
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(
            num_heads=num_head, embed_dim=d_model, dropout=dropout)
        self.qkv = nn.Linear(d_model, 3*d_model) # Query, key, value projection
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model), nn.GELU())
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res_stream = self.norm1(x)
        Q, K, V = self.qkv(res_stream).chunk(3, dim=-1)
        res_stream = self.mha(Q, K, V, need_weights=False)
        x = x+res_stream[0]
        res_stream = self.norm2(x)
        res_stream = self.ff(res_stream)
        x = x+res_stream
        return x

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x



class ViTBackbone(nn.Module):
    """
    transformers based module
    """

    def __init__(self,d_model:int,d_hidden:int,num_head:int,num_blocks:int,patch_size:int,num_patches:int,num_chanel:Optional[int]=1) -> None:
        super().__init__()
        network = [(Block(num_head=num_head, d_model=d_model,d_hidden=d_hidden)) for _ in range(num_blocks)]
        self.network = nn.Sequential(*network)
        self.embedding = nn.Parameter(torch.randn(1,1+num_patches,d_model))
        self.token = nn.Parameter(torch.randn(1,1,d_model))
        self.pre = nn.Linear(patch_size**2*num_chanel,d_model)
        self.patch_size = patch_size

    def forward(self, x):
        x = img_to_patch(x,self.patch_size)
        B, T, _ = x.shape
        x = self.pre(x)
        class_token = self.token.repeat(B, 1, 1)
        x = torch.cat((class_token,x),dim=1)
        x += self.embedding[:,:T+1]
        x = x.transpose(0, 1)
        x = self.network(x)
        return x
    
class BirdAttention(nn.Module):
    def __init__(self, num_class:int,d_model:int,backbone:ViTBackbone) -> None:
        super().__init__()
        self.mlp_head=nn.Linear(d_model,num_class)
        self.backbone=backbone

    def forward(self,x):
        x = self.backbone(x)
        x = self.mlp_head(x[0])
        return x
