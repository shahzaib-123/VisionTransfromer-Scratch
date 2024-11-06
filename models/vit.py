import torch
import torch.nn as nn
import numpy as np
from utilities.UTILS import patchify, get_positional_embeddings
from models.msa import MSA
from models.vit_block import ViTBlock

class ViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        super(ViT,self).__init__()
        
        self.chw = chw 
        self.n_patches = n_patches
        
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        self.hidden_d=hidden_d
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        
        #linear mapping
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        #classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False
        
        #Transformer Encoder Block
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        #MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
        
    def forward(self,images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(images.device)
        
        tokens=self.linear_mapper(patches)
        
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        pos_embed = self.pos_embed.to(images.device).repeat(n, 1, 1)
        out = tokens + pos_embed
        
        for block in self.blocks:
            out = block(out)
            
        out = out[:, 0]
        
        return self.mlp(out)
        
if __name__=='__main__':
    pass