# Standard Library Imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prov-gigapath')))
from functools import partial

# External Library Imports
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Timm Library Imports
import timm
from timm.models.registry import register_model

# Project-Specific Imports
import LongNetDecoderConfig
from gigapath.slide_encoder import LongNetViT
from gigapath.torchscale.architecture.config import DecoderConfig
from gigapath.torchscale.model.LongNet import LongNetDecoder

# Custom Class to Override the Forward Method
class CustomLongNetViT(LongNetViT):
    """Masked LongNetViT model"""
    def __init__(self, **kwargs):
        # call LongNetViT constructor
        print(f"[INFO] Loading LongNetViT")
        super().__init__(**kwargs)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        decoder_embed_dim = 768
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        print(f"[INFO] Loading LongNetDecoder with args:\n {LongNetDecoderConfig.LongNetDecoder_12_layers_768_dim}")
        decoder_args = LongNetDecoderConfig.LongNetDecoder_12_layers_768_dim
        decoder_config = DecoderConfig(**decoder_args)
        self.decoder = LongNetDecoder(decoder_config)
        print('Number of trainable LongNetDecoder parameters: ', sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
        # --------------------------------------------------------------------------

    def random_masking(self, x, mask_ratio):
        """
        Masking logic from models_mae.py
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, coords):
        """Adapted from forward method of LongNetViT"""
        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords)  # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        x = self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, coords):
        """Adapted from models_mae.py"""

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle

        # apply positional embeddings (re-use the same positional embeddings)
        pos = self.coords_to_pos(coords)  # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # Pass through the decoder
        out, _ = self.decoder(prev_output_tokens=x, token_embeddings=x, features_only=True)
        out = self.norm(out)

        return out
    
    def forward_loss(self, orig_embeddings, pred, mask):
        """
        orig_embeddings: [N, L, D], original tile embeddings
        pred: [N, L, D], predicted embeddings
        mask: [N, L], mask where 0 is keep, 1 is remove
        """

        # Normalize embeddings for cosine similarity
        pred_normalized = F.normalize(pred, dim=-1)  
        orig_normalized = F.normalize(orig_embeddings, dim=-1) 

        # Compute cosine similarity between predictions and original embeddings
        cos_sim = (pred_normalized * orig_normalized).sum(dim=-1)  # Cosine similarity, shape: [N, L]
        loss = 1 - cos_sim 

        # Calculate mean loss over the masked tokens
        loss = (loss * mask).sum() / mask.sum()  

        return loss

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask_ratio=0.75):
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        if x.dim() != 3:
            x = x.unsqueeze(0)
        assert len(x.shape) == 3

        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, coords)
        pred = self.forward_decoder(latent, ids_restore, coords)
        orig_embeddings = self.patch_embed(x)
        loss = self.forward_loss(orig_embeddings, pred, mask)
        
        return loss, pred, mask

def create_model(model_arch: str = "custom_gigapath_slide_enc12l768d", in_chans: int = 1536):
    model = timm.create_model(model_arch, pretrained=False, in_chans=in_chans)
    return model

@register_model
def custom_gigapath_slide_enc12l768d(**kwargs): 
    """Custom LongNetViT model with 12 layers and 768 dimensions"""
    model = CustomLongNetViT(embed_dim=768, depth=12, mlp_ratio=4, global_pool = True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
