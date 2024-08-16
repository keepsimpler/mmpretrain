import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from .vision_transformer import VisionTransformer
from ..utils import build_norm_layer, resize_pos_embed


class CrossAttention(nn.Module):
    def __init__(self, 
                 embed_dims, 
                 image_dims,
                 scale=None, 
                 qkv_bias=True, 
                 drop_rate=0., 
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                ):
        super().__init__()
        self.scale = image_dims ** -0.5 if scale is None else scale
        # self.to_q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.to_k = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        # self.to_v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.drop = nn.Dropout(drop_rate)
        self.ln = build_norm_layer(norm_cfg, image_dims)

    def forward(self, p, k_v, with_softmax=True):
        # q = self.to_q(q)
        k = self.to_k(k_v)
        v = self.to_k(k_v)
        # print(p.shape, k.shape, v.shape)
        
        out = torch.einsum('b i j, b j d -> b i d', p, v)
        sim = torch.einsum('b i d, b j d -> b i j', out, k) * self.scale
        sim = self.ln(sim)
        if with_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim
        attn = self.drop(attn)

        return attn


@MODELS.register_module()
class BayesDecoderVisionTransformer(VisionTransformer):
    """
    """
    def __init__(self, arch='deit-tiny', scale=None, num_feature_tokens=1, *args, **kwargs):
        super(BayesDecoderVisionTransformer, self).__init__(
            arch=arch,
            # with_cls_token=True,
            # out_type='raw',
            final_norm=False,  # donot need the final norm
            *args,
            **kwargs,
        )
        self.image_dims = self.patch_resolution[0] * self.patch_resolution[1] + self.num_extra_tokens
        self.feature_token = nn.Parameter(torch.ones(1, num_feature_tokens, self.image_dims) / self.image_dims)
        self.digup = CrossAttention(embed_dims=self.embed_dims, image_dims=self.image_dims, scale=scale)
    
    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        # x = x + self.pos_embed
        x = self.drop_after_pos(x)

        cls_tokens = self.feature_token.expand(B, -1, -1)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # k_v = self.ln1(x)
            if i < self.num_layers - 1:
                cls_tokens = self.digup(cls_tokens, x)
            else:
                cls_tokens = self.digup(cls_tokens, x, with_softmax=False)
            # cls_tokens = self.ln1(cls_tokens)

        # cls_tokens = self.ln1(cls_tokens)
        
        return (cls_tokens.mean(dim=1),)  # cls_tokens[:,0]   torch.cat([x, cls_tokens], dim=1).mean(dim=1)

