from collections import OrderedDict
from functools import partial
from typing import Tuple, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.layers import Mlp, DropPath
from timm.models.layers.helpers import to_2tuple
from torchvision.ops import roi_align

# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.layers import Mlp, DropPath
from timm.models.layers.helpers import to_2tuple
from torchvision.ops import roi_align

from util.misc import LayerNorm


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.vis = False

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        weights = attn if self.vis else None
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        return x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, weights = self.attn(x)
        x = h + self.drop_path1(self.ls1(x))

        h = x
        x = self.norm2(x)
        x = self.mlp(x)

        x = h + self.drop_path2(self.ls2(x))
        return x, weights


class CellViT(nn.Module):
    """CellViT model for cell segmentation. U-Net like network with vision transformer as backbone encoder

    Skip connections are shared between branches

    The model is having multiple branches:
        * tissue_types: Tissue prediction based on global class token
        * nuclei_binary_map: Binary nuclei prediction
        * hv_map: HV-prediction to separate isolated instances
        * nuclei_type_map: Nuclei instance-prediction

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        embed_dim (int): Embedding dimension of backbone ViT
        input_channels (int): Number of input channels
        depth (int): Depth of the backbone ViT
        num_heads (int): Number of heads of the backbone ViT
        extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens. First blocks starts with 1, and maximum is N=depth.
            Is used for skip connections. At least 4 skip connections needs to be returned.
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone ViT. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone ViT. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        embed_dim: int,
        extract_layers: List,
        drop_rate: float,
        encoder):
        # For simplicity, we will assume that extract layers must have a length of 4
        super().__init__()
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.patch_size = 16
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate

        self.encoder = encoder

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        self.common_decoder = self.create_upsampling_branch()

        self.apply(self._init_weights)

        self.box_embed = PatchEmbed(img_size=16, patch_size=8,
                                    in_chans=embed_dim, embed_dim=embed_dim)

    def forward(self, x, boxes, mask):
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (batch_size, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (batch_size, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (batch_size, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (batch_size, num_nuclei_classes, H, W)
                * (optinal) tokens
        """
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"

        _, _, z = self.encoder(x)
        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        _, b1 = self._forward_upsample(z0, z1, z2, z3, z4, self.common_decoder)
        boxes_features = self.extract_box_feature(x=b1, boxes_info=boxes, scale_factor=1., mask=mask)
        boxes_features = self.box_embed(boxes_features).squeeze()
        return boxes_features

    def extract_box_feature(self, x, boxes_info, scale_factor, mask):
        batch_size, h, w, embed_dim = x.shape
        num_box = boxes_info.shape[1]
        #batch_size = x.shape[0]
        #embed_dim = x.shape[]
        x = x.view(batch_size, h, w, embed_dim).permute(0, 3, 1, 2)

        batch_index = torch.arange(0.0, batch_size).repeat(num_box).view(num_box, -1) \
            .transpose(0, 1).flatten(0, 1).to(x.device)
        roi_box_info = boxes_info.view(-1, 4).to(x.device)

        roi_info = torch.stack((batch_index, roi_box_info[:, 0],
                                roi_box_info[:, 1], roi_box_info[:, 2],
                                roi_box_info[:, 3]), dim=1).to(x.device)
        aligned_out = roi_align(input=x, boxes=roi_info, spatial_scale=scale_factor,
                                output_size=8)

        aligned_out = aligned_out.view(batch_size, num_box, embed_dim, 8, 8)[mask]
        aligned_out.view(-1, embed_dim, 8, 8)

        return aligned_out

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential):
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)

        return b0, b1

    def create_upsampling_branch(self) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                ]
            )
        )

        return decoder

    def freeze_encoder(self):
        """Freeze encoder to not train it"""
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] not in ["head", "fc_norm"]:  # do not freeze head
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model"""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        init_values = kwargs.pop('init_values')

        self.extract_layers = kwargs.pop('extract_layers')
        super(VisionTransformer, self).__init__(**kwargs)

        self.patch_size = kwargs['patch_size']

        drop_path_rate = kwargs['drop_path_rate']
        depth = kwargs['depth']
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'],
                init_values=init_values, norm_layer=kwargs['norm_layer'], drop_path=dpr[i])
            for i in range(kwargs['depth'])])

        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)

        del self.norm  # remove the original norm

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

    def forward(self, x):
        extract_layers = []

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        hidden_states = x

        for depth, blk in enumerate(self.blocks):
            hidden_states, _ = blk(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)
            if depth == len(self.blocks) - 1:
                outcome = hidden_states

        outcome = self.fc_norm(outcome)
        output = self.head(outcome[:, 0])

        return output, outcome[:, 0], extract_layers


def cell_vit_base_patch16(num_nuclei_classes, embed_dim, extract_layers, drop_rate, encoder):

    model = CellViT(num_nuclei_classes=num_nuclei_classes,
                    embed_dim=embed_dim,
                    extract_layers=extract_layers,
                    encoder=encoder,
                    drop_rate=drop_rate)
    return model


def unetr_vit_small_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=256, extract_layers=[3, 6, 9, 12], patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


def unetr_vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=256, extract_layers=[3, 6, 9, 12], patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


def unetr_vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


def unetr_vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model
