import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Optional, Sequence, Tuple

from timm.models.layers import DropPath
from physics_jepa.utils.attentive_pooler_modules import Block as AttnBlock
from physics_jepa.utils.tensors import trunc_normal_

class PatchEmbed3D(nn.Module):
    """
    patchify input 3D video, then embed
    """

    def __init__(
        self,
        patch_size=16,
        num_frames_per_patch=None,
        in_chans=2,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames_per_patch = num_frames_per_patch

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(num_frames_per_patch, patch_size, patch_size),
            stride=(num_frames_per_patch, patch_size, patch_size),
        ) # (B, C, T, H, W) -> (B, embed_dim, T//num_frames_per_patch, H//patch_size, W//patch_size)

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            weight_expanded = self.weight.view(self.weight.shape[0], *([1] * (x.dim() - 2)))
            bias_expanded = self.bias.view(self.bias.shape[0], *([1] * (x.dim() - 2)))
            x = weight_expanded * x + bias_expanded
            return x

class ResidualBlock(nn.Module):
    def __init__(self, embed_dim, num_spatial_dims=3, layer_scale_init_value=1e-6, drop_path=0.):
        super().__init__()
        self.num_spatial_dims = num_spatial_dims

        self.conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim) if num_spatial_dims == 3 else nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)
        self.norm = LayerNorm(embed_dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(embed_dim, 4 * embed_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * embed_dim, embed_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((embed_dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1) if self.num_spatial_dims == 3 else x.permute(0, 2, 3, 1) # (N, C, T, H, W) -> (N, T, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) if self.num_spatial_dims == 3 else x.permute(0, 3, 1, 2) # (N, T, H, W, C) -> (N, C, T, H, W)

        x = input + self.drop_path(x)
        return x

class ConvEncoder(nn.Module):
    def __init__(self,
                 in_chans=2,
                 num_res_blocks=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 num_frames=4,
                 attn_stages: Optional[Sequence[int]] = None,
                 attn_num_heads: int = 4,
                 attn_mlp_ratio: float = 4.0,
                 attn_depth: int = 1):
        super().__init__()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1, 4, 4), padding='same'),
            LayerNorm(dims[0], data_format="channels_first"),
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)

        if num_frames == 16:
            for i in range(len(dims)-1):
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm(dims[i], data_format="channels_first"),
                        nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=2, stride=2)
                    )
                )
    
            self.res_blocks = nn.ModuleList()
            for i in range(len(dims)):
                self.res_blocks.append(
                    nn.Sequential(
                        *[ResidualBlock(dims[i], num_spatial_dims=3 if i < len(dims)-1 else 2) for _ in range(num_res_blocks[i])]
                    )
                )

        elif num_frames == 4:
            for i in range(3):
                stride = 2 if i % 2 == 0 else (1, 2, 2) # downsample time every other layer
                kernel_size = 2 if i % 2 == 0 else (1, 2, 2)
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm(dims[i], data_format="channels_first"),
                        nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, stride=stride)#, padding=(1, 0, 0)),
                    )
                )
            for i in range(3, len(dims)-1): # downsample spatial only
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm(dims[i], data_format="channels_first"),
                        nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=2, stride=2)
                    )
                )
            
            self.res_blocks = nn.ModuleList()
            for i in range(len(dims)):
                self.res_blocks.append(
                    nn.Sequential(
                        *[ResidualBlock(dims[i], num_spatial_dims=3 if i < 3 else 2) for _ in range(num_res_blocks[i])]
                    )
                )

        else:
            raise ValueError(f"Currently supports 4 and 16 frames, input num_frames: {num_frames}")

        self.dims = dims

        # Stage indices (0..len(dims)-1) after whose ResidualBlock stack a
        # transformer `Block` is inserted. Empty -> no attention modules;
        # self.attn_blocks stays empty and adds no parameters.
        self.attn_stages = tuple(sorted(set(int(i) for i in (attn_stages or []))))
        self.attn_depth = max(1, int(attn_depth))
        self.attn_blocks = nn.ModuleDict()
        for stage_idx in self.attn_stages:
            if stage_idx < 0 or stage_idx >= len(dims):
                raise ValueError(f"attn_stages index {stage_idx} out of range [0, {len(dims)-1}]")
            # Stack `attn_depth` transformer Blocks at this stage.
            # IMPORTANT: when attn_depth == 1, store the bare AttnBlock so
            # state-dict keys (`attn_blocks.<i>.norm1.weight`, ...) match
            # checkpoints saved before this parameter existed. Deeper stacks
            # use a Sequential, which renames keys to `attn_blocks.<i>.<k>.*`
            # — a fresh layout, so deeper-attn checkpoints never collide
            # with the old single-block ones.
            if self.attn_depth == 1:
                self.attn_blocks[str(stage_idx)] = AttnBlock(
                    dim=dims[stage_idx],
                    num_heads=attn_num_heads,
                    mlp_ratio=attn_mlp_ratio,
                    qkv_bias=True,
                )
            else:
                self.attn_blocks[str(stage_idx)] = nn.Sequential(*[
                    AttnBlock(
                        dim=dims[stage_idx],
                        num_heads=attn_num_heads,
                        mlp_ratio=attn_mlp_ratio,
                        qkv_bias=True,
                    )
                    for _ in range(self.attn_depth)
                ])

    def _apply_attn(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Apply the stage's attention block to (B, C, *) tensors.

        Flattens spatial (+optional temporal) dims into a token sequence,
        runs self-attention, and reshapes back. No-op when the stage has
        no attention block.
        """
        key = str(stage_idx)
        if key not in self.attn_blocks:
            return x
        blk = self.attn_blocks[key]
        if x.ndim == 5:
            B, C, T, H, W = x.shape
            tokens = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            tokens = blk(tokens)
            return tokens.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        elif x.ndim == 4:
            B, C, H, W = x.shape
            tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            tokens = blk(tokens)
            return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"unexpected tensor rank in _apply_attn: {x.ndim}")

    def forward(self, x, **kwargs):
        for i in range(len(self.dims)):
            x = self.downsample_layers[i](x)
            x = x.squeeze(2)
            x = self.res_blocks[i](x)
            if i in self.attn_stages:
                x = self._apply_attn(x, i)
        return x

class ConvEncoderViTStem(nn.Module):
    """ViT-style 3D patchify -> single attention block. No pos embed.

    Tokenization ablation between `vit3d` (full transformer over 3D patches,
    learnable pos embed, depth N) and `conv3d_next_attn` (ConvNeXt encoder
    with attention at the last stage). This isolates the *tokenization* step:
      - same Conv3d patchify (kernel=stride=patch_size) as `ViT3DEncoder`,
      - channels-first LayerNorm in place of `nn.LayerNorm` post-patch,
      - **no** learnable positional embedding,
      - **0** residual blocks,
      - **1** transformer Block (vs ViT3D's `depth`).

    Output contract matches ConvEncoder/ViT3DEncoder: (B, embed_dim, H', W')
    via mean-pool over the temporal patch axis, so `ConvPredictor` wires up
    unchanged. Exposes `dims = [embed_dim, embed_dim]` for the same reason.
    """

    def __init__(
        self,
        in_chans: int = 11,
        num_frames: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        patch_size: Tuple[int, int, int] = (4, 16, 16),   # (pt, ph, pw)
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        pt, ph, pw = patch_size
        if num_frames % pt != 0:
            raise ValueError(f"num_frames ({num_frames}) must be divisible by patch_size[0] ({pt})")
        if img_size[0] % ph != 0 or img_size[1] % pw != 0:
            raise ValueError(f"img_size {img_size} must be divisible by spatial patch {(ph, pw)}")

        self.in_chans = in_chans
        self.num_frames = num_frames
        self.img_size = tuple(img_size)
        self.patch_size = (pt, ph, pw)
        self.T_out = num_frames // pt
        self.H_out = img_size[0] // ph
        self.W_out = img_size[1] // pw
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )
        # Channels-first LayerNorm matches the ConvNet style (rather than
        # nn.LayerNorm applied to flattened tokens, which is what ViT3D does).
        self.norm_pre_attn = LayerNorm(embed_dim, data_format="channels_first")
        # No learnable pos embed: we want positional info to come purely from
        # the conv stem's spatial structure, so this ablation tests whether
        # 1 attn block over conv-stem tokens (no posemb) is enough to match
        # vit3d's depth-6 stack.
        self.attn = AttnBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
        )

        # Match ViT3DEncoder/ConvEncoder dims contract for ConvPredictor wiring.
        self.dims = [embed_dim, embed_dim]

    def forward(self, x, **kwargs):
        # x: (B, C, T, H, W). Pad/crop time like ViT3DEncoder so this drops
        # in for the same input contract.
        B, C, T, H, W = x.shape
        if T < self.num_frames:
            x = F.pad(x, (0, 0, 0, 0, 0, self.num_frames - T))
        elif T > self.num_frames:
            x = x[:, :, : self.num_frames]
        x = self.patch_embed(x)               # (B, D, T', H', W')
        x = self.norm_pre_attn(x)             # channels-first LN
        B, D, Tp, Hp, Wp = x.shape
        # Tokenize -> single attention block -> reshape back. No pos embed.
        tokens = x.flatten(2).transpose(1, 2).contiguous()    # (B, T'*H'*W', D)
        tokens = self.attn(tokens)
        x = tokens.transpose(1, 2).reshape(B, D, Tp, Hp, Wp).contiguous()
        # Collapse time to match ViT3DEncoder's (B, D, H', W') output.
        if Tp == 1:
            x = x.squeeze(2)
        else:
            x = x.mean(dim=2)
        return x


class ConvEncoderViTTiny(nn.Module):
    def __init__(self,
                 in_chans=2,
                 num_res_blocks=[3, 3, 9, 3],
                 dims=[48, 96, 192, 384]):
        super().__init__()
        
        # Stem: 11 -> 48 channels, no spatial downsampling
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1, 4, 4), padding='same'),
            LayerNorm(dims[0], data_format="channels_first"),
        )
        
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        
        # Layer 1: Time downsampling (4 -> 2), 48 -> 96 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[0], data_format="channels_first"),
                nn.Conv3d(in_channels=dims[0], out_channels=dims[1], 
                         kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # downsample time only
            )
        )
        
         # Layer 2: Spatial downsampling (224 -> 112), 96 -> 192 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[1], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[1], out_channels=dims[2], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Layer 3: Spatial downsampling (112 -> 56), 192 -> 384 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[2], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[2], out_channels=dims[3], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Layer 4: Spatial downsampling (56 -> 28), keep 384 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[3], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[3], out_channels=dims[3], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Layer 5: Spatial downsampling (28 -> 14), keep 384 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[3], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[3], out_channels=dims[3], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Residual blocks for each stage
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.downsample_layers)):
            if i == 0:
                channels = dims[0]
            elif i <= 3:
                channels = dims[i]
            else:
                channels = dims[3]  # 384 for final layers
            
            self.res_blocks.append(
                nn.Sequential(
                    *[ResidualBlock(channels, num_spatial_dims=3 if i <= 1 else 2) for _ in range(num_res_blocks[min(i, len(num_res_blocks)-1)])]
                )
            )
        
        self.dims = dims

    def forward(self, x, **kwargs):
        # Input: (B, 11, 4, 224, 224)
        # Layer 0: (B, 48, 4, 224, 224) - stem
        # Layer 1: (B, 96, 2, 224, 224) - time downsampling
        # Layer 2: (B, 192, 2, 112, 112) - spatial downsampling
        # Layer 3: (B, 384, 2, 56, 56) - spatial downsampling
        # Layer 4: (B, 384, 2, 28, 28) - spatial downsampling
        # Layer 5: (B, 384, 2, 14, 14) - spatial downsampling
        b, c0, t0, h0, w0 = x.shape
        for i in range(len(self.downsample_layers)):
            if i == 2:
                # flatten time dimension to use conv2ds
                b, c, t, h, w = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
                x = x.view(b*t, c, h, w)  # (B*T, C, H, W)
            x = self.downsample_layers[i](x)
            x = self.res_blocks[i](x)
        # reshape back to (B, C, T, H, W)
        _, c, h, w = x.shape
        x = x.view(b, t0//2, c, h, w)  # (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        return x

class ViT3DEncoder(nn.Module):
    """Simple ViT3D encoder: 3D patchify, transformer stack, collapse time.

    Output contract matches ConvEncoder: (B, C_out, H', W') so the existing
    ConvPredictor (Conv2d-based) keeps working. When the temporal patching
    leaves T'>1, the time dim is reduced by mean-pool; when T'==1, it's
    simply squeezed.

    Intentionally lightweight (no cls token, learnable pos embed). Matches
    the JEPA loss contract with a single (B, C, H, W) tensor.
    """

    def __init__(
        self,
        in_chans: int = 11,
        num_frames: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        patch_size: Tuple[int, int, int] = (4, 16, 16),   # (pt, ph, pw)
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        pt, ph, pw = patch_size
        if num_frames % pt != 0:
            raise ValueError(f"num_frames ({num_frames}) must be divisible by patch_size[0] ({pt})")
        if img_size[0] % ph != 0 or img_size[1] % pw != 0:
            raise ValueError(f"img_size {img_size} must be divisible by spatial patch {(ph, pw)}")

        self.in_chans = in_chans
        self.num_frames = num_frames
        self.img_size = tuple(img_size)
        self.patch_size = (pt, ph, pw)
        self.T_out = num_frames // pt
        self.H_out = img_size[0] // ph
        self.W_out = img_size[1] // pw
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )
        self.num_tokens = self.T_out * self.H_out * self.W_out
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            AttnBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Expose `dims` so callers constructing ConvPredictor(encoder.dims[-2:])
        # see a compatible structure. The list length doesn't change the
        # predictor, but keeping two entries preserves semantics.
        self.dims = [embed_dim, embed_dim]

    def forward(self, x, **kwargs):
        # x: (B, C, T, H, W). Light padding of time if < expected.
        B, C, T, H, W = x.shape
        if T < self.num_frames:
            x = F.pad(x, (0, 0, 0, 0, 0, self.num_frames - T))
        elif T > self.num_frames:
            x = x[:, :, : self.num_frames]
        x = self.patch_embed(x)  # (B, D, T', H', W')
        B, D, Tp, Hp, Wp = x.shape
        # (B, D, T', H', W') -> (B, T'*H'*W', D)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        # reshape back to (B, D, T', H', W')
        x = tokens.transpose(1, 2).reshape(B, D, Tp, Hp, Wp).contiguous()
        # Collapse time: squeeze if T'==1 else mean over time -> (B, D, H', W')
        if Tp == 1:
            x = x.squeeze(2)
        else:
            x = x.mean(dim=2)
        return x


class ConvPredictorViTTiny(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # self.thw = (time_dim, height_dim, width_dim)
        self.scale_factor = 2
        self.conv = nn.Sequential(
            nn.Conv3d(dims[0], dims[0]*self.scale_factor, kernel_size=2, padding=1),
            ResidualBlock(dims[0]*self.scale_factor, num_spatial_dims=3),
            nn.Conv3d(dims[0]*self.scale_factor, dims[0], kernel_size=2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvPredictor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # self.thw = (time_dim, height_dim, width_dim)
        self.scale_factor = 2

        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[0]*self.scale_factor, kernel_size=2, padding=1),
            ResidualBlock(dims[0]*self.scale_factor, num_spatial_dims=2),
            nn.Conv2d(dims[0]*self.scale_factor, dims[0], kernel_size=2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvDecoder(nn.Module):
    # image input: 128x128x3 channels -> embedding: 4x4x768 channels -> input size
    def __init__(self,
                 out_chans=2,
                 num_res_blocks=[3, 9, 3, 3],
                 dims=[768, 384, 192, 96],
        ):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.upsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], data_format="channels_first"),
                    nn.ConvTranspose3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=2, stride=2),
                )
            )
        
        self.res_blocks = nn.ModuleList()
        for i in range(len(dims)-1):
            self.res_blocks.append(
                nn.Sequential(
                    *[ResidualBlock(dims[i]) for _ in range(num_res_blocks[i])]
                )
            )

        self.final_conv = nn.Conv3d(dims[-1], out_chans, kernel_size=1)
        
        self.dims = dims
    
    def forward(self, x):
        for i in range(len(self.dims)-1):
            x = self.res_blocks[i](x)
            x = self.upsample_layers[i](x)
        x = self.final_conv(x)
        return x

class RegressionHead(nn.Module):
    def __init__(self, in_dim, out_dim, flatten_first=False, add_dropout=False, dropout_rate=0.4):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.flatten_first = flatten_first
        self.add_dropout = add_dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.flatten_first:
            x = x.flatten(1, -1)
        if self.add_dropout:
            x = self.dropout(x)
        return self.fc(x)

class RegressionMLP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim=32,
                 num_hidden_layers=1,
                 flatten_first=False,
                 add_dropout=False,
                 dropout_rate=0.2):
        super().__init__()

        self.add_dropout = add_dropout
        self.flatten_first = flatten_first
        
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp_block = lambda in_dim, out_dim: nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            self.dropout if add_dropout else nn.Identity(),
        )
        self.layers = []
        self.stem = self.mlp_block(in_dim, hidden_dim)
        self.hidden_layers = nn.Sequential(
            *[self.mlp_block(hidden_dim, hidden_dim) for _ in range(num_hidden_layers-1)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        if self.flatten_first:
            x = x.flatten(1, -1)
        x = self.stem(x)
        if self.add_dropout:
            x = self.dropout(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)

class Projector3D(nn.Module):
    """
    Projector that takes embeddings of shape (B, C, T, H, W) and projects them to (B, C', T, H, W)
    using 1D Conv3D operations: C -> C' -> C'
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.proj = nn.Sequential(
            # First projection: C -> C'
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            # Second projection: C' -> C'
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        Returns:
            Projected tensor of shape (B, C', T, H, W)
        """
        return self.proj(x)

# from convnext
def cosine_schedule_array(
    base_value,            # peak value after warmup (e.g., max LR)
    final_value,           # floor value at the very end (e.g., 1e-6)
    epochs=0,
    niter_per_ep=0,
    steps=0,
    warmup_epochs=0,
    start_warmup_value=0.0,
    warmup_steps=-1        # if >0, overrides warmup_epochs
):
    assert (epochs > 0 and niter_per_ep > 0) or steps > 0, "either (epochs and niter_per_ep) or steps must be provided"
    if steps == 0:
        total_steps = int(epochs * niter_per_ep)
        if total_steps <= 0:
            return np.array([], dtype=np.float32)
    else:
        total_steps = steps

    # Compute warmup iters (steps), prefer explicit steps if provided
    warmup_iters = int(warmup_steps) if warmup_steps > 0 else int(warmup_epochs * niter_per_ep)
    warmup_iters = max(0, min(warmup_iters, total_steps))

    # Warmup schedule (linear from start_warmup_value -> base_value)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters, dtype=np.float32
        )
    else:
        warmup_schedule = np.array([], dtype=np.float32)

    # Cosine decay schedule (base_value -> final_value)
    remain = total_steps - warmup_iters
    if remain > 0:
        # i ranges [0 .. remain-1]; use N-1 in denominator to hit final_value exactly at the last step
        if remain == 1:
            cos_factors = np.array([1.0], dtype=np.float32)  # single step = exactly base_value
        else:
            t = np.arange(remain, dtype=np.float32) / (remain - 1)
            # cosine from 0 -> pi
            cos_factors = 0.5 * (1.0 + np.cos(np.pi * t))
        decay_schedule = final_value + (base_value - final_value) * cos_factors
    else:
        decay_schedule = np.array([], dtype=np.float32)

    schedule = np.concatenate([warmup_schedule, decay_schedule]).astype(np.float32)
    # Safety clamp and length assert
    schedule = np.clip(schedule, min(start_warmup_value, final_value), max(base_value, final_value))
    assert len(schedule) == total_steps, f"len(schedule)={len(schedule)} != total_steps={total_steps}"
    return schedule

class CosineLRScheduler:
    def __init__(self, optimizer, step=0, **kwargs):
        self.optimizer = optimizer
        self.schedule = cosine_schedule_array(**kwargs)
        self.idx = step

    def step(self):
        if self.idx < len(self.schedule):
            lr = float(self.schedule[self.idx])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            self.idx += 1
        # if idx >= schedule length, lr stays constant at final_value

    def get_last_lr(self):
        if self.idx == 0:
            return [pg["lr"] for pg in self.optimizer.param_groups]
        return [float(self.schedule[min(self.idx - 1, len(self.schedule) - 1)])]

    def state_dict(self):
        return {"idx": self.idx, "schedule": self.schedule.tolist()}

    def load_state_dict(self, state_dict):
        self.idx = state_dict["idx"]
        self.schedule = np.array(state_dict["schedule"], dtype=np.float32)


def linear_schedule_array(
    base_value, final_value, steps, warmup_steps=0, start_warmup_value=0.0,
):
    """Linear warmup then linear decay to final_value over `steps` updates."""
    steps = int(steps)
    warmup_steps = max(0, min(int(warmup_steps), steps))
    if warmup_steps > 0:
        warmup = np.linspace(start_warmup_value, base_value, warmup_steps, dtype=np.float32)
    else:
        warmup = np.array([], dtype=np.float32)
    remain = steps - warmup_steps
    if remain > 0:
        if remain == 1:
            decay = np.array([base_value], dtype=np.float32)
        else:
            t = np.arange(remain, dtype=np.float32) / (remain - 1)
            decay = (base_value + (final_value - base_value) * t).astype(np.float32)
    else:
        decay = np.array([], dtype=np.float32)
    schedule = np.concatenate([warmup, decay]).astype(np.float32)
    assert len(schedule) == steps
    return schedule


def constant_schedule_array(base_value, steps, warmup_steps=0, start_warmup_value=0.0):
    """Constant LR after optional linear warmup."""
    steps = int(steps)
    warmup_steps = max(0, min(int(warmup_steps), steps))
    if warmup_steps > 0:
        warmup = np.linspace(start_warmup_value, base_value, warmup_steps, dtype=np.float32)
    else:
        warmup = np.array([], dtype=np.float32)
    remain = steps - warmup_steps
    flat = np.full((remain,), float(base_value), dtype=np.float32)
    return np.concatenate([warmup, flat]).astype(np.float32)


class ArrayLRScheduler:
    """Generic scheduler that steps through a precomputed schedule array.

    Used for `linear` and `constant` schedules. `CosineLRScheduler` is a
    separate class so its state_dict shape stays stable for resumes.
    """

    def __init__(self, optimizer, schedule, step=0):
        self.optimizer = optimizer
        self.schedule = np.asarray(schedule, dtype=np.float32)
        self.idx = int(step)

    def step(self):
        if self.idx < len(self.schedule):
            lr = float(self.schedule[self.idx])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            self.idx += 1

    def get_last_lr(self):
        if self.idx == 0:
            return [pg["lr"] for pg in self.optimizer.param_groups]
        return [float(self.schedule[min(self.idx - 1, len(self.schedule) - 1)])]

    def state_dict(self):
        return {"idx": self.idx, "schedule": self.schedule.tolist()}

    def load_state_dict(self, state_dict):
        self.idx = state_dict["idx"]
        self.schedule = np.array(state_dict["schedule"], dtype=np.float32)


def build_lr_scheduler(optimizer, name, base_value, final_value, steps,
                       warmup_steps=0, start_warmup_value=0.0, start_step=0):
    """Factory: returns a scheduler matching `name` (cosine|linear|constant) or None."""
    if name is None:
        return None
    name = str(name).lower()
    if name == "cosine":
        return CosineLRScheduler(
            optimizer,
            step=start_step,
            base_value=base_value,
            final_value=final_value,
            steps=steps,
            warmup_steps=warmup_steps,
            start_warmup_value=start_warmup_value,
        )
    if name == "linear":
        sched = linear_schedule_array(
            base_value=base_value,
            final_value=final_value,
            steps=steps,
            warmup_steps=warmup_steps,
            start_warmup_value=start_warmup_value,
        )
        return ArrayLRScheduler(optimizer, sched, step=start_step)
    if name == "constant":
        sched = constant_schedule_array(
            base_value=base_value,
            steps=steps,
            warmup_steps=warmup_steps,
            start_warmup_value=start_warmup_value,
        )
        return ArrayLRScheduler(optimizer, sched, step=start_step)
    raise ValueError(f"unknown lr_scheduler: {name!r}; expected cosine|linear|constant")


def build_optimizer(params, train_cfg):
    """Build optimizer from `train_cfg`.

    Defaults reproduce the prior hard-coded behavior:
      - AdamW with lr=train_cfg.lr, weight_decay=train_cfg.weight_decay|0.05,
        betas=(0.9, 0.95).

    Optional `train.optim` block overrides:
      optim:
        name: adamw | lion       # default adamw
        betas: [b1, b2]          # default [0.9, 0.95] for adamw, [0.9, 0.99] for lion
    """
    optim_cfg = train_cfg.get("optim", None) if hasattr(train_cfg, "get") else None
    optim_cfg = optim_cfg if optim_cfg is not None else {}

    name = optim_cfg.get("name", "adamw") if hasattr(optim_cfg, "get") else optim_cfg.get("name", "adamw")
    name = str(name).lower()

    lr = train_cfg.lr
    weight_decay = train_cfg.get("weight_decay", 0.05) if hasattr(train_cfg, "get") else 0.05

    if name == "adamw":
        betas = tuple(optim_cfg.get("betas", [0.9, 0.95]))
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as e:
            raise ImportError(
                "train.optim.name=lion requires `lion-pytorch`; "
                "pip install lion-pytorch"
            ) from e
        betas = tuple(optim_cfg.get("betas", [0.9, 0.99]))
        return Lion(params, lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"unknown train.optim.name: {name!r}; expected adamw|lion")