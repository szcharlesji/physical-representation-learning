import copy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from einops import rearrange
from collections import defaultdict

from physics_jepa.utils.model_utils import ConvEncoder, ConvPredictor, ConvDecoder, ViT3DEncoder


def build_encoder(model_cfg, num_frames: int, in_chans: int, img_size=None):
    """Factory for the pretrain encoder selected by `model_cfg.backbone`.

    - `conv3d_next` (default): ConvEncoder with no attention modules.
    - `conv3d_next_attn`: ConvEncoder with a transformer Block inserted
      after the ResidualBlock stack at each index in `model_cfg.attn_stages`.
      An empty list produces a structurally identical encoder to
      `conv3d_next` (no added parameters).
    - `vit3d`: ViT3DEncoder with 3D patch embedding and transformer stack.
    """
    backbone = None
    if hasattr(model_cfg, "get"):
        backbone = model_cfg.get("backbone", None)
    backbone = str(backbone or "conv3d_next").lower()

    if backbone in ("conv3d_next", "conv3d_next_attn"):
        attn_stages = list(model_cfg.get("attn_stages", []) or []) if backbone == "conv3d_next_attn" else []
        encoder = ConvEncoder(
            dims=list(model_cfg.dims),
            in_chans=in_chans,
            num_res_blocks=list(model_cfg.num_res_blocks),
            num_frames=num_frames,
            attn_stages=attn_stages,
            attn_num_heads=int(model_cfg.get("attn_num_heads", 4)),
            attn_mlp_ratio=float(model_cfg.get("attn_mlp_ratio", 4.0)),
        )
        return encoder

    if backbone == "vit3d":
        vit_cfg = model_cfg.get("vit3d", {}) if hasattr(model_cfg, "get") else {}
        vit_cfg = vit_cfg or {}
        if img_size is None:
            img_size = 256
        patch = list(vit_cfg.get("patch_size", [4, 16, 16]))
        encoder = ViT3DEncoder(
            in_chans=in_chans,
            num_frames=num_frames,
            img_size=img_size,
            patch_size=tuple(patch),
            embed_dim=int(vit_cfg.get("embed_dim", model_cfg.dims[-1] if hasattr(model_cfg, "dims") else 384)),
            depth=int(vit_cfg.get("depth", 6)),
            num_heads=int(vit_cfg.get("num_heads", 6)),
            mlp_ratio=float(vit_cfg.get("mlp_ratio", 4.0)),
        )
        return encoder

    raise ValueError(f"unknown model.backbone: {backbone!r}; expected conv3d_next|conv3d_next_attn|vit3d")


def get_model_and_loss_cnn(dims, num_res_blocks, num_frames, in_chans=2, sim_coeff=25, std_coeff=25, cov_coeff=1,
                           model_cfg=None, img_size=None):
    """Build (encoder, predictor, loss) tuple.

    Pass `model_cfg` to dispatch on `cfg.model.backbone`. Without it, the
    function ignores backbone selection and always builds a plain
    ConvEncoder from the positional `dims` / `num_res_blocks` / `num_frames`.
    """
    if model_cfg is not None:
        encoder = build_encoder(model_cfg, num_frames=num_frames, in_chans=in_chans, img_size=img_size)
    else:
        encoder = ConvEncoder(
            dims=dims,
            in_chans=in_chans,
            num_res_blocks=num_res_blocks,
            num_frames=num_frames,
        )
    loss = partial(vicreg_loss_3d,
                sim_coeff=sim_coeff,
                std_coeff=std_coeff,
                cov_coeff=cov_coeff,
                n_chunks=5)
    predictor = ConvPredictor(dims=list(reversed(encoder.dims))[:2])
    return encoder, predictor, loss

def vicreg_loss_3d(
    x, y, sim_coeff, std_coeff, cov_coeff, n_chunks=10,
    num_groups=1,
    fp32_stats=False,
    zscore_for_cov=False,
    adaptive_cov_scale=False
):
    """
    x,y: (B, C, T, H, W)
    """

    # Under bf16 autocast the variance/cov epsilon and small-difference math
    # are numerically unstable; promote stats to fp32.
    if x.dtype == torch.bfloat16 or y.dtype == torch.bfloat16:
        fp32_stats = True

    # Flatten to (N, C) where N = B*T*H*W
    x = rearrange(x, 'b c t h w -> (b t h w) c')
    y = rearrange(y, 'b c t h w -> (b t h w) c')

    N = x.shape[0]

    # Shuffle rows to decorrelate neighborhoods, then chunk
    shuffle_idx = torch.randperm(N, device=x.device)
    x_shuffled = x[shuffle_idx]
    y_shuffled = y[shuffle_idx]

    # Ensure chunks are valid (keep chunks >=1 and not smaller than ~C_group*8)
    n_chunks = max(1, int(n_chunks))
    x_chunks = x_shuffled.chunk(n_chunks, dim=0)
    y_chunks = y_shuffled.chunk(n_chunks, dim=0)

    # Return mean over chunks
    losses = defaultdict(list)
    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        out = vicreg_loss(
            x_chunk, y_chunk, sim_coeff, std_coeff, cov_coeff,
            num_groups=num_groups, fp32_stats=fp32_stats,
            zscore_for_cov=zscore_for_cov, adaptive_cov_scale=adaptive_cov_scale
        )
        (loss, repr_loss, std_loss, cov_loss,
         std_loss_x, std_loss_y, cov_loss_x, cov_loss_y) = out

        losses['loss'].append(loss)
        losses['repr_loss'].append(repr_loss)
        losses['std_loss'].append(std_loss)
        losses['cov_loss'].append(cov_loss)
        losses['std_loss_x'].append(std_loss_x)
        losses['std_loss_y'].append(std_loss_y)
        losses['cov_loss_x'].append(cov_loss_x)
        losses['cov_loss_y'].append(cov_loss_y)

    return {k: torch.stack(v).mean() for k, v in losses.items()}


def vicreg_loss(
    x, y, sim_coeff, std_coeff, cov_coeff,
    num_groups=8, fp32_stats=True, zscore_for_cov=False, adaptive_cov_scale=False
):
    """
    x, y: (N, C)
    Group-wise covariance penalty for stability when N << C.
    """
    N, C = x.shape
    assert C % num_groups == 0, f"C={C} must be divisible by num_groups={num_groups}"
    Cg = C // num_groups

    def off_diagonal(m):
        n, m_ = m.shape
        assert n == m_
        return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # ---- representation loss (keep in original dtype) ----
    repr_loss = F.mse_loss(x, y)

    # For stats, optionally upcast to fp32 for stability
    xs = x.float() if fp32_stats else x
    ys = y.float() if fp32_stats else y

    # Center
    xs = xs - xs.mean(dim=0)
    ys = ys - ys.mean(dim=0)

    # ---- variance (same as your impl) ----
    std_x = torch.sqrt(xs.var(dim=0, unbiased=False) + 1e-4)
    std_y = torch.sqrt(ys.var(dim=0, unbiased=False) + 1e-4)
    std_loss_x = torch.mean(F.relu(1.0 - std_x)) / 2.0
    std_loss_y = torch.mean(F.relu(1.0 - std_y)) / 2.0
    std_loss = std_loss_x + std_loss_y

    # ---- (optional) z-score before covariance so cov ~ correlations ----
    if zscore_for_cov:
        sx = std_x.detach().clamp_min(1e-3)
        sy = std_y.detach().clamp_min(1e-3)
        xs = xs / sx
        ys = ys / sy

    # ---- group-wise covariance ----
    cov_loss_x = xs.new_tensor(0.0)
    cov_loss_y = ys.new_tensor(0.0)

    # Optional: adapt cov weight when N is small relative to Cg
    # scale ~ 1 when N >= 8*Cg, smaller otherwise
    if adaptive_cov_scale:
        scale = min(1.0, float(N) / float(8 * Cg))
    else:
        scale = 1.0

    for g in range(num_groups):
        xg = xs[:, g*Cg:(g+1)*Cg]
        yg = ys[:, g*Cg:(g+1)*Cg]

        # covariance within group (unbiased=False to match above var)
        cov_xg = (xg.T @ xg) / max(1, (N - 1))
        cov_yg = (yg.T @ yg) / max(1, (N - 1))

        cov_loss_x = cov_loss_x + off_diagonal(cov_xg).pow_(2).sum().div(Cg)
        cov_loss_y = cov_loss_y + off_diagonal(cov_yg).pow_(2).sum().div(Cg)

    cov_loss_x = cov_loss_x / num_groups
    cov_loss_y = cov_loss_y / num_groups
    cov_loss = scale * (cov_loss_x + cov_loss_y)

    total_loss = (
        sim_coeff * repr_loss
        + std_coeff * std_loss
        + cov_coeff * cov_loss
    )

    return total_loss, repr_loss, std_loss, cov_loss, std_loss_x, std_loss_y, cov_loss_x, cov_loss_y

# randall's method: match distribution of embeddings to isotropic Gaussian
class BCS(torch.nn.Module):
    def __init__(self, num_slices=1024):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0

    @staticmethod
    def epps_pulley(x):
        def all_reduce(x, op):
            if dist.is_available() and dist.is_initialized():
                op = dist.nn.ReduceOp.__dict__[op]
                dist.nn.all_reduce(x, op=op)
                return x
            else:
                return x

        # integration points
        t = torch.linspace(-4, 4, 17, device=x.device)
        # theoretical CF for N(0, 1)
        exp_f = torch.exp(-0.5 * t**2)
        # ECF
        x_t = x.unsqueeze(2) * t  # (N, M, T)
        ecf = (1j * x_t).exp().mean(0)
        ecf = all_reduce(ecf, op="AVG")
        # weighted L2 distance
        err = exp_f * (ecf - exp_f).abs() ** 2
        T = torch.trapz(err, t, dim=1)
        return T

    def forward(self, x, y):
        views = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        with torch.no_grad():
            dev = views.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (views.size(2), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        views_A = views @ A
        self.step += 1
        return sum(self.epps_pulley(v).mean() for v in views_A) / len(views)

def vicreg_loss_bcs(x, y, sim_coeff, bcs_coeff, num_slices=1024):
    bcs = BCS(num_slices=num_slices)

    # Flatten to (N, C) where N = B*T*H*W
    x = rearrange(x, 'b c t h w -> b (t h w c)')
    y = rearrange(y, 'b c t h w -> b (t h w c)')

    sim_loss = F.mse_loss(x, y)
    bcs_loss = bcs(x, y)

    loss_dict = {
        'loss': sim_coeff * sim_loss + bcs_coeff * bcs_loss,
        'sim_loss': sim_loss,
        'bcs_loss': bcs_loss,
    }
    return loss_dict

def get_decoder(dims):
    return ConvDecoder(dims=dims)

def get_autoencoder(dims, in_chans=2):
    encoder = ConvEncoder(dims=dims, in_chans=in_chans)
    decoder = ConvDecoder(dims=list(reversed(dims)), out_chans=in_chans)
    return encoder, decoder