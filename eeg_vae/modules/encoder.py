import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


__all__ = ["Encoder"]


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=8):
    return nn.GroupNorm(
        num_groups=math.gcd(num_groups, in_channels),
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        return nn.functional.avg_pool1d(x, kernel_size=2, stride=2)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)

        q = self.q(h).permute(0, 2, 1)
        k = self.k(h)
        v = self.v(h)

        w = torch.bmm(q, k)
        w = w * (x.shape[1] ** -0.5)
        w = torch.softmax(w, dim=2)

        h = torch.bmm(v, w.permute(0, 2, 1))
        h = self.proj_out(h)

        return x + h


class MambaEEGBlock(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "sequence_block='mamba' requires the `mamba_ssm` package. "
                "Install it or use sequence_block='attention'."
            )

        self.norm = Normalize(in_channels)
        self.mamba = Mamba(d_model=in_channels, use_fast_path=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError(
                "sequence_block='mamba' requires CUDA with the installed "
                "mamba_ssm/causal_conv1d kernels. Move the model and inputs to CUDA "
                "or use sequence_block='attention'."
            )
        h = self.norm(x).transpose(1, 2)
        h = self.mamba(h).transpose(1, 2)
        return x + self.dropout(h)


def normalize_sequence_block(sequence_block):
    sequence_block = str(sequence_block).lower()
    if sequence_block not in {"attention", "mamba"}:
        raise ValueError(
            f"Unknown sequence_block={sequence_block!r}. "
            "Expected 'attention' or 'mamba'."
        )
    return sequence_block


def make_sequence_block(sequence_block, in_channels, dropout=0.0):
    sequence_block = normalize_sequence_block(sequence_block)
    if sequence_block == "attention":
        return AttnBlock(in_channels)
    return MambaEEGBlock(in_channels, dropout=dropout)


class CrissCrossEEGAttention(nn.Module):
    """CBraMod-style factorized attention over EEG channel/time patches.

    The old implementation attended over raw time steps and built attention
    tensors proportional to B*C*T. This version first patches the temporal
    axis, applies spatial attention over channels and temporal attention over
    patches, then projects the patch tokens back to a residual signal with the
    original (B, C, T) shape.
    """

    def __init__(
        self,
        in_channels,
        attn_channels=None,
        max_temporal_tokens=256,
        patch_size=200,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        del max_temporal_tokens

        attn_channels = (
            min(64, max(16, in_channels))
            if attn_channels is None
            else attn_channels
        )
        if attn_channels % 2 != 0:
            attn_channels += 1

        half_channels = attn_channels // 2
        num_heads = min(num_heads, half_channels)
        while half_channels % num_heads != 0:
            num_heads -= 1

        self.patch_size = patch_size
        self.attn_channels = attn_channels

        self.patch_proj = nn.Linear(patch_size, attn_channels)
        self.spectral_proj = nn.Linear(patch_size // 2 + 1, attn_channels)
        self.positional_encoding = nn.Conv2d(
            attn_channels,
            attn_channels,
            kernel_size=(19, 7),
            padding=(9, 3),
            groups=attn_channels,
        )

        self.spatial_norm = nn.LayerNorm(half_channels)
        self.temporal_norm = nn.LayerNorm(half_channels)
        self.spatial_attn = nn.MultiheadAttention(
            half_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_attn = nn.MultiheadAttention(
            half_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff_norm = nn.LayerNorm(attn_channels)
        self.ff = nn.Sequential(
            nn.Linear(attn_channels, 4 * attn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * attn_channels, attn_channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(attn_channels, patch_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def _patchify(self, x):
        b, c, t = x.shape
        pad_len = (self.patch_size - (t % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        patch_count = x.shape[-1] // self.patch_size
        x = x.reshape(b, c, patch_count, self.patch_size)
        return x, t

    def forward(self, x):
        patches, original_t = self._patchify(x)
        b, c, patch_count, patch_size = patches.shape

        tokens = self.patch_proj(patches)

        spectra = torch.fft.rfft(patches, dim=-1, norm="forward").abs()
        tokens = tokens + self.spectral_proj(spectra)

        pos = self.positional_encoding(tokens.permute(0, 3, 1, 2))
        tokens = tokens + pos.permute(0, 2, 3, 1)

        spatial_tokens, temporal_tokens = tokens.chunk(2, dim=-1)

        spatial_shape = spatial_tokens.shape
        spatial_tokens = spatial_tokens.transpose(1, 2).reshape(
            b * patch_count,
            c,
            spatial_shape[-1],
        )
        spatial_in = self.spatial_norm(spatial_tokens)
        spatial_delta = self.spatial_attn(
            spatial_in,
            spatial_in,
            spatial_in,
            need_weights=False,
        )[0]
        spatial_tokens = spatial_tokens + self.dropout(spatial_delta)
        spatial_tokens = spatial_tokens.reshape(
            b,
            patch_count,
            c,
            spatial_shape[-1],
        ).transpose(1, 2)

        temporal_shape = temporal_tokens.shape
        temporal_tokens = temporal_tokens.reshape(
            b * c,
            patch_count,
            temporal_shape[-1],
        )
        temporal_in = self.temporal_norm(temporal_tokens)
        temporal_delta = self.temporal_attn(
            temporal_in,
            temporal_in,
            temporal_in,
            need_weights=False,
        )[0]
        temporal_tokens = temporal_tokens + self.dropout(temporal_delta)
        temporal_tokens = temporal_tokens.reshape(
            b,
            c,
            patch_count,
            temporal_shape[-1],
        )

        tokens = torch.cat((spatial_tokens, temporal_tokens), dim=-1)
        tokens = tokens + self.dropout(self.ff(self.ff_norm(tokens)))

        residual = self.out_proj(tokens)
        residual = residual.reshape(b, c, patch_count * patch_size)[..., :original_t]

        return x + self.gate * residual

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.shortcut(x) + h


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        z_channels,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        resolution=None,
        dropout=0.0,
        resamp_with_conv=True,
        double_z=True,
        use_checkpoint=False,
        sequence_block="attention",
        use_criss_cross_attention=True,
        criss_cross_attn_channels=None,
        criss_cross_max_temporal_tokens=256,
        criss_cross_patch_size=200,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.sequence_block = normalize_sequence_block(sequence_block)

        if not use_criss_cross_attention:
            self.criss_cross_attn = nn.Identity()
        elif self.sequence_block == "mamba":
            self.criss_cross_attn = MambaEEGBlock(in_channels, dropout=dropout)
        else:
            self.criss_cross_attn = CrissCrossEEGAttention(
                in_channels=in_channels,
                attn_channels=criss_cross_attn_channels,
                max_temporal_tokens=criss_cross_max_temporal_tokens,
                patch_size=criss_cross_patch_size,
                dropout=dropout,
            )

        self.conv_in = nn.Conv1d(
            in_channels,
            ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                if curr_res is not None and curr_res in attn_resolutions:
                    attn.append(
                        make_sequence_block(
                            self.sequence_block,
                            block_in,
                            dropout=dropout,
                        )
                    )

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                if curr_res is not None:
                    curr_res = curr_res // 2

            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = make_sequence_block(
            self.sequence_block,
            block_in,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _run(self, module, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x):
        x = self._run(self.criss_cross_attn, x)
        h = self._run(self.conv_in, x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self._run(self.down[i_level].block[i_block], h)

                if len(self.down[i_level].attn) > 0:
                    h = self._run(self.down[i_level].attn[i_block], h)

            if i_level != self.num_resolutions - 1:
                h = self._run(self.down[i_level].downsample, h)

        h = self._run(self.mid.block_1, h)
        h = self._run(self.mid.attn_1, h)
        h = self._run(self.mid.block_2, h)

        h = self._run(self.norm_out, h)
        h = nonlinearity(h)
        h = self._run(self.conv_out, h)

        return h