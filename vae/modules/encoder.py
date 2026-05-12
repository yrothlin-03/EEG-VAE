import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


__all__ = ["Encoder"]


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=8):
    return nn.GroupNorm(
        num_groups=min(num_groups, in_channels),
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
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

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
                    attn.append(AttnBlock(block_in))

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
        self.mid.attn_1 = AttnBlock(block_in)
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