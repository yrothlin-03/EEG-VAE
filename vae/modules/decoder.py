import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .encoder import AttnBlock, Normalize, nonlinearity, ResnetBlock

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        z_channels,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        resolution=None,
        dropout=0.0,
        resamp_with_conv=True,
        tanh_out=False,
        give_pre_end=False,
        use_checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out
        self.give_pre_end = give_pre_end
        self.resolution = resolution

        block_in = ch * ch_mult[-1]

        self.conv_in = nn.Conv1d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)

        self.up = nn.ModuleList()

        curr_res = None
        if resolution is not None:
            curr_res = resolution // (2 ** (self.num_resolutions - 1))

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks + 1):
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

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                if curr_res is not None:
                    curr_res = curr_res * 2

            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(
            block_in,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _run(self, module, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, z):
        h = self._run(self.conv_in, z)

        h = self._run(self.mid.block_1, h)
        h = self._run(self.mid.attn_1, h)
        h = self._run(self.mid.block_2, h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self._run(self.up[i_level].block[i_block], h)

                if len(self.up[i_level].attn) > 0:
                    h = self._run(self.up[i_level].attn[i_block], h)

            if i_level != 0:
                h = self._run(self.up[i_level].upsample, h)

        if self.give_pre_end:
            return h

        h = self._run(self.norm_out, h)
        h = nonlinearity(h)
        h = self._run(self.conv_out, h)

        if self.tanh_out:
            h = torch.tanh(h)

        return h