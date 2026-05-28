import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.config_mamba import MambaConfig
from .modules.mixer_seq_simple import MixerModel
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder
from einops import rearrange
from torch.nn.parameter import UninitializedParameter

class EEGMamba(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        config = MambaConfig()
        config.ssm_cfg = {
            "layer": "Mamba2",
            "headdim": 50,
            "d_state": 64,
        }
        self.encoder = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=None,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
        )
        # encoder_layer = TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
        #     activation=F.gelu
        # )
        # self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward_features(self, x, mask=None):
        bz, ch_num, seq_len, patch_size = x.shape
        h = self.patch_embedding(x, mask=mask)
        h = rearrange(h, 'b c l d -> b (c l) d')
        h = self.encoder(h)
        h = rearrange(h, 'b (c l) d -> b c l d', l=seq_len)
        return h  # ← AVANT proj_out

    def forward(self, x, mask=None, return_features=False):
        h = self.forward_features(x, mask)
        if return_features:
            return h
        out = self.proj_out(h)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        # self.norm = nn.InstanceNorm2d(200)
        self.positional_encoding = nn.Sequential(
            # nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4),
            #           groups=d_model, bias=False),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
                      groups=d_model, bias=False),
            # nn.GroupNorm(40, d_model),
            # nn.GELU(),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24), bias=False),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            # nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            # nn.GroupNorm(5, 25),
            # nn.GELU(),
            #
            # nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            # nn.GroupNorm(5, 25),
            # nn.GELU(),
        )
        self.token_proj = nn.LazyLinear(d_model)
        self.spectral_proj = nn.LazyLinear(d_model)

        with torch.no_grad():
            x_init = torch.randn(1, 19, seq_len, in_dim)
            _ = self.forward(x_init, mask=None)

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = rearrange(mask_x, 'b c l d -> b d c l')
        # norm_x = self.norm(mask_x)
        # norm_x = mask_x
        time_x = rearrange(mask_x, 'b d c l -> b (c l) d').unsqueeze(1)

        time_feat = self.proj_in(time_x)                               # [B,25,CL,W]
        time_feat = time_feat.permute(0, 2, 1, 3).contiguous()         # [B,CL,25,W]
        time_feat = time_feat.flatten(start_dim=2)                     # [B,CL,25*W]
        time_emb = self.token_proj(time_feat)                          # [B,CL,d_model]
        time_emb = time_emb.view(bz, ch_num, patch_num, self.d_model)  # [B,C,L,d_model]

        flat = rearrange(mask_x, 'b d c l -> (b c l) d').contiguous()   # [B*C*L, P]
        spec = torch.fft.rfft(flat, dim=-1, norm='forward').abs()       # [B*C*L, P//2+1]
        spec_emb = self.spectral_proj(spec)                             # [B*C*L, d_model]
        spec_emb = spec_emb.view(bz, ch_num, patch_num, self.d_model)   # [B,C,L,d_model]

        patch_emb = time_emb + spec_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb



def _weights_init(m):
    if hasattr(m, "weight") and isinstance(m.weight, UninitializedParameter):
        return

    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = EEGMamba(in_dim=200, out_dim=200, d_model=200,
                        dim_feedforward=800, seq_len=30, n_layer=12, nhead=8).to(device)
    a = torch.randn((64, 19, 30, 200)).to(device)


    with torch.no_grad():
        feats = backbone.forward_features(a)
        out = backbone(a)

    print("Before proj_out:", feats.shape)
    print("After proj_out :", out.shape)
