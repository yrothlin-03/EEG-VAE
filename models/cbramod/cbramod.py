import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import UninitializedParameter
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.d_model = d_model

        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model,
                      kernel_size=(19, 7), stride=(1, 1),
                      padding=(9, 3), groups=d_model),
        )

        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )

        # proj “agnostique”: infère 25*W automatiquement au 1er forward
        self.token_proj = nn.LazyLinear(d_model)
        self.spectral_proj = nn.LazyLinear(d_model)

    def forward(self, x, mask=None):
        # x: [B, C, P, S]
        bz, ch_num, patch_num, patch_size = x.shape

        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # ---- time/proj path ----
        mask_x2d = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)  # [B,1,N,S]
        z = self.proj_in(mask_x2d)  # [B,25,N,W]

        # flatten [25,W] -> [B,N,25*W] puis proj -> d_model
        z = z.permute(0, 2, 1, 3).contiguous()          # [B,N,25,W]
        z = z.flatten(start_dim=2)                      # [B,N,25*W]
        patch_emb = self.token_proj(z)                  # [B,N,d_model]
        patch_emb = patch_emb.view(bz, ch_num, patch_num, self.d_model)

        # ---- spectral path ----
        flat = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spec = torch.fft.rfft(flat, dim=-1, norm='forward').abs()   # [B*C*P, S//2+1]
        spec_emb = self.spectral_proj(spec)                          # LazyLinear -> d_model
        spec_emb = spec_emb.view(bz, ch_num, patch_num, self.d_model)

        patch_emb = patch_emb + spec_emb

        # ---- positional encoding (depthwise conv2d) ----
        pos = self.positional_encoding(patch_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        patch_emb = patch_emb + pos

        return patch_emb


def _weights_init(m):
    # skip Lazy modules not initialized yet
    if hasattr(m, "weight") and isinstance(m.weight, UninitializedParameter):
        return

    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)




class CBraMod(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8):
        super().__init__()
        self.seq_len = seq_len
        self.patch_embedding = PatchEmbedding(in_dim, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        x_init = torch.randn((1, 16, seq_len, in_dim))
        with torch.no_grad():
            _ = self.patch_embedding(x_init)
        self.apply(_weights_init)

    def forward(self, x, mask=None, return_features=False):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        if return_features:
            return feats
            
        out = self.proj_out(feats)

        return out
    



if __name__ == '__main__':

    model = CBraMod(in_dim=50, out_dim=50, d_model=200, dim_feedforward=800, seq_len=10, n_layer=12,
                    nhead=8)

    a = torch.randn((8, 16, 10, 50))
    b = model(a, return_features=True)
    c= model(a)
    print(a.shape, b.shape, c.shape)
