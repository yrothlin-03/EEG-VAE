import torch.nn as nn
import torch.nn.functional as F


class ArbitraryChannelAdaptor(nn.Module):
    """Adapt EEG tensors from any input channel count to a fixed channel count.

    Args:
        input: eeg signal with shape (B, C, T), where C can vary at runtime
        output: eeg signal with shape (B, C', T)

    The public interface mirrors ``ChannelAdaptor`` so it can be swapped into
    ``EEGVAE``. ``in_channels`` is accepted for compatibility, but the forward
    pass does not require the runtime input to have that exact number of
    channels.
    """

    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()

        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")

        self.in_channels = int(in_channels) if in_channels is not None else None
        self.out_channels = int(out_channels)
        self.n_layers = int(n_layers)

        layers = []
        current_channels = self.out_channels

        for i in range(self.n_layers):
            next_channels = (
                self.out_channels
                if i == self.n_layers - 1
                else max(1, self.out_channels // self.n_layers * (i + 1))
            )

            layers.append(nn.Conv1d(current_channels, next_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(next_channels))
            layers.append(nn.ReLU(inplace=True))

            current_channels = next_channels

        self.model = nn.Sequential(*layers)

    def _resize_channels(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")

        if x.shape[1] == self.out_channels:
            return x

        x = x.unsqueeze(1)
        x = F.interpolate(
            x,
            size=(self.out_channels, x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(1)

    def forward(self, x):
        x = self._resize_channels(x)
        return self.model(x)
