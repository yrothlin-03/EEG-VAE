import torch
import torch.nn as nn


class ChannelAdaptor(nn.Module):
    """
    Args:
        input: eeg signal with shape (B, C, T)
        output: eeg signal with shape (B, C', T)
    """

    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(n_layers):
            next_channels = out_channels if i == n_layers - 1 else out_channels // n_layers * (i + 1)

            layers.append(nn.Conv1d(current_channels, next_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(next_channels))
            layers.append(nn.ReLU(inplace=True))

            current_channels = next_channels

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)