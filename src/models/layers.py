from typing import List

import torch
import torch.nn as nn


def get_f(name: str = "lrelu") -> nn.Module:
    # choose activation function
    activations = nn.ModuleDict(
        [
            ["lrelu", nn.LeakyReLU(0.1)],
            ["relu", nn.ReLU()],
            ["tanh", nn.Tanh()],
            ["sigmoid", nn.Sigmoid()],
            ["gelu", nn.GELU()],
        ]
    )
    return activations[name]


def dense_block(layers: List, activation: str = None, f_last: bool = True) -> nn.Sequential:
    """stack layers of MLP with activation function."""
    out = list()
    for i, (in_f, out_f) in enumerate(zip(layers, layers[1:])):
        out.append(nn.Linear(in_f, out_f))
        if activation is not None:
            out.append(get_f(activation))
    if not f_last:
        return nn.Sequential(*out[:-1])
    return nn.Sequential(*out)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int) -> None:
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = 1 / (in_channels * out_channels)
        # (in_channel, out_channel, mode)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input: torch.tensor, weights: torch.tensor) -> torch.tensor:
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (batch, width, resolution)
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft: (batch, width, modes)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        # out_ft: (batch, out_channel, 0.5*resolution + 1)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, : self.modes1], self.weights1)

        # Return to physical space
        # x: (batch, width, resolution)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
