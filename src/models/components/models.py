from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.layers import SpectralConv1d, dense_block, get_f


class BaseNN(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.apply(self._init_weights)

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _init_weights(self, module):
        """Define how to initialize weights and biases for each type of layer."""
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.data.size()

            # [Option 1]
            # --- Xavier "truncated" normal + zero bias
            std = np.sqrt(2.0 / (fan_in + fan_out))

            # # [Option 2]
            # # # --- He "truncated" normal + zero bias
            # std = np.sqrt(2.0 / (fan_in))

            torch.nn.init.trunc_normal_(module.weight.data, std=std, mean=0, a=-2, b=2)
            if module.bias is not None:
                module.bias.data.zero_()


class DNN(BaseNN):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 20,
        output_dim: int = 1,
        n_layers: int = 8,
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        layers = [input_dim] + [hidden_dim] * n_layers + [output_dim]

        self.net = dense_block(layers, activation)
        self.apply(self._init_weights)

    def forward(self, xt: list) -> torch.Tensor:
        return self.net(torch.cat(xt, dim=1))


class PINN(BaseNN):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 20,
        output_dim: int = 1,
        n_layers: int = 8,
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        layers = [input_dim] + [hidden_dim] * n_layers + [output_dim]

        self.net = dense_block(layers, activation)
        self.apply(self._init_weights)

    def forward(self, xt: list) -> torch.Tensor:
        return self.net(torch.cat(xt, dim=1))


class DeepONet(BaseNN):
    def __init__(
        self,
        branch_dim: int = 128,
        branch_layers: int = 3,
        trunk_dim: int = 2,
        trunk_layers: int = 3,
        hidden_dim: int = 20,
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        branch_layer = [branch_dim] + [hidden_dim] * branch_layers
        trunk_layer = [trunk_dim] + [hidden_dim] * trunk_layers
        self.num_sensor: int = branch_dim

        self.branch_net = dense_block(branch_layer, activation, f_last=False)
        self.trunk_net = dense_block(trunk_layer, activation, f_last=False)
        self.b0 = nn.Parameter(torch.zeros((1,)), requires_grad=True)

        self.apply(self._init_weights)

    # def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    def forward(self, uy: list) -> torch.Tensor:
        u, y = uy  # (batch, num_input_sensor) / (batch, num_output_sensor, dim_output_sensors)
        if y.ndim == 2:
            y = y.unsqueeze(0)
        # (batch, num_input_sensors) -> (batch, hidden_dim)
        b = self.branch_net(u)
        # (batch, num_output_sensors, dim_output_sensors) -> (batch, num_output_sensors, hidden_dim)
        t = self.trunk_net(y)
        # (batch, num_output_sensors)
        s = torch.einsum("bi, bni -> bn", b, t) + self.b0
        return s


class FNO(BaseNN):
    def __init__(
        self,
        num_step: int = 1,
        n_dimension: int = 1,
        modes: int = 16,
        width: int = 32,
        hidden_dim: int = 128,
        activation="gelu",
    ) -> None:
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.num_step = num_step
        self.n_dimension = n_dimension

        self.modes1: int = modes
        self.width: int = width
        self.fc0 = nn.Linear(
            num_step + n_dimension, self.width
        )  # input channel is : (a(x,t[ti ~ to]), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.activation = get_f(activation)

    def forward(self, axt: list):
        ax, total_step = axt
        if not isinstance(total_step, int):
            total_step = int(total_step[0])

        # step = 1
        preds = ax[0]  # t = t_init (batch, grid_x, num_step)
        for t in range(total_step - self.num_step):
            # (batch, grid_x, 1)
            im = self.forward_step(ax)
            # (batch, grid_x, num_step + t)
            preds = torch.cat([preds, im], -1)
            # (batch, grid_x, num_step), (batch, grid_x, n_dimension)
            ax = (preds[..., t + 1 :], ax[1])

        return preds  # (batch, grid_x, total_step)

    def forward_step(self, ax: list) -> torch.tensor:
        # (batch, resolution, num_step + n_dimension) -> (batch, resolution, width)
        x = self.fc0(torch.cat(ax, -1))
        # (batch, width, resolution)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # (batch, resolution, width)
        x = x.permute(0, 2, 1)
        # (batch, resolution, hidden_dim)
        x = self.fc1(x)
        x = self.activation(x)
        # (batch, resolution, 1)
        x = self.fc2(x)
        return x
