from typing import Any, Optional, Tuple

import torch

from src.data.components.utils import make_mesh


class IC:
    """Initial Condition."""

    def __init__(
        self,
        xs: torch.tensor,
        values: torch.tensor,
        num_data: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # device: torch.device = torch.device("cpu"),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        PENDING: how to generate data points?
        - option 1: sample new data when training starts only <- current method(1 vote)
        - option 2: sample new data every step
        PENDING: how to choose data points?
        - option 1: select randomly from fixed given values <- current method(1 vote)
        - option 2: select any random values in domain, with interpolation

        xs: (Nx,) spatial grid
        values: (Nx,) initial value
        num_data: number of data to sample
        """
        assert num_data <= len(
            values
        ), "Collocation number(%d) should be less than or equal to number of values(%d)" % (
            num_data,
            len(values),
        )

        t0: torch.tensor = torch.zeros((1,)).to(xs.device)  # (1,)
        mesh: torch.tensor = make_mesh(xs, t0)  # (Nx, 2)

        idx: torch.tensor = torch.randperm(len(xs))[:num_data]  # (num_data,)
        data = mesh[idx]
        target = values[idx].reshape(-1, 1)
        self.data: torch.tensor = data.to(device)  # (num_data, 2)
        self.target: torch.tensor = target.to(device)  # (num_data, 1)

    def push_data(self):
        return (self.data[:, :1], self.data[:, 1:]), self.target


class BC:
    """(Periodic) Boundary Condition."""

    def __init__(
        self,
        xs: torch.tensor,
        ts: torch.tensor,
        num_data: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        PENDING: how to generate data points?
        - option 1: sample new data when training starts only <- current method
        - option 2: sample new data every step
        PENDING: how to choose data points?
        - option 1: select randomly from fixed given values <- current method
        - option 2: select any random values in domain, with interpolation

        xs: (Nx,) spatial grid
        ts: (Nt,) temporal grid
        num_data: number of data to sample
        """
        assert num_data <= len(
            ts
        ), "Collocation number(%d) should be less than or equal to number of time grids(%d)" % (
            num_data,
            len(ts),
        )
        left: torch.tensor = xs[0].clone()  # (1,)
        right: torch.tensor = xs[-1].clone()  # (1,)
        left_boundary: torch.tensor = make_mesh(left, ts)  # (Nt, 2)
        right_boundary: torch.tensor = make_mesh(right, ts)  # (Nt, 2)

        idx: torch.tensor = torch.randperm(len(ts))[:num_data]  # (num_data,)
        left_data = left_boundary[idx]
        right_data = right_boundary[idx]

        self.left_data: torch.tensor = left_data.to(device)  # (num_data, 2)
        self.right_data: torch.tensor = right_data.to(device)  # (num_data, 2)

    def push_data(self):
        return (self.left_data[:, :1], self.left_data[:, 1:]), (
            self.right_data[:, :1],
            self.right_data[:, 1:],
        )


class Collocator:
    """Collocation method."""

    def __init__(
        self,
        xs: torch.tensor,
        ts: torch.tensor,
        num_data: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        *args: Any,
        **kwds: Any,
    ) -> None:
        """
        PENDING: how to generate data points?
        - option 1: sample new data when training starts only <- current method
        - option 2: sample new data every step

        xs: (Nx,) spatial grid
        ts: (Nt,) temporal grid
        num_data: number of data to sample
        """
        mesh: torch.tensor = make_mesh(xs, ts)  # (Nx x Nt, 2)
        assert num_data <= len(
            mesh
        ), "Collocation number(%d) should be less than or equal to number of total grids(%d)" % (
            num_data,
            len(mesh),
        )

        idx: torch.tensor = torch.randperm(len(mesh))[:num_data]  # (num_data,)
        data = mesh[idx]

        self.data: torch.tensor = data.to(device)  # (num_data, 2)

    def push_data(self):
        return (self.data[:, :1], self.data[:, 1:])
