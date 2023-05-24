# import libs
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pathos.pools import ProcessPool
from scipy import integrate

import src.data.components.solver as solver

# import user-defined libs
from src.data.components.function_spaces import GRF
from src.data.components.utils import make_mesh
from src.utils.utils import timing


@dataclass
class Data_Generator:
    # pde params
    target_pde: str = "Burgers_spectral"
    xlim: Tuple[float, float] = (0, 1)
    tlim: Tuple[float, float] = (0, 1)
    Nx: int = 1024
    Nt: int = 512
    coefficient: float = 0.01
    backend: str = "torch"
    device: torch.device = torch.device("cpu")
    # data params
    pde_solver: callable = getattr(solver, target_pde)
    space: callable = GRF(backend)
    m: float = 0.0
    sigma: float = 49.0
    tau: float = 7.0
    gamma: float = 2.5
    is_parallel: bool = False
    data_dir: Path = Path("data")

    @property
    def full_name(self):
        return {
            "DNN": "Deep Neural Network",
            "PINN": "Physics Informed Neural Network",
            "DeepONet": "Deep Operator Network",
            "FNO": "Fourier Neural Operator",
        }

    @property
    def xs(self):
        return torch.linspace(*self.xlim, self.Nx)

    @property
    def ts(self):
        return torch.linspace(*self.tlim, self.Nt)

    @property
    def mesh(self):
        return make_mesh(self.xs, self.ts)

    @property
    def data_params(self):
        return {
            "m": self.m,
            "sigma": self.sigma,
            "tau": self.tau,
            "gamma": self.gamma,
        }

    def sample_random_function(self, num_data: int = 1) -> torch.tensor:
        """
        return: torch.tensor (num_data, Nx,)
        """
        if num_data == 1:
            u = self.space(N=self.Nx // 2, **self.data_params)
            return u[: self.Nx].unsqueeze(0)  # u[:-1]
        else:
            u = torch.cat(
                [self.sample_random_function(**self.data_params) for _ in range(num_data)],
                dim=0,
            )
            return u

    def solve_PDE(self, u_0: torch.Tensor) -> torch.tensor:
        """
        u_0: initial condition (Nx,)
        return: total solution (Nx, Nt)
        """
        # [Option 1]odeint
        s = integrate.odeint(
            self.pde_solver,
            y0=u_0,
            t=self.ts,
            args=(self.xlim[1] - self.xlim[0], self.coefficient),
            tfirst=True,
        ).T

        # # [Option 2]solve_ivp
        # # Not working on Windows
        # s = integrate.solve_ivp(
        #     self.pde_solver,
        #     t_span=self.tlim,
        #     y0=(u_0),
        #     args=(self.xlim[1] - self.xlim[0], self.coefficient),
        #     method="LSODA",
        #     t_eval=self.ts,
        # ).y
        return torch.Tensor(s)

    @timing
    def create_data(self, num_data: int = 1000):
        u = self.sample_random_function(num_data)

        if self.is_parallel:
            p = ProcessPool(nodes=os.cpu_count())
            s = p.map(self.solve_PDE, u.cpu())
        else:
            s = [self.solve_PDE(u_i) for u_i in u.cpu()]
        s = torch.stack(s, dim=0)
        return s

    def save_data(self, filename: str, data: torch.tensor) -> None:
        file_path = (Path(self.data_dir) / filename).with_suffix(".npz")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            file_path,
            xs=self.xs,
            ts=self.ts,
            ys=data,
            coefficient=self.coefficient,
            data_params=self.data_params,
        )
        print(f"Data saved at {file_path}")
