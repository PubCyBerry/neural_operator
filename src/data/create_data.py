# import libs
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from pathos.pools import ProcessPool
from scipy import integrate

import src.data.solver as solver

# import user-defined libs
from src.data.function_spaces import GRF
from src.utils.utils import timing


@dataclass
class Data_Generator:
    target_pde: str = "Burgers_spectral"
    xlim: Tuple[float, float] = (0.0, +1.0)
    tlim: Tuple[float, float] = (0.0, +1.0)
    Nx: int = 2**10  # 2^10 = 1024
    Nt: int = 2**9  # 2^9  = 512
    coefficient: float = 0.1
    data_dir: str = "data"
    backend: str = "torch"
    space: callable = GRF(backend)
    pde_solver: callable = getattr(solver, target_pde)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def xs(self):
        return torch.linspace(*self.xlim, self.Nx)

    @property
    def ts(self):
        return torch.linspace(*self.tlim, self.Nt)

    def sample_random_function(
        self,
        num_data: int = 1,
        m: float = 0,
        sigma: float = 7**2,
        tau: float = 7,
        gamma: float = 2.5,
    ) -> torch.tensor:
        """
        return: torch.tensor (num_data, Nx,)
        """
        if num_data == 1:
            u = self.space(
                N=self.Nx // 2, m=m, sigma=sigma, tau=tau, gamma=gamma, device=self.device
            )
            return u[: self.Nx].unsqueeze(0)  # u[:-1]
        else:
            u = torch.cat(
                [
                    self.sample_random_function(m=m, sigma=sigma, tau=tau, gamma=gamma)
                    for _ in range(num_data)
                ],
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
            t=torch.linspace(*self.tlim, self.Nt),
            args=(self.xlim[1] - self.xlim[0], self.coefficient),
            tfirst=True,
        ).T

        # # [Option 2]solve_ivp
        # s = integrate.solve_ivp(
        #     self.pde_solver,
        #     t_span=self.tlim,
        #     y0=(u_0),
        #     args=(self.xlim[1] - self.xlim[0], self.coefficient),
        #     method="LSODA",
        #     t_eval=self.ts,
        # ).y
        return torch.tensor(s)

    @timing
    def create_data(
        self,
        num_data: int = 1000,
        m: float = 0,
        sigma: float = 7**2,
        tau: float = 7,
        gamma: float = 2.5,
        is_parallel: bool = False,
    ):
        u = self.sample_random_function(num_data, m, sigma, tau, gamma)

        if is_parallel:
            p = ProcessPool(nodes=os.cpu_count())
            s = p.map(self.solve_PDE, u.cpu())
        else:
            s = [self.solve_PDE(u_i) for u_i in u.cpu()]
        s = torch.stack(s, dim=0)
        return s

    def save_data(self, filename: str, data: np.array) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        np.savez(
            os.path.join(self.data_dir, filename + ".npz"),
            xs=self.xs,
            ts=self.ts,
            ys=data,
            coefficient=self.coefficient,
        )
        print(f"Data saved at {os.path.join(self.data_dir, filename+'.npz')}")