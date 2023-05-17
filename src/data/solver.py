import os.path as osp
from typing import Tuple

import numpy as np
from scipy import fftpack, integrate

from src.utils.utils import timing


# Define pseudo-spectral solver
# PDE -> FFT -> ODE
def Burgers_spectral(t: float, u: np.array, period: float, coefficient: float) -> np.array:
    """solve Burgers equation with spectral method.

    u_t + u * u_x = \nu u_xx
    u: u(x, t_i), shape: (num_x)
    period: assumed period of sequence
    coefficient, nu: kinematic viscosity
    """
    # [option 1]
    u_x = fftpack.diff(u, order=1, period=period)  # first derivative
    u_xx = fftpack.diff(u, order=2, period=period)  # second derivative
    # u_x = fftpack.diff(u, order=1)  # first derivative
    # u_xx = fftpack.diff(u, order=2)  # second derivative
    u_t = -u * u_x + coefficient * u_xx

    # # [option 2]
    # u2_x = 0.5*fftpack.diff(u**2, period=period)
    # u_xx = fftpack.diff(u, order=2, period=period)  # second derivative
    # u_t = -u2_x+ coefficient * u_xx

    return u_t


# Define pseudo-spectral solver
# PDE -> FFT -> ODE
def KdV_spectral(t: float, u: np.array, period: float, coefficient: float) -> np.array:
    r"""
    solve Korteweg de Vries equation with spectral method
    u_t + u * u_x + \delta^2 * u_xxx = 0
    u: u(x, t_i), shape: (num_x)
    period: assumed period of sequence
    coefficient: kinematic viscosity
    """
    u_x = fftpack.diff(u, order=1, period=period)
    u_xxx = fftpack.diff(u, order=3, period=period)
    u_t = -u * u_x - coefficient * u_xxx
    return u_t


@timing
def solve_equation(
    target_pde: str = KdV_spectral,
    xlim: Tuple[float, float] = (0.0, +2.0),
    tlim: Tuple[float, float] = (0.0, +8.0),
    Nx: int = 128,
    Nt: int = 500,
    coefficient: float = 0.022**2,
    u_0: callable = lambda x: np.cos(np.pi * x),
    data_dir: str = "data",
    filename: str = None,
    mode: str = "save",
):
    if filename is not None:
        file_path: str = osp.join(data_dir, filename + ".npz")

    if isinstance(target_pde, str):
        target_pde = globals()[target_pde]

    # if mode is [load], retrieve values from existing file.
    if mode == "load":
        if osp.exists(file_path):
            with np.load(osp.join(data_dir, filename + ".npz")) as data:
                xs, ts, U = data["xs"], data["ts"], data["U"]
            return xs, ts, U

    # create spatial & temporal grid
    period = np.diff(xlim).item()
    xs = np.linspace(*xlim, Nx, endpoint=False)
    ts = np.linspace(*tlim, Nt)

    # # solve equation with solve_ivp()
    # U = integrate.solve_ivp(
    #     fun=KdV_spectral,
    #     t_span=tlim,
    #     y0=u_0(xs) if callable(u_0) else u_0,
    #     args=(period, coefficient),
    #     method="RK45",
    #     t_eval=ts,
    # ).y

    # solve equation with odeint()
    U = integrate.odeint(
        func=target_pde,
        y0=u_0(xs) if callable(u_0) else u_0,
        t=ts,
        args=(period, coefficient),
        tfirst=True,
    ).T

    # if mode is [save], save values as .npz file.
    if mode == "save":
        np.savez(file_path, xs=xs, ts=ts, U=U)
    return xs, ts, U


if __name__ == "__main__":
    pass
