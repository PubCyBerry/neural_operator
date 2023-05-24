# Gaussian Random Field(GRF)
from typing import Any

import numpy as np
import torch


class GRF:
    def __init__(self, backend: str = "torch") -> None:
        """
        Gaussian Random Field(GRF)
        N: number of grids
        m: mean vector
        gamma: regularity of random field
        tau: inverse length scale of random field
        sigma: scaling factor
        condition: boundary condition
        """
        if backend == "numpy":
            self.grf_func: callable = self.GRF_numpy
        elif backend == "torch":
            self.grf_func: callable = self.GRF_torch
        # elif backend == 'jax':
        #     self.grf_func:callable = self.GRF_jax
        else:
            raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.grf_func(*args, **kwargs)

    def GRF_numpy(
        self,
        N: int = 2**13,
        m: float = 0,
        gamma: float = 2.5,
        tau: float = 7,
        sigma: float = 7**2,
        condition: str = "periodic",
        *args: Any,
        **kwargs: Any
    ) -> np.array:
        eigs = np.sqrt(2) * sigma * ((2 * np.pi * np.arange(N)) ** 2 + tau**2) ** (-gamma / 2)

        xi_alpha = np.random.normal(loc=0, scale=1, size=(N))
        alpha = eigs * xi_alpha

        xi_beta = np.random.normal(loc=0, scale=1, size=(N))
        beta = eigs * xi_beta

        a = +0.5 * alpha
        b = -0.5 * beta

        c = np.concatenate([np.flipud(a) - 1j * np.flipud(b), [m + 0j], a + 1j * b])
        c = np.fft.ifftshift(c)

        mu = np.fft.ifft(c).real * N
        return mu

    def GRF_torch(
        self,
        N: int = 2**13,
        m: float = 0,
        gamma: float = 2.5,
        tau: float = 7,
        sigma: float = 7**2,
        condition: str = "periodic",
        device: torch.device = torch.device("cuda"),
        *args: Any,
        **kwargs: Any
    ) -> torch.tensor:
        eigs = (
            torch.sqrt(torch.tensor(2))
            * sigma
            * ((2 * torch.tensor(np.pi).to(device) * torch.arange(N).to(device)) ** 2 + tau**2)
            ** (-gamma / 2)
        )

        xi_alpha = torch.randn(N).to(device)
        alpha = eigs * xi_alpha
        xi_beta = torch.randn(N).to(device)
        beta = eigs * xi_beta

        a = +0.5 * alpha
        b = -0.5 * beta

        c = torch.cat(
            [
                torch.flip(a, dims=[0]) - 1j * torch.flip(b, dims=[0]),
                torch.tensor([m + 0j], dtype=torch.complex64).to(device),
                a + 1j * b,
            ]
        )
        c = torch.fft.ifftshift(c)

        mu = torch.fft.ifft(c).real * N
        return mu

    # import jax.numpy as jnp
    # from jax import random
    # def GRF_jax(self, key, freqs=jnp.arange(0, 1024+1), m=0, gamma=2.5, tau=7, sigma=7**2):
    #     # jax does not allow dynamic shape array
    #     eigs = jnp.sqrt(2) * sigma * ((2 * jnp.pi * freqs) ** 2 + tau**2) ** (-gamma / 2)

    #     xi_alpha = random.normal(key, shape=(len(freqs),))
    #     alpha = eigs * xi_alpha

    #     xi_beta = random.normal(key, shape=(len(freqs),))
    #     beta = eigs * xi_beta

    #     a = +0.5 * alpha
    #     b = -0.5 * beta

    #     c = jnp.concatenate([jnp.flipud(a) - 1j * jnp.flipud(b), jnp.array([m + 0j]), a + 1j * b])
    #     c = jnp.fft.ifftshift(c)

    #     mu = jnp.fft.ifft(c).real * len(freqs)
    #     return mu
