from typing import Tuple

import pytest

from src.data.create_data import Data_Generator


@pytest.mark.parametrize("num_data", [3])
def test_generator(num_data: int):
    cfg_generator = {
        "target_pde": "Burgers_spectral",
        "xlim": (0.0, +1.0),
        "tlim": (0.0, +1.0),
        "Nx": 512,
        "Nt": 128,
        "coefficient": 0.01,
        "data_dir": "data",
        "backend": "torch",
    }

    data_params = {
        "num_data": num_data,
        "m": 0,
        "sigma": 7**2,
        "tau": 7,
        "gamma": 2.5,
        "is_parallel": False,
    }

    generator: object = Data_Generator(**cfg_generator)
    data = generator.create_data(**data_params)
