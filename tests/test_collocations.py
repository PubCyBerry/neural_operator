import pytest
import torch

from src.data.collocations import BC, IC, Collocator


@pytest.mark.parametrize("num_data", [100])
def test_collocations(num_data: int):
    device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nx: int = 512
    Nt: int = 512

    xlim: tuple = (0, 1)
    tlim: tuple = (0, 1)

    xs = torch.linspace(*xlim, Nx).to(device)
    ts = torch.linspace(*tlim, Nt).to(device)
    values = torch.sin(xs)

    Ni: int = 100
    Nb: int = 100
    Nu: int = 5000
    ic = IC(xs, values, Ni)
    bc = BC(xs, ts, Nb)
    cols = Collocator(xs, ts, Nu)
    for data in [ic, bc, cols]:
        print(data.__class__.__name__, "batch", end=" ")
        for d in data.push_data():
            if isinstance(d, tuple):
                print(d[0].size(), d[1].size(), end=" ")
            else:
                print(d.size(), end=" ")
        print()
