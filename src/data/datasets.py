import os.path as osp
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.collocations import BC, IC, Collocator
from src.utils.utils import make_mesh


class BaseDataset(Dataset):
    """Dataset Baseline All Datasets are inherited from this class."""

    def __init__(self, data_dir: str, filename: str, *args: Any, **kwargs: Any) -> None:
        '''
        all Dataset inherit this class
        Note: torch.tensor has operand dtype(could be 64bit), but torch.Tensor has 32bit dtype
        '''
        data_path: str = osp.join(data_dir, filename + ".npz")
        if not osp.exists(data_path):
            raise FileNotFoundError

        data = np.load(data_path)
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xs: torch.tensor = torch.Tensor(data["xs"]).to(device)  # (Nx,)
        self.ts: torch.tensor = torch.Tensor(data["ts"]).to(device)  # (Nt,)
        self.ys: torch.tensor = torch.Tensor(data["ys"]).to(device)  # (num_data, Nx, Nt)
        self.coefficient: float = data["coefficient"]

        self.mesh: torch.tensor = make_mesh(self.xs, self.ts)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def reduce_data(self, num_data: int, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        Nx: int = len(self.xs)
        Nt: int = len(self.ts)
        idx: torch.tensor = torch.randperm(Nx * Nt)[:num_data]
        x: torch.tensor = x[idx, :]
        y: torch.tensor = y[idx, :]
        return x, y


class DNNDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        filename: str,
        num_data: int = 0,
        idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_dir, filename, *args, **kwargs)

        # (Nx x Nt, 2)
        data = make_mesh(self.xs, self.ts)

        # (Nx x Nt, 1)
        target = self.ys[idx].reshape(-1, 1)

        if num_data > 0:
            assert num_data <= len(data), "maximum number of data is %d" % (len(data))
            data, target = self.reduce_data(num_data, data, target)

        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, t = self.data[idx, :1], self.data[idx, 1:]
        target = self.target[idx]
        return (x, t), target


class PINNDataset(DNNDataset):
    def __init__(
        self,
        data_dir: str,
        filename: str,
        num_data: int = 0,
        idx: int = 0,
        Ni: int = 100,
        Nb: int = 100,
        Nu: int = 100,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_dir, filename, num_data, idx, *args, **kwargs)
        # (Nx x Nt, 2)
        data = make_mesh(self.xs, self.ts)

        # (Nx x Nt, 1)
        target = self.ys[idx].reshape(-1, 1)

        # IC
        self.ic_data = IC(self.xs, self.ys[idx][:, :1], num_data=Ni)

        # BC
        self.bc_data = BC(self.xs, self.ts, num_data=Nb)

        # Collocation Data
        self.col_data = Collocator(self.xs, self.ts, num_data=Nu)

        if num_data > 0:
            assert num_data <= len(data), "maximum number of data is %d" % (len(data))
            data, target = self.reduce_data(num_data, data, target)

        self.data = data
        self.target = target

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        x, t = self.data[idx, :1], self.data[idx, 1:]
        target = self.target[idx]
        return (x, t), target


class DeepONetDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        filename: str,
        num_input_sensors: int = 128,
        num_output_sensors: int = 100,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        num_input_sensors = train data resolution
        num_output_sensors = number of target points
        """
        # TODO: If we re-sample indices each time, is it unaligned data?
        super().__init__(data_dir, filename, *args, **kwargs)

        Nx: int = len(self.xs)
        Nt: int = len(self.ts)

        # prepare u
        # (num_data, num_input_sensors)
        sub: int = len(self.xs) // num_input_sensors  # subsampling
        self.u = self.ys[:, ::sub, 0]  # initial condition

        assert self.u.shape[1] == num_input_sensors, "size mismatch: expected %d, took %d" % (
            num_input_sensors,
            self.u.shape[1],
        )

        # prepare y
        # (Nx x Nt, 2)
        sensor_idx: torch.tensor = torch.randperm(Nx * Nt)[:num_output_sensors]
        self.y: torch.tensor = self.mesh[sensor_idx]

        # prepare s
        # (num_data, num_output_sensors)
        s = self.ys.reshape(-1, Nx * Nt)
        self.s: torch.tensor = s[:, sensor_idx]

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = self.u[idx]
        y = self.y
        s = self.s[idx]
        return (u, y), s


if __name__ == "__main__":
    pass
