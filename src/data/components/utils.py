import torch

def make_mesh(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    create 2d coordinate
    return: (Nx x Nt, 2)
    """
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx, yy], axis=2)
    return xy.reshape(-1, 2)