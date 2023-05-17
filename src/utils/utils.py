import os.path as osp
import random
import shutil
import sys
import time
from functools import wraps
from typing import Any, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def set_progress_bar():
    # Define custom progress bar
    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[progress.description]{task.description}"),
    )
    return progress_bar


def set_seed(seed: int = 41) -> None:
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def timing(f: callable) -> callable:
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        print(f"{f.__name__!r} took {end_time - start_time:f} s\n")
        sys.stdout.flush()
        return result

    return wrapper


def save_checkpoint(
    epoch: Optional[int],
    model: torch.nn.Module,
    best_metric: float,
    optimizer: torch.optim.Optimizer,
    is_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint.pth",
    model_dir: str = None,
    cfg: DictConfig = None,
) -> None:
    """save model.

    # ref:
    # https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L287
    """

    checkpoint_path: str = osp.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    if is_best:
        shutil.copyfile(checkpoint_path, checkpoint_path.replace(filename, "best_" + filename))
        if model_dir is not None:
            shutil.copyfile(checkpoint_path, osp.join(model_dir, filename))
            OmegaConf.save(cfg, osp.join(model_dir, filename.replace(".pth", "yaml")))


def load_checkpoint(
    cfg: DictConfig,
    ckpt_path: str = "checkpoint.pth",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.nn.Module:
    """load model."""
    checkpoint = torch.load(ckpt_path)
    model = hydra.utils.instantiate(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def make_mesh(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    create 2d coordinate
    return: (Nx x Nt, 2)
    """
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx, yy], axis=2)
    return xy.reshape(-1, 2)
