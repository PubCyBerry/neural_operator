import random
import sys
import time
from functools import wraps
from typing import Any, Optional

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def make_mesh(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    create 2d coordinate
    return: (Nx x Nt, 2)
    """
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx, yy], axis=2)
    return xy.reshape(-1, 2)


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
