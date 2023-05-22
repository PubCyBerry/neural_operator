from typing import Optional
from pathlib import Path
import shutil

import torch
import hydra
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    epoch: Optional[int],
    model: torch.nn.Module,
    best_metric: float,
    optimizer: torch.optim.Optimizer,
    is_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint",
    model_dir: str = None,
    cfg: DictConfig = None,
) -> None:
    """save model.

    # ref:
    # https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L287
    """
    if model_dir is not None:
        model_path: Path = Path(model_dir) / filename

    checkpoint_path: Path = (Path(checkpoint_dir) / filename).with_suffix(".pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    shutil.copyfile(checkpoint_path, model_path)
    if is_best:
        shutil.copyfile(checkpoint_path, checkpoint_path.with_stem("best_" + checkpoint_path.stem))
        shutil.copyfile(checkpoint_path, model_path.with_stem("best_" + model_path.stem))
        OmegaConf.save(cfg, model_path.with_suffix(".yaml"))


def load_checkpoint(
    model_name: str = "checkpoint",
    ckpt_dir: Path = ("models"),
    model_cfg: DictConfig = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    is_training: bool = False,
) -> torch.nn.Module:
    """load model."""
    model_path: Path = Path(ckpt_dir) / model_name
    if model_cfg is None:
        model_cfg: DictConfig = OmegaConf.load(
            model_path.with_stem(model_path.stem.replace("best_", "")).with_suffix(".yaml")
        )

    checkpoint = torch.load(model_path.with_suffix(".pth"))
    model = hydra.utils.instantiate(model_cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if is_training:
        model.train()
    else:
        model.eval()
    return model
