import os
import os.path as osp
from typing import Any, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

# user-defined libs
from src.utils.closures import Closure
from src.utils.utils import save_checkpoint, set_progress_bar, set_seed

# set pythonpath
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    closure: callable,
    progress_bar: Any,
    task: Any,
):
    """
    model: model to train
    loader: dataloader for training
    optimizer: optimizer for training
    writer: logger (Tensorborad / WandB / ...)
    epoch: current epoch
    device: device to run (cpu/gpu)
    closure: Loss function corresponding to each model
    progress_bar: progress bar counter
    task: label text for progress bar
    """
    # loss calculation reference:
    # https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L287
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        loss = closure(model, data, target, device)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(data)
        progress_bar.update(task, advance=1, description=f"[green]Train Loss: {loss.item():.4e}")

    train_loss /= len(loader.dataset)
    # log metrics
    writer.add_scalar("Loss/train", train_loss, epoch)


def test(
    model: torch.nn.Module,
    loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    closure: callable,
    progress_bar: Any,
    task: Any,
):
    """
    model: model to test
    loader: dataloader for testing
    writer: logger (Tensorborad / WandB / ...)
    epoch: current epoch
    device: device to run (cpu/gpu)
    closure: Loss function corresponding to each model
    progress_bar: progress bar counter
    task: label text for progress bar
    """
    # loss calculation reference:
    # https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L287
    model.eval()
    test_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        loss = closure(model, data, target, device)
        test_loss += loss.item() * len(data)

        progress_bar.update(task, advance=1, description=f"[purple]Test Loss: {loss.item():.4e}")
    test_loss /= len(loader.dataset)
    metric = test_loss

    # log metrics
    writer.add_scalar("Loss/test", test_loss, epoch)
    return metric


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # set seed for reproducibility
    set_seed(cfg.seed)

    # set train parameters
    train_test_split: Tuple[float, float] = (0.8, 0.2)
    best_metric: float = 1e10  # [Choice] Current metric : <Test Loss>

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(
        osp.join(cfg.paths.output_dir, "tensorboard"), comment=cfg.get("comment", "")
    )

    model: torch.nn.Module = hydra.utils.instantiate(cfg.model).to(device)
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=model.parameters()
    )
    scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
        cfg.scheduler, optimizer=optimizer
    )

    # create checkpoint_dir
    checkpoint_dir: str = osp.join(cfg.paths.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)
    closure: callable = Closure(model_name=model.__class__.__name__, dataset=dataset)
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=train_test_split,
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = hydra.utils.instantiate(cfg.loader, dataset=train_dataset, shuffle=True)
    test_loader = hydra.utils.instantiate(cfg.loader, dataset=test_dataset, shuffle=False)

    with set_progress_bar() as p:
        main_task = p.add_task("Main Loop", total=cfg.epochs)
        train_task = p.add_task("Train Loop", total=len(train_loader))
        test_task = p.add_task("Test Loop", total=len(test_loader))
        for e in range(cfg.epochs):
            # train model
            train(model, train_loader, optimizer, writer, e, device, closure, p, train_task)
            if (e + 1) < cfg.epochs:
                p.reset(train_task)

            # validate model
            metric = test(model, test_loader, writer, e, device, closure, p, test_task)
            if (e + 1) < cfg.epochs:
                p.reset(test_task)

            # update learning rate
            scheduler.step()
            # check improvements
            is_best = metric < best_metric
            best_metric = min(metric, best_metric)
            # save last & best model
            save_checkpoint(
                e + 1,
                model,
                best_metric,
                optimizer,
                is_best,
                checkpoint_dir,
                f"{model.__class__.__name__}.pth",
                cfg.path.model_dir,
                cfg.model,
            )
            p.update(main_task, advance=1, description=f"[yellow]Best Metric: {best_metric:.4e}")
    return best_metric


if __name__ == "__main__":
    main()
