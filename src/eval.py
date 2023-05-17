import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.plotting import animate_solution, plot_solution
from src.utils.utils import load_checkpoint


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.ckpt_path, "Checkpoint path should be given explicitly"

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = load_checkpoint(cfg.model, cfg.ckpt_path, device)
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    # Coordinates
    xs: torch.tensor = dataset.xs.cpu()
    ts: torch.tensor = dataset.ts.cpu()
    mesh: torch.tensor = dataset.mesh

    # Total data
    ys: torch.tensor = dataset.ys  # (num_data, Nx, Nt)
    # Initial values
    init: torch.tensor = ys[..., :1]  # (num_data, Nx, 1)

    idx: torch.tensor = torch.randperm(len(ys))[0]

    target: torch.tensor = ys[idx]

    # perform inference
    preds = model(mesh[:, :1], mesh[:, 1:]).detach().cpu().numpy().reshape(len(xs), len(ts))

    # plot result
    img = plot_solution(ts=ts, xs=xs, U=target, U_pred=preds, **cfg.plot)

    # animate result
    anim = animate_solution(x=xs, y=target, y_pred=preds, dt=ts[1] - ts[0], **cfg.animate)


if __name__ == "__main__":
    main()
