# Type Hinting
from typing import Tuple

# Config
import hydra

# set pythonpath
import pyrootutils

# PyTorch
import torch
from omegaconf import DictConfig

# user-defined libs
from src.utils.plotting import animate_solution, plot_solution

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def infer(
    model: torch.nn.Module | str,
    xlim: Tuple[float, float] = (0, 1),
    tlim: Tuple[float, float] = (0, 1),
    Nx: int = 1024,
    Nt: int = 512,
    solution: torch.tensor = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    # if isinstance(model, str):
    #     model: torch.nn.Module = load_checkpoint(model)
    model = None
    model_name: str = model.__class__.__name__

    xs: torch.tensor = torch.linspace(*xlim, Nx).to(device)
    ts: torch.tensor = torch.linspace(*tlim, Nt).to(device)
    # y: torch.tensor = make_mesh(xs, ts)
    y = None

    if model_name == "DeepONet":
        u = solution[:: solution.shape[0] // model.num_sensor, 0].unsqueeze(0).to(device)
        preds = model((u, y))
    elif model_name == "FNO":
        u = solution[:, : model.num_step].unsqueeze(0).to(device)
        y = torch.linspace(*xlim, u.shape[1]).view(1, -1, 1).to(device)
        preds = model(((u, y), Nt))
    else:
        preds = model((y[:, :1], y[:, 1:]))

    preds = preds.view(Nx, Nt).detach().cpu()
    return preds


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # Load Model
    model_name: str = cfg.model_name
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = torch.nn.Module

    # Generate Random Data
    helper: object = hydra.utils.instantiate(cfg.generator)
    solution: torch.tensor = helper.create_data(num_data=1).squeeze()

    # Perform Inference
    preds: torch.tensor = infer(model, helper.xlim, helper.tlim, helper.Nx, helper.Nt, solution)

    # Plot Result
    import matplotlib.pyplot as plt

    fig = plot_solution(helper.xlim, helper.tlim, solution, preds, **cfg.plot)
    plt.close()
    # Animate Result
    anim = animate_solution(helper.xlim, helper.tlim, solution, preds, **cfg.animate)
    plt.close()


if __name__ == "__main__":
    main()
