# Path
from pathlib import Path

# Config
import hydra
import pyrootutils

# PyTorch
import torch
from omegaconf import DictConfig, OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Plotting
import matplotlib.pyplot as plt

from src.utils.plotting import animate_solution, plot_solution


class Helper:
    def __init__(self, cfg: DictConfig) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.load_generator()

    def load_generator(self):
        self.generator = hydra.utils.instantiate(self.cfg.generator)
        self.shape = (self.generator.Nx, self.generator.Nt)

    def load_model(self):
        model_name = self.cfg.model_name
        which = self.cfg.which
        device = self.device
        model_path = Path(self.cfg.ckpt_dir) / model_name
        ckpt_path = model_path.with_stem(model_path.stem + f"_{which}").with_suffix(".ckpt")
        yaml_path = model_path.with_suffix(".yaml")
        model_cfg = OmegaConf.load(yaml_path)
        self.model = (
            hydra.utils.instantiate(model_cfg).load_from_checkpoint(ckpt_path).net.to(device)
        )

    def load_data(self):
        device = self.device
        model_name = self.cfg.model_name
        s = self.generator.create_data(num_data=1).squeeze()
        model_name = model_name.lower()
        if model_name == "deeponet":
            m = self.model.num_sensor  # num_input_sensors
            sub = len(s) // m
            u = s[::sub, 0].unsqueeze(0)
            y = self.generator.mesh
            self.data = (u.to(device), y.to(device))
        elif model_name == "fno":
            init_t = self.model.num_step
            u = s[:, :init_t].unsqueeze(0)
            y = self.generator.xs.view(1, -1, 1)
            self.data = ((u.to(device), y.to(device)), self.generator.Nt)
        else:
            u = None
            y = self.generator.mesh
            self.data = (y[:, :1].to(device), y[:, 1:].to(device))
        self.target = s

    def predict(self):
        device = self.device
        data = self.data
        model = self.model.to(device)
        model.eval()
        preds = model(data)
        self.preds = preds.view(self.shape).detach().cpu()

    def plot_output(self):
        gen = self.generator
        xlim = gen.xlim
        tlim = gen.tlim

        target = self.target
        preds = self.preds

        plot_solution(xlim, tlim, target, preds, **self.cfg.plot)
        animate_solution(xlim, tlim, target, preds, **self.cfg.animate)


@hydra.main(config_path="../configs", config_name="eval.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    helper = Helper(cfg=cfg)
    helper.load_model()
    helper.load_data()
    helper.predict()
    helper.plot_output()


if __name__ == "__main__":
    main()
