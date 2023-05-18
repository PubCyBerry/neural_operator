import hydra
import torch
from omegaconf import DictConfig

# set pythonpath
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="create.yaml")
def main(cfg: DictConfig) -> None:
    """
    1. generate PDE data
    2. save output to data directory
    - hyperparameter for this task refers configs/create.yaml
    """
    generator: object = hydra.utils.instantiate(cfg.generator)
    data: torch.tensor = generator.create_data(**cfg.data_params)
    generator.save_data(cfg.filename, data)


if __name__ == "__main__":
    main()
