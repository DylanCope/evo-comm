from pathlib import Path

import hydra
from omegaconf import OmegaConf


def get_current_hydra_output_dir() -> Path:
    conf = hydra.core.hydra_config.HydraConfig.get()
    return Path(conf.runtime.output_dir)


def load_config(experiment_dir: str):
    return OmegaConf.load(f'{experiment_dir}/.hydra/config.yaml')
