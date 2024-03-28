from pathlib import Path
import hydra


def get_current_hydra_output_dir() -> Path:
    conf = hydra.core.hydra_config.HydraConfig.get()
    return Path(conf.runtime.output_dir)
