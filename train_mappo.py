import json
import hydra
from omegaconf import OmegaConf

from jaxevocomm.train.mappo import MAPPO
from jaxevocomm.utils.hydra_utils import get_current_hydra_output_dir
from jaxevocomm.train.callback import (
    ChainedCallback, WandbCallback, Checkpointer, MetricsLogger
)


@hydra.main(version_base=None,
            config_path="config/mappo_homogenous_rnn",
            config_name="mpe")
def main(config):
    config = OmegaConf.to_container(config)
    print('Config:\n', json.dumps(config, indent=4))

    output_dir = get_current_hydra_output_dir()
    cb = ChainedCallback(
        WandbCallback(tags=["MAPPO", "RNN", config["ENV_NAME"]]),
        Checkpointer(
            output_dir / 'checkpoints',
            max_to_keep=config.get('KEEP_CHECKPOINTS', 1),
            save_interval_steps=config.get('CHECKPOINT_INTERVAL', 20)
        ),
        MetricsLogger(output_dir)
    )

    trainer = MAPPO(config, cb)
    trainer.run()


if __name__=="__main__":
    main()
