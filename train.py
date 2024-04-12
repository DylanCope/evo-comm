import json
import hydra
from omegaconf import OmegaConf

from jaxevocomm.train.evo.evo_runner import EvoRunner
from jaxevocomm.train.evo.ckpt_cb import EvoCheckpointer
from jaxevocomm.train.mappo import MAPPO
from jaxevocomm.train.mappo.mappo_no_share import MAPPONoShare
from jaxevocomm.train.mappo.ckpt_cb import MAPPOCheckpointer
from jaxevocomm.train.mappo.metrics import performance_metrics, mce_metrics
from jaxevocomm.utils.hydra_utils import get_current_hydra_output_dir
from jaxevocomm.train.callback import (
    ChainedCallback, WandbCallback, MetricsLogger
)


def create_mappo_trainer(config: dict):

    output_dir = config['OUTPUT_DIR']
    cb = ChainedCallback(
        WandbCallback(tags=[config['ALGORITHM'], "RNN", config["ENV_NAME"]]),
        MAPPOCheckpointer(
            output_dir / 'checkpoints',
            max_to_keep=config.get('KEEP_CHECKPOINTS', 1),
            save_interval_steps=config.get('CHECKPOINT_INTERVAL', 20)
        ),
        MetricsLogger(output_dir)
    )

    metrics = [performance_metrics]

    if config['ENV_NAME'] == 'MimicryCommEnvGridworld':
        metrics.append(mce_metrics)

    return MAPPO(config, metrics, cb)


def create_mappo_no_share_trainer(config: dict):

    output_dir = config['OUTPUT_DIR']
    cb = ChainedCallback(
        WandbCallback(tags=[config['ALGORITHM'], "RNN", config["ENV_NAME"]]),
        MAPPOCheckpointer(
            output_dir / 'checkpoints',
            max_to_keep=config.get('KEEP_CHECKPOINTS', 1),
            save_interval_steps=config.get('CHECKPOINT_INTERVAL', 20)
        ),
        MetricsLogger(output_dir)
    )

    metrics = [performance_metrics]

    if config['ENV_NAME'] == 'MimicryCommEnvGridworld':
        metrics.append(mce_metrics)

    return MAPPONoShare(config, metrics, cb)


def create_evo_runner(config: dict):
    output_dir = config['OUTPUT_DIR']
    cb = ChainedCallback(
        WandbCallback(tags=[config['ALGORITHM'], "RNN", config["ENV_NAME"]]),
        EvoCheckpointer(
            output_dir / 'checkpoints',
            max_to_keep=config.get('KEEP_CHECKPOINTS', 1),
            save_interval_steps=config.get('CHECKPOINT_INTERVAL', 20)
        ),
        MetricsLogger(output_dir)
    )

    return EvoRunner(config, cb)


@hydra.main(version_base=None,
            config_path="config",
            config_name="mcrg_mappo")
def main(config):
    config = OmegaConf.to_container(config)
    print('Config:\n', json.dumps(config, indent=4))

    config['OUTPUT_DIR'] = get_current_hydra_output_dir()

    algorithm = config.get('ALGORITHM', 'MAPPO')
    if algorithm == 'MAPPO':
        trainer = create_mappo_trainer(config)
    elif algorithm == 'MAPPO_NoShare':
        trainer = create_mappo_no_share_trainer(config)
    elif algorithm == 'EVO':
        trainer = create_evo_runner(config)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    trainer.run()


if __name__=="__main__":
    main()
