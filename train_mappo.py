import json

import hydra
from omegaconf import OmegaConf
from jaxevocomm.env.mimicry_comm_env import MimicryCommEnvGridworld
import jaxmarl

from jaxevocomm.callback import WandbCallback
from jaxevocomm.mappo import MAPPO
from jaxevocomm.mappo_state_wrapper import MAPPOWorldStateWrapper
from jaxmarl.wrappers.baselines import MPELogWrapper


def make_env(config: dict):
    env_name = config['ENV_NAME']
    if env_name.startswith("MPE_"):
        env = jaxmarl.make(config["ENV_NAME"],
                        **config["ENV_KWARGS"])
        env = MAPPOWorldStateWrapper(env)
        env = MPELogWrapper(env)
        return env

    if env_name == "MimicryCommEnvGridworld":
        env = MimicryCommEnvGridworld(grid_size=config['GRID_SIZE'],
                                      n_agents=config['N_AGENTS'],
                                      n_prey=config['N_PREY'],
                                      n_overlapping_sounds=config['N_OVERLAPPING_SOUNDS'],)
        env = MAPPOWorldStateWrapper(env)
        env = MPELogWrapper(env)
        return env

    raise ValueError(f"Unknown environment {env_name}")


@hydra.main(version_base=None,
            config_path="config",
            config_name="mappo_homogenous_rnn_mpe")
def main(config):
    config = OmegaConf.to_container(config)
    print('Config:\n', json.dumps(config, indent=4))
    env = make_env(config)
    cb = WandbCallback()
    trainer = MAPPO(env, config, cb)
    trainer.run()


if __name__=="__main__":
    main()
