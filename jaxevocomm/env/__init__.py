import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper

from jaxevocomm.env.mimicry_comm_env import MimicryCommEnvGridworld
from jaxevocomm.train.mappo import MAPPOWorldStateWrapper


def make_env(config: dict):
    env_name = config['ENV_NAME']
    if env_name.startswith("MPE_"):
        env = jaxmarl.make(config["ENV_NAME"],
                        **config["ENV_KWARGS"])
        env = MAPPOWorldStateWrapper(env)
        env = MPELogWrapper(env)
        return env

    elif env_name == "MimicryCommEnvGridworld":
        env = MimicryCommEnvGridworld(grid_size=config['GRID_SIZE'],
                                      n_agents=config['N_AGENTS'],
                                      n_prey=config['N_PREY'],
                                      n_overlapping_sounds=config['N_OVERLAPPING_SOUNDS'],)
        env = MAPPOWorldStateWrapper(env)
        env = MPELogWrapper(env)
        return env

    raise ValueError(f"Unknown environment {env_name}")
