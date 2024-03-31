import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper

from jaxevocomm.env.mimicry_comm_env import MimicryCommEnvGridworld


def make_env(config: dict):
    env_name = config['ENV_NAME']
    if env_name.startswith("MPE_"):
        env = jaxmarl.make(config["ENV_NAME"],
                        **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
        return env

    elif env_name == "MimicryCommEnvGridworld":
        overlapping_sounds = config.get('N_OVERLAPPING_SOUNDS', 0)
        total_sounds = config['N_TOTAL_SOUNDS']
        agent_sounds = (total_sounds - overlapping_sounds) // 2
        prey_sounds = total_sounds - agent_sounds - overlapping_sounds

        config['N_AGENT_SOUNDS'] = agent_sounds
        config['N_PREY_SOUNDS'] = prey_sounds

        env = MimicryCommEnvGridworld(grid_size=config['GRID_SIZE'],
                                      n_agents=config['N_AGENTS'],
                                      n_prey=config['N_PREY'],
                                      n_overlapping_sounds=overlapping_sounds,
                                      n_agent_only_sounds=agent_sounds,
                                      n_prey_only_sounds=prey_sounds,)
        env = MPELogWrapper(env)
        return env

    raise ValueError(f"Unknown environment {env_name}")
