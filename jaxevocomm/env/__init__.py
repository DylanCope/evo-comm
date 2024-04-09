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
        agent_only_sounds = config['N_AGENT_SOUNDS'] - overlapping_sounds
        prey_only_sounds = config['N_PREY_SOUNDS'] - overlapping_sounds
        prey_noise_prob = config.get('PREY_NOISE_PROB', 0.25)

        env = MimicryCommEnvGridworld(
            grid_size=config['GRID_SIZE'],
            n_agents=config['N_AGENTS'],
            n_prey=config['N_PREY'],
            prey_audible_range=config.get('PREY_AUDIBLE_RANGE', 5),
            prey_visible_range=config.get('PREY_VISIBLE_RANGE', 2),
            prey_noise_prob=prey_noise_prob,
            n_overlapping_sounds=overlapping_sounds,
            n_agent_only_sounds=agent_only_sounds,
            n_prey_only_sounds=prey_only_sounds,
            observe_other_agents_pos=config.get('OBSERVE_OTHER_AGENTS_POS', False),
            on_prey_reward=config.get('ON_PREY_REWARD', 0.0),
        )
        env = MPELogWrapper(env)
        return env

    raise ValueError(f"Unknown environment {env_name}")
