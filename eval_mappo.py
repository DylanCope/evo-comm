from jaxevocomm.train.mappo import MAPPO
import jax


experiment_dir = 'multirun/2024-03-30/15-34-43/0'
trainer, runner_state = MAPPO.restore(experiment_dir)
print('Successfully restored trainer and runner_state')

N_STEPS = 100
print('Running evaluations...')
trajectories, metrics = trainer.rollout(runner_state, n_envs=128, n_steps=N_STEPS)

print('Mean reward:', metrics['mean_total_reward'])
print('Mean episode length:', metrics['mean_episode_length'])
print('Number of episodes:', metrics['num_episodes'])


agent_0, *_ = trainer.env.agents
ENV_IDX = 0

transitions = [
    jax.tree_map(lambda x: x[step, ENV_IDX],
                 trajectories[agent_0])
    for step in range(N_STEPS)
]

env_states = [
    transition.env_state.env_state
    for transition in transitions
]

from jaxevocomm.env.mce_visualiser import MCEVisualiser

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

visualiser = MCEVisualiser(trainer.env)
visualiser.animate(env_states, f'{experiment_dir}/animation', fps=1)
