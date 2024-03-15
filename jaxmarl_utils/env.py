import jax
import jax.numpy as jnp
import numpy as onp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.mpe.default_params import *
import chex
from gymnax.environments.spaces import Box, Discrete
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib


@struct.dataclass
class State:
    """Basic MPE State"""

    agent_pos: chex.Array  # [n_agents, [x, y]]
    prey_pos: chex.Array  # [n_prey, [x, y]]
    c: chex.Array  # communication state [n_agents + n_prey,]  (int32)
    done: chex.Array  # bool [num_agents, ]
    step: int  # current step


ENV_ACTIONS = [
    'noop', 'up', 'down', 'left', 'right'
]


class CommEnvGridworld(MultiAgentEnv):

    def __init__(self,
                 grid_size: int = 10,
                 n_agents: int = 2,
                 n_prey: int = 1,
                 n_overlapping_sounds: int = 2,
                 n_agent_only_sounds: int = 2,
                 n_prey_only_sounds: int = 2,
                 prey_vision_range: int = 1,
                 prey_sound_range: int = 3,
                 max_steps: int = 50,
                 agents_to_capture_prey: int = 2,
                 capture_reward: float = 10.0,
                 time_penalty: float = -0.1,
                 **kwargs):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_prey = n_prey
        self.n_overlapping_sounds = n_overlapping_sounds
        self.n_agent_only_sounds = n_agent_only_sounds
        self.n_prey_only_sounds = n_prey_only_sounds
        self.prey_sound_range = prey_sound_range
        self.prey_vision_range = prey_vision_range
        self.max_steps = max_steps
        self.agents_to_capture_prey = agents_to_capture_prey
        self.capture_reward = capture_reward
        self.time_penalty = time_penalty

        self.agents = [
            f'agent_{i}' for i in range(self.n_agents)
        ]

        self.action_spaces = {
            agent: Discrete(len(ENV_ACTIONS) + self.n_sounds)
            for agent in self.agents
        }

    @property
    def n_sounds(self):
        return (
            self.n_overlapping_sounds +
            self.n_agent_only_sounds +
            self.n_prey_only_sounds
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        agent_pos_key, prey_pos_key = jax.random.split(key)

        agent_pos = jax.random.randint(
            agent_pos_key, (self.n_agents, 2), 0, self.grid_size
        )
        prey_pos = jax.random.randint(
            prey_pos_key, (self.n_prey, 2), 0, self.grid_size
        )
        state = State(
            agent_pos=agent_pos,
            prey_pos=prey_pos,
            c=jnp.zeros((self.n_agents + self.n_prey,), dtype=jnp.int32),
            done=jnp.full((self.n_agents), False),
            step=0,
        )

        return self.get_obs(state), state

    def get_agent_obs(self, state: State, agent: str) -> jnp.ndarray:
        features = []

        agent_idx = self.agents.index(agent)
        agent_pos = state.agent_pos[agent_idx]
        # agent_pos_1h = jnp.zeros((self.grid_size * 2,))
        # print(agent_pos[0])
        # agent_pos_1h = agent_pos_1h.at(agent_pos[0]).set(1)
        # agent_pos_1h = agent_pos_1h.at(self.grid_size + agent_pos[1]).set(1)

        agent_pos_1h = jax.nn.one_hot(agent_pos, self.grid_size).flatten()

        features.append(agent_pos_1h)

        for prey_pos in state.prey_pos:
            if jnp.sum(jnp.abs(agent_pos - prey_pos)) <= self.prey_vision_range:
                prey_pos_1h = jax.nn.one_hot(prey_pos, self.grid_size).flatten()
            else:
                prey_pos_1h = jnp.zeros((self.grid_size * 2,))
            features.append(prey_pos_1h)

        for axis in range(2):
            sound = jnp.zeros((self.n_sounds,))
            for other_agent_idx in range(self.n_agents):
                agent_sound = state.c[other_agent_idx]
                if agent_idx != other_agent_idx and agent_sound > 0:
                    other_agent_pos = state.agent_pos[other_agent_idx]
                    axis_factor = jnp.sign(agent_pos[axis] - other_agent_pos[axis])
                    sound = sound.at(agent_sound).set(axis_factor)
    
        features.append(sound.flatten())

        return jnp.concatenate(features)

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        pass

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        return {
            agent: self.get_agent_obs(state, agent)
            for agent in self.agents
        }


if __name__ == '__main__':
    env = CommEnvGridworld()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    print(obs)
