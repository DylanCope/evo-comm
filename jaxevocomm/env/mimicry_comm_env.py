import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.mpe.default_params import *
import chex
from gymnax.environments.spaces import Discrete, Box
from flax import struct

from typing import List, Tuple, Dict
from functools import partial


@struct.dataclass
class State:
    agent_pos: chex.Array  # [n_agents, [x, y]]
    prey_pos: chex.Array  # [n_prey, [x, y]]
    c: chex.Array  # communication state [n_agents + n_prey,]  (int32)
    step: int  # current step
    prey_captured: int  # number of prey captured throughout the game


class GridAction:
    N_ACTIONS = 5

    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class MimicryCommEnvGridworld(MultiAgentEnv):

    def __init__(self,
                 grid_size: int = 10,
                 n_agents: int = 2,
                 n_prey: int = 1,
                 n_overlapping_sounds: int = 2,
                 n_agent_only_sounds: int = 2,
                 n_prey_only_sounds: int = 2,
                 prey_vision_range: int = 1,
                 prey_sound_range: int = 5,
                 prey_noise_prob: float = 0.25,
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
        self.prey_noise_prob = prey_noise_prob
        self.max_steps = max_steps
        self.agents_to_capture_prey = agents_to_capture_prey
        self.capture_reward = capture_reward
        self.time_penalty = time_penalty

        self.agents = [
            f'agent_{i}' for i in range(self.n_agents)
        ]

        self.action_spaces = {
            agent: Discrete(GridAction.N_ACTIONS + self.n_agent_sounds)
            for agent in self.agents
        }

        obs, _ = self.reset(jax.random.PRNGKey(0))

        self.observation_spaces = {
            agent: Box(-1.0, 1.0, shape=agent_obs.shape)
            for agent, agent_obs in obs.items()
        }

    @property
    def n_sounds(self):
        # first sound is special silent symbol
        return 1 + (
            self.n_overlapping_sounds +
            self.n_agent_only_sounds +
            self.n_prey_only_sounds
        )

    @property
    def n_prey_sounds(self):
        return (
            self.n_overlapping_sounds +
            self.n_prey_only_sounds
        )

    @property
    def n_agent_sounds(self):
        return (
            self.n_overlapping_sounds +
            self.n_agent_only_sounds
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
            step=0,
            prey_captured=0
        )

        return self.get_obs(state), state

    def _create_sound_feats(self, state: State, agent_idx: int) -> jnp.ndarray:
        agent_pos = state.agent_pos[agent_idx]
        features = []
        for axis in range(2):
            axis_sound_feat = jnp.zeros((self.n_sounds,))

            for other_agent_idx in range(self.n_agents):
                if agent_idx != other_agent_idx:
                    sound_idx = state.c[other_agent_idx]
                    other_agent_sound = jax.nn.one_hot(sound_idx,
                                                       self.n_sounds)
                    is_silent_sound = sound_idx == 0
                    other_agent_pos = state.agent_pos[other_agent_idx]
                    axis_factor = jnp.sign(agent_pos[axis] - other_agent_pos[axis])
                    axis_sound_feat = axis_sound_feat + (
                        (1.0 - is_silent_sound) * axis_factor * other_agent_sound
                    )

            for prey_idx in range(self.n_prey):
                sound_idx = state.c[self.n_agents + prey_idx]
                prey_sound = jax.nn.one_hot(sound_idx, self.n_sounds)
                is_silent_sound = sound_idx == 0
                prey_pos = state.prey_pos[prey_idx]
                prey_dist = jnp.sum(jnp.abs(agent_pos - prey_pos))
                in_audible_range = prey_dist <= self.prey_sound_range
                axis_factor = jnp.sign(agent_pos[axis] - prey_pos[axis])
                axis_sound_feat = axis_sound_feat + (
                    (1.0 - is_silent_sound) * in_audible_range * axis_factor * prey_sound
                )

            features.append(axis_sound_feat)

        return jnp.concatenate(features)

    def get_agent_obs(self, state: State, agent: str) -> jnp.ndarray:
        features = []

        agent_idx = self.agents.index(agent)
        agent_pos = state.agent_pos[agent_idx]

        agent_pos_1h = jax.nn.one_hot(agent_pos, self.grid_size).flatten()

        features.append(agent_pos_1h)

        for prey_pos in state.prey_pos:
            manhattan_dist = jnp.sum(jnp.abs(agent_pos - prey_pos))
            is_visible_mask = manhattan_dist <= self.prey_vision_range
            prey_pos_1h = is_visible_mask * jax.nn.one_hot(prey_pos, self.grid_size).flatten()
            features.append(prey_pos_1h)

        features.append(self._create_sound_feats(state, agent_idx))

        return jnp.concatenate(features)

    def convert_agent_action_to_sound(self, action: jnp.array) -> jnp.array:
        return jnp.clip(action - GridAction.N_ACTIONS, 0, self.n_agent_sounds)

    def update_sound_state(self, key: chex.PRNGKey, actions: Dict[str, chex.Array]) -> chex.Array:
        agent_sounds = jnp.array([
            self.convert_agent_action_to_sound(a)
            for a in actions.values()
        ])
        key, noise_k1, noise_k2 = jax.random.split(key, 3)
        prey_sounds = (
            jax.random.bernoulli(noise_k1, self.prey_noise_prob, (self.n_prey,))
            * jax.random.randint(noise_k2, (self.n_prey,), 0, self.n_prey_sounds + 1)
        )

        return jnp.concatenate([agent_sounds, prey_sounds])

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        return {
            agent: self.get_agent_obs(state, agent)
            for agent in self.agents
        }

    def _move_agents(self,
                     state: State,
                     actions: Dict[str, chex.Array]) -> List[chex.Array]:

        new_agent_positions = []
        for agent in self.agents:
            agent_idx = self.agents.index(agent)
            agent_pos = state.agent_pos[agent_idx]
            action = actions[agent]

            UP_DELTA = jnp.array([0, 1])
            DOWN_DELTA = jnp.array([0, -1])
            LEFT_DELTA = jnp.array([-1, 0])
            RIGHT_DELTA = jnp.array([1, 0])

            up_mask = action == GridAction.UP
            down_mask = action == GridAction.DOWN
            left_mask = action == GridAction.LEFT
            right_mask = action == GridAction.RIGHT

            new_agent_pos = agent_pos + (
                up_mask * UP_DELTA +
                down_mask * DOWN_DELTA +
                left_mask * LEFT_DELTA +
                right_mask * RIGHT_DELTA
            )
            new_agent_positions.append(new_agent_pos)
    
        return jnp.vstack(new_agent_positions)

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        key, sound_key = jax.random.split(key)
        new_sound_state = self.update_sound_state(sound_key, actions)

        new_agent_positions = self._move_agents(state, actions)

        reward = jnp.ones((1,)) * (-self.time_penalty)
        total_prey_captured = state.prey_captured.copy()

        new_prey_positions = []
        for prey_pos in state.prey_pos:
            n_agent_on_prey = jnp.sum(jnp.apply_along_axis(
                lambda x: jnp.all(x == prey_pos), 1, new_agent_positions
            ))
            prey_captured = n_agent_on_prey >= self.agents_to_capture_prey
            total_prey_captured = total_prey_captured + prey_captured
            reward = reward + prey_captured * self.capture_reward
            key, new_prey_pos_key = jax.random.split(key)
            new_prey_pos = prey_pos * (1 - prey_captured) + (
                prey_captured * jax.random.randint(
                    new_prey_pos_key, (2,), 0, self.grid_size
                )
            )
            new_prey_positions.append(new_prey_pos)

        new_prey_positions = jnp.vstack(new_prey_positions)

        next_step = state.step + 1

        new_state = state.replace(
            agent_pos=new_agent_positions,
            prey_pos=new_prey_positions,
            c=new_sound_state,
            step=next_step,
        )

        done = jnp.full((self.n_agents), state.step >= self.max_steps)
        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return self.get_obs(new_state), new_state, reward, dones, {}


if __name__ == '__main__':
    env = MimicryCommEnvGridworld()

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    print(state)
    print(obs)

    action_keys = jax.random.split(key, env.n_agents)
    actions = {
        agent: env.action_spaces[agent].sample(k)
        for agent, k in zip(env.agents, action_keys)
    }
    print('Actions:', actions)

    key, step_key = jax.random.split(key)

    obs, state, reward, dones, infos = env.step(step_key, state, actions)

    print(state)
    print(obs)
