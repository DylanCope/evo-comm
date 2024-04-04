import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments.spaces import Discrete, Box

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.mpe.default_params import *

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
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    N_ACTIONS = 5
    ACTIONS = [
        NOOP, UP, DOWN, LEFT, RIGHT
    ]


X = 0
Y = 1


class MimicryCommEnvGridworld(MultiAgentEnv):

    def __init__(self,
                 grid_size: int = 10,
                 n_agents: int = 2,
                 n_prey: int = 1,
                 n_overlapping_sounds: int = 2,
                 n_agent_only_sounds: int = 2,
                 n_prey_only_sounds: int = 2,
                 prey_visible_range: int = 1,
                 prey_audible_range: int = 5,
                 prey_noise_prob: float = 0.25,
                 max_steps: int = 50,
                 agents_to_capture_prey: int = 2,
                 observe_other_agents_pos: bool = False,
                 capture_reward: float = 10.0,
                 time_penalty: float = 0.1,
                 **kwargs):
        super(MimicryCommEnvGridworld, self).__init__(num_agents = n_agents)
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_prey = n_prey
        self.n_overlapping_sounds = n_overlapping_sounds
        self.n_agent_only_sounds = n_agent_only_sounds
        self.n_prey_only_sounds = n_prey_only_sounds
        self.prey_audible_range = prey_audible_range
        self.prey_visible_range = prey_visible_range
        self.prey_noise_prob = prey_noise_prob
        self.max_steps = max_steps
        self.agents_to_capture_prey = agents_to_capture_prey
        self.capture_reward = capture_reward
        self.time_penalty = time_penalty
        self.observe_other_agents_pos = observe_other_agents_pos

        self.agents = [
            f'agent_{i}' for i in range(self.n_agents)
        ]

        self.action_spaces = {
            agent: Discrete(
                GridAction.N_ACTIONS
                + 1  # silent sound
                + self.n_agent_sounds
            )
            for agent in self.agents
        }

        obs, _ = self.reset(jax.random.PRNGKey(0))

        self.observation_spaces = {
            agent: Box(-1.0, 1.0, shape=agent_obs.shape)
            for agent, agent_obs in obs.items()
        }

    @property
    def n_sounds(self):
        return (
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
            c=jnp.zeros((self.n_agents + self.n_prey,),
                        dtype=jnp.int32),
            step=0,
            prey_captured=0
        )

        return self.get_obs(state), state

    def _create_sound_feats(self,
                            state: State,
                            agent: int | str) -> Dict[str, chex.Array]:
        """
        Creates sound features for a given agent.

        Sound features correspond to the sounds heard by the agent along
        the X and Y axes. For each axis there are two vectors corresponding
        to sounds coming from ahead or behind the agent along that axis.

        Each sound vector is a one-hot vector with an dimension for each
        possible sound that can be heard by the agent. The vector is all
        zeros if the sound is silent or not sensed by the agent.
        """
        if isinstance(agent, str):
            agent_idx = self.agents.index(agent)
        else:
            agent_idx = agent

        agent_pos = state.agent_pos[agent_idx]
        features = dict()

        feat_names = {
            'right': (X, 1),
            'left': (X, -1),
            'up': (Y, 1),
            'down': (Y, -1),
        }

        # each axis and direction combination specifies a sensor feature
        # that detects sounds coming from either side of the agent along each axis
        for feat_name, (axis, direction) in feat_names.items():

            sound_feat = jnp.zeros((self.n_sounds,))

            def _create_sound_vec(sound_val, source_pos):
                # sounds are between 1 and n_sounds, and 0 if silent
                sound_1h_idx = jnp.clip(sound_val - 1, 0, self.n_sounds)
                sound_feat = jax.nn.one_hot(sound_1h_idx, self.n_sounds)

                # +1 if or -1 if sound along the given axis, 0 if not
                axis_factor = jnp.sign(agent_pos[axis] - source_pos[axis])

                # +1 if sound is in the given direction along the given axis 
                correct_direction = jnp.clip(direction * axis_factor, 0, 1)

                # mask for silent sound value
                not_silent = sound_val != 0

                # vector is all zeros if silent sound or not sensed by
                # this sound feature
                return not_silent * correct_direction * sound_feat     

            # add sounds emitted by other agents
            for other_agent_idx in range(self.n_agents):
                if agent_idx != other_agent_idx:
                    sound_val = state.c[other_agent_idx]
                    other_agent_pos = state.agent_pos[other_agent_idx]
                    sound_feat = sound_feat + _create_sound_vec(sound_val,
                                                                other_agent_pos)

            # add sounds emitted by prey
            for prey_idx in range(self.n_prey):
                sound_val = state.c[self.n_agents + prey_idx]
                prey_pos = state.prey_pos[prey_idx]
                prey_dist = jnp.sum(jnp.abs(agent_pos - prey_pos))
                in_audible_range = prey_dist <= self.prey_audible_range
                prey_sound = _create_sound_vec(sound_val, prey_pos)
                sound_feat = sound_feat + in_audible_range * prey_sound

            features[f'sound_feat_{feat_name}'] = sound_feat

        return features

    def _get_position_feats(self,
                            state: State,
                            agent: str) -> Dict[str, chex.Array]:
        features = {}

        agent_idx = self.agents.index(agent)
        agent_pos = state.agent_pos[agent_idx]
        agent_pos_x_1h = jax.nn.one_hot(agent_pos[X], self.grid_size).flatten()
        agent_pos_y_1h = jax.nn.one_hot(agent_pos[Y], self.grid_size).flatten()
        features['agent_pos_x'] = agent_pos_x_1h
        features['agent_pos_y'] = agent_pos_y_1h

        if self.observe_other_agents_pos:
            for other_agent_idx in range(self.n_agents):
                if other_agent_idx != agent_idx:
                    other_agent_pos = state.agent_pos[other_agent_idx]
                    other_agent_pos_x_1h = jax.nn.one_hot(other_agent_pos[X],
                                                          self.grid_size).flatten()
                    other_agent_pos_y_1h = jax.nn.one_hot(other_agent_pos[Y],
                                                          self.grid_size).flatten()
                    features[f'other_agent_{other_agent_idx}_pos_x'] = other_agent_pos_x_1h
                    features[f'other_agent_{other_agent_idx}_pos_y'] = other_agent_pos_y_1h

        for prey_i, prey_pos in enumerate(state.prey_pos):
            manhattan_dist = jnp.sum(jnp.abs(agent_pos - prey_pos))
            is_visible_mask = manhattan_dist <= self.prey_visible_range
            prey_pos_x_1h = jax.nn.one_hot(prey_pos[X], self.grid_size).flatten()
            prey_pos_y_1h = jax.nn.one_hot(prey_pos[Y], self.grid_size).flatten()
            features[f'prey_{prey_i}_pos_x'] = is_visible_mask * prey_pos_x_1h
            features[f'prey_{prey_i}_pos_y'] = is_visible_mask * prey_pos_y_1h

        return features

    def _get_agent_obs_feats(self,
                             state: State,
                             agent: str) -> Dict[str, jnp.ndarray]:
        """
        Creates observation features for a given agent, stored as a dictionary
        of feature names to feature vectors.

        Consists of agent position, prey positions, and sound features.
        Position features are represented as one-hot vectors of the X and Y grid coordinates.
        See _create_sound_feats for sound features.
        
        Args:
            state: current environment state
            agent: agent ID

        Returns:
            features: dictionary of feature names to feature vectors
        """
        features = self._get_position_feats(state, agent)
        features.update(self._create_sound_feats(state, agent))

        return features

    def get_agent_obs(self, state: State, agent: str) -> chex.Array:
        """Applies observation function to state."""
        obs_feat = self._get_agent_obs_feats(state, agent)
        feats = list(obs_feat.values())
        return jnp.concatenate(feats)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        return {
            agent: self.get_agent_obs(state, agent)
            for agent in self.agents
        }

    def convert_agent_action_to_sound(self, action: chex.Array) -> chex.Array:
        return jnp.clip(action - GridAction.N_ACTIONS, 0, self.n_agent_sounds)

    def update_sound_state(self,
                           key: chex.PRNGKey,
                           actions: Dict[str, chex.Array]) -> chex.Array:
        # Get agent sounds from actions
        agent_sounds = jnp.array([
            self.convert_agent_action_to_sound(a)
            for a in actions.values()
        ]).reshape((self.n_agents,))

        # Create prey Sounds
        key, noise_k1, noise_k2 = jax.random.split(key, 3)

        # Prey randomly create one of their possible sounds
        # The prey_noise_prob is the probability that a prey will make a sound
        # If it makes a sound, it will choose one of the sounds from 1 to
        # n_prey_sounds with equal probability
        prey_sounds = (
            jax.random.bernoulli(noise_k1, self.prey_noise_prob, (self.n_prey,))
            * jax.random.randint(noise_k2, (self.n_prey,), 1, self.n_prey_sounds + 1)
        )

        # Agent and prey sounds need to be in the same alphabet, so we add
        # the number of agent only sounds to the prey sounds 
        is_making_sound = prey_sounds != 0
        prey_sounds = (prey_sounds + self.n_agent_only_sounds) * is_making_sound

        # final sound state is concatenation of agent and prey sounds
        return jnp.concatenate([agent_sounds, prey_sounds])

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
            new_agent_pos = jnp.clip(new_agent_pos, 0, self.grid_size - 1)
            new_agent_positions.append(new_agent_pos)
    
        return jnp.vstack(new_agent_positions)

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""

        # if agents or prey are making sounds, update the sound state
        key, sound_key = jax.random.split(key)
        new_sound_state = self.update_sound_state(sound_key, actions)

        # if agents have chosen movement actions, update positions
        new_agent_positions = self._move_agents(state, actions)

        reward = -self.time_penalty
        total_prey_captured = state.prey_captured.copy()

        new_prey_positions = []
        for prey_pos in state.prey_pos:
            n_agent_on_prey = jnp.sum(jnp.apply_along_axis(
                lambda x: jnp.all(x == prey_pos), 1, new_agent_positions
            ))
            prey_captured = n_agent_on_prey >= self.agents_to_capture_prey
            total_prey_captured = total_prey_captured + prey_captured
            reward = reward + prey_captured.squeeze() * self.capture_reward

            # randomly move prey to a new position if captured
            key, new_prey_pos_key = jax.random.split(key)
            new_prey_pos = (
                # same position as previously if not captured
                prey_pos * (1 - prey_captured) + 
                # new random position if captured
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
            prey_captured=total_prey_captured
        )

        obs = self.get_obs(new_state)
        info = {}

        done = next_step >= self.max_steps
        dones = {a: done for a in self.agents}
        dones.update({"__all__": done})

        rewards = {a: reward for a in self.agents}

        return obs, new_state, rewards, dones, info


if __name__ == '__main__':

    env = MimicryCommEnvGridworld()

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    print(state)
    print(obs)

    for _ in range(10):
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
        print(dones)
