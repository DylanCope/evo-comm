import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct
from gymnax.environments.spaces import Discrete, Box

from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from typing import Tuple, Dict
from functools import partial


@struct.dataclass
class State:
    # latent state between 1 and n_latent_vars (inclusive)
    # dim: (2,)
    z: chex.Array
    
    # communication state between 1 and n_signals (inclusive)
    # 0 means no communication (default state)
    # dim: (2,)
    c: chex.Array 

    step: int  # current step


class MimicryCommReferentialGameSymmetric(MultiAgentEnv):

    def __init__(self,
                 n_actions: int = 10,
                 external_source_prob: float = 0.5,
                 **kwargs):
        super(MimicryCommReferentialGameSymmetric, self).__init__(num_agents = 2)
        self.n_actions = n_actions
        self.n_signals = n_actions
        self.n_latent_vars = n_actions
        self.external_source_prob = external_source_prob

        self.correct_answer_reward = 1.0
        self.n_agents = 2
        self.agents = ['agent_0', 'agent_1']

        self.action_spaces = {
            'agent_0': Discrete(self.n_signals + self.n_actions),
            'agent_1': Discrete(self.n_signals + self.n_actions)
        }

        obs, _ = self.reset(jax.random.PRNGKey(0))

        self.observation_spaces = {
            agent: Box(-1.0, 1.0, shape=agent_obs.shape)
            for agent, agent_obs in obs.items()
        }

    def agent_idx(self, agent_name: str):
        return self.agents.index(agent_name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        z = 1 + jax.random.randint(key, (2,), 0, self.n_latent_vars)
        c = jnp.zeros((2,), dtype=jnp.int32)
        state = State(z=z, c=c, step=0)
        return self.get_obs(state), state

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        return {
            agent: self.get_agent_obs(state, agent)
            for agent in self.agents
        }

    def get_agent_obs(self, state: State, agent: str) -> chex.Array:
        agent_i = self.agent_idx(agent)

        latent_var = jax.nn.one_hot(state.z[agent_i] - 1, self.n_latent_vars)

        other_agent_i = 1 - agent_i
        signal_from_other = state.c[other_agent_i]
        is_signal = signal_from_other != 0
        signal = is_signal * jax.nn.one_hot(signal_from_other - 1,
                                            self.n_signals)

        step_vec = jax.nn.one_hot(state.step % 2, 2)

        return jnp.concatenate([step_vec, latent_var, signal], axis=-1)

    def update_sound_state(self,
                           key: chex.PRNGKey,
                           state: State,
                           agent_i: int,
                           speaker_action: chex.Array) -> chex.Array:

        key, _key = jax.random.split(key)
        use_external_source = jax.random.bernoulli(_key,
                                                   self.external_source_prob,
                                                   (1,))

        # speaker actions are in the range [0, n_actions + n_signals)
        # first n_actions are for guessing the latent state
        # next n_signals are for communication
        is_comm_action = speaker_action >= self.n_actions

        # if the action is a communication action, the signal >= 0
        agent_signal = jnp.clip(speaker_action - self.n_actions,
                                0, self.n_signals)

        # external signal just copies the latent state known
        # by agent i and sends it to the other agent
        external_signal = state.z[agent_i]

        new_c = (
            use_external_source * external_signal +
            (1 - use_external_source) * is_comm_action * (1 + agent_signal)
        )

        return new_c

    def compute_reward(self,
                       state: State,
                       done: chex.Array,
                       agent_i: int,
                       agent_act: chex.Array):
        return (
            done                                   # only give reward if the episode is done
            * (agent_act + 1 == state.z[agent_i])  # reward for correct guess
            * self.correct_answer_reward           # reward value
        ).squeeze()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array],  # observations
               State,                  # next state
               Dict[str, float],       # rewards
               Dict[str, bool],        # dones
               Dict                    # infos
               ]:
        """
        Args:
            key: random key
            state: current state
            actions: dict of actions for each agent

        Returns:
            Tuple of (observations, next state, rewards, dones, infos)
        """

        key, *sound_keys = jax.random.split(key, self.n_agents + 1)

        new_sound_state = jnp.concatenate([
            self.update_sound_state(_key, state, i, actions[agent])
            for (i, agent), _key in zip(enumerate(self.agents), sound_keys)
        ])

        next_step = state.step + 1

        new_random_z = 1 + jax.random.randint(key, (2,), 0, self.n_latent_vars)
        need_new_z = next_step % 2 == 0
        new_z = need_new_z * new_random_z + (1 - need_new_z) * state.z

        new_state = state.replace(
            z=new_z,
            c=new_sound_state,
            step=next_step,
        )

        done = next_step >= 10
        dones = {a: done for a in self.agents}
        dones.update({"__all__": done})

        reward = (
            self.compute_reward(state, need_new_z, 0, actions['agent_0'])
            + self.compute_reward(state, need_new_z, 1, actions['agent_1'])
        ) / 2.0

        rewards = {a: reward for a in self.agents}

        return self.get_obs(new_state), new_state, rewards, dones, {}


if __name__ == '__main__':

    env = MimicryCommReferentialGameSymmetric(2, 0.0)

    print('Action Space Sizes', env.action_spaces['agent_0'].n)

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    print('Initial State:', state)
    print('Initial Obs:', obs)
    print()

    for _ in range(20):
        action_keys = jax.random.split(key, env.n_agents)
        actions = {
            agent: env.action_spaces[agent].sample(k)
            for agent, k in zip(env.agents, action_keys)
        }
        print('Current', state)
        print('Obs:', obs)
        print('Actions:', actions)

        key, step_key = jax.random.split(key)

        obs, state, reward, dones, infos = env.step(step_key, state, actions)

        print('Reward:', reward)

        if dones['__all__']:
            print('Episode done')
            print('\n')
        print()
