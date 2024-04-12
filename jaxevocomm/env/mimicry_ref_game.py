import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments.spaces import Discrete, Box

from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from typing import Tuple, Dict
from functools import partial


@struct.dataclass
class State:
    # latent state between 1 and n_latent_vars (inclusive)
    z: chex.Array
    
    # communication state between 1 and n_signals (inclusive)
    # 0 means no communication (default state)
    c: chex.Array 

    step: int  # current step


class MimicryCommReferentialGame(MultiAgentEnv):

    def __init__(self,
                 n_actions: int = 10,
                 external_source_prob: float = 0.5,
                 **kwargs):
        super(MimicryCommReferentialGame, self).__init__(num_agents = 2)
        self.n_actions = n_actions
        self.n_signals = n_actions
        self.n_latent_vars = n_actions
        self.external_source_prob = external_source_prob

        self.correct_answer_reward = 1.0
        self.n_agents = 2
        self.agents = ['speaker', 'listener']

        self.action_spaces = {
            'speaker': Discrete(self.n_signals),
            'listener': Discrete(self.n_actions)
        }

        obs, _ = self.reset(jax.random.PRNGKey(0))

        self.observation_spaces = {
            agent: Box(-1.0, 1.0, shape=agent_obs.shape)
            for agent, agent_obs in obs.items()
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        z = 1 + jax.random.randint(key, (1,), 0, self.n_latent_vars)
        c = jnp.zeros((1,), dtype=jnp.int32)
        state = State(z=z, c=c, step=0)
        return self.get_obs(state), state

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        return {
            'speaker': self.get_speaker_obs(state),
            'listener': self.get_listener_obs(state)
        }

    def get_speaker_obs(self, state: State) -> chex.Array:
        latent_var = jax.nn.one_hot(state.z - 1, self.n_latent_vars)
        # id_vec = jnp.array([1.0])
        # step_vec = jax.nn.one_hot(state.step, 3)
        # return jnp.concatenate([id_vec, step_vec, latent_var.squeeze()])
        return latent_var

    def get_listener_obs(self, state: State) -> chex.Array:
        is_signal = state.c != 0
        signal = is_signal * jax.nn.one_hot(state.c - 1, self.n_signals)
        # id_vec = jnp.array([0.0])
        # step_vec = jax.nn.one_hot(state.step, 3)
        # return jnp.concatenate([id_vec, step_vec, signal.squeeze()])
        return signal

    def update_sound_state(self,
                           key: chex.PRNGKey,
                           state: State,
                           speaker_action: chex.Array) -> chex.Array:

        key, _key = jax.random.split(key)
        use_external_source = jax.random.bernoulli(_key,
                                                   self.external_source_prob,
                                                   state.z.shape)

        new_c = (
            use_external_source * state.z +
            (1 - use_external_source) * (1 + speaker_action)
        )

        return new_c

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
        speaker_act = actions['speaker']
        listener_act = actions['listener']

        key, sound_key = jax.random.split(key)
        new_sound_state = self.update_sound_state(sound_key,
                                                  state,
                                                  speaker_act)

        next_step = state.step + 1
        new_state = state.replace(
            c=new_sound_state,
            step=next_step,
        )

        done = next_step >= 2
        dones = {a: done for a in self.agents}
        dones.update({"__all__": done})

        reward = (
            done
            * (listener_act + 1 == state.z)
            * self.correct_answer_reward
        ).squeeze()

        rewards = {a: reward for a in self.agents}

        return self.get_obs(new_state), new_state, rewards, dones, {}


if __name__ == '__main__':

    env = MimicryCommReferentialGame(2)

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
