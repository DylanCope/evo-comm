import jax
import jax.numpy as jnp

from functools import partial
from jaxmarl.wrappers.baselines import JaxMARLWrapper


class MPEWorldStateWrapper(JaxMARLWrapper):
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs = self.pad_longest(obs)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs = self.pad_longest(obs)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def pad_longest(self, obs):
        
        longest_len = max([obs[agent].shape[-1] for agent in self._env.agents])

        def pad(x):
            return jnp.pad(x, ((0, longest_len - x.shape[-1])))

        return {agent: pad(obs[agent]) for agent in self._env.agents}

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]
        """
        
        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs
            
        all_obs = jnp.concatenate([obs[agent].flatten() for agent in self._env.agents])
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def get_agent_obs_dim(self):
        return max([self._env.observation_space(agent).shape[-1] for agent in self._env.agents])

    def world_state_size(self):
        # spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        # return sum([space.shape[-1] for space in spaces])
        max_space_dim = max([self._env.observation_space(agent).shape[-1] for agent in self._env.agents])
        return max_space_dim * self._env.num_agents
