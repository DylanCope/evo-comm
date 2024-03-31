from jaxevocomm.env import make_env
from jaxevocomm.models.actor_rnn import ActorRNN
from jaxevocomm.models.scanned_rnn import ScannedRNN

from functools import partial
import numpy as np
from typing import Any, List, NamedTuple, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct, core
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from evosax.core import ParameterReshaper


class FitnessEvaluatorEnvWrapper(JaxMARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs = self.pad_longest(obs)
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
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def pad_longest(self, obs):
        
        longest_len = max([
            obs[agent].shape[-1] for agent in self._env.agents
        ])

        def pad(x):
            return jnp.pad(x, ((0, longest_len - x.shape[-1])))

        return {agent: pad(obs[agent]) for agent in self._env.agents}

    def get_agent_obs_dim(self):
        return max([
            self._env.observation_space(agent).shape[-1]
            for agent in self._env.agents
        ])


class RolloutState(struct.PyTreeNode):
    """
    Class to hold the rollout state of the MAPPO algorithm.
    """
    rng: jnp.ndarray = struct.field(pytree_node=True)

    actor_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    actor_hstate: jnp.ndarray = struct.field(pytree_node=True)

    env_state: struct.dataclass = struct.field(pytree_node=True)
    last_obs: Dict[str, jnp.ndarray] = struct.field(pytree_node=True)
    last_done: jnp.ndarray = struct.field(pytree_node=True)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: core.FrozenDict[str, jnp.ndarray]
    info: jnp.ndarray
    env_state: struct.dataclass


def batchify(x: Dict[str, jnp.ndarray],
             agent_ids: List[str],
             num_actors: int) -> jnp.ndarray:
    x = jnp.stack([x[a] for a in agent_ids])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray,
               agent_ids: List[str],
               num_envs: int) -> Dict[str, jnp.ndarray]:
    num_agents = len(agent_ids)
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_ids)}


def unbatchify_traj_batch(traj_batch: Transition,
                          agent_ids: List[str]) -> Dict[str, Transition]:
    n_agents = len(agent_ids)
    n_steps, batch_size = traj_batch.done.shape
    n_envs = batch_size // n_agents

    def _unbatchify(x):
        return x.reshape((n_steps, n_agents, n_envs, -1)).squeeze()

    global_done = _unbatchify(traj_batch.global_done)
    done = _unbatchify(traj_batch.done)
    action = _unbatchify(traj_batch.action)
    reward = _unbatchify(traj_batch.reward)
    log_prob = _unbatchify(traj_batch.log_prob)
    info = jax.tree_map(_unbatchify, traj_batch.info)

    return {
        a: Transition(
            global_done=global_done[:, i],
            done=done[:, i],
            action=action[:, i],
            reward=reward[:, i],
            log_prob=log_prob[:, i],
            obs=traj_batch.obs[:, i],
            info={
                k: v[:, i] for k, v in info.items()
            },
            env_state=traj_batch.env_state
        )
        for i, a in enumerate(agent_ids)
    }


class FitnessEvaluator:

    def __init__(self, config: dict):
        self.config = config
        self.env = FitnessEvaluatorEnvWrapper(make_env(self.config))
        self.actor_network = ActorRNN(self.action_space_dim,
                                      config=self.config)

        ac_init_x = (
            jnp.zeros((1, self.config["NUM_ENVS"],
                       self.env.get_agent_obs_dim())),
            jnp.zeros((1, self.config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ENVS"], 128)
        actor_network_params = self.actor_network.init(jax.random.PRNGKey(0),
                                                       ac_init_hstate,
                                                       ac_init_x)

        self.params_reshaper = ParameterReshaper(actor_network_params, n_devices=1)
        self.n_params = self.params_reshaper.total_params

    @property
    def action_space_dim(self):
        return self.env.action_space(self.env.agents[0]).n

    def reset_env(self,
                  n_envs: int,
                  rng: jnp.ndarray
    ) -> Tuple[core.FrozenDict[str, Any],
               struct.dataclass,
               chex.Array]:
        reset_rng = jax.random.split(rng, n_envs)
        init_obs, env_state = jax.vmap(self.env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(n_envs * len(self.env.agents), 128)
        return init_obs, env_state, ac_init_hstate

    def _collect_trajectories(self,
                              n_envs: int,
                              n_steps: int,
                              rollout_state: RolloutState) -> Tuple[RolloutState, Transition]:

        n_actors = len(self.env.agents) * n_envs

        @jax.jit
        def _env_step(rollout_state: RolloutState, _) -> Tuple[RolloutState, Transition]:

            # SELECT ACTION
            rng, _rng = jax.random.split(rollout_state.rng)

            obs_batch = batchify(
                rollout_state.last_obs,
                self.env.agents,
                n_actors
            )
            ac_in = (
                obs_batch[np.newaxis, :],
                rollout_state.last_done[np.newaxis, :],
            )

            ac_hstate, pi = self.actor_network.apply(rollout_state.actor_params,
                                                     rollout_state.actor_hstate,
                                                     ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, self.env.agents, n_envs
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, n_envs)

            obsv, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0)
            )(rng_step, rollout_state.env_state, env_act)

            info = jax.tree_map(lambda x: x.reshape((n_actors)), info)
            done_batch = batchify(
                done, self.env.agents, n_actors
            ).squeeze()

            transition = Transition(
                jnp.tile(done["__all__"],
                         self.env.num_agents),
                done_batch,
                action.squeeze(),
                batchify(
                    reward, self.env.agents, n_actors
                ).squeeze(),
                log_prob.squeeze(),
                obs_batch,
                info,
                env_state,
            )

            rollout_state = rollout_state.replace(
                env_state=env_state,
                last_obs=obsv,
                last_done=done_batch,
                actor_hstate=ac_hstate,
                rng=rng,
            )

            return rollout_state, transition

        rollout_state, traj_batch = jax.lax.scan(
            _env_step, rollout_state, None, n_steps
        )

        return rollout_state, traj_batch

    def rollout_single(self,
                       rng: jnp.ndarray,
                       params: chex.Array,
                       n_envs: int,
                       n_steps: int) -> Dict[str, Transition]:
        init_obs, env_state, hstate = self.reset_env(n_envs, rng)
        init_rollout_state = RolloutState(
            rng=rng,
            actor_params=params,
            actor_hstate=hstate,
            env_state=env_state,
            last_obs=init_obs,
            last_done=jnp.zeros((n_envs * len(self.env.agents)), dtype=bool),
        )
        _, traj_batch = self._collect_trajectories(
            n_envs, n_steps, init_rollout_state
        )
        trajectories = unbatchify_traj_batch(traj_batch, self.env.agents)
        return trajectories

    def _compute_total_rewards(self,
                               trajectories: Dict[str, Transition]) -> Dict[str, jnp.ndarray]:
        metrics = {}
        agent_0, *_ = self.env.agents
        num_episodes = trajectories[agent_0].global_done.sum()

        def _create_row_done_mask(dones):
            return jax.lax.scan(
                lambda found_done, is_done: (is_done | found_done, is_done | found_done),
                False, dones, reverse=True
            )[::-1]

        dones_mask, _ = jax.vmap(
            _create_row_done_mask, in_axes=(1,)
        )(trajectories[agent_0].global_done)

        dones_mask = dones_mask.transpose()

        mean_total_reward = (
            (dones_mask * trajectories[agent_0].reward).sum() / num_episodes
        )
        return mean_total_reward

    def rollout(self,
                rng: jnp.ndarray,
                population: chex.Array) -> Dict[str, float]:
        """
        Completes rollouts for fitness evaluation of the population.
        """
        @jax.jit
        def _compute_fitness(params):
            trajs = self.rollout_single(rng,
                                        params,
                                        self.config["NUM_ENVS"],
                                        self.config["NUM_STEPS"])
            return self._compute_total_rewards(trajs)

        population_params = self.params_reshaper.reshape(population)
        fitness_vals = jax.vmap(_compute_fitness, in_axes=(0,))(population_params)
        return fitness_vals
