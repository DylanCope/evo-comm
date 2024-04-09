from typing import List, NamedTuple, Dict

import jax
import jax.numpy as jnp
from flax import struct, core


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: core.FrozenDict[str, jnp.ndarray]
    world_state: jnp.ndarray
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
    value = _unbatchify(traj_batch.value)
    reward = _unbatchify(traj_batch.reward)
    log_prob = _unbatchify(traj_batch.log_prob)
    world_state = _unbatchify(traj_batch.world_state)
    info = jax.tree_map(_unbatchify, traj_batch.info)

    return {
        a: Transition(
            global_done=global_done[:, i],
            done=done[:, i],
            action=action[:, i],
            value=value[:, i],
            reward=reward[:, i],
            log_prob=log_prob[:, i],
            obs=traj_batch.obs[:, i],
            world_state=world_state[:, i],
            info={
                k: v[:, i] for k, v in info.items()
            },
            env_state=traj_batch.env_state
        )
        for i, a in enumerate(agent_ids)
    }    
