from jaxevocomm.train.mappo.mappo_transition import Transition
from jaxevocomm.env.mimicry_comm_env import MimicryCommEnvGridworld

from typing import Dict

import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


def episode_metrics(env: MultiAgentEnv,
                    trajectories: Dict[str, Transition]) -> Dict[str, jnp.ndarray]:

    metrics = {}
    agent_0, *_ = env.agents
    metrics['num_episodes'] = trajectories[agent_0].global_done.sum()

    def _create_row_done_mask(dones):
        return jax.lax.scan(
            lambda found_done, is_done: (is_done | found_done, is_done | found_done),
            False, dones, reverse=True
        )[::-1]

    dones_mask, _ = jax.vmap(
        _create_row_done_mask, in_axes=(1,)
    )(trajectories[agent_0].global_done)
    dones_mask = dones_mask.transpose()

    metrics['mean_total_reward'] = (
        (dones_mask * trajectories[agent_0].reward).sum() / metrics['num_episodes']
    )
    metrics['mean_episode_length'] = (
        dones_mask.sum() / metrics['num_episodes']
    )
    return metrics


def mce_metrics(env: MimicryCommEnvGridworld,
                trajectories: Dict[str, Transition]) -> Dict[str, jnp.ndarray]:
    metrics = {}
    agent_0, *_ = env.agents

    # comm state same for all agents
    # history of communication states [n_steps, n_agents + n_prey]
    # each row is a step, and the first columns are the agent communication states
    comm_state = trajectories[agent_0].env_state.c

    # for agent_i, agent in enumerate(env.agents):
    #     for sound_val in env.get_agent_sounds():
    #         agent_comm_state = comm_state[:, agent_i]
    #         # ignore silent states to just get messages
    #         agent_msgs = agent_comm_state[agent_comm_state != 0]
    #         sound_freq = (agent_msgs == sound_val).mean()
    #         metrics[f'{agent}_sound_{sound_val}_freq'] = sound_freq

    agents_comm_state = comm_state[:, :env.n_agents]
    overlap_sounds = jnp.array(env.get_overlapping_sounds())
    n_overlap_use = jnp.isins(agents_comm_state, overlap_sounds).sum()
    n_not_silent = agents_comm_state[agents_comm_state != 0].sum()
    overlap_freq = n_overlap_use / n_not_silent
    metrics['use_overlap_freq'] = overlap_freq

    metrics['silent_freq'] = (agents_comm_state == 0).mean()

    return metrics
