from jaxevocomm.env.mimicry_comm_env import (
    MimicryCommEnvGridworld, State as MimicryCommEnvState
)

import jax
import jax.numpy as jnp

from hypothesis import given
from hypothesis.strategies import integers


def create_key():
    return jax.random.PRNGKey(0)


@given(integers(min_value=1, max_value=10),
       integers(min_value=1, max_value=10),
       integers(min_value=1, max_value=100))
def test_reset(n_agents: int, n_prey: int, grid_size: int):
    env = MimicryCommEnvGridworld(grid_size=grid_size,
                                  n_agents=n_agents,
                                  n_prey=n_prey)
    assert len(env.agents) == n_agents

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    assert isinstance(state, MimicryCommEnvState)
    assert isinstance(obs, dict)
    assert isinstance(state.agent_pos, jnp.array)
    assert state.agent_pos.shape == (n_agents, 2)

    for agent_idx, agent_id in enumerate(env.agents):
        assert isinstance(agent_id, str)
        assert agent_id in obs
        agent_x, agent_y = state.agent_pos[agent_idx]
        assert 0 <= agent_x < grid_size
        assert 0 <= agent_y < grid_size
