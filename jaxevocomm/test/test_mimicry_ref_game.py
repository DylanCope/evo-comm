from jaxevocomm.env.mimicry_ref_game import (
    MimicryCommReferentialGame,
    State as MCRGState
)

import chex
import jax
import jax.numpy as jnp


def test_reset():
    n_actions = 10
    env = MimicryCommReferentialGame(n_actions)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert isinstance(state.c, chex.Array)
    assert isinstance(state.z, chex.Array)
    assert (state.c == 0).all()
    assert (state.z != 0).all()
    assert 'speaker' in obs
    assert 'listener' in obs


def create_state(c, z, step):
    return MCRGState(
        c=jnp.array(c),
        z=jnp.array(z),
        step=step
    )


def test_external_source():
    n_actions = 10
    env = MimicryCommReferentialGame(n_actions,
                                     external_source_prob=1.0)
    key = jax.random.PRNGKey(0)
    for n in range(n_actions):
        z = n + 1
        state = create_state(0, z, 0)
        not_n = (n + 1) % n_actions
        actions = {
            'speaker': jnp.array(not_n),
            'listener': jnp.array(0),  # doesn't matter
        }
        obs, new_state, *_ = env.step_env(key, state, actions)
        assert new_state.c == z
        assert obs['listener'][n] == 1.0
        assert obs['listener'].sum() == 1.0


def test_speaker_source():
    n_actions = 10
    env = MimicryCommReferentialGame(n_actions,
                                     external_source_prob=0.0)
    key = jax.random.PRNGKey(0)
    for n in range(n_actions):
        not_n = (n + 1) % n_actions
        z = not_n + 1
        state = create_state(0, z, 0)
        actions = {
            'speaker': jnp.array(n),
            'listener': jnp.array(0),  # doesn't matter
        }
        obs, new_state, *_ = env.step_env(key, state, actions)
        assert new_state.c == n + 1
        assert obs['listener'][n] == 1.0
        assert obs['listener'].sum() == 1.0



def test_listener_answer():
    n_actions = 5
    env = MimicryCommReferentialGame(n_actions)
    key = jax.random.PRNGKey(0)
    for z in range(n_actions):
        for a in range(n_actions):
            state = create_state(0, z + 1, 1)
            actions = {
                'speaker': jnp.array(0),  # doesn't matter
                'listener': jnp.array(a)
            }
            obs, new_state, rewards, *_ = env.step_env(key, state, actions)
            if a == z:
                assert rewards['speaker'] == 1.0
                assert rewards['listener'] == 1.0
            else:
                assert rewards['speaker'] == 0.0
                assert rewards['listener'] == 0.0
