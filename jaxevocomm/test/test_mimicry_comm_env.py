from itertools import product
from typing import Dict
from jaxevocomm.env.mimicry_comm_env import (
    MimicryCommEnvGridworld,
    State as MimicryCommEnvState,
    GridAction
)

import chex
import jax


def test_reset():
    n_agents = 3
    n_prey = 2
    grid_size = 5

    env = MimicryCommEnvGridworld(grid_size=grid_size,
                                  n_agents=n_agents,
                                  n_prey=n_prey)
    assert len(env.agents) == n_agents

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    assert isinstance(state, MimicryCommEnvState)
    assert isinstance(obs, dict)
    assert isinstance(state.agent_pos, chex.Array)
    assert state.agent_pos.shape == (n_agents, 2)

    for agent_idx, agent_id in enumerate(env.agents):
        assert isinstance(agent_id, str)
        assert agent_id in obs
        agent_x, agent_y = state.agent_pos[agent_idx]
        assert 0 <= agent_x < grid_size
        assert 0 <= agent_y < grid_size


    assert (state.c == 0).all()


def create_state(agent_positions: Dict[str, tuple],
                 prey_positions: Dict[str, tuple],
                 messages: Dict[str, int] = None,
                 step: int = 0,
                 prey_captured: int = 0):

    messages = messages or {}
    msg_state = jax.numpy.array([
        messages.get(agent_id, 0)
        for agent_id in agent_positions
    ] + [
        messages.get(prey_id, 0)
        for prey_id in prey_positions
    ])

    return MimicryCommEnvState(
        agent_pos=jax.numpy.array([
            agent_positions[agent_id] for agent_id in agent_positions
        ]),
        prey_pos=jax.numpy.array([
            prey_positions[prey_id] for prey_id in prey_positions
        ]),
        c=msg_state,
        step=jax.numpy.array([step]),
        prey_captured=jax.numpy.array([prey_captured])
    )


def test_move_actions_no_boundary():
    n_agents = 1
    n_prey = 1
    grid_size = 10

    env = MimicryCommEnvGridworld(grid_size=grid_size,
                                  n_agents=n_agents,
                                  n_prey=n_prey)

    init_x, init_y = 5, 5
    agent_pos = {
        'agent_0': (init_x, init_y)
    }
    prey_pos = {
        'prey_0': (4, 5)  # arbitrary position
    }

    state = create_state(agent_pos, prey_pos)

    key = jax.random.PRNGKey(0)

    deltas = {
        GridAction.UP:    ( 0,  1),
        GridAction.DOWN:  ( 0, -1),
        GridAction.LEFT:  (-1,  0),
        GridAction.RIGHT: ( 1,  0),
        GridAction.NOOP:  ( 0,  0),
    }

    for action in GridAction.ACTIONS:
        actions = {
            'agent_0': jax.numpy.array([action])
        }
        obs, new_state, reward, dones, infos = env.step_env(key, state, actions)
        pos_x, pos_y = new_state.agent_pos[0]
        delta_x, delta_y = deltas[action]
        assert pos_x == init_x + delta_x
        assert pos_y == init_y + delta_y
        assert new_state.c[0] == 0


def test_sound_actions():
    """
    Checks that the agents performing sound actions correctly
    updates the state with the sound value.

    The test first checks that the "silent" action (always action 5) does not
    move the agent and leaves the sound state for the agent as 0.

    Then for each of the agent sounds, it checks that the agent again does move
    when the corresponding action is taken, and the sound state is set to the
    appropriate value.
    """
    n_agents = 1
    n_prey = 1
    grid_size = 10
    n_agent_only_sounds = 3
    n_overlapping_sounds = 1
    n_prey_only_sounds = 2

    env = MimicryCommEnvGridworld(grid_size=grid_size,
                                  n_agents=n_agents,
                                  n_prey=n_prey,
                                  n_agent_only_sounds=n_agent_only_sounds,
                                  n_overlapping_sounds=n_overlapping_sounds,
                                  n_prey_sounds=n_prey_only_sounds)

    assert env.n_sounds == n_agent_only_sounds + n_overlapping_sounds + n_prey_only_sounds
    assert env.n_agent_sounds == n_agent_only_sounds + n_overlapping_sounds
    assert env.n_prey_sounds == n_prey_only_sounds + n_overlapping_sounds

    init_x, init_y = 5, 5
    agent_pos = {
        'agent_0': (init_x, init_y)
    }
    prey_pos = {
        'prey_0': (4, 5)  # arbitrary position
    }

    state = create_state(agent_pos, prey_pos)
    key = jax.random.PRNGKey(0)

    # check silent action (effectively the same as noop)
    # but kept as it is easier to handle

    # Grid actions are:
    # NOOP (0), UP (1), DOWN (2), LEFT (3), RIGHT (4)
    # So silent sound index is at N_ACTIONS = 5
    silent_action = GridAction.N_ACTIONS
    actions = {
        'agent_0': jax.numpy.array([silent_action])
    }
    obs, new_state, reward, dones, infos = env.step_env(key, state, actions)

    def check_position_not_changed(new_state):
        pos_x, pos_y = new_state.agent_pos[0]
        assert pos_x == init_x
        assert pos_y == init_y

    def check_sound_correct(sound_val, new_state):
        assert new_state.c[0] == sound_val

    check_position_not_changed(new_state)
    check_sound_correct(0, new_state)

    for sound_val in range(1, 1 + env.n_agent_sounds):
        actions = {
            'agent_0': jax.numpy.array([GridAction.N_ACTIONS + sound_val])
        }
        obs, new_state, reward, dones, infos = env.step_env(key, state, actions)
        check_position_not_changed(new_state)
        check_sound_correct(sound_val, new_state)


def test_agent_obs_sound_feats():
    n_agents = 2
    n_prey = 1
    grid_size = 3

    env = MimicryCommEnvGridworld(grid_size=grid_size,
                                  n_agents=n_agents,
                                  n_prey=n_prey)

    grid_positions = list(product(range(grid_size), range(grid_size)))

    ego_agent_sound = 1
    other_agent_sound = 2
    prey_sound = 3

    sounds = {
        'agent_0': ego_agent_sound,
        'agent_1': other_agent_sound,
        'prey_0': prey_sound
    }

    ego_x, ego_y = 1, 1

    def check_sounds_correct(feats, source_x, source_y, sound_val):
        if source_x < ego_x:
            assert feats['sound_feat_right'][sound_val - 1] == 1
        elif source_x == ego_x:
            assert feats['sound_feat_right'][sound_val - 1] == 0
            assert feats['sound_feat_left'][sound_val - 1] == 0
        else:
            assert feats['sound_feat_left'][sound_val - 1] == 1

        if source_y < ego_y:
            assert feats['sound_feat_up'][sound_val - 1] == 1
        elif source_y == ego_y:
            assert feats['sound_feat_up'][sound_val - 1] == 0
            assert feats['sound_feat_down'][sound_val - 1] == 0
        else:
            assert feats['sound_feat_down'][sound_val - 1] == 1

    # create a state with all possible positions of prey and other agent
    # and check the prey and other agent sounds are are correctly encoded
    # depending on their relative positions to the ego agent
    for other_x, other_y in grid_positions:
        for prey_x, prey_y in grid_positions:
            agent_pos = {
                'agent_0': (ego_x, ego_y),
                'agent_1': (other_x, other_y)
            }
            prey_pos = {
                'prey_0': (prey_x, prey_y)
            }

            state = create_state(agent_pos, prey_pos, sounds)

            feats = env._create_sound_feats(state, 'agent_0')

            assert len(feats.keys()) == 4

            # should never hear sound from itself
            for sound_feat in feats.values():
                assert sound_feat[ego_agent_sound - 1] == 0

            check_sounds_correct(feats, other_x, other_y, other_agent_sound)
            check_sounds_correct(feats, prey_x, prey_y, prey_sound)


def test_observe_prey_vision():
    n_agents = 1
    n_prey = 1
    grid_size = 5

    agent_x, agent_y = 2, 2
    grid_positions = list(product(range(grid_size), range(grid_size)))

    for prey_visible_range in range(0, grid_size + 1):
        env = MimicryCommEnvGridworld(grid_size=grid_size,
                                      n_agents=n_agents,
                                      n_prey=n_prey,
                                      prey_visible_range=prey_visible_range)

        for prey_x, prey_y in grid_positions:
            state = create_state(
                agent_positions={'agent_0': (agent_x, agent_y)},
                prey_positions={'prey_0': (prey_x, prey_y)}
            )

            pos_feats = env._get_position_feats(state, 'agent_0')
            prey_feat_x = pos_feats['prey_0_pos_x']
            prey_feat_y = pos_feats['prey_0_pos_y']

            dist = abs(prey_x - agent_x) + abs(prey_y - agent_y)

            if dist <= prey_visible_range:
                assert prey_feat_x[prey_x] == 1
                assert prey_feat_y[prey_y] == 1
            else:
                assert (prey_feat_x == 0).all()
                assert (prey_feat_y == 0).all()
