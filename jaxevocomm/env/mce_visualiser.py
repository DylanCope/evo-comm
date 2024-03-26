from typing import Callable, List, Optional

import pygame
from pygame import gfxdraw
import seaborn as sns
import numpy as onp

from jaxevocomm.env.mimicry_comm_env import (
    MimicryCommEnvGridworld, State as MCEState
)
from jaxevocomm.utils.video import save_video

RenderLayers = List[Callable[[pygame.Surface], None]]

CELL_SIZE = 48
WHITE = (255, 255, 255)


def to_rgb255(colour: tuple) -> tuple:
    return tuple(int(255 * c) for c in colour)


AGENT_COLOUR = to_rgb255(sns.color_palette('pastel', n_colors=2)[1])
AGENT_COLOUR_OUTLINE = to_rgb255(sns.color_palette('bright', n_colors=2)[1])

PREY_COLOUR = to_rgb255(sns.color_palette(n_colors=2)[0])
PREY_COLOUR_OUTLINE = to_rgb255(sns.color_palette('bright', n_colors=2)[0])

pygame.font.init()
TEXT_FONT = pygame.font.SysFont(None, int(CELL_SIZE * 0.5))
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def draw_circle(surface: pygame.Surface,
                colour: tuple,
                x: int, y: int, radius: int,
                outline_colour: tuple = None):
    outline_colour = outline_colour or colour
    gfxdraw.filled_circle(surface, x, y, radius, colour)
    gfxdraw.aacircle(surface, x, y, radius, outline_colour)


def draw_diamond(surface: pygame.Surface,
                 colour: tuple,
                 x: int, y: int, radius: int,
                 outline_colour: tuple = None):
    outline_colour = outline_colour or colour
    points = [(x, y - radius),
                (x + radius, y),
                (x, y + radius),
                (x - radius, y)]
    gfxdraw.filled_polygon(surface, points, colour)
    gfxdraw.aapolygon(surface, points, outline_colour)


def get_cell_circle(x: int, y: int):
    centre_0 = y * CELL_SIZE + CELL_SIZE // 2
    centre_1 = x * CELL_SIZE + CELL_SIZE // 2
    r = int(0.6 * CELL_SIZE // 2)
    return (centre_0, centre_1, r)


def fill_bg_white(pygame_window: pygame.Surface):
    pygame_window.fill(WHITE)


def draw_sound(sound: int,
               screen_x: int, screen_y: int, screen_r: int,
               pygame_window: pygame.Surface):
    if sound != 0:
        text = TEXT_FONT.render(ALPHABET[sound - 1], True, WHITE)
        text_pos = (
            int(screen_x - 0.4*screen_r),
            int(screen_y - 0.45*screen_r)
        )
        pygame_window.blit(text, text_pos)


def draw_agents(state: MCEState,
                pygame_window: pygame.Surface):
    for agent_i, agent_pos in enumerate(state.agent_pos):
        screen_x, screen_y, screen_r = get_cell_circle(*agent_pos)
        draw_circle(pygame_window,
                    AGENT_COLOUR,
                    screen_x, screen_y, screen_r,
                    outline_colour=AGENT_COLOUR_OUTLINE)

        agent_sound = state.c[agent_i]
        draw_sound(agent_sound,
                   screen_x, screen_y, screen_r,
                   pygame_window)


def draw_prey(state: MCEState,
              pygame_window: pygame.Surface):
    n_agents, *_ = state.agent_pos.shape

    for prey_i, prey_pos in enumerate(state.prey_pos):
        screen_x, screen_y, screen_r = get_cell_circle(*prey_pos)
        draw_diamond(pygame_window,
                     PREY_COLOUR,
                     screen_x, screen_y, screen_r,
                     outline_colour=PREY_COLOUR_OUTLINE)

        prey_sound = state.c[n_agents + prey_i]
        draw_sound(prey_sound,
                   screen_x, screen_y, screen_r,
                   pygame_window)


class MCEVisualiser(object):
    """
    Visualise Mimicry Communications Environments (MCEs) with pygame.
    """

    def __init__(self, env: MimicryCommEnvGridworld):
        self.env = env
        window_size = int(CELL_SIZE * self.env.grid_size)
        self.pygame_window = pygame.display.set_mode((window_size, window_size))

        self.render_layers = [
            lambda _, surf: fill_bg_white(surf),
            draw_prey,
            draw_agents,
        ]

    def render(self, state: MCEState) -> onp.ndarray:
        for render_layer in self.render_layers:
            render_layer(state, self.pygame_window)
        frame = pygame.surfarray.array3d(self.pygame_window)
        return onp.rot90(frame[::-1, :, :], -1)

    def animate(self,
                states: List[MCEState],
                file_path: str,
                fps: int = 15):
        frames = [self.render(state) for state in states]
        save_video(frames,
                   file_path=file_path,
                   format='mp4',
                   fps=fps)


if __name__ == '__main__':
    import jax
    import tqdm

    STEPS = 50

    env = MimicryCommEnvGridworld(2)

    progbar = tqdm.tqdm(range(STEPS), desc='Running MCE', unit='step')

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    progbar.update()

    states = [state]
    total_rewards = 0

    for step in range(STEPS):
        action_keys = jax.random.split(key, env.n_agents)
        actions = {
            agent: env.action_spaces[agent].sample(k)
            for agent, k in zip(env.agents, action_keys)
        }

        key, step_key = jax.random.split(key)

        obs, state, rewards, dones, infos = env.step(step_key, state, actions)
        total_rewards += sum(rewards.values())

        states.append(state)
        progbar.update()

    progbar.close()
    print(f"Finished running MCE for {STEPS} steps.")
    print(f"Total rewards: {total_rewards}")
    print('Final state:', state)

    from pathlib import Path
    tmp_dir = Path('tmp')
    tmp_dir.mkdir(exist_ok=True)

    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    visualiser = MCEVisualiser(env)
    visualiser.animate(states, tmp_dir / 'mce_visualisation', fps=1)
    print(f"Saved visualisation to {tmp_dir / 'mce_visualisation.mp4'}")
