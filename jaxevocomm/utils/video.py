from pathlib import Path
from typing import List
import numpy as np
import imageio


def save_video(frames: List[np.ndarray],
               file_path: str = 'video',
               format='mp4',
               fps: int = 30) -> str:
    """
    Saves a video of the policy being evaluating in an environment.

    Args:
        frames (List[np.ndarray]): The frames of the video.
        file_path (str): The path of the file to save the video to.
        fps (int): The frames per second of the video.

    Returns:
        str: The path to the saved video.
    """
    assert len(frames) > 0, "No frames to save."
    if frames[0].dtype != np.uint8:
        frames = [frame.astype(np.uint8) for frame in frames]

    if isinstance(file_path, Path):
        file_path = str(file_path)

    if not file_path.endswith(f'.{format}'):
        file_path = f'{file_path}.{format}'

    imageio.mimwrite(file_path, frames, fps=fps)

    return file_path
