from .callback import TrainerCallback

from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax import struct, core


class Checkpointer(TrainerCallback):

    def __init__(self,
                 ckpt_dir : str | Path,
                 **ckpt_kwargs):
        self.ckpt_dir = ckpt_dir
        self.checkpoint_manager = ocp.CheckpointManager(
            ckpt_dir,
            options=ocp.CheckpointManagerOptions(**ckpt_kwargs)
        )

    def on_iteration_end(self,
                         iteration: int,
                         training_state: struct.PyTreeNode,
                         metric: core.FrozenDict[str, Any]):
        if isinstance(iteration, np.ndarray):
            iteration = int(iteration.item())

        self.checkpoint_manager.save(
            iteration,
            args=ocp.args.StandardSave(training_state),
            metrics=metric
        )
        self.checkpoint_manager.wait_until_finished()
