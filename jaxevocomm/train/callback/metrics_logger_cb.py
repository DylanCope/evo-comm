from .callback import TrainerCallback

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from flax import struct, core


class MetricsLogger(TrainerCallback):

    def __init__(self, log_dir : str | Path):
        self.log_dir = log_dir
        self.metrics_history = []

    def on_iteration_end(self,
                         iteration: int,
                         training_state: struct.PyTreeNode,
                         metric: core.FrozenDict[str, Any]):
        if isinstance(iteration, np.ndarray):
            iteration = int(iteration.item())

        self.metrics_history.append({
            'iteration': iteration,
            **{k: v.tolist() for k, v in metric.items()}
        })

    def on_train_end(self, training_state):
        data = pd.DataFrame(self.metrics_history)
        data.to_csv(self.log_dir / "metrics.csv", index=False)
