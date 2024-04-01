from jaxevocomm.train.callback.callback import TrainerCallback
from jaxevocomm.train.evo.fitness_evalutator import FitnessEvaluator

from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from flax import struct
from evosax.strategy import EvoState
from evosax import DE, FitnessShaper, Strategies



class EvoRunnerState(struct.PyTreeNode):
    """
    Class to hold the training state of the MAPPO algorithm.
    """

    generation: int
    evo_state: EvoState
    rng: jnp.ndarray = struct.field(pytree_node=True)

    @classmethod
    def create(cls, evo_state, rng):
        return cls(
            generation=0,
            evo_state=evo_state,
            rng=rng
        )

    def next_rng(self):
        rng, _rng = jax.random.split(self.rng)
        return _rng, self.replace(rng=rng)

    def iterate(self, next_state: EvoState):
        return self.replace(evo_state=next_state,
                            generation=self.generation + 1)


class EvoRunner:

    def __init__(self,
                 config: dict,
                 callback: TrainerCallback | List[TrainerCallback] = None):
        self.config = config
        self.fitness_evaluator = FitnessEvaluator(config)
        self.fit_shaper = FitnessShaper(maximize=True)
        self.callback = callback

        strategy_name = config.get('EVO_STRATEGY', 'SimpleGA')
        if strategy_name in Strategies:
            strategy_cls = Strategies[strategy_name]
            self.strategy = strategy_cls(popsize=self.config['POP_SIZE'],
                                         num_dims=self.fitness_evaluator.n_params)
        else:
            raise ValueError(f"Unknown EVO strategy {config['EVO_STRATEGY']}")

    @partial(jax.jit, static_argnums=0)
    def _run_generation(self, runner_state: EvoRunnerState):
        rng, runner_state = runner_state.next_rng()
        population, evo_state = self.strategy.ask(rng,
                                                  runner_state.evo_state,
                                                  self.strategy.default_params)
        rng, runner_state = runner_state.next_rng()
        mean_rewards = self.fitness_evaluator.rollout(rng, population)
        fitness = self.fit_shaper.apply(population, mean_rewards)
        evo_state = self.strategy.tell(population,
                                       fitness,
                                       evo_state,
                                       self.strategy.default_params)

        runner_state = runner_state.iterate(evo_state)

        metrics = self._compute_metrics(mean_rewards)

        jax.experimental.io_callback(self.callback.on_iteration_end,
                                     None, runner_state.generation,
                                     runner_state, metrics)

        return runner_state, metrics

    def _compute_metrics(self, mean_rewards):
        return {
            'mean_reward': jnp.mean(mean_rewards),
            'std_reward': jnp.std(mean_rewards),
            'max_reward': jnp.max(mean_rewards),
            'min_reward': jnp.min(mean_rewards)
        }

    def run(self):
        rng = jax.random.PRNGKey(self.config.get('SEED', 0))
        rng, _rng = jax.random.split(rng)
        evo_state = self.strategy.initialize(_rng, self.strategy.default_params)
        runner_state = EvoRunnerState.create(evo_state, rng)

        self.callback.on_train_begin(self.config)

        with jax.disable_jit(False):
            final_state, _ = jax.lax.scan(
                lambda s, _: self._run_generation(s),
                runner_state, None, self.config["TOTAL_GENERATIONS"]
            )

        self.callback.on_train_end(final_state)
