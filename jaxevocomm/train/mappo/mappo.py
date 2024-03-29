"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.

Adapted from https://github.com/FLAIROx/JaxMARL/blob/main/baselines/MAPPO/mappo_rnn_mpe.py
"""

from pathlib import Path
import numpy as np
from typing import Any, List, NamedTuple, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import optax
from flax.training.train_state import TrainState
from flax import struct, core

from jaxevocomm.env import make_env
from jaxevocomm.train.callback import ChainedCallback, TrainerCallback
from jaxevocomm.models import ScannedRNN, ActorRNN, CriticRNN
from jaxevocomm.train.callback.ckpt_cb import load_best_ckpt


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


class MAPPOTrainState(struct.PyTreeNode):
    """
    Class to hold the training state of the MAPPO algorithm.
    """

    training_iteration: int
    actor_train_state: TrainState = struct.field(pytree_node=True)
    critic_train_state: TrainState = struct.field(pytree_node=True)

    env_state: struct.dataclass = struct.field(pytree_node=True)
    last_obs: core.FrozenDict[str, jnp.ndarray] = struct.field(pytree_node=True)
    last_done: jnp.ndarray = struct.field(pytree_node=True)

    actor_hstate: jnp.ndarray = struct.field(pytree_node=True)
    critic_hstate: jnp.ndarray = struct.field(pytree_node=True)

    rng: jnp.ndarray = struct.field(pytree_node=True)

    @classmethod
    def create(cls, actor_train_state, critic_train_state, env_state,
               last_obs, last_done, actor_hstate, critic_hstate, rng):
        return cls(
            training_iteration=0,
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            env_state=env_state,
            last_obs=last_obs,
            last_done=last_done,
            actor_hstate=actor_hstate,
            critic_hstate=critic_hstate,
            rng=rng,
        )

    def next_rng(self):
        rng, _rng = jax.random.split(self.rng)
        return _rng, self.replace(rng=rng)


class RolloutState(struct.PyTreeNode):
    """
    Class to hold the rollout state of the MAPPO algorithm.
    """

    actor_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    critic_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    env_state: struct.dataclass = struct.field(pytree_node=True)
    last_obs: Dict[str, jnp.ndarray] = struct.field(pytree_node=True)
    last_done: jnp.ndarray = struct.field(pytree_node=True)

    actor_hstate: jnp.ndarray = struct.field(pytree_node=True)
    critic_hstate: jnp.ndarray = struct.field(pytree_node=True)

    rng: jnp.ndarray = struct.field(pytree_node=True)


class MAPPO:

    def __init__(self,
                 config: dict,
                 callback: TrainerCallback | List[TrainerCallback] = None):
        self.config = config
        self.env = make_env(config)
        self.rng = jax.random.PRNGKey(config["SEED"])

        if isinstance(callback, list):
            self.callback = ChainedCallback(*callback)
        else:
            self.callback = callback or TrainerCallback()

        config["NUM_ACTORS"] = self.env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )

        if config["SCALE_CLIP_EPS"]:
            config["CLIP_EPS"] = config["CLIP_EPS"] / self.env.num_agents

    def _next_rng(self, n: int = 1):
        self.rng, _rng = jax.random.split(self.rng)
        if n <= 1:
            return _rng

        return jax.random.split(_rng, n)

    @property
    def action_space_dim(self):
        return self.env.action_space(self.env.agents[0]).n

    def _create_networks(self):
        self.actor_network = ActorRNN(self.action_space_dim,
                                      config=self.config)
        self.critic_network = CriticRNN()

    def _create_init_network_states(self):
        _rng_actor, _rng_critic = self._next_rng(2)
        ac_init_x = (
            jnp.zeros((1, self.config["NUM_ENVS"],
                       self.env.get_agent_obs_dim())),
            jnp.zeros((1, self.config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ENVS"], 128)
        actor_network_params = self.actor_network.init(_rng_actor,
                                                       ac_init_hstate,
                                                       ac_init_x)

        cr_init_x = (
            jnp.zeros((1, self.config["NUM_ENVS"], self.env.world_state_size(),)),
            jnp.zeros((1, self.config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ENVS"], 128)
        critic_network_params = self.critic_network.init(_rng_critic,
                                                         cr_init_hstate,
                                                         cr_init_x)

        return actor_network_params, critic_network_params

    def _create_optimisers(self):
        if self.config["ANNEAL_LR"]:

            def linear_schedule(count):
                total_steps = (
                    self.config["NUM_MINIBATCHES"]
                    * self.config["UPDATE_EPOCHS"]
                )
                frac = (
                    1.0 - (count // total_steps) / self.config["NUM_UPDATES"]
                )
                return self.config["LR"] * frac

            lr = linear_schedule
        else:
            lr = self.config["LR"]

        actor_tx = optax.chain(
            optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
            optax.adam(lr, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
            optax.adam(lr, eps=1e-5),
        )

        actor_network_params, critic_network_params = self._create_init_network_states()

        actor_train_state = TrainState.create(
            apply_fn=self.actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=self.actor_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        return actor_train_state, critic_train_state

    def setup(self):
        self._create_networks()
        return self._create_init_runner_state()

    def run(self):
        self.callback.on_train_begin(self.config)
        with jax.disable_jit(False):
            final_state = self._train()
        self.callback.on_train_end(final_state)

    def _init_env(self, n_envs: int) -> Tuple[core.FrozenDict[str, Any],
                                              struct.dataclass]:
        reset_rng = self._next_rng(n_envs)
        init_obs, env_state = jax.vmap(self.env.reset, in_axes=(0,))(reset_rng)
        return init_obs, env_state

    def _create_init_runner_state(self):
        ac_train_state, cr_train_state = self._create_optimisers()

        init_obs, env_state = self._init_env(self.config["NUM_ENVS"])

        ac_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ACTORS"], 128)
        cr_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ACTORS"], 128)

        runner_state = MAPPOTrainState.create(
            actor_train_state=ac_train_state,
            critic_train_state=cr_train_state,
            env_state=env_state,
            last_obs=init_obs,
            last_done=jnp.zeros((self.config["NUM_ACTORS"]), dtype=bool),
            actor_hstate=ac_init_hstate,
            critic_hstate=cr_init_hstate,
            rng=self._next_rng(), 
        )

        return runner_state

    @partial(jax.jit, static_argnums=0)
    def _train(self):
        runner_state = self.setup()
        runner_state, _ = jax.lax.scan(
            lambda s, _: self._training_iteration(s),
            runner_state, None, self.config["NUM_UPDATES"]
        )
        return runner_state

    def _collect_trajectories(self,
                              n_envs: int,
                              n_steps: int,
                              rollout_state: RolloutState) -> Tuple[RolloutState, Transition]:

        n_actors = len(self.env.agents) * n_envs

        def _env_step(rollout_state: RolloutState, _) -> Tuple[RolloutState, Transition]:

            # SELECT ACTION
            rng, _rng = jax.random.split(rollout_state.rng)

            obs_batch = batchify(
                rollout_state.last_obs,
                self.env.agents, n_actors
            )
            ac_in = (
                obs_batch[np.newaxis, :],
                rollout_state.last_done[np.newaxis, :],
            )

            ac_hstate, pi = self.actor_network.apply(rollout_state.actor_params,
                                                     rollout_state.actor_hstate,
                                                     ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, self.env.agents, n_envs
            )

            # VALUE
            world_state = rollout_state.last_obs["world_state"].reshape((n_actors, -1))
            cr_in = (
                world_state[None, :],
                rollout_state.last_done[np.newaxis, :],
            )
            cr_hstate, value = self.critic_network.apply(rollout_state.critic_params,
                                                         rollout_state.critic_hstate,
                                                         cr_in)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, n_envs)

            obsv, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0)
            )(rng_step, rollout_state.env_state, env_act)

            info = jax.tree_map(lambda x: x.reshape((n_actors)), info)
            done_batch = batchify(done, self.env.agents, n_actors).squeeze()
            transition = Transition(
                jnp.tile(done["__all__"], self.env.num_agents),
                done_batch,
                action.squeeze(),
                value.squeeze(),
                batchify(reward, self.env.agents, n_actors).squeeze(),
                log_prob.squeeze(),
                obs_batch,
                world_state,
                info,
                env_state,
            )

            rollout_state = rollout_state.replace(
                env_state=env_state,
                last_obs=obsv,
                last_done=done_batch,
                actor_hstate=ac_hstate,
                critic_hstate=cr_hstate,
                rng=rng,
            )

            return rollout_state, transition

        rollout_state, traj_batch = jax.lax.scan(
            _env_step, rollout_state, None, n_steps
        )

        return rollout_state, traj_batch

    def rollout(self, runner_state: MAPPOTrainState, n_envs: int, n_steps: int) -> Dict[str, Transition]:
        init_obs, env_state = self._init_env(n_envs)
        init_rollout_state = RolloutState(
            actor_params=runner_state.actor_train_state.params,
            critic_params=runner_state.critic_train_state.params,
            env_state=env_state,
            last_obs=init_obs,
            last_done=jnp.zeros((n_envs * len(self.env.agents)), dtype=bool),
            actor_hstate=runner_state.actor_hstate,
            critic_hstate=runner_state.critic_hstate,
            rng=runner_state.rng,
        )
        _, traj_batch = self._collect_trajectories(
            n_envs, n_steps, init_rollout_state
        )
        trajectories = unbatchify_traj_batch(traj_batch, self.env.agents)
        return trajectories, self._compute_metrics(trajectories)

    def _compute_advantages(self,
                            runner_state: MAPPOTrainState,
                            traj_batch):

        last_world_state = (
            runner_state.last_obs["world_state"].reshape((self.config["NUM_ACTORS"], -1))
        )
        cr_in = (
            last_world_state[None, :],
            runner_state.last_done[np.newaxis, :],
        )
        _, last_val = self.critic_network.apply(runner_state.critic_train_state.params,
                                                runner_state.critic_hstate,
                                                cr_in)
        last_val = last_val.squeeze()

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + self.config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        return advantages, targets

    def _actor_loss_fn(self, actor_params, init_hstate, traj_batch, gae):
        # RERUN NETWORK
        _, pi = self.actor_network.apply(
            actor_params,
            init_hstate.transpose(),
            (jax.lax.stop_gradient(traj_batch.obs), traj_batch.done),
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - self.config["CLIP_EPS"],
                1.0 + self.config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean(where=(1 - traj_batch.done))
        entropy = pi.entropy().mean(where=(1 - traj_batch.done))
        actor_loss = (
            loss_actor
            - self.config["ENT_COEF"] * entropy
        )
        return actor_loss, (loss_actor, entropy)

    def _critic_loss_fn(self, critic_params, init_hstate, traj_batch, targets):
        # RERUN NETWORK
        _, value = self.critic_network.apply(critic_params, init_hstate.transpose(),
                                             (traj_batch.world_state,  traj_batch.done)) 

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-self.config["CLIP_EPS"], self.config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=(1 - traj_batch.done))
        )
        critic_loss = self.config["VF_COEF"] * value_loss
        return critic_loss, (value_loss)

    def _learn_step(self, train_states, batch_info):
        actor_train_state, critic_train_state = train_states
        ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

        actor_grad_fn = jax.value_and_grad(self._actor_loss_fn, has_aux=True)
        actor_loss, actor_grads = actor_grad_fn(
            actor_train_state.params, ac_init_hstate, traj_batch, advantages
        )
        critic_grad_fn = jax.value_and_grad(self._critic_loss_fn, has_aux=True)
        critic_loss, critic_grads = critic_grad_fn(
            critic_train_state.params, cr_init_hstate, traj_batch, targets
        )

        actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
        critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
        
        total_loss = actor_loss[0] + critic_loss[0]
        loss_info = {
            "total_loss": total_loss,
            "actor_loss": actor_loss[0],
            "critic_loss": critic_loss[0],
            "entropy": actor_loss[1][1],
        }

        return (actor_train_state, critic_train_state), loss_info

    def _learn(self,
               init_rollout_state: RolloutState,
               advantages,
               targets,
               traj_batch: Transition,
               runner_state: MAPPOTrainState):

        ac_init_hstate = init_rollout_state.actor_hstate[None, :].squeeze().transpose()
        cr_init_hstate = init_rollout_state.critic_hstate[None, :].squeeze().transpose()
        init_hstates = (ac_init_hstate, cr_init_hstate)

        init_hstates = jax.tree_map(lambda x: jnp.reshape(
            x, (self.config["NUM_STEPS"], self.config["NUM_ACTORS"])
        ), init_hstates)
        
        batch = (
            init_hstates[0],
            init_hstates[1],
            traj_batch,
            advantages.squeeze(),
            targets.squeeze(),
        )
        rng, runner_state = runner_state.next_rng()
        permutation = jax.random.permutation(rng, self.config["NUM_ACTORS"])


        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=1), batch
        )

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x,
                    [x.shape[0], self.config["NUM_MINIBATCHES"], -1]
                    + list(x.shape[2:]),
                ),
                1,
                0,
            ),
            shuffled_batch,
        )

        train_states = (runner_state.actor_train_state,
                        runner_state.critic_train_state)
        (new_actor_train_state, new_critic_train_state), loss_info = jax.lax.scan(
            self._learn_step, train_states, minibatches
        )

        runner_state = runner_state.replace(
            actor_train_state=new_actor_train_state,
            critic_train_state=new_critic_train_state,
        )

        return runner_state, loss_info

    def _compute_metrics(self, trajectories: Dict[str, Transition]) -> Dict[str, jnp.ndarray]:
        metrics = {}
        mean_agent_rewards = [
            trajectories[a].reward.sum(-1) for a in trajectories
        ]
        metrics['mean_total_team_reward'] = jnp.mean(jnp.concatenate(mean_agent_rewards))
        for agent, reward in zip(self.env.agents, mean_agent_rewards):
            metrics[f'mean_{agent}_total_reward'] = reward
        agent_0, *_ = trajectories.keys()
        metrics['num_episodes'] = trajectories[agent_0].global_done.sum()
        metrics['n_env_steps'] = np.prod(trajectories[agent_0].global_done.shape)
        metrics['mean_episode_length'] = metrics['n_env_steps'] / metrics['num_episodes']
        return metrics

    def _training_iteration(self, runner_state: MAPPOTrainState):
        # COLLECT TRAJECTORIES
        init_rollout_state = RolloutState(
            actor_params=runner_state.actor_train_state.params,
            critic_params=runner_state.critic_train_state.params,
            env_state=runner_state.env_state,
            last_obs=runner_state.last_obs,
            last_done=runner_state.last_done,
            actor_hstate=runner_state.actor_hstate,
            critic_hstate=runner_state.critic_hstate,
            rng=runner_state.rng,
        )
        rollout_state, traj_batch = self._collect_trajectories(
            self.config["NUM_ENVS"], self.config["NUM_STEPS"], init_rollout_state
        )
        runner_state = runner_state.replace(
            env_state=rollout_state.env_state,
            last_obs=rollout_state.last_obs,
            last_done=rollout_state.last_done,
            actor_hstate=rollout_state.actor_hstate,
            critic_hstate=rollout_state.critic_hstate,
            rng=rollout_state.rng,
        )

        # CALCULATE ADVANTAGE
        advantages, targets = self._compute_advantages(runner_state, traj_batch)

        runner_state, loss_info = jax.lax.scan(
            lambda state, _: self._learn(init_rollout_state,
                                         advantages,
                                         targets,
                                         traj_batch,
                                         state),
            runner_state, None, self.config["UPDATE_EPOCHS"]
        )
        loss_info = jax.tree_map(lambda x: x.mean(), loss_info)

        training_iteration = runner_state.training_iteration + 1

        metrics = self._compute_metrics(traj_batch)
        metrics["training_iteration"] = training_iteration
        metrics["total_env_steps"] = (
            training_iteration
            * self.config["NUM_ENVS"]
            * self.config["NUM_STEPS"]
        )
        runner_state = runner_state.replace(training_iteration=training_iteration)

        jax.experimental.io_callback(self.callback.on_iteration_end,
                                     None, training_iteration, runner_state, metrics)

        return runner_state, metrics

    def restore_best_training_state(self, experiment_dir: str | Path, **kwargs) -> MAPPOTrainState:
        return load_best_ckpt(Path(experiment_dir) / 'checkpoints',
                              self.setup(),
                              **kwargs)

    @classmethod
    def restore(cls, experiment_dir: str | Path, **kwargs):
        config = OmegaConf.load(Path(experiment_dir) / '.hydra/config.yaml')
        trainer = cls(config)
        runner_state = trainer.restore_best_training_state(experiment_dir, **kwargs)
        return trainer, runner_state
