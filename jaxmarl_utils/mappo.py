"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict

from flax.training.train_state import TrainState
import distrax
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper

import wandb
import functools

from jaxmarl_utils.callback import TrainerCallback


class MPEWorldStateWrapper(JaxMARLWrapper):
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]
        """
        
        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs
            
        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs
    
    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


class MAPPOTrainer:

    def __init__(self, config: dict):
        self.config = config
        self.rng = jax.random.PRNGKey(config["SEED"])
        self.callback = config.get('callback_cls', TrainerCallback)()

        self.env = self._create_env()    
        config["NUM_ACTORS"] = self.env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )
        config["CLIP_EPS"] = config["CLIP_EPS"] / self.env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    def _next_rng(self, n: int = 1):
        self.rng, _rng = jax.random.split(self.rng)
        if n <= 1:
            return _rng

        return jax.random.split(_rng, n)

    def _create_env(self):
        env = jaxmarl.make(self.config["ENV_NAME"],
                           **self.config["ENV_KWARGS"])
        env = MPEWorldStateWrapper(env)
        env = MPELogWrapper(env)
        return env

    @property
    def action_space_dim(self):
        return self.env.action_space(self.env.agents[0]).n

    def _create_networks(self):
        self.actor_network = ActorRNN(self.action_space_dim,
                                      config=self.config)
        self.critic_network = CriticRNN()
        _rng_actor, _rng_critic = self._next_rng(2)
        ac_init_x = (
            jnp.zeros((1, self.config["NUM_ENVS"],
                       self.env.observation_space(self.env.agents[0]).shape[0])),
            jnp.zeros((1, self.config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ENVS"], 128)
        self.actor_network_params = self.actor_network.init(_rng_actor,
                                                            ac_init_hstate,
                                                            ac_init_x)

        cr_init_x = (
            jnp.zeros((1, self.config["NUM_ENVS"], self.env.world_state_size(),)),  #  + env.observation_space(env.agents[0]).shape[0]
            jnp.zeros((1, self.config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ENVS"], 128)
        self.critic_network_params = self.critic_network.init(_rng_critic,
                                                              cr_init_hstate,
                                                              cr_init_x)

    def _create_optimisers(self):
        if self.config["ANNEAL_LR"]:

            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (self.config["NUM_MINIBATCHES"] * self.config["UPDATE_EPOCHS"]))
                    / self.config["NUM_UPDATES"]
                )
                return self.config["LR"] * frac

            actor_tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(self.config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(self.config["LR"], eps=1e-5),
            )

        self.actor_train_state = TrainState.create(
            apply_fn=self.actor_network.apply,
            params=self.actor_network_params,
            tx=actor_tx,
        )
        self.critic_train_state = TrainState.create(
            apply_fn=self.actor_network.apply,
            params=self.critic_network_params,
            tx=critic_tx,
        )

    def setup(self):
        self._create_networks()
        self._create_optimisers()

    def _collect_trajectories(self, runner_state, update_steps):
        def _env_step(runner_state, _):
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            obs_batch = batchify(last_obs, self.env.agents, self.config["NUM_ACTORS"])
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            ac_hstate, pi = self.actor_network.apply(train_states[0].params, hstates[0], ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, self.env.agents, self.config["NUM_ENVS"], self.env.num_agents
            )

            # VALUE
            world_state = last_obs["world_state"].reshape((self.config["NUM_ACTORS"],-1))
            cr_in = (
                world_state[None, :],
                last_done[np.newaxis, :],
            )
            cr_hstate, value = self.critic_network.apply(train_states[1].params,
                                                         hstates[1],
                                                         cr_in)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, self.config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)
            info = jax.tree_map(lambda x: x.reshape((self.config["NUM_ACTORS"])), info)
            done_batch = batchify(done, self.env.agents, self.config["NUM_ACTORS"]).squeeze()
            transition = Transition(
                jnp.tile(done["__all__"], self.env.num_agents),
                done_batch,
                action.squeeze(),
                value.squeeze(),
                batchify(reward, self.env.agents, self.config["NUM_ACTORS"]).squeeze(),
                log_prob.squeeze(),
                obs_batch,
                world_state,
                info,
            )
            runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
            return runner_state, transition

        initial_hstates = runner_state[-2]
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, self.config["NUM_STEPS"]
        )

        return initial_hstates, runner_state, traj_batch

    def _compute_advantages(self, runner_state, traj_batch):
    
        train_states, env_state, last_obs, last_done, hstates, rng = runner_state
    
        last_world_state = last_obs["world_state"].reshape((self.config["NUM_ACTORS"],-1))
        cr_in = (
            last_world_state[None, :],
            last_done[np.newaxis, :],
        )
        _, last_val = self.critic_network.apply(train_states[1].params, hstates[1], cr_in)
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
            (traj_batch.obs, traj_batch.done),
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
        _, value = self.critic_network.apply(critic_params, init_hstate.transpose(), (traj_batch.world_state,  traj_batch.done)) 

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

    def _learn(self, update_state):
        (
            train_states,
            init_hstates,
            traj_batch,
            advantages,
            targets,
            rng,
        ) = update_state

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
        permutation = jax.random.permutation(self._next_rng(),
                                                self.config["NUM_ACTORS"])

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

        train_states, loss_info = jax.lax.scan(
            self._learn_step, train_states, minibatches
        )

        update_state = (
            train_states,
            init_hstates,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        return update_state, loss_info

    def _training_iteration(self, update_runner_state):
        # COLLECT TRAJECTORIES
        runner_state, update_steps = update_runner_state
        
        initial_hstates, runner_state, traj_batch = self._collect_trajectories(runner_state, update_steps)

        train_states, env_state, last_obs, last_done, hstates, rng = runner_state

        # CALCULATE ADVANTAGE
        advantages, targets = self._compute_advantages(runner_state, traj_batch)

        # UPDATE NETWORK
        ac_init_hstate = initial_hstates[0][None, :].squeeze().transpose()
        cr_init_hstate = initial_hstates[1][None, :].squeeze().transpose()

        update_state = (
            train_states,
            (ac_init_hstate, cr_init_hstate),
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            lambda state, _: self._learn(state),
            update_state, None, self.config["UPDATE_EPOCHS"]
        )
        loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
        
        train_states = update_state[0]
        metric = traj_batch.info
        rng = update_state[-1]
        metric["update_steps"] = update_steps
        jax.experimental.io_callback(self.callback.on_iteration_end,
                                     None, metric)
        update_steps = update_steps + 1
        runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
        return (runner_state, update_steps), metric

    def run(self):
        self.callback.on_train_begin(self.config)
        with jax.disable_jit(False):
            self._train()
        self.callback.on_train_end()

    def _create_init_runner_state(self):
        reset_rng = self._next_rng(self.config["NUM_ENVS"])
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ACTORS"], 128)
        cr_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ACTORS"], 128)

        runner_state = (
            (self.actor_train_state, self.critic_train_state),
            env_state,
            obsv,
            jnp.zeros((self.config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            self._next_rng(),
        )

        return runner_state

    @partial(jax.jit, static_argnums=0)
    def _train(self):
        self.setup()
        runner_state = self._create_init_runner_state()
        runner_state, metric = jax.lax.scan(
            lambda s, _: self._training_iteration(s),
            (runner_state, 0), None, self.config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}
