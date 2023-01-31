"""Implementations of behavioral cloning in continuous action spaces."""
import functools
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax

from jaxrl_m.typing import *
from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy, WithEncoder

import ml_collections

class BCAgent(flax.struct.PyTreeNode):
    model: TrainState

    @functools.partial(jax.pmap, axis_name='pmap')
    def update(agent, batch: Batch):
        def loss_fn(model_params):
            dist = agent.model(batch['observations'], params=model_params)
            log_probs = dist.log_prob(batch['actions'])
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            return actor_loss, {
                'actor_loss': actor_loss,
                'mse_loss': ((dist.mode() - batch['actions']) ** 2).sum(-1).mean(),
                'mean_std': actor_std.mean(),
                'max_std': actor_std.max(),
                'min_std': actor_std.min(),
            }

        new_model, info = agent.model.apply_loss_fn(loss_fn=loss_fn, has_aux=True, pmap_axis='pmap')
        return agent.replace(model=new_model), info

    @functools.partial(jax.jit, static_argnames=('argmax'))
    def sample_actions(agent,
                       observations: np.ndarray,
                       *,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       argmax=False) -> jnp.ndarray:
        dist = agent.model(observations, temperature=temperature)
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

def create_bc_learner(
            seed: int,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            encoder_def: nn.Module,
            actor_kwargs: dict = {
                'hidden_dims': [256, 256],
                'tanh_squash_distribution': False,
                'state_dependent_std': False,
            },
            optim_kwargs: dict = {
                'learning_rate': 6e-5,
            },
            **kwargs):
    print('Extra kwargs:', kwargs)
    rng = jax.random.PRNGKey(seed)

    actor_def = Policy(action_dim=actions.shape[-1], **actor_kwargs)
    model_def = WithEncoder(encoder=encoder_def, model=actor_def)

    params = model_def.init(rng, observations)['params']
    model = TrainState.create(model_def, params, tx=optax.adam(**optim_kwargs))

    return BCAgent(model)

def get_default_config():
    config = ml_collections.ConfigDict({
        'algo': 'bc',
        'optim_kwargs': {
            'learning_rate': 6e-5,
        },
        'actor_kwargs': {
            'hidden_dims': [256, 256],
            'tanh_squash_distribution': False,
            'state_dependent_std': False,
        }
    })

    return config