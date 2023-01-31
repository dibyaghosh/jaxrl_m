"""Implementations of behavioral cloning in discrete action spaces."""
import functools
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
import distrax

from jaxrl_m.typing import *
from jaxrl_m.common import TrainState
from jaxrl_m.networks import WithEncoder, DiscreteCritic

import ml_collections


class BCAgent(flax.struct.PyTreeNode):
    model: TrainState
    
    @functools.partial(jax.pmap, axis_name='pmap')
    def update(agent, batch: Batch):

        def loss_fn(params):
            logits = agent.model(batch['observations'], params=params)
            dist = distrax.Categorical(logits=logits)
            probs = jax.nn.softmax(logits)
            accuracy = jnp.mean(jnp.equal(jnp.argmax(probs, axis=1), batch['actions']))
            actor_loss = -1 * dist.log_prob(batch['actions']).mean()

            return actor_loss, {
                'actor_loss': actor_loss,
                'accuracy': accuracy,
                'entropy': dist.entropy().mean(),
            }

        new_model, info = agent.model.apply_loss_fn(loss_fn=loss_fn, has_aux=True, pmap_axis='pmap')
        return agent.replace(model=new_model), info

    @functools.partial(jax.jit, static_argnames=('argmax'))
    def sample_actions(agent, observations, *, seed, temperature=1.0, argmax=False):
        logits = agent.model(observations)
        dist = distrax.Categorical(logits=logits / temperature)

        if argmax:
            return dist.mode()
        else:
            return dist.sample(seed=seed)

def create_bc_learner(
                seed: int,
                observations: jnp.ndarray,
                n_actions: int,
                encoder_def: nn.Module,
                network_kwargs: dict = {
                    'hidden_dims': [256, 256],
                },
                optim_kwargs: dict = {
                    'learning_rate': 6e-5,
                },
                **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)

        network_def = DiscreteCritic(n_actions=n_actions, **network_kwargs)
        model_def = WithEncoder(encoder=encoder_def, model=network_def)
                
        params = model_def.init(rng, observations)['params']
        model = TrainState.create(model_def, params, tx=optax.adam(**optim_kwargs))

        return BCAgent(model)


def get_default_config():
    config = ml_collections.ConfigDict({
        'algo': 'bc',
        'optim_kwargs': {
            'learning_rate': 6e-5,
        },
        'network_kwargs': {
            'hidden_dims': (256, 256),
        }
    })

    return config