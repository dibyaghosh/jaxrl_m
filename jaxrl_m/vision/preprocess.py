import flax.linen as nn
import jax
import jax.numpy as jnp
import functools as ft

import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import flax
import logging
from typing import Union

def preprocess_observations(obs,
                            normalize: bool = True,
                            normalization_params: Union[str, dict] = None,
                            resize: bool = False,
                            resize_shape: tuple = (224, 224),
    ):
    if resize:
        if obs.shape[-3] != resize_shape[0] or obs.shape[-2] != resize_shape[1]: # Already resized
            if obs.shape[-3] != resize_shape[0] or obs.shape[-2] != resize_shape[1]:
                logging.info('Resizing to %s' % str(resize_shape))
                obs = jax.image.resize(obs, (*obs.shape[:-3], *resize_shape, obs.shape[-1]), method='bilinear')

    if normalize:
        obs = obs / 255.0

        if isinstance(normalization_params, str):
            normalization_params = DEFAULT_NORMALIZATIONS[normalization_params]
            
        if normalization_params is not None:
            obs = obs - jnp.array(normalization_params['mean'])
            obs = obs / jnp.array(normalization_params['std'])

    return obs

IMAGENET_NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

VIT_NORMALIZATION = {
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
}

DEFAULT_NORMALIZATIONS = {
    'imagenet': IMAGENET_NORMALIZATION,
    'vit': VIT_NORMALIZATION,
    'none': None,
}

class PreprocessEncoder(nn.Module):
    """
        Preprocesses image observations and passes them to an encoder.
        Expects unnormalized images in [0, 255] range.

        By default, this will only normalize [0, 255] images to [0, 1].

        You may additionally specify options that will:
            - normalize images differently (e.g. ImageNet normalization),
            - resize images to a specific size before passing them to the encoder.
            - freeze the encoder (useful for finetuning)
            - pass additional kwargs to the encoder

    """
    encoder: nn.Module

    normalize: bool = True
    normalization_params: Union[str, dict] = None
    resize: bool = False # If true, will resize images to resize_shape
    resize_shape: tuple = (224, 224)


    freeze_encoder: bool = False
    default_kwargs: dict = None # Default arguments, will be overriden by kwargs passed to __call__
    force_kwargs: dict = None # Will override arguments passed to __call__

    @nn.compact
    def __call__(self, observations, **kwargs):
        no_batch_dim = len(observations.shape) == 3
        if no_batch_dim:
            observations = jnp.expand_dims(observations, 0)

        observations = preprocess_observations(observations, self.normalize, self.normalization_params, self.resize, self.resize_shape)

        default_kwargs = self.default_kwargs if self.default_kwargs else {}
        force_kwargs = self.force_kwargs if self.force_kwargs else {}
        kwargs = {**default_kwargs, **kwargs, **force_kwargs}

        output = self.encoder(observations, **kwargs)
        if self.freeze_encoder:
            output = jax.lax.stop_gradient(output)        

        if no_batch_dim:
            output = jnp.squeeze(output, 0)

        return output

# Basic preprocessor that only normalizes [0, 255] images to [0, 1]
BasicPreprocessEncoder = ft.partial(PreprocessEncoder, normalize=True, normalization_params=None, resize=False, freeze_encoder=False, default_kwargs=None, force_kwargs=None)
