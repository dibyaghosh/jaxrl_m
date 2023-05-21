from jaxrl_m.vision import resnet_v1
import flax.linen as nn
import jax
import jax.numpy as jnp
import functools as ft

import flax.linen as nn
from flax.core import freeze, unfreeze
import pickle
import flax.training.checkpoints as checkpoints
from flax.traverse_util import flatten_dict, unflatten_dict

def preprocess_observations(obs,
                            normalize_imagenet=False,
                            resize=False,
                            final_shape=(224, 224),
                            center_crop=False,
                            pre_crop_shape=(256, 256)):
    if resize:
        if obs.shape[-3] != final_shape[0] or obs.shape[-2] != final_shape[1]: # Already resized
            resize_shape = pre_crop_shape if center_crop else final_shape
            if obs.shape[-3] != resize_shape[0] or obs.shape[-2] != resize_shape[1]:
                print('Resizing to %s' % str(resize_shape))
                obs = jax.image.resize(obs, (*obs.shape[:-3], *resize_shape, 3), method='bilinear')

            if center_crop:
                start_y, start_x = (pre_crop_shape[0] - final_shape[0]) // 2, (pre_crop_shape[1] - final_shape[1]) // 2
                obs = obs[..., start_y:start_y + final_shape[0], start_x:start_x + final_shape[1], :]
                print('Cropping to %s' % str(obs.shape))

    if normalize_imagenet:
        obs = obs / 255.0
        obs = obs - jnp.array([0.485, 0.456, 0.406])
        obs = obs / jnp.array([0.229, 0.224, 0.225])

    return obs

class ResizingEncoder(nn.Module):
    encoder: nn.Module
    normalize_imagenet: bool = False

    resize: bool = True
    final_shape: tuple = (224, 224)
    center_crop: bool = False
    pre_crop_shape: tuple = (256, 256)

    freeze_encoder: bool = False
    default_kwargs: dict = None

    @nn.compact
    def __call__(self, observations, **kwargs):
        no_batch_dim = len(observations.shape) == 3
        if no_batch_dim:
            print('Adding batch dimension')
            observations = jnp.expand_dims(observations, 0)
        observations = preprocess_observations(observations, self.normalize_imagenet, self.resize, self.final_shape, self.center_crop, self.pre_crop_shape)
        if self.default_kwargs is not None:
            kwargs = {**self.default_kwargs, **kwargs}
        output = self.encoder(observations, **kwargs)
        if self.freeze_encoder:
            output = jax.lax.stop_gradient(output)        

        if no_batch_dim:
            print('Removing batch dimension')
            output = jnp.squeeze(output, 0)
        return output

def merge_dicts(new_dict, restore_from, allow_extra=True, allow_missing=True):
    new_dict_flat = flatten_dict(new_dict)
    restore_from_flat = flatten_dict(restore_from)
    
    missing_from_new = set(restore_from_flat.keys()) - set(new_dict_flat.keys())
    missing_from_restore = set(new_dict_flat.keys()) - set(restore_from_flat.keys())
    if not allow_extra:
        assert len(missing_from_new) == 0, 'Keys missing from new dict: %s' % str(missing_from_new)
    elif len(missing_from_new) > 0:
        print('Keys missing from target dict: %s' % str(missing_from_new))
    
    if not allow_missing:
        assert len(missing_from_restore) == 0, 'Keys missing from restore dict: %s' % str(missing_from_restore)
    elif len(missing_from_restore) > 0:
        print('Keys missing from restore dict: %s' % str(missing_from_restore))
    
    new_dict_flat.update(restore_from_flat)
    return unflatten_dict(new_dict_flat)

def load_pretrained_params(pretrained_params, pretrained_extra_variables, params, extra_variables, prefix_key='encoder/encoder'):
    params, extra_variables = unfreeze(params), unfreeze(extra_variables)

    sp = params
    prefix_list = prefix_key.split('/') if prefix_key != '' else []
    for k in prefix_list:
        sp = sp[k]
    assert sp.keys() == pretrained_params.keys(), (sp.keys(), pretrained_params.keys())
    merge_dicts(sp, pretrained_params, True, True) # Just checking
    sp.update(pretrained_params)

    for k in pretrained_extra_variables:
        sp = extra_variables[k]
        for kk in prefix_list:
            sp = sp[kk]
        assert sp.keys() == pretrained_extra_variables[k].keys(), (sp.keys(), pretrained_extra_variables[k].keys())
        sp.update(pretrained_extra_variables[k])

    return freeze(params), freeze(extra_variables)

