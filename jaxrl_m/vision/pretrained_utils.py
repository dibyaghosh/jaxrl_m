"""
    Defines `load_pretrained_params`, a convenience utility for loading pretrained parameters into a model.

    This is useful for splicing pretrained encoders into models with a different downstream head.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import functools as ft

import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import flax
import logging

def _pretty_repr_missing_keys(missing_keys_flat):
    from collections import defaultdict
    missing_keys = defaultdict(list)
    for k in missing_keys_flat:
        missing_keys[k[:-1]].append(k[-1])

    missing_keys = unflatten_dict({k: tuple(v) for k, v in missing_keys.items()})
    return flax.core.frozen_dict.pretty_repr(missing_keys)

def _merge_dicts(new_dict, restore_from, allow_extra=True, allow_missing=True, prefix=tuple()):
    new_dict_flat_full = flatten_dict(new_dict)
    restore_from_flat = flatten_dict(restore_from)

    new_dict_flat = {
        k[len(prefix):]: v
        for k, v in new_dict_flat_full.items()
        if k[:len(prefix)] == prefix
    }
    
    missing_from_new = set(restore_from_flat.keys()) - set(new_dict_flat.keys())
    missing_from_restore = set(new_dict_flat.keys()) - set(restore_from_flat.keys())

    if len(missing_from_new) > 0:
        logging.error('Keys missing from target dict: ')
        logging.error(_pretty_repr_missing_keys(missing_from_new))
        if not allow_extra:
            raise ValueError()

    elif len(missing_from_restore) > 0:
        logging.error('Keys missing from restore dict:')
        logging.error(_pretty_repr_missing_keys(missing_from_restore))
        if not allow_missing:
            raise ValueError()
    
    keys_to_update = set(new_dict_flat.keys()) & set(restore_from_flat.keys())
    new_dict_flat.update({k: restore_from_flat[k] for k in keys_to_update})
    new_dict_flat_full.update({prefix + k: v for k, v in new_dict_flat.items()})

    return unflatten_dict(new_dict_flat_full)

def load_pretrained_params(pretrained_params, pretrained_extra_variables, params, extra_variables, prefix_key='encoder/encoder'):
    """
        Args:
            pretrained_params: parameters for the pretrained encoder
            pretrained_extra_variables: extra variables for the pretrained encoder
            params: parameters for the downstream model being loaded into
            extra_variables: extra variables for the downstream model being loaded into
            prefix_key: where the encoder parameters should go in the downstream model

        Returns:
            params, extra_variables: the updated parameters and extra variables
    """
    params, extra_variables = unfreeze(params), unfreeze(extra_variables)
    prefix_tuple = tuple(prefix_key.split('/')) if prefix_key != '' else tuple()
    params = _merge_dicts(params, pretrained_params, True, True, prefix_tuple) # Just checking

    if not (set(pretrained_extra_variables.keys()) <= set(extra_variables.keys())):
        logging.error('Extra variables missing from target dict: ')
        logging.error(set(pretrained_extra_variables.keys()) - set(extra_variables.keys()))

    for k in pretrained_extra_variables:
        extra_variables[k] = _merge_dicts(
            extra_variables[k],
            pretrained_extra_variables[k],
            True, True, prefix_tuple)

    return freeze(params), freeze(extra_variables)


