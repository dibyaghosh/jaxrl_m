# Copyright 2022 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BiT models as in the paper (ResNet V2) w/ loading of public weights.

See reproduction proof: http://(internal link)
"""

import functools
import re
from typing import Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
import jax
import logging

def standardize(x, axis, eps):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x


# Defined our own, because we compute normalizing variance slightly differently,
# which does affect performance when loading pre-trained weights!
class GroupNorm(nn.Module):
  """Group normalization (arxiv.org/abs/1803.08494)."""
  ngroups: int = 32

  @nn.compact
  def __call__(self, x):

    input_shape = x.shape
    group_shape = x.shape[:-1] + (self.ngroups, x.shape[-1] // self.ngroups)

    x = x.reshape(group_shape)

    # Standardize along spatial and group dimensions
    x = standardize(x, axis=[1, 2, 4], eps=1e-5)
    x = x.reshape(input_shape)

    bias_scale_shape = tuple([1, 1, 1] + [input_shape[-1]])
    x = x * self.param('scale', nn.initializers.ones, bias_scale_shape)
    x = x + self.param('bias', nn.initializers.zeros, bias_scale_shape)
    return x


class StdConv(nn.Conv):

  def param(self, name, *a, **kw):
    param = super().param(name, *a, **kw)
    if name == 'kernel':
      param = standardize(param, axis=[0, 1, 2], eps=1e-10)
    return param


class RootBlock(nn.Module):
  """Root block of ResNet."""
  width: int

  @nn.compact
  def __call__(self, x):
    x = StdConv(self.width, (7, 7), (2, 2), padding=[(3, 3), (3, 3)],
                use_bias=False, name='conv_root')(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
    return x


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""
  nmid: Optional[int] = None
  strides: Sequence[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    nmid = self.nmid or x.shape[-1] // 4
    nout = nmid * 4
    conv = functools.partial(StdConv, use_bias=False)

    residual = x
    x = GroupNorm(name='gn1')(x)
    x = nn.relu(x)

    if x.shape[-1] != nout or self.strides != (1, 1):
      residual = conv(nout, (1, 1), self.strides, name='conv_proj')(x)

    x = conv(nmid, (1, 1), name='conv1')(x)
    x = GroupNorm(name='gn2')(x)
    x = nn.relu(x)
    x = conv(nmid, (3, 3), self.strides, padding=[(1, 1), (1, 1)],
             name='conv2')(x)
    x = GroupNorm(name='gn3')(x)
    x = nn.relu(x)
    x = conv(nout, (1, 1), name='conv3')(x)

    return x + residual


class ResNetStage(nn.Module):
  """A stage (sequence of same-resolution blocks)."""
  block_size: int
  nmid: Optional[int] = None
  first_stride: Sequence[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    out = {}
    x = out['unit01'] = ResidualUnit(
        self.nmid, strides=self.first_stride, name='unit01')(x)
    for i in range(1, self.block_size):
      x = out[f'unit{i+1:02d}'] = ResidualUnit(
          self.nmid, name=f'unit{i+1:02d}')(x)
    return x, out


class Model(nn.Module):
  """ResNetV2."""
  num_classes: int
  width: int = 1
  depth: Union[int, Sequence[int]] = 50  # 5/101/152, or list of block depths.

  @nn.compact
  def __call__(self, x, *, train=False):
    if x.dtype in [jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]:
        logging.warning('Observations seem to be [0, 255] int encoded images. This encoder expects float-normalized inputs.')
        logging.warning('Use PreprocessEncoder to wrap this encoder to automatically normalize uint images to floats.')

    blocks = get_block_desc(self.depth)
    width = int(64 * self.width)
    out = {}

    x = out['stem'] = RootBlock(width=width, name='root_block')(x)

    # Blocks
    x, out['stage1'] = ResNetStage(blocks[0], nmid=width, name='block1')(x)
    for i, block_size in enumerate(blocks[1:], 1):
      x, out[f'stage{i + 1}'] = ResNetStage(
          block_size, width * 2 ** i,
          first_stride=(2, 2), name=f'block{i + 1}')(x)

    # Pre-head
    x = out['norm_pre_head'] = GroupNorm(name='norm-pre-head')(x)
    x = out['pre_logits_2d'] = nn.relu(x)
    x = out['pre_logits'] = jnp.mean(x, axis=(1, 2))

    # Head
    if self.num_classes:
      head = nn.Dense(self.num_classes, name='head',
                      kernel_init=nn.initializers.zeros)
      out['logits_2d'] = head(out['pre_logits_2d'])
      x = out['logits'] = head(out['pre_logits'])

    return x#, out

def get_block_desc(depth):
  if isinstance(depth, list):  # Be robust to silly mistakes.
    depth = tuple(depth)
  return {
      26: [2, 2, 2, 2],  # From timm, gets ~75% on ImageNet.
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }.get(depth, depth)

import functools as ft
resnetv2_configs = {
  'resnetv2-26-1': ft.partial(Model, num_classes=None, depth=26),
  'resnetv2-50-1': ft.partial(Model, num_classes=None, depth=50),
}