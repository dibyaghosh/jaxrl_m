"""Flax implementation of ResNet V1. Stolen from tps://github.com/google/flax/blob/master/examples/imagenet/models.py"""
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides, padding=((1, 1), (1, 1)))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), padding=((1, 1), (1, 1)))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides, padding=((1, 1), (1, 1)))(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, x, train: bool = True):

    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    # x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    # x = jnp.asarray(x, self.dtype)
    # # x = nn.log_softmax(x)  # to match the Torch implementation at https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    return x



# function to translate pytorch param keys to jax:
def translate_key(pytorch_name, resnet="50"):
    if resnet == "50":
      block_name = "BottleneckResNetBlock" 
      layer_list = [3, 4, 6, 3]
    elif resnet == "18":
      block_name = "ResNetBlock"
      layer_list = [2, 2, 2, 2]
    else:
      raise RuntimeError("Choose one of {'18', '50'}.")

    split = pytorch_name.split('.')
    
    # fc.{weight|bias} -> (params, Dense_0, {kernel|bias})
    if len(split) == 2 and split[0] == 'fc':
      return ("params", "Dense_0", "bias" if split[1] == "bias" else "kernel")

    # layer{i}.{j}.bn{k}.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, BottleneckBlock_{}, BatchNorm_{}, {scale|bias|mean|var})
    if len(split) == 4 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2][:-1] == 'bn':
      if split[3] in ['num_batches_tracked']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params" if split[3] in ["weight", "bias"] else "batch_stats",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              f"BatchNorm_{int(split[2][-1]) - 1}",
              "scale" if split[3] == "weight" else split[3][8:] if split[3] in ["running_mean", "running_var"] else "bias")

    # layer{i}.{j}.conv{k}.weight -> (params, BottleneckBlock_{}, Conv_{}, kernel)
    if len(split) == 4 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2][:-1] == 'conv':
      if split[3] in ['bias']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              f"Conv_{int(split[2][-1]) - 1}",
              "kernel")

    # bn1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, bn_init, {scale|bias|mean|var})
    if len(split) == 2 and split[0] == "bn1":
      if split[1] in ['num_batches_tracked']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params" if split[1] in ["weight", "bias"] else "batch_stats",
              "bn_init",
              "scale" if split[1] == "weight" else split[1][8:] if split[1] in ["running_mean", "running_var"] else "bias")

    # conv1.weight -> (params, conv_init, kernel)
    if len(split) == 2 and split[0] == "conv1":
      if split[1] in ['bias']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params", "conv_init", "kernel")

    # layer{i}.{j}.downsample.0.weight -> (params, BottleneckBlock_{}, conv_proj, kernel)
    if len(split) == 5 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2] == 'downsample' and split[3] == '0':
      if split[4] in ['bias']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              "conv_proj",
              "kernel")

    # layer{i}.{j}.downsample.1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, BottleneckBlock_{}, norm_proj, {scale|bias|mean|var})
    if len(split) == 5 and split[0][:-1] == 'layer' and split[1].isdigit() and split[2] == 'downsample' and split[3] == '1':
      if split[4] in ['num_batches_tracked']:
        print(f"NO PATTERN MATCHES: {pytorch_name}")
        return None

      return ("params" if split[4] in ["weight", "bias"] else "batch_stats",
              f"{block_name}_{sum(layer_list[:int(split[0][-1]) - 1]) + int(split[1])}",
              "norm_proj",
              "scale" if split[4] == "weight" else split[4][8:] if split[4] in ["running_mean", "running_var"] else "bias")


    print(f"NO PATTERN MATCHES: {pytorch_name}")
    return None

def convert_pytorch_to_jax(pytorch_statedict, jax_variables, resnet_type="50"):
  from flax.traverse_util import flatten_dict, unflatten_dict
  from flax.core import freeze, unfreeze


  jax_params = flatten_dict(unfreeze(jax_variables))
  # create a new dict the same shape as the original jax params dict but filled with the (transposed) pytorch weights
  jax2pytorch = {translate_key(key, resnet_type): key for key in pytorch_statedict.keys() if translate_key(key) is not None}
  pytorch_params = {k: v.numpy().T if len(v.shape) != 4 else v.numpy().transpose((2, 3, 1, 0))
                    for k, v in pytorch_statedict.items()}
  new_jax_params = freeze(unflatten_dict({key: pytorch_params[jax2pytorch[key]] for key in jax_params.keys()}))
  return new_jax_params

vanilla_resnetv1_configs = {
    'resnetv1-18': partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock),
    'resnetv1-34': partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock),
    'resnetv1-50': partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock),
    'resnetv1-101': partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock),
    'resnetv1-152': partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock),
    'resnetv1-200': partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock)
}
