# Implementation modified from https://github.com/young-geng/m3ae_public/tree/master/m3ae

from typing import Callable, Optional, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from functools import partial

LayerNorm = partial(nn.LayerNorm, epsilon=1e-6)

def extract_patches(inputs, patch_size):
    batch, height, width, channels = inputs.shape
    height, width = height // patch_size, width // patch_size
    x = jnp.reshape(inputs, (batch, height, patch_size, width, patch_size, channels))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * width, patch_size**2 * channels))
    return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0
    )


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    return jnp.expand_dims(pos_embed, 0)

def get_fixed_pos_embed(embed_dim, length):
    pos_embed_nocls = get_1d_sincos_pos_embed(embed_dim, length - 1)
    pos_embed = jnp.concatenate([
        jnp.zeros([1, 1, embed_dim], dtype=jnp.float32),
        pos_embed_nocls
    ], axis=1)
    return pos_embed

def index_sequence(x, ids):
    return x[:, ids, ...]


def random_masking(x, rng, keep_len, padding_mask=None):
    batch, length, dim = x.shape
    noise = jax.random.uniform(rng, (length,), dtype=jnp.float32)
    ids_shuffle = jnp.argsort(noise, axis=0)
    ids_restore = jnp.argsort(ids_shuffle, axis=0)
    kept = index_sequence(x, ids_shuffle[:keep_len])
    mask = jnp.ones([batch, length], dtype=jnp.float32)
    mask = mask.at[:, :keep_len].set(0.0)
    mask = index_sequence(mask, ids_restore)

    if padding_mask is None:
        return kept, mask, ids_restore

    padding_mask_kept = index_sequence(padding_mask, ids_shuffle[:keep_len])
    return kept, mask, ids_restore, padding_mask_kept

class PatchEmbed(nn.Module):
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768

    def setup(self):
        self.proj = nn.Conv(features=self.embed_dim,
                            kernel_size=(self.patch_size, self.patch_size),
                            strides=(self.patch_size, self.patch_size),
                            padding=0)


    def __call__(self, x):
        B, H, W, C = x.shape
        x = self.proj(x) # B, H // patch_size, W // patch_size, self.embed_dim
        x = jnp.reshape(x, (B, -1, self.embed_dim))
        return x


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    depth: int
    input_norm: bool = True

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        if self.input_norm:
            x = LayerNorm()(x)

        for i in range(self.depth):
            y = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            y = nn.gelu(y, approximate=False)
            y = LayerNorm()(y)
            if i > 0:
                x = x + y
            else:
                x = y

        x = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class DropPath(nn.Module):
    dropout_prob: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        if deterministic or self.dropout_prob == 0.0:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = keep_prob + jax.random.uniform(rng, shape, dtype=jnp.float32)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = nn.Dense(
            self.dim, kernel_init=self.kernel_init, name="fc1"
        )(inputs)

        x = nn.gelu(x, approximate=False)
        if self.dropout > 0.0:
            x = nn.Dropout(self.dropout)(x, deterministic)

        x = nn.Dense(
            self.out_dim, kernel_init=self.kernel_init, name="fc2"
        )(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    """Modified from flax_models to support mask"""

    dim: int
    num_heads: int = 8
    use_bias: bool = False
    att_drop: float = 0
    proj_drop: float = 0
    kernel_init: Callable = nn.initializers.xavier_uniform()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None, padding_mask=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        batch, n, channels = inputs.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            name='qkv'
        )(inputs)
        qkv = jnp.reshape(
            qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale

        if padding_mask is not None:
            padding_mask = jnp.expand_dims(jnp.expand_dims(padding_mask, 1), 1)
            padding_mask = jnp.broadcast_to(padding_mask, attention.shape)
            attention = jnp.where(padding_mask > 0, jnp.array(-1e7), attention)

        attention = nn.softmax(attention, axis=-1)
        attention = nn.Dropout(self.att_drop)(attention, deterministic)

        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = nn.Dense(
            self.dim, kernel_init=nn.initializers.xavier_uniform(),
            name='proj'
        )(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


class Block(nn.Module):
    emb_dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    att_drop: float = 0.0
    drop: float = 0.0
    drop_path: float = 0.0

    @nn.compact
    def __call__(self, inputs, deterministic=False, padding_mask=None):
        x = LayerNorm(name='norm1')(inputs)
        x = Attention(
            self.emb_dim, self.num_heads, True, self.att_drop, self.drop,
            name='attn'
        )(x, deterministic, padding_mask)
        x = DropPath(self.drop_path)(x, deterministic)
        inputs = inputs + x

        x = LayerNorm(name='norm2')(inputs)
        x = TransformerMLP(
            self.emb_dim * self.mlp_ratio, self.emb_dim, self.drop,
            name='mlp'
        )(x, deterministic)

        x = DropPath(self.drop_path)(x, deterministic)

        return inputs + x


class Transformer(nn.Module):
    emb_dim: int = 1024
    depth: int = 24
    att_drop: float = 0
    drop: float = 0
    drop_path: float = 0
    num_heads: int = 16
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, deterministic=False, padding_mask=None):
        for n in range(self.depth):
            x = Block(
                self.emb_dim,
                self.num_heads,
                self.mlp_ratio,
                self.att_drop,
                self.drop,
                self.drop_path,
                name=f'blocks_{n}'
            )(x, deterministic, padding_mask)
        x = LayerNorm(name='norm')(x)
        return x

class MaskedAutoencoder(nn.Module):
    emb_dim: int
    dec_emb_dim: int
    depth: int
    dec_depth: int
    num_heads: int
    dec_num_heads: int
    mlp_ratio: int

    output_head_depth: int = 0
    att_drop: float = 0.0
    drop: float = 0.0
    drop_path: float = 0.0

    image_mask_ratio: float = 0.75
    use_type_embedding: bool = False
    image_output_dim: int = 224
    patch_size: int = 16

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise', 'drop_path', 'dropout')

    @nn.nowrap
    def no_decay_list(self):
        # model specific no decay list
        no_decay = [
            'cls_token', 'encoder_image_type_embedding', 'image_mask_embedding',
            'bias',
        ]
        return no_decay

    def setup(self):
        self.patch_embed = PatchEmbed(embed_dim=self.emb_dim)
        

        # CLS and masks
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.emb_dim),
        )
        n_patches = (self.image_size // self.patch_size) ** 2
        self.positional_embedding = self.param(
            "pos_embed",
            lambda rng, shape: get_fixed_pos_embed(shape[2], shape[1]),
            (1, n_patches + 1, self.emb_dim)
        )

        self.image_mask_embedding = self.param(
            "mask_token",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.dec_emb_dim),
        )

        self.encoder = Transformer(
            emb_dim=self.emb_dim,
            depth=self.depth,
            att_drop=self.att_drop,
            drop=self.drop,
            drop_path=self.drop_path,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )

        self.decoder = Transformer(
            emb_dim=self.dec_emb_dim,
            depth=self.dec_depth,
            att_drop=self.att_drop,
            drop=self.drop,
            drop_path=self.drop_path,
            num_heads=self.dec_num_heads,
            mlp_ratio=self.mlp_ratio,
        )

        self.decoder_input_projection = nn.Dense(
            self.dec_emb_dim,
            kernel_init=nn.initializers.xavier_uniform()
        )

        self.decoder_image_output = MLP(
            self.dec_emb_dim,
            self.image_output_dim,
            self.output_head_depth,
            input_norm=self.output_head_depth > 0,
            name='decoder_image_output',
        )

    def forward_representation(self, image, train=True):
        batch_size = image.shape[0]
        # image_x = self.image_embedding(image)
        image_x = self.patch_embed(image)
        cls_token = jnp.broadcast_to(
            self.cls_token, (batch_size, 1, self.emb_dim)
        )
        x = jnp.concatenate([cls_token, image_x], axis=1)
        x = x + self.positional_embedding

        x = self.encoder(x, not train)
        return x

    def forward_encoder(self, image, deterministic=False):
        batch_size = image.shape[0]
        # image_x = self.image_embedding(image)
        image_x = self.patch_embed(image)
        image_keep_length = int(image_x.shape[1] * (1.0 - self.image_mask_ratio))

        image_x = (
            image_x
            + self.positional_embedding[:, 1:, :]
        )
        image_x, image_mask, image_ids_restore = random_masking(
            image_x, self.make_rng("noise"), image_keep_length
        )
        cls_token = jnp.broadcast_to(
            self.cls_token + self.positional_embedding[:, :1, :],
            (batch_size, 1, self.emb_dim)
        )
        x = jnp.concatenate([cls_token, image_x], axis=1)
        x = self.encoder(x, deterministic)

        return x, image_mask, image_ids_restore

    def forward_decoder(self, x, image_ids_restore, deterministic=False):
        batch_size = x.shape[0]
        image_keep_length = int(image_ids_restore.shape[0] * (1.0 - self.image_mask_ratio))
        x = self.decoder_input_projection(x)
        encoder_cls = x[:, :1, :]
        image_x = x[:, 1:, :]

        masked_image_x = jnp.broadcast_to(
            self.image_mask_embedding,
            (
                batch_size,
                image_ids_restore.shape[0] - image_keep_length,
                self.dec_emb_dim,
            ),
        )

        image_x = index_sequence(
            jnp.concatenate([image_x, masked_image_x], axis=1), image_ids_restore
        )

        image_x = (
            image_x
            + get_2d_sincos_pos_embed(self.dec_emb_dim, image_ids_restore.shape[0])
        )

        x = jnp.concatenate([encoder_cls, image_x], axis=1)
        x = self.decoder(x, deterministic)
        image_x = x[:, 1:, :]
        image_output = self.decoder_image_output(image_x)

        return image_output

    def __call__(self, image, deterministic=False):
        return self.forward_representation(image, deterministic)
    
    def encode_and_decode(self, image, deterministic=False):
        x, image_mask, image_ids_restore = self.forward_encoder(image, deterministic)
        image_output = self.forward_decoder(x, image_ids_restore, deterministic)
        return image_output, image_mask, x

class MaskedAutoencoderVIT(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int

    att_drop: float = 0.0
    drop: float = 0.0
    drop_path: float = 0.0

    patch_size: int = 16
    image_size: int = 224
    output_type: str = 'tokens'


    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise', 'drop_path', 'dropout')

    @nn.nowrap
    def no_decay_list(self):
        # model specific no decay list
        no_decay = [
            'cls_token', 'encoder_image_type_embedding', 'image_mask_embedding',
            'bias',
        ]
        return no_decay

    def setup(self):
        self.patch_embed = PatchEmbed(patch_size=self.patch_size,
                                      embed_dim=self.emb_dim)
        # CLS and masks
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02, dtype=jnp.float32),
            (1, 1, self.emb_dim),
        )

        n_patches = (self.image_size // self.patch_size) ** 2
        self.positional_embedding = self.param(
            "pos_embed",
            lambda rng, shape: get_fixed_pos_embed(shape[2], shape[1]),
            (1, n_patches + 1, self.emb_dim)
        )

        self.encoder = Transformer(
            emb_dim=self.emb_dim,
            depth=self.depth,
            att_drop=self.att_drop,
            drop=self.drop,
            drop_path=self.drop_path,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )

    def __call__(self, image, train=True):
        batch_size = image.shape[0]

        image_x = self.patch_embed(image)

        cls_token = jnp.broadcast_to(
            self.cls_token, (batch_size, 1, self.emb_dim)
        )
        x = jnp.concatenate([cls_token, image_x], axis=1)

        x = x + self.positional_embedding

        x = self.encoder(x, not train)

        if self.output_type == 'tokens':
            x = x
        elif self.output_type == 'cls':
            x = x[:, 0]
        elif self.output_type == 'pooled':
            x = x[:, 1:, :].mean(axis=1)
        else:
            raise ValueError(f'Unknown output type {self.output_type}')

        return x

def map_to_jax(pytorch_key):
    if 'blocks' in pytorch_key[0]:
        if 'decoder' in pytorch_key[0]:
            jax_key = ('decoder', f'blocks_{pytorch_key[1]}') + pytorch_key[2:]
        else:
            jax_key = ('encoder', f'blocks_{pytorch_key[1]}') + pytorch_key[2:]
    else:
        if pytorch_key[0] == 'decoder_pred' :
            jax_key = ('decoder_image_output', 'Dense_0', *pytorch_key[1:])
#         elif 'patch_embed' == pytorch_key[0]:
#             jax_key = ('image_embedding', *pytorch_key[1:])
        elif 'decoder_embed' == pytorch_key[0]:
            jax_key = ('decoder_input_projection', *pytorch_key[1:])
        elif 'decoder' in pytorch_key[0]:
            jax_key = ('decoder', pytorch_key[0].partition('_')[2], *pytorch_key[1:])
        else:
            if pytorch_key[0] in ['cls_token', 'mask_token', 'patch_embed', 'pos_embed']:
                jax_key = pytorch_key
            else:
                jax_key = ('encoder', *pytorch_key)
        
    
    if jax_key[-1] == "weight":
        if 'norm' in jax_key[-2]:
            jax_key = jax_key[:-1] + ("scale",)
        else:
            jax_key = jax_key[:-1] + ("kernel",)
    return jax_key


def pytorch_statedict_to_jax(state_dict):
    pytorch_dict = {tuple(k.split('.')): v for k, v in state_dict['model'].items()}
    
    jax_flat_dict = {map_to_jax(k): jnp.asarray(v) for k, v in pytorch_dict.items()}
    for k in jax_flat_dict:
        if k[-1] == 'kernel':
            kernel = jax_flat_dict[k]
            if kernel.ndim > 2: # Conv
                kernel = jnp.transpose(kernel, (2, 3, 1, 0))
            else:
                kernel = jnp.transpose(kernel, (1, 0))
            jax_flat_dict[k] = kernel
    return flax.traverse_util.unflatten_dict(jax_flat_dict)


transformer_config_dicts = {
    'small': {
        'emb_dim': 384,
        'dec_emb_dim': 512,
        'depth': 12,
        'dec_depth': 8,
        'num_heads': 6,
        'dec_num_heads': 16,
        'mlp_ratio': 4,
    },

    'base': {
        'emb_dim': 768,
        'dec_emb_dim': 512,
        'depth': 12,
        'dec_depth': 8,
        'num_heads': 12,
        'dec_num_heads': 16,
        'mlp_ratio': 4,
    },

    'large': {
        'emb_dim': 1024,
        'dec_emb_dim': 512,
        'depth': 24,
        'dec_depth': 8,
        'num_heads': 16,
        'dec_num_heads': 16,
        'mlp_ratio': 4,
    },

    'huge': {
        'emb_dim': 1280,
        'dec_emb_dim': 512,
        'depth': 32,
        'dec_depth': 8,
        'num_heads': 16,
        'dec_num_heads': 16,
        'mlp_ratio': 4,
    },

    'debug': {
        'emb_dim': 1024,
        'dec_emb_dim': 512,
        'depth': 2,
        'dec_depth': 2,
        'num_heads': 16,
        'dec_num_heads': 16,
        'mlp_ratio': 4,
    }
}

def remove_decoder_config(kwargs):
    return {k: v for k, v in kwargs.items() if not k.startswith('dec_')}



mae_model_configs = {
    **{
    f'mae_{size}': partial(MaskedAutoencoder, **config)
    for size, config in transformer_config_dicts.items() 
    },
    **{
    f'maerep_{size}': partial(MaskedAutoencoderVIT, **remove_decoder_config(config))
    for size, config in transformer_config_dicts.items() 
    },
}


