"""MLP based Variational AutoEncoder"""
from typing import List, NamedTuple, Tuple, Callable

import jax
import jax.numpy as jnp
from jax import nn
import haiku as hk

from sb3_jax.common.jax_layers import init_weights, create_mlp

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class VAEOutput(NamedTuple):
    mean: jax.Array
    std: jax.Array
    logits: jax.Array


class Encoder(hk.Module):
    def __init__(
        self,
        net_arch: List[int],
        latent_dim: int, 
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
    ):
        super().__init__()
        self.net_arch = net_arch
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = create_mlp(-1, self.net_arch, self.activation_fn, batch_norm=False)(x)
        
        mean = hk.Linear(self.latent_dim, **init_weights())(x)
        log_std = hk.Linear(self.latent_dim, **init_weights())(x)
        log_std = jnp.clip(log_std, a_min=LOG_STD_MIN, a_max=LOG_STD_MAX)
        std = jnp.exp(log_std)
        return mean, std


class Decoder(hk.Module):
    def __init__(
        self,
        net_arch: List[int],
        output_size: int,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
    ):
        super().__init__()
        self.net_arch = net_arch
        self.output_size = output_size
        self.activation_fn = activation_fn
    
    def __call__(self, z: jax.Array) -> jax.Array:
        z = create_mlp(-1, self.net_arch, self.activation_fn, batch_norm=False)(z)
        logits = hk.Linear(self.output_size)(z)
        return logits


class VariationalAutoEncoder(hk.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        closed_loop: bool = False,
        squash_z: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.closed_loop = closed_loop
        self.squash_z = squash_z
        
    def __call__(
        self, 
        obs: jax.Array = None, 
        act: jax.Array = None,
        obt: jax.Array = None,
        mean: jax.Array = None, 
        std: jax.Array = None,
        deterministic: bool = False
    ) -> VAEOutput:
        if obs is not None:
            x = jnp.concatenate([obs, act], axis=2)
            mean, std = self.encoder(x)
        
        z = mean if deterministic else \
            mean + std * jax.random.normal(hk.next_rng_key(), mean.shape)
        if self.squash_z: z = nn.tanh(z)

        if self.closed_loop:
            _size = int(obt.shape[0] / z.shape[0])
            z = jnp.repeat(z, _size, axis=0)
            z = jnp.concatenate([obt, z], axis=1)

        logits = self.decoder(z)
        return VAEOutput(mean, std, logits)


class ContrastiveEncoder(hk.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
    
    def __call__(self, obs: jax.Array):
        out = hk.Linear(self.hidden_dim, **init_weights())(obs)
        out = nn.relu(out)
        out = hk.Linear(self.hidden_dim, **init_weights())(out)
        out = nn.relu(out)
        out = hk.Linear(self.feature_dim, **init_weights())(out)
        return out


class ContrastiveFutureState(hk.Module):
    def __init__(
        self,
        encoder: ContrastiveEncoder,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim

    def __call__(self, obs: jax.Array, obh: jax.Array, ret: bool = False):
        z_obs = self.encoder(obs)
        if ret:
            return z_obs
        z_obh = self.encoder(obh)
        
        W = hk.get_parameter("W", \
            [self.feature_dim, self.feature_dim], init=hk.initializers.RandomUniform())
        Wz = jnp.matmul(W, z_obh.T) # [feature_dim, batch_size]
        logits = jnp.matmul(z_obs, Wz) # [batch_size, batch_size]
        logits = logits - jnp.max(logits, 1, keepdims=True)
        return logits
