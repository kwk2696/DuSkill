"""LSTM based sequence encoder for the skill VAE.

Only the recognition (encoder) side lives here; the matching decoders are in
`duskill.common.mlp_vae` (MLP) and `duskill.common.du_vae` (diffusion).
"""
from typing import Callable

import jax
import jax.numpy as jnp
from jax import nn
import haiku as hk

from sb3_jax.common.jax_layers import init_weights

from duskill.common.mlp_vae import LOG_STD_MIN, LOG_STD_MAX


class LSTMEncoder(hk.Module):
    """Encodes an h-step (observation, action) sequence into a Gaussian skill posterior."""
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn

    def __call__(self, x: jax.Array):
        """x: [batch, h, obs_dim + act_dim] -> (mean, std), each [batch, latent_dim]."""
        # embed inputs
        x = hk.Linear(self.embed_dim, **init_weights())(x)
        
        # run the LSTM over the h steps of every trajectory. `x` is batch-major,
        # so `time_major=False` is required (Haiku defaults to time-major input).
        batch = x.shape[0]
        core = hk.LSTM(self.hidden_dim)
        x, state = hk.dynamic_unroll(core, x, core.initial_state(batch), time_major=False)
        x = x[:, -1, :]  # last output: [batch, h, dim] -> [batch, dim]

        mean = hk.Linear(self.latent_dim, **init_weights())(x)
        log_std = hk.Linear(self.latent_dim, **init_weights())(x)
        log_std = jnp.clip(log_std, a_min=LOG_STD_MIN, a_max=LOG_STD_MAX)
        std = jnp.exp(log_std)
        return mean, std
