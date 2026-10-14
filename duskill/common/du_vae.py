"""Diffusion based Variational AutoEncoder"""
from typing import Dict, List, NamedTuple, Tuple, Union, Callable

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import nn
import haiku as hk

from sb3_jax.common.jax_layers import init_weights, create_mlp
from sb3_jax.common.jax_utils import stop_grad

from duskill.common.mlp_vae import Encoder, LOG_STD_MIN, LOG_STD_MAX

# Residual branches are averaged rather than summed, so each branch is divided
# by sqrt(2) to keep the activation variance constant across depth.
_RESIDUAL_SCALE = math.sqrt(2.0)


def ddpm_schedules(beta1, beta2, T, is_linear=True):
    """
    Returns pre-computed schedules for DDPM samplings, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    if is_linear: # linear scheduler
        beta_t = (beta2 - beta1) * jnp.arange(-1, T, dtype=np.float32) / (T - 1) + beta1
    else: # squared scheduler
        beta_t = (beta2 - beta1) * jnp.square(jnp.arange(-1, T, dtype=np.float32)) / \
            jnp.max(jnp.square(jnp.arange(-1, T, dtype=np.float32))) + beta1
    # index 0 is the placeholder for t=0; training only samples t in [1, T]
    beta_t = beta_t.at[0].set(beta1)

    sqrt_beta_t = jnp.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = jnp.log(alpha_t)
    alphabar_t = jnp.exp(jnp.cumsum(log_alpha_t, axis=0))
    alphabar_t_prev = jnp.append(1., alphabar_t[:-1])
    alphabar_t_next = jnp.append(alphabar_t[1:], 0.)

    sqrtab = jnp.sqrt(alphabar_t)
    sqrtab_p = jnp.sqrt(alphabar_t_prev)
    oneover_sqrta = 1 / jnp.sqrt(alpha_t)
    oneover_sqrtab = 1 / jnp.sqrt(alphabar_t)

    sqrtmab = jnp.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "oneover_sqrtab": oneover_sqrtab,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "alphabar_t_prev": alphabar_t_prev,
        "alphabar_t_next": alphabar_t_next,
        "sqrtab": sqrtab,
        "sqrtab_p": sqrtab_p,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


# =================================================================== #
#                           Naive Diffusion                           #
# =================================================================== #


class MLPDiffusionModel(hk.Module):
    """ MLP based diffusion model. """
    def __init__(
        self,
        net_arch: List[int],
        embed_dim: int,
        output_size: int,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
    ):
        super().__init__()
        self.net_arch = net_arch
        self.embed_dim = embed_dim
        self.output_size = output_size      
        self.activation_fn = activation_fn

    def __call__(
        self,
        x_t: jax.Array, # latent variables
        z: jax.Array,   # skill embeddings
        o_t: jax.Array, # current observation
        t: jax.Array,   # denoising step
    ) -> jax.Array:

        # x_t embeddings
        x_embed = hk.Linear(self.embed_dim, name="xe1", **init_weights())(x_t)
        x_embed = self.activation_fn(x_embed)
        x_embed = hk.Linear(self.embed_dim, name="xe2", **init_weights())(x_embed)
        
        # z embeddings
        z_embed = hk.Linear(self.embed_dim, name="ze1", **init_weights())(z)
        z_embed = self.activation_fn(z_embed)
        z_embed = hk.Linear(self.embed_dim, name="ze2", **init_weights())(z_embed)
        
        # o_t embeddings
        o_embed = hk.Linear(self.embed_dim, name="oe1", **init_weights())(o_t)
        o_embed = self.activation_fn(o_embed)
        o_embed = hk.Linear(self.embed_dim, name="oe2", **init_weights())(o_embed)

        # t embeddings
        t_embed = hk.Linear(self.embed_dim, name="te1", **init_weights())(t)
        t_embed = jnp.sin(t_embed)
        t_embed = hk.Linear(self.embed_dim, name="te2", **init_weights())(t_embed)

        # first layer
        nn_input = jnp.concatenate((x_embed, z_embed, o_embed, t_embed), axis=1)
        out = hk.Linear(self.net_arch[0], name="fc0", **init_weights())(nn_input)
        out = nn.gelu(out)

        # middle layers: the conditioning embeddings are re-injected at every layer
        for i, size in enumerate(self.net_arch[1:-1]):
            nn_input = jnp.concatenate((out/_RESIDUAL_SCALE, z_embed, o_embed, t_embed), axis=1)
            _out = hk.Linear(size, name=f"fc{i+1}", **init_weights())(nn_input)
            out = nn.gelu(_out) + out/_RESIDUAL_SCALE

        # last layer
        nn_input = jnp.concatenate((out, z_embed, o_embed, t_embed), axis=1)
        out = hk.Linear(self.output_size, name="out", **init_weights())(nn_input)
        return out


# =================================================================== #
#                           Guided Diffusion                          #
# =================================================================== #

class PrefEncoder(hk.Module):
    """Preference posterior q(pf_z | sk_embed, p).

    Conditions on the (detached) skill embedding and the scalar preference value
    and returns a diagonal Gaussian over the preference latent.
    """
    def __init__(
        self,
        net_arch: List[int],
        embed_dim: int,
        latent_dim: int,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
    ):
        super().__init__()
        self.net_arch = net_arch
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        
    def __call__(self, sk_embed: jax.Array, p: jax.Array) -> jax.Array:
        # skill embeddings (module names kept as "ae*" for checkpoint compatibility)
        s_embed = hk.Linear(self.embed_dim, name='ae1', **init_weights())(sk_embed)
        s_embed = self.activation_fn(s_embed)
        s_embed = hk.Linear(self.embed_dim, name='ae2', **init_weights())(s_embed)
        
        # preference embeddings
        p_embed = hk.Linear(self.embed_dim, name='pe1', **init_weights())(p)
        p_embed = self.activation_fn(p_embed)
        p_embed = hk.Linear(self.embed_dim, name='pe2', **init_weights())(p_embed)

        x = jnp.concatenate((s_embed, p_embed), axis=1)
        x = create_mlp(-1, self.net_arch, self.activation_fn, batch_norm=False)(x)
        
        mean = hk.Linear(self.latent_dim, **init_weights())(x)
        log_std = hk.Linear(self.latent_dim, **init_weights())(x)
        log_std = jnp.clip(log_std, a_min=LOG_STD_MIN, a_max=LOG_STD_MAX)
        std = jnp.exp(log_std)

        return mean, std


class PrefPrior(hk.Module):
    """Preference prior p(pf_z | sk_embed).

    Maps a skill embedding to a diagonal Gaussian over the preference latent;
    the pref posterior is regularized towards it.
    """
    def __init__(
        self,
        net_arch: List[int],
        embed_dim: int,
        latent_dim: int,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
    ):
        super().__init__()
        self.net_arch = net_arch
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn

    def __call__(self, p: jax.Array) -> jax.Array:
        p_embed = hk.Linear(self.embed_dim, name='pe1', **init_weights())(p)
        p_embed = self.activation_fn(p_embed)
        p_embed = hk.Linear(self.embed_dim, name='pe2', **init_weights())(p_embed)

        mean = hk.Linear(self.latent_dim, **init_weights())(p_embed)
        log_std = hk.Linear(self.latent_dim, **init_weights())(p_embed)
        log_std = jnp.clip(log_std, a_min=LOG_STD_MIN, a_max=LOG_STD_MAX)
        std = jnp.exp(log_std)
        return mean, std


class GuidedDiffusionDecoder(hk.Module):
    """Classifier-free guided DDPM decoder over actions.

    Two noise predictors (skill-conditioned `sk_du` and pref-conditioned `pf_du`)
    are combined as `eps = eps_skill + w * (eps_pref - eps_skill)` with guidance
    weight `w = guide_wei`. During training the guidance term is dropped with
    probability `drop_rate`; at evaluation it is always applied.
    """
    def __init__(
        self,
        sk_du: Union[MLPDiffusionModel],
        pf_du: Union[MLPDiffusionModel],
        n_denoise: int,
        guide_wei: float,
        drop_rate: float,
        ddpm_dict: Dict[str, jnp.ndarray],
        freeze_sk_dec: bool = False,
    ):
        super().__init__()
        self.sk_du = sk_du
        self.pf_du = pf_du
        self.guide_wei = guide_wei
        self.drop_rate= drop_rate
        self.n_denoise = n_denoise
        self.output_size = sk_du.output_size
        self.freeze_sk_dec = freeze_sk_dec

        # scheduler params
        self.alpha_t = ddpm_dict["alpha_t"]
        self.oneover_sqrta = ddpm_dict["oneover_sqrta"]
        self.oneover_sqrtab = ddpm_dict["oneover_sqrtab"]
        self.sqrt_beta_t = ddpm_dict["sqrt_beta_t"]
        self.alphabar_t = ddpm_dict["alphabar_t"]
        self.alphabar_t_prev = ddpm_dict["alphabar_t_prev"]
        self.alphabar_t_next = ddpm_dict["alphabar_t_next"]
        self.sqrtab = ddpm_dict["sqrtab"]
        self.sqrtab_p = ddpm_dict["sqrtab_p"]
        self.sqrtmab = ddpm_dict["sqrtmab"]
        self.mab_over_sqrtmab = ddpm_dict["mab_over_sqrtmab"]

    def __call__(
        self,
        x_t: jax.Array,
        sk_z: jax.Array,
        pf_z: jax.Array,
        o_t: jax.Array,
        t: jax.Array,
        denoise: bool = False,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        
        if denoise:
            n_batch = sk_z.shape[0]
            # use given x_t is available
            x_i = x_t
            if x_t is None:
                x_i = jax.random.normal(hk.next_rng_key(), shape=(n_batch, self.output_size))
            x_i_trace = dict()
            for i in range(self.n_denoise, 0, -1):
                t_i = jnp.array([[i / self.n_denoise]])
                t_i = jnp.repeat(t_i, n_batch, axis=0)
                noise = jax.random.normal(hk.next_rng_key(), shape=(n_batch, self.output_size)) if i > 1 else 0
                eps = self._predict_eps(x_i, sk_z, pf_z, o_t, t_i, deterministic)
                if deterministic : noise = 0
                
                # ddpm reverse process
                x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * noise
                x_i_trace[i-1] = [x_i, eps] # return action, eps
            return x_i, x_i_trace
        return self._predict_eps(x_t, sk_z, pf_z, o_t, t, deterministic), {}

    def _predict_eps(
        self, 
        x_i: jax.Array, 
        sk_z: jax.Array, 
        pf_z: jax.Array, 
        o_t: jax.Array, 
        t_i: jax.Array,
        deterministic: bool,
    ) -> jax.Array:
        # guide_wei == 0: no guidance; a single network is conditioned on the
        # concatenated [skill, pref] embeddings (ablation setting)
        if self.guide_wei == 0.0:
            return self.sk_du(x_i, jnp.concatenate([sk_z, pf_z], axis=-1), o_t, t_i)

        # guided diffusion model
        if self.freeze_sk_dec:
            sk_eps = self.sk_du(x_i, sk_z, o_t, t_i)
            sk_eps = stop_grad(sk_eps)
        else:
            sk_eps = self.sk_du(x_i, sk_z, o_t, t_i)

        # use preference embedding
        pf_eps = self.pf_du(x_i, pf_z, o_t, t_i)
        
        # drop rate
        drop = jax.random.choice(hk.next_rng_key(), 
            jnp.array([0, 1]), p=jnp.array([self.drop_rate, 1-self.drop_rate]))
        
        if deterministic:
            eps = sk_eps + self.guide_wei * (pf_eps - sk_eps)
        else:
            eps = sk_eps + self.guide_wei * drop * (pf_eps - sk_eps)
        return eps


# =================================================================== #
#                     Hierarchical Guided Diffusion                   #
# =================================================================== #


class HGDAEOutput(NamedTuple):
    sk_mean: jax.Array
    sk_std: jax.Array 
    pf_mean: jax.Array
    pf_std: jax.Array
    logits: jax.Array
    info: Dict[str, jax.Array]


class HierarchicalGuidedDiffusionAutoEncoder(hk.Module): 
    """Hierarchical auto-encoder behind the DuSkill prior.

    The skill encoder maps an h-step trajectory to a skill latent; the pref
    encoder maps the detached skill mean plus the preference value to a pref
    latent; both latents condition the guided diffusion decoder that predicts
    the action noise.
    """
    def __init__(
        self,
        sk_encoder: Encoder,
        pf_encoder: PrefEncoder,
        decoder: GuidedDiffusionDecoder, # pref.
        squash_z: bool = False
    ):
        super().__init__()
        self.sk_encoder = sk_encoder
        self.pf_encoder = pf_encoder
        self.decoder = decoder
        self.squash_z = squash_z

    def __call__(
        self,
        obs: jax.Array = None,  # observation sequence
        act: jax.Array = None,  # action sequence
        o_t: jax.Array = None,  # current observation
        x_t: jax.Array = None,  # diffusion noise
        t: jax.Array = None,    # diffusion timestep
        p: jax.Array = None,    # preference info
        denoise: bool = False,
        sk_mean: jax.Array = None,
        sk_std: jax.Array = None,
        pf_mean: jax.Array = None,
        pf_std: jax.Array = None,
        deterministic: bool = False,
    ) -> HGDAEOutput:
        
        if obs is not None: # get skill embedding
            sk_x = jnp.concatenate([obs, act], axis=2)
            sk_mean, sk_std = self.sk_encoder(sk_x)

        sk_z = sk_mean if deterministic else \
            sk_mean + sk_std * jax.random.normal(hk.next_rng_key(), sk_mean.shape)
        if self.squash_z: sk_z = nn.tanh(sk_z)

        # Motivated from Hierarchical VAE: the pref encoder is conditioned on the
        # detached skill *mean* (not the sampled skill), so the pref branch never
        # back-propagates into the skill encoder.
        if p is not None: # get pref embedding
            pf_mean, pf_std = self.pf_encoder(stop_grad(sk_mean), p)
        
        pf_z = pf_mean if deterministic else \
            pf_mean + pf_std * jax.random.normal(hk.next_rng_key(), pf_mean.shape)
        if self.squash_z: pf_z = nn.tanh(pf_z)

        # Repeat as o_t 
        _size = int(o_t.shape[0] / sk_z.shape[0])
        sk_z = jnp.repeat(sk_z, _size, axis=0)
        pf_z = jnp.repeat(pf_z, _size, axis=0)

        logits, info = self.decoder(x_t, sk_z, pf_z, o_t, t, denoise, deterministic=deterministic)
        
        return HGDAEOutput(sk_mean, sk_std, pf_mean, pf_std, logits, info) 
