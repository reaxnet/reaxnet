"""
JAX implementation of DFT-D3 long-range dispersion correction
"""

import jax
import jax.numpy as jnp
from jax_md import partition
from functools import partial
from typing import Callable

Array = jnp.ndarray
f32 = jnp.float32

BOHR2ANGSTROM = 0.52917726  
HATREE2EV = 27.21138505  

d3_params = jnp.load(__file__.replace("jax_d3.py", "dftd3_params.npz"))
c6ab = jnp.asarray(d3_params["c6ab"])
r0ab = jnp.asarray(d3_params["r0ab"])
rcov = jnp.asarray(d3_params["rcov"])
r2r4 = jnp.asarray(d3_params["r2r4"])
d3_k1 = 16.000
d3_k2 = 4 / 3
d3_k3 = -4.000

def neighborlist_disp(displacement_fn: Callable):
    def distance_fn(positions: Array,
                    neighbor: partition.NeighborList,
                    **kwargs):
        receivers, senders = neighbor.idx
        Rb = positions[senders]
        Ra = positions[receivers]
        mask = partition.neighbor_list_mask(neighbor, True)
        d = jax.vmap(partial(displacement_fn, **kwargs))
        dR = d(Ra, Rb)
        dR = jnp.where(mask[:, None], dR, 1)
        return dR
    return distance_fn

def _poly_smoothing(cutoff: float, r):
    cuton = cutoff - 1
    x = (cutoff - r) / (cutoff - cuton)
    x2 = x**2
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return jnp.where(
        r <= cuton,
        jnp.ones_like(x),
        jnp.where(r >= cutoff, jnp.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3),
    )

def _multi_iso_smoothing(cutoff: float, dr):
    r_onset = cutoff - 1
    r_c = cutoff ** f32(2)
    r_o = r_onset ** f32(2)
    r = dr ** f32(2)
    inner = jnp.where(dr < cutoff,
                     (r_c - r)**2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o)**3,
                     0)
    return jnp.where(dr < r_onset, 1, inner)

def _ncoord(k1, r, rco):
    return 1.0 / (1.0 + jnp.exp(-k1 * (rco / r - 1.0)))

vmap_ncoord = jax.vmap(partial(_ncoord, d3_k1), in_axes=(0, 0))

def _getc6(c6ab: Array, k3: float, Zi: Array, Zj: Array, nci: Array, ncj: Array) -> Array:
    c6ab_ = c6ab[Zi, Zj]
    cn0 = c6ab_[:, :, :, 0]
    r = jnp.power((c6ab_[:, :, :, 1] - nci[:, None, None]), 2.0) + jnp.power((c6ab_[:, :, :, 2] - ncj[:, None, None]), 2.0)
    n_edges = r.shape[0]
    n_c6ab = r.shape[1] * r.shape[2]
    if cn0.shape[0] == 0:
        k3_rnc = (k3 * r).reshape(n_edges, n_c6ab)
    else:
        k3_rnc = jnp.where(cn0 > 0.0, k3 * r, -1.0e20 * jnp.ones_like(r)).reshape(n_edges, n_c6ab)
    r_ratio = jax.nn.softmax(k3_rnc, axis=1)
    c6 = jnp.sum(r_ratio * cn0.reshape(n_edges, n_c6ab), axis=1)
    return c6

get_c6 = partial(_getc6, c6ab, d3_k3)
