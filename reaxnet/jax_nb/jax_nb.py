"""
JAX implementation of non-bond potential.
"""

import jax
import jax.numpy as jnp
from jax_md import space, partition
from typing import Callable, Dict
from .jax_d3 import get_c6, vmap_ncoord, HATREE2EV, BOHR2ANGSTROM, r0ab, rcov, r2r4, neighborlist_disp

Array = jnp.ndarray
LAMBDA = 0.462770
COULCONSTANTKCAL = 332.063710
EV2KCALPMOL = 23.060549
COULCONSTANTEV = 14.399645

def taper(r: Array, 
          swa: float, 
          swb: float) -> Array:
    '''Taper function for smooth cutoff. The taper smooth is close to Ewald summation'''
    r = r - swa
    rc = swb - swa
    tap7 =  20.0 / jnp.power(rc, 7)
    tap6 = -70.0 / jnp.power(rc, 6)
    tap5 =  84.0 / jnp.power(rc, 5)
    tap4 = -35.0 / jnp.power(rc, 4)
    return 1.0 + tap4 * jnp.power(r, 4) + tap5 * jnp.power(r, 5) + tap6 * jnp.power(r, 6) + tap7 * jnp.power(r, 7)

def dtaper(r: Array, 
           swa: float, 
           swb: float) -> Array:
    '''Analytical derivative of taper function'''
    r = r - swa
    rc = swb - swa
    tap7 =  20.0 / jnp.power(rc, 7)
    tap6 = -70.0 / jnp.power(rc, 6)
    tap5 =  84.0 / jnp.power(rc, 5)
    tap4 = -35.0 / jnp.power(rc, 4) 
    return 7.0 * tap7 * jnp.power(r, 6) + 6.0 * tap6 * jnp.power(r, 5) + 5.0 * tap5 * jnp.power(r, 4) + 4.0 * tap4 * jnp.power(r, 3)

def coulombev(r: Array, 
              alpha: Array, 
              cutoff: float) -> Array:
    return jax.scipy.special.erf(alpha * r) * COULCONSTANTEV / r * taper(r, 0.0, cutoff) 

def coulombkcal(r: Array, 
                alpha: Array, 
                cutoff: float):
    return jax.scipy.special.erf(alpha * r) * COULCONSTANTKCAL / r * taper(r, 0.0, cutoff)

def _dcoulombkcal(r: Array, 
                  alpha: Array, 
                  cutoff: float) -> Array:
    screen = jax.scipy.special.erf(alpha * r)
    dscreen = 2.0 * alpha * jnp.exp(- alpha * alpha * r * r) / jnp.sqrt(jnp.pi)
    coul  =   COULCONSTANTKCAL / r 
    dcoul = - COULCONSTANTKCAL / (r * r) 
    _taper = taper(r, 0.0, cutoff)
    _dtaper = dtaper(r, 0.0, cutoff)

    return dcoul * screen * _taper + coul * dscreen * _taper + coul * screen * _dtaper

def _sforce_dcoulombkcal(r: Array, 
                         alpha: Array, 
                         cutoff: float) -> Array:
    scalar_r = space.distance(r)
    _d = _dcoulombkcal(scalar_r, alpha, cutoff)
    return _d * r / scalar_r

# vectorinzed the coulombic and dcoulombic functions
vmap_coulombev = jax.vmap(coulombev, in_axes=(0, 0, None))
vmap_coulombkcal = jax.vmap(coulombkcal, in_axes=(0, 0, None))
vmap_sforce_dcoulobmkcal = jax.vmap(_sforce_dcoulombkcal, in_axes=(0, 0, None))  

# Build A matrix
def build_A_matrix(displacement_fn: Callable,
                   positions: Array, 
                   neighbor: partition.NeighborList,
                   alpha: Array,
                   eta0: Array,
                   cutoff: float = 12.5,
                   **kwargs) -> (Array, Array, Array):            

    mat_A = jnp.zeros((len(positions)+1, len(positions)+1))
    di = jnp.diag_indices(len(positions))
    dR = neighborlist_disp(displacement_fn)(positions, neighbor, **kwargs)
    iidx, jidx = neighbor.idx
    scalar_dr = space.distance(dR)
    mat_A = mat_A.at[iidx, jidx].set(vmap_coulombkcal(scalar_dr, alpha[iidx, jidx], cutoff))
    mat_A = mat_A.at[di].set(eta0 * EV2KCALPMOL)
    mat_A = mat_A.at[:-1,-1].set(-1.0)
    mat_A = mat_A.at[-1,:-1].set(1.0)

    # JAX-MD dose not mask the ghost atom-atom self interaction
    # See jax_md/partition.py 995-997
    mat_A = mat_A.at[-1,-1].set(0.0)
    return mat_A, dR, scalar_dr


def compute_shell(dRcc: Array,
                  positions: Array,
                  neighbor: partition.NeighborList,
                  rs_old: Array,
                  charges: Array,
                  alpha: Array,
                  z: Array,
                  Ks: Array,
                  cutoff: float = 12.5,
                  ) -> Array:
   
    qc = charges + z
    qs = -z
    iidx, jidx = neighbor.idx
    dRsc = dRcc + rs_old[iidx]
    dRss = dRcc + rs_old[iidx] - rs_old[jidx]
    sforce = jnp.zeros_like(positions)
    sforce = sforce.at[iidx].add( -vmap_sforce_dcoulobmkcal(dRss, alpha[iidx, jidx], cutoff) * qs[iidx][:, None] * qs[jidx][:, None] )
    sforce = sforce.at[iidx].add( -vmap_sforce_dcoulobmkcal(dRsc, alpha[iidx, jidx], cutoff) * qs[iidx][:, None] * qc[jidx][:, None] )
    rs_new = sforce / Ks[:, None]
    return rs_new


def build_b_vector(dRcc: Array,
                   scalar_dRcc: Array,
                   positions: Array,
                   neighbor: partition.NeighborList,
                   shell_positions: Array,
                   alpha: Array,
                   chi0: Array,
                   z: Array,
                   net_charge: float = 0.0,
                   cutoff: float = 12.5,
                   ) -> Array:

    b = jnp.zeros(len(positions)+1)
    b = b.at[:-1].set(-chi0 * EV2KCALPMOL)
    iidx, jidx = neighbor.idx
    dRcs = dRcc - shell_positions[jidx]
    scalar_dRcs = space.distance(dRcs)
    b = b.at[iidx].add((-vmap_coulombkcal(scalar_dRcc, 
                                          alpha[iidx, jidx], 
                                          cutoff) +
                        vmap_coulombkcal(scalar_dRcs, 
                                         alpha[iidx, jidx], 
                                         cutoff)
                        ) * z[jidx])
    b = b.at[-1].set(net_charge)

    return b

def pqeq_fori_loop(displacement_fn: Callable,
                   positions: Array,
                   neighbor: partition.NeighborList,
                   alpha: Array,
                   eta0: Array,
                   chi0: Array,
                   z: Array,
                   Ks: Array,
                   net_charge: float = 0.0,
                   cutoff: float = 12.5,
                   iterations: int = 2,
                   ) -> (Array, Array):

    mat_A, dRcc, scalar_dRcc = build_A_matrix(displacement_fn, 
                                              positions, 
                                              neighbor, 
                                              alpha, 
                                              eta0, 
                                              cutoff)
    r_shell = jnp.zeros_like(positions)
    if iterations == 1:
        vec_b = build_b_vector(dRcc, scalar_dRcc, positions, neighbor, r_shell, alpha, chi0, z, net_charge, cutoff)
        charges = jnp.linalg.solve(mat_A, vec_b)[:-1]
        r_shell = compute_shell(dRcc, positions, neighbor, r_shell, charges, alpha, z, Ks, cutoff)
        return charges, r_shell
    
    elif iterations == 2:
        vec_b = build_b_vector(dRcc, scalar_dRcc, positions, neighbor, r_shell, alpha, chi0, z, net_charge, cutoff)
        charges = jnp.linalg.solve(mat_A, vec_b)[:-1]
        r_shell = compute_shell(dRcc, positions, neighbor, r_shell, charges, alpha, z, Ks, cutoff)
        vec_b = build_b_vector(dRcc, scalar_dRcc, positions, neighbor, r_shell, alpha, chi0, z, net_charge, cutoff)
        charges = jnp.linalg.solve(mat_A, vec_b)[:-1]
        r_shell = compute_shell(dRcc, positions, neighbor, r_shell, charges, alpha, z, Ks, cutoff)
        return charges, r_shell

    else:
        for _ in range(iterations):
            vec_b = build_b_vector(dRcc, scalar_dRcc, positions,  neighbor, r_shell, alpha, chi0, z, net_charge, cutoff)
            charges = jax.scipy.linalg.solve(mat_A, vec_b)[:-1]
            r_shell = compute_shell(dRcc, positions, neighbor, r_shell, charges, alpha, z, Ks, cutoff)
        return charges, r_shell


def nonbond_potential(displacement_fn: Callable,
                      positions: Array,
                      neighbor: partition.NeighborList,
                      shell_positions: Array,
                      charges: Array,
                      alpha: Array,
                      z: Array,
                      eta0: Array,
                      chi0: Array,
                      Ks: Array,
                      atomic_numbers: Array,
                      d3_params: Dict[str, float],
                      cutoff: float = 12.5,
                      compute_d3: bool=True,
                      r0ab: Array=r0ab,
                      rcov: Array=rcov,
                      r2r4: Array=r2r4,
                      damping: str = "zero",
                      smooth_fn: Callable = None,
                      **kwargs) -> Array:
    
    qc = charges + z
    qs = -z
    dRcc = neighborlist_disp(displacement_fn)(positions, neighbor, **kwargs)
    iidx, jidx = neighbor.idx
    dRsc = dRcc + shell_positions[iidx]
    dRss = dRcc + shell_positions[iidx] - shell_positions[jidx]
    dRcs = dRcc - shell_positions[jidx]
    mask = partition.neighbor_list_mask(neighbor, True)

    # First order, second order and core-shell oscillator
    pot = jnp.dot(charges, chi0) + 0.5 * jnp.dot(charges**2, eta0) + \
          0.5 * jnp.dot(jnp.sum(jnp.square(shell_positions), axis=1), Ks) / EV2KCALPMOL

    # PQEq electrostatic energy
    dr = space.distance(dRcc)
    pot += jnp.sum(vmap_coulombev(dr,                   alpha[iidx, jidx], cutoff) * qc[iidx] * qc[jidx] * mask ) / 2.0
    pot += jnp.sum(vmap_coulombev(space.distance(dRsc), alpha[iidx, jidx], cutoff) * qs[iidx] * qc[jidx] * mask ) / 2.0
    pot += jnp.sum(vmap_coulombev(space.distance(dRss), alpha[iidx, jidx], cutoff) * qs[iidx] * qs[jidx] * mask ) / 2.0
    pot += jnp.sum(vmap_coulombev(space.distance(dRcs), alpha[iidx, jidx], cutoff) * qc[iidx] * qs[jidx] * mask ) / 2.0
    
    if not compute_d3:
        return pot

    # DFT-D3 correction energy
    dr = dr / BOHR2ANGSTROM  
    r6 = jnp.power(dr, 6)
    r8 = r6 * jnp.power(dr, 2)
    Zi = atomic_numbers[iidx]
    Zj = atomic_numbers[jidx]
    _nc = vmap_ncoord(dr, rcov[Zi] + rcov[Zj])
    if smooth_fn is not None:
        assert Callable(smooth_fn), f"[ERROR] smooth_fn must be a callable function"
        _nc = _nc * smooth_fn(dr)
    nc = jnp.zeros((atomic_numbers.shape[0],))
    nc = nc.at[iidx].add(_nc)
    c6 = get_c6(Zi, Zj, nc[iidx], nc[jidx])
    c8 = 3 * c6 * r2r4[Zi] * r2r4[Zj]
    s6 = d3_params["s6"]
    s8 = d3_params["s18"]
    if damping in ["bj", "bjm"]:
        a1 = d3_params["rs6"]
        a2 = d3_params["rs18"]
        tmp = a1 * jnp.sqrt(c8 / c6) + a2
        tmp2 = tmp ** 2
        tmp6 = tmp2 ** 3
        tmp8 = tmp6 * tmp2
        e6 = 1 / (r6 + tmp6)
        e8 = 1 / (r8 + tmp8)
    elif damping == "zero":
        rs6 = d3_params["rs6"]
        rs8 = d3_params["rs18"]
        alp = d3_params["alp"]
        alp6 = alp
        alp8 = alp + 2.0
        tmp2 = r0ab[Zi, Zj]
        rr = tmp2 / dr
        e6 = 1.0 / (1.0 + 6.0 * (rs6 * rr) ** alp6) / r6
        e8 = 1.0 / (1.0 + 6.0 * (rs8 * rr) ** alp8) / r8
    elif damping == "zerom":
        rs6 = d3_params["rs6"]
        rs8 = d3_params["rs18"]
        alp = d3_params["alp"]
        alp6 = alp
        alp8 = alp + 2.0
        tmp2 = r0ab[Zi, Zj]
        r0_beta = rs8 * tmp2
        rr = dr / tmp2
        tmp = rr / rs6 + r0_beta
        damp6 = 1.0 / (1.0 + 6.0 * tmp ** (-alp6))
        tmp = rr + r0_beta
        damp8 = 1.0 / (1.0 + 6.0 * tmp ** (-alp8))
        e6 = damp6 / r6
        e8 = damp8 / r8
    else:
        raise ValueError(f"[ERROR] Unexpected value damping={damping}")

    e6 = -0.5 * s6 * c6 * e6  # (n_edges,)
    e8 = -0.5 * s8 * c8 * e8  # (n_edges,)
    e68 = e6 + e8
    if smooth_fn is not None:
        g = jnp.sum(e68 * smooth_fn(dr) * mask)
    else:
        g = jnp.sum(e68 * mask)

    return g * HATREE2EV + pot