'''
This file contains the energy function for the neural network model.
'''

from typing import Dict, Union
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import jraph

Array = Union[np.ndarray, jnp.ndarray]
@jax.jit
def periodic_displacement_fn(Ra, Rb, unit_shift, cell, strains):
    edges = Ra - Rb + jnp.dot(unit_shift, cell)
    edges = jnp.dot(edges, jnp.eye(3) + strains)
    return edges

vmap_pbc_displacement_fn = jax.vmap(periodic_displacement_fn, in_axes=(0, 0, 0, 0, 0))

@jax.jit
def vmap_edges_fn(positions: Array,
                  senders: Array,
                  receivers: Array,
                  shifts: Array,
                  cell: Array,
                  n_edge: Array,
                  strains: Array
    ) -> Array:
    num_edges = receivers.shape[0]
    return vmap_pbc_displacement_fn(positions[senders], 
                                    positions[receivers], 
                                    shifts, 
                                    jnp.repeat(cell, n_edge, axis=0, total_repeat_length=num_edges),
                                    jnp.repeat(strains, n_edge, axis=0, total_repeat_length=num_edges))

def edges_fn(positions: Array,     
             senders: Array,       
             receivers: Array,     
             shifts: Array,        
             cell: Array,           
             n_edge: Array, 
             strain: Array         
    ) -> Array:
    """
    Recommend to install opt_einsum for better performance.
    """
    num_edges = receivers.shape[0]
    shifts = jnp.einsum("ei,eij->ej", shifts, jnp.repeat(cell, n_edge, axis=0, total_repeat_length=num_edges))  # [n_edges, 3]
    edges = positions[senders] - positions[receivers] + shifts
    edges = jnp.einsum("ei,eij->ej", edges, jnp.repeat(jnp.repeat(jnp.eye(3)[None, :, :], len(cell), axis=0) + strain, n_edge, axis=0, total_repeat_length=num_edges))
    return edges

def compute_fn(model: nn.Module, 
               params, 
               graph: jraph.GraphsTuple) -> Dict[str, jnp.ndarray]:
    
    def energy_fn(positions, strains):
        edges = vmap_edges_fn(positions, senders=graph.senders, 
                         receivers=graph.receivers,
                         shifts=graph.edges.shifts,
                         cell=graph.globals.cell,
                         n_edge=graph.n_edge,
                         strains=strains)
        node_energies = model.apply(params, edges, graph.nodes.species, graph.senders, graph.receivers)
        assert node_energies.shape == (len(positions),) , "model output needs to be an array of shape (n_nodes, )"
        return jnp.sum(node_energies), node_energies
    
    (dEdR, dEdV), node_energies = jax.grad(energy_fn, (0, 1), 
                                           has_aux=True)(graph.nodes.positions, 
                                                         jnp.zeros_like(graph.globals.cell))
                                           
    graph_energies = e3nn.scatter_sum(node_energies, nel=graph.n_node)
    volumes = jnp.linalg.det(graph.globals.cell)[:, None, None]
    volumes = jnp.where(volumes > 0.0, volumes, 1.0)
   
    # Note that the stress in ASE is negative of the stress in VASP / LAMMPS
    return dict(energy = graph_energies, 
                forces = -dEdR, 
                stress = dEdV / volumes)