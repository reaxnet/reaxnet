from typing import Any
import jax.numpy as jnp
import numpy as np
import e3nn_jax as e3nn
import jraph

def _safe_divide(x, y):
    return jnp.where(y == 0.0, 0.0, x / jnp.where(y == 0.0, 1.0, y))

class WeightedLossFunction:
    def __init__(self, 
                 energy_weight: float=1.0,
                 forces_weight: float=10.0,
                 stress_weight: float=100.0):
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        
    def __call__(self, ref: jraph.GraphsTuple, predictions: dict[str, jnp.ndarray]):
        
        ref_energy = ref.globals.energy.flatten()
        ref_forces = ref.nodes.forces
        ref_stress = ref.globals.stress

        mask = jraph.get_graph_padding_mask(ref)
        
        pred_energy = predictions['energy']
        pred_forces = predictions['forces']
        pred_stress = predictions['stress']
        
        peratom_e_loss = jnp.sqrt(jnp.sum(jnp.square(_safe_divide(pred_energy - ref_energy, ref.n_node))) / jnp.sum(mask))
        
        f_loss = jnp.sqrt(jnp.sum(_safe_divide(e3nn.scatter_sum(
                         jnp.mean(jnp.square(ref_forces - pred_forces), axis=1), nel=ref.n_node), 
                                      ref.n_node)
                         ) / jnp.sum(mask))
        
        s_loss = jnp.sqrt(jnp.sum(jnp.mean(jnp.square(ref_stress - pred_stress), axis=(1, 2))) / jnp.sum(mask))
        
        return self.energy_weight * peratom_e_loss + self.forces_weight * f_loss + self.stress_weight * s_loss
    
class EvaluationLossFunction:
    def __init__(self, 
                 energy_weight: float=1.0,
                 forces_weight: float=10.0,
                 stress_weight: float=100.0):
        
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        
    def __call__(self, ref: jraph.GraphsTuple, predictions: dict[str, jnp.ndarray]):
        
        ref_energy = ref.globals.energy.flatten()
        ref_forces = ref.nodes.forces
        ref_stress = ref.globals.stress

        mask = jraph.get_graph_padding_mask(ref)
        
        pred_energy = predictions['energy']
        pred_forces = predictions['forces']
        pred_stress = predictions['stress']
       
        peratom_e_loss_rmse  = jnp.sqrt(jnp.sum(jnp.square(_safe_divide(pred_energy - ref_energy, ref.n_node))) / jnp.sum(mask))
        f_loss_rmse = jnp.sqrt(jnp.sum(_safe_divide(e3nn.scatter_sum(
                         jnp.mean(jnp.square(ref_forces - pred_forces), axis=1), nel=ref.n_node), ref.n_node)) / jnp.sum(mask))
        
        s_loss_rmse = jnp.sqrt(jnp.sum(jnp.mean(jnp.square(ref_stress - pred_stress), axis=(1, 2))) / jnp.sum(mask)) 
        
        peratom_e_loss_mae = jnp.sum(_safe_divide(jnp.abs(pred_energy - ref_energy), ref.n_node)) / jnp.sum(mask)
        f_loss_mae = jnp.sum(_safe_divide(e3nn.scatter_sum(
                         jnp.mean(jnp.abs(ref_forces - pred_forces), axis=1), nel=ref.n_node), ref.n_node)) / jnp.sum(mask)
        
        s_loss_mae = jnp.sum(jnp.mean(jnp.abs(ref_stress - pred_stress), axis=(1, 2))) / jnp.sum(mask)
        
        total_loss = self.energy_weight * peratom_e_loss_rmse + self.forces_weight * f_loss_rmse + self.stress_weight * s_loss_rmse
        
        return total_loss, peratom_e_loss_rmse, f_loss_rmse, s_loss_rmse, peratom_e_loss_mae, f_loss_mae, s_loss_mae