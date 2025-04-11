"""Neural Network Primitives.
The modules in this file are modifications from original code below.

https://github.com/e3nn/e3nn-jax
and 
https://github.com/jax-md/jax-md"""

import functools
from typing import Callable, Dict, Optional, Tuple, Union, List
import flax.linen as nn
from jax import vmap
from jax.nn import initializers
import jax.numpy as jnp
from jax_md import energy, partition, util
import e3nn_jax as e3nn
import numpy as np
import jraph
from jraph import GraphsTuple
from ml_collections import ConfigDict

Array = util.Array
FeaturizerFn = Callable[
    [GraphsTuple, Array, Array, Optional[Array]], GraphsTuple
]
f32 = jnp.float32
partial = functools.partial
normal = lambda var: initializers.variance_scaling(var, 'fan_in', 'normal')
UnaryFn = Callable[[Array], Array]

class BetaSwish(nn.Module):

  @nn.compact
  def __call__(self, x):
    features = x.shape[-1]
    beta = self.param('Beta', nn.initializers.ones, (features,))
    return x * nn.sigmoid(beta * x)

NONLINEARITY = {
    'none': lambda x: x,
    'relu': nn.relu,
    'swish': BetaSwish(),
    'raw_swish': nn.swish,
    'tanh': nn.tanh,
    'sigmoid': nn.sigmoid,
    'silu': nn.silu,
}

def get_nonlinearity_by_name(name: str) -> UnaryFn:
  if name in NONLINEARITY:
    return NONLINEARITY[name]
  raise ValueError(f'Nonlinearity "{name}" not found.')

class MLP(nn.Module):
  """Multilayer Perceptron."""
  
  features: Tuple[int, ...]
  nonlinearity: str

  use_bias: bool = True
  scalar_mlp_std: Optional[float] = None

  @nn.compact
  def __call__(self, x):
    features = self.features

    dense = partial(nn.Dense, use_bias=self.use_bias)
    phi = get_nonlinearity_by_name(self.nonlinearity)

    kernel_init = normal(self.scalar_mlp_std)

    for h in features[:-1]:
      x = phi(dense(h, kernel_init=kernel_init)(x))

    return dense(features[-1], kernel_init=normal(1.0))(x)

def mlp(
    hidden_features: Union[int, Tuple[int, ...]], nonlinearity: str, **kwargs
) -> Callable[..., Array]:
  if isinstance(hidden_features, int):
    hidden_features = (hidden_features,)

  def mlp_fn(*args):
    fn = MLP(hidden_features, nonlinearity, **kwargs)
    return jraph.concatenated_args(fn)(*args)

  return mlp_fn

def neighbor_list_featurizer(displacement_fn):
  def featurize(atoms, position, neighbor, **kwargs):
    N = position.shape[0]
    graph = partition.to_jraph(neighbor, nodes=atoms)
    mask = partition.neighbor_list_mask(neighbor, True)

    Rb = position[graph.senders]
    Ra = position[graph.receivers]

    d = vmap(partial(displacement_fn, **kwargs))
    dR = d(Ra, Rb)
    dR = jnp.where(mask[:, None], dR, 1)

    return graph._replace(edges=dR)

  return featurize

# Bessel Functions


# Similar to the original Behler-Parinello features. Used by Nequip [1] and
# Schnet [2] to encode distance information.

def bessel(r_c, frequencies, r):
  rp = jnp.where(r > f32(1e-5), r, f32(1000.0))
  b = 2 / r_c * jnp.sin(frequencies * rp / r_c) / rp
  return jnp.where(r > f32(1e-5), b, 0)

class BesselEmbedding(nn.Module):
  count: int
  inner_cutoff: float
  outer_cutoff: float

  @nn.compact
  def __call__(self, rs: Array) -> Array:
    def init_fn(key, shape):
      del key
      assert len(shape) == 1
      n = shape[0]
      return jnp.arange(1, n + 1) * jnp.pi

    frequencies = self.param('frequencies', init_fn, (self.count,))
    bessel_fn = partial(bessel, self.outer_cutoff, frequencies)
    bessel_fn = vmap(
        energy.multiplicative_isotropic_cutoff(
            bessel_fn, self.inner_cutoff, self.outer_cutoff
        )
    )
    return bessel_fn(rs)


# Scale and Shifts
DATASET_SHIFT_SCALE = {'harder_silicon': (2.2548, 0.8825)}

def get_shift_and_scale(cfg: ConfigDict) -> Tuple[float, float]:
  if hasattr(cfg, 'scale') and hasattr(cfg, 'shift'):
    return cfg.shift, cfg.scale
  elif hasattr(cfg, 'train_dataset'):
    return DATASET_SHIFT_SCALE[cfg.train_dataset[0]]
  else:
    raise ValueError()

def tp_out_irreps_with_instructions(irreps1: e3nn.Irreps, irreps2: e3nn.Irreps, target_irreps: e3nn.Irreps
) -> Tuple[e3nn.Irreps, List]:
    
    mode = 'uvu'
    trainable = 'True'
    irreps_after_tp = []
    instructions = []
    for i, (mul_in1, irreps_in1) in enumerate(irreps1):
      for j, (_, irreps_in2) in enumerate(irreps2):
        for curr_irreps_out in irreps_in1 * irreps_in2:
          if curr_irreps_out in target_irreps:
            k = len(irreps_after_tp)
            irreps_after_tp += [(mul_in1, curr_irreps_out)]
            instructions += [(i, j, k, mode, trainable)]
            
    irreps_after_tp, p, _ = e3nn.Irreps(irreps_after_tp).sort()
    sorted_instructions = []
    for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
        sorted_instructions += [(
          irreps_in1,
          irreps_in2,
          p[irreps_out],
          mode,
          trainable,
      )]
    return irreps_after_tp, sorted_instructions

def random_split(num, ratio):
    indices = np.random.permutation(num)
    split = int(num * ratio)
    train_idx = indices[split:].tolist()
    val_idx = indices[:split].tolist()
    return train_idx, val_idx
