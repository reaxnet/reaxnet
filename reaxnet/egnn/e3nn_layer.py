"""The modules in this file are modifications from original code below.

https://github.com/e3nn/e3nn-jax
and 
https://github.com/jax-md/jax-md
"""

from typing import Optional

import e3nn_jax as e3nn
from e3nn_jax import FunctionalLinear
from e3nn_jax import Irreps
from e3nn_jax import IrrepsArray
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
from e3nn_jax.utils import vmap
import flax.linen as nn
import jax.numpy as jnp


class FullyConnectedTensorProductE3nn(nn.Module):
  """Flax module of an equivariant Fully-Connected Tensor Product."""

  irreps_out: Irreps
  irreps_in1: Optional[Irreps] = None
  irreps_in2: Optional[Irreps] = None

  @nn.compact
  def __call__(self, x1: IrrepsArray, x2: IrrepsArray, **kwargs) -> IrrepsArray:
    irreps_out = Irreps(self.irreps_out)
    irreps_in1 = (
        Irreps(self.irreps_in1) if self.irreps_in1 is not None else None
    )
    irreps_in2 = (
        Irreps(self.irreps_in2) if self.irreps_in2 is not None else None
    )

    x1 = e3nn.as_irreps_array(x1)
    x2 = e3nn.as_irreps_array(x2)

    leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
    x1 = x1.broadcast_to(leading_shape + (-1,))
    x2 = x2.broadcast_to(leading_shape + (-1,))

    if irreps_in1 is not None:
      x1 = x1.rechunk(irreps_in1)
    if irreps_in2 is not None:
      x2 = x2.rechunk(irreps_in2)

    x1 = x1.remove_zero_chunks().simplify()
    x2 = x2.remove_zero_chunks().simplify()

    tp = FunctionalFullyConnectedTensorProduct(
        x1.irreps, x2.irreps, irreps_out.simplify()
    )

    ws = [
        self.param(
            (
                f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                f"{tp.irreps_in1[ins.i_in1]},"
                f"{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
            ),
            nn.initializers.normal(stddev=ins.weight_std),
            ins.path_shape,
        )
        for ins in tp.instructions
    ]

    f = lambda x1, x2: tp.left_right(ws, x1, x2, **kwargs)

    for _ in range(len(leading_shape)):
      f = e3nn.utils.vmap(f)

    output = f(x1, x2)
    return output.rechunk(irreps_out)


class Linear(nn.Module):
  """Flax module of an equivariant linear layer."""

  irreps_out: Irreps
  irreps_in: Optional[Irreps] = None

  @nn.compact
  def __call__(self, x: IrrepsArray) -> IrrepsArray:
    irreps_out = Irreps(self.irreps_out)
    irreps_in = Irreps(self.irreps_in) if self.irreps_in is not None else None

    if self.irreps_in is None and not isinstance(x, IrrepsArray):
      raise ValueError(
          "the input of Linear must be an IrrepsArray, or "
          "`irreps_in` must be specified"
      )

    if irreps_in is not None:
      x = IrrepsArray(irreps_in, x)

    x = x.remove_nones().simplify()

    lin = FunctionalLinear(x.irreps, irreps_out, instructions=None, biases=None)

    w = [
        self.param(  # pylint:disable=g-long-ternary
            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
            nn.initializers.normal(stddev=ins.weight_std),
            ins.path_shape,
        )
        if ins.i_in == -1
        else self.param(
            f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},"
            f"{lin.irreps_out[ins.i_out]}",
            nn.initializers.normal(stddev=ins.weight_std),
            ins.path_shape,
        )
        for ins in lin.instructions
    ]

    f = lambda x: lin(w, x)
    for _ in range(x.ndim - 1):
      f = vmap(f)
    return f(x)
