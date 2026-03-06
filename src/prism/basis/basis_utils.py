import jax
import jax.numpy as jnp
from typing import Tuple
import itertools
from typing import List

from functools import reduce # Make sure to import reduce

def _combine_bases(*M: jax.Array) -> jax.Array:
    """Pointwise outer-product -> Nx(∏Mi)."""

    def combine_point(*row: jax.Array) -> jax.Array:
        # Use reduce to apply kron sequentially to all rows.
        # This works for 2, 3, or N dimensions.
        return reduce(jnp.kron, row)

    # The vmap part is correct and remains unchanged.
    return jax.vmap(combine_point, in_axes=(0,) * len(M))(*M)


# def _combine_bases(*M: jax.Array) -> jax.Array:
#     """Pointwise outer-product -> Nx(∏Mi)."""
#     def combine_point(*row):
#         return jnp.kron(*row) if len(row)==2 else \
#                _combine_bases(row[0], *_combine_bases(*row[1:]))
#     return jax.vmap(combine_point, in_axes=(0,) * len(M))(*M)

def _get_idxs(self, order: int) -> List[Tuple[int,...]]:
        N = len(self.degs)
        idxs = [
            idx for idx in itertools.product(range(order+1), repeat=N)
            if sum(idx) <= order
        ]
        idxs.sort(
            key=lambda idx: (sum(idx),) + tuple(-i for i in idx)
        )
        return idxs
    