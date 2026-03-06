import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable, Sequence, Union
import equinox as eqx

from .basis_fns import (
    vectorized_chebyshev_basis,
    vectorized_fourier_basis,
    vectorized_cosine_basis,
    vectorized_legendre_basis
)

from .basis_utils import (
    _combine_bases,
    _get_idxs
)

class BasisND(eqx.Module):
    """
    basis_fns[i](coords[i], degs[i], order)
      returns a tuple of 1D basis/derivative mats for that axis.
    """
    basis_fns: Sequence[Callable]
    degs: Sequence[int]

    def __init__(self, basis_fns: Sequence[Callable], degs: Sequence[int]):
        if len(basis_fns) != len(degs):
            raise ValueError("must supply one fn + deg per dim")
        self.basis_fns = tuple(basis_fns)
        self.degs = tuple(degs)

    def basis_type(self):
        return type(self).__name__

    def __call__(self, *coords: Union[jax.Array, jnp.ndarray,np.ndarray], order: Union[int, Tuple[int,...]]) -> Union[jax.Array, Tuple[jax.Array,...]]:
        """
        coords: N arrays, either each of length Ni (grid) or length P (scattered).
        order: tuple of max total derivative order (0,1,2,...) per dim.
        grid: if True do pointwise outer; else full grid tensor-product.
        """
        if isinstance(order, int):
            return self.build_all(*coords, order=order)
        else:
            return self.build(*coords, order=order)
    
    def build_all(self, *coords: Union[jax.Array, jnp.ndarray,np.ndarray], order: int=0) -> Tuple[jax.Array,...]:
        idxs = _get_idxs(self, order)
        return tuple(self.build(*coords, order=idx) for idx in idxs)
    
    @eqx.filter_jit
    def build(self, *coords: Union[jax.Array, jnp.ndarray,np.ndarray], order: Tuple[int,...]) -> jax.Array:
        N = len(self.degs)

        if len(coords) != N:
            raise ValueError("must supply one coord per dim")
        if len(order) != N:
            raise ValueError("must supply one order per dim")
        
        per_dim: Sequence[Tuple[jnp.ndarray, ...]] = [
            fn(jnp.asarray(c).reshape(-1), deg, order=o)
            for fn, deg, c, o in zip(self.basis_fns, self.degs, coords, order)
        ]
        # enumerate all multi-indices (i0,...,iN) with sum <= order
        
        mats_i = [ per_dim[dim][order[dim]] for dim in range(N) ]
        result = _combine_bases(*mats_i)
        return result

class ChebyshevBasis2D(BasisND):
    def __init__(self, deg: Tuple[int, int]):
        super().__init__([vectorized_chebyshev_basis, vectorized_chebyshev_basis], deg)

class FourierBasis2D(BasisND):
    def __init__(self, deg: Tuple[int, int]):
        super().__init__([vectorized_fourier_basis, vectorized_fourier_basis], deg)

class FourierChebyshevBasis2D(BasisND):
    def __init__(self, deg: Tuple[int, int]):
        super().__init__([vectorized_fourier_basis, vectorized_chebyshev_basis], deg)

class CosineChebyshevBasis2D(BasisND):
    def __init__(self, deg: Tuple[int, int]):
        super().__init__([vectorized_cosine_basis, vectorized_chebyshev_basis], deg)

class LegendreBasis2D(BasisND):
    def __init__(self, deg: Tuple[int, int]):
        super().__init__([vectorized_legendre_basis, vectorized_legendre_basis], deg)

class CosineLegendreBasis2D(BasisND):
    def __init__(self, deg: Tuple[int, int]):
        super().__init__([vectorized_cosine_basis, vectorized_legendre_basis], deg)