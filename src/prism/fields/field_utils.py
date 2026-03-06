import jax
import jax.numpy as jnp
from typing import Tuple
from ..basis.basis_nd import BasisND
from .field import BasisField, LogBasisField, Coeffs, FactorizedCoeffs, StaticCoeffs, PreconditionedChebyshevCoeffs
import equinox as eqx

def fit_basis_field_from_data(basis: BasisND, coords: Tuple[jax.Array, ...], values: jax.Array, trainable: bool = True, log_transform: bool = False, factorize: bool = False, r: int = 1, precondition: bool = False) -> BasisField:
    mask = jnp.isnan(values)
    for c in coords:
        mask |= jnp.isnan(c)
    mask = ~mask
    values = values[mask].reshape(-1)
    coords = tuple(c[mask].reshape(-1) for c in coords)
    if log_transform:
        values = jnp.log(values)
    coeffs = _jit_fit_basis_field_from_data(basis, coords, values)
    if factorize:
        basis_deg = basis.degs
        coeffs = coeffs.reshape(basis_deg)
        coeffs = FactorizedCoeffs.factorize(coeffs, r)
    elif not trainable:
        coeffs = StaticCoeffs(coeffs)
    elif precondition:
        coeffs = PreconditionedChebyshevCoeffs.make_coeffs(basis.degs, coeffs)
    else:
        coeffs = Coeffs(coeffs)
    if log_transform:
        return LogBasisField(basis, coeffs)
    else:
        return BasisField(basis, coeffs)

@eqx.filter_jit
def _jit_fit_basis_field_from_data(basis: BasisND, coords: Tuple[jax.Array, ...], values: jax.Array) -> jax.Array:
    order = tuple(0 for _ in range(len(coords)))
    vander = basis.build(*coords, order=order)
    coeffs, *_ = jnp.linalg.lstsq(vander, values, rcond=None)
    
    return coeffs
