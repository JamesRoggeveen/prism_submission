import pytest
import numpy as np
import jax.numpy as jnp
import jax

# Assuming the function is in a file named chebyshev_lib.py
from prism.basis.basis_fns import (
    vectorized_chebyshev_basis,
    vectorized_fourier_basis,
    vectorized_cosine_basis,
    vectorized_legendre_basis
)
from reference_basis import (
    get_expected_chebyshev_basis,
    get_expected_fourier_basis,
    get_expected_cosine_basis,
    get_expected_legendre_basis
)

jax.config.update("jax_enable_x64", True)

# --- Test Cases ---

# Define a set of points to test, including endpoints and interior points
test_points = [
    0.5,  # Scalar interior
    0.0,  # Scalar center
    1.0,  # Scalar right endpoint
    -1.0, # Scalar left endpoint
    np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), # Array of points
    jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0]), # JAX array of points
]

# Define a set of degrees to test
test_degrees = [0, 1, 5, 10]

# Define the derivative orders to test
test_orders = [0, 1, 2]

@pytest.mark.parametrize("functions", [
    (vectorized_chebyshev_basis, get_expected_chebyshev_basis),
    (vectorized_fourier_basis, get_expected_fourier_basis),
    (vectorized_cosine_basis, get_expected_cosine_basis),
    (vectorized_legendre_basis, get_expected_legendre_basis)
])
@pytest.mark.parametrize("x", test_points)
@pytest.mark.parametrize("deg", test_degrees)
@pytest.mark.parametrize("order", test_orders)
def test_vectorized_against_reference(functions, x, deg, order):
    vectorized_fn, reference_fn = functions
    vectorized_results = vectorized_fn(x, deg, order)
    reference_results = reference_fn(x, deg, order)
    for i in range(len(vectorized_results)):
        np.testing.assert_allclose(vectorized_results[i], reference_results[i])

@pytest.mark.parametrize("func", [vectorized_fourier_basis, vectorized_chebyshev_basis, vectorized_cosine_basis, vectorized_legendre_basis])
@pytest.mark.parametrize("deg, order", [
    (-1, 0),  # Invalid deg
    (5, -1),  # Invalid order
    (5, 3),   # Invalid order
])
def test_invalid_inputs_raise_value_error(func, deg, order):
    """
    Tests that the function raises a ValueError for invalid N or order inputs.
    """
    with pytest.raises(ValueError):
        func(jnp.array([0.5]), deg, order)

def _to_tuple(x):
    if isinstance(x, float):
        return (x,)
    if isinstance(x, jax.Array):
        return tuple(x.tolist())
    if isinstance(x, np.ndarray):
        return tuple(x.tolist())
    raise ValueError(f"Unsupported type: {type(x)}")