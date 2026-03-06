import pytest
import jax.numpy as jnp
import jax
import numpy as np

from prism.basis.basis_nd import (
    BasisND, 
    ChebyshevBasis2D, 
    FourierBasis2D, 
    FourierChebyshevBasis2D, 
    CosineChebyshevBasis2D
)
from prism.basis.basis_fns import (
    vectorized_chebyshev_basis, 
    vectorized_fourier_basis, 
    vectorized_cosine_basis
)
from reference_basis import (
    get_expected_2d_basis
)

jax.config.update("jax_enable_x64", True)

@pytest.fixture
def generic_basis():
    def _mock_basis(x_in, deg, order=0):
        x_vec = jnp.atleast_1d(x_in)
        num_x = x_vec.shape[0]
        
        # Create predictable outputs based on inputs
        results = []
        
        # Value (order 0)
        basis_values = jnp.zeros((num_x, deg + 1))
        for i in range(deg + 1):
            basis_values = basis_values.at[:, i].set(i * x_vec + 0.5)
        results.append(jnp.array(basis_values))
        
        # First derivative (order 1)
        if order > 0:
            dx_values = jnp.zeros((num_x, deg + 1))
            for i in range(deg + 1):
                dx_values = dx_values.at[:, i].set(i)  # Derivative of i*x + 0.5
            results.append(jnp.array(dx_values))
        
        # Second derivative (order 2)
        if order > 1:
            ddx_values = jnp.zeros((num_x, deg + 1))
            # Second derivative of i*x + 0.5 is 0
            results.append(jnp.array(ddx_values))
        
        return tuple(results)
    
    myBasis = BasisND([_mock_basis,_mock_basis], [2,2])
    def mockBasis(x, y, order): return get_expected_2d_basis(x,y,[2,2],[_mock_basis,_mock_basis],order,scattered=True)
    return myBasis, mockBasis

test_orders = [0, 1, 2]

@pytest.mark.parametrize("order", test_orders)
def test_2d_bases_generic(generic_basis, order):
    """
    A single comprehensive test for all 2D basis functions, for both
    grid and scattered points.
    """
    x = jnp.array([0.5, 1.0])
    y = jnp.array([0.5, 1.0])
    basis, mockBasis = generic_basis
    actual_results = basis(x,y,order=order)

    # 2. Get expected results from the reference implementation
    expected_results = mockBasis(x,y,order)
    # 3. Assertions
    assert len(actual_results) == len(expected_results)
    
    for actual, expected in zip(actual_results, expected_results):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

# --- Test Parameters ---
test_setups = [
    (ChebyshevBasis2D, [vectorized_chebyshev_basis, vectorized_chebyshev_basis]),
    (FourierBasis2D, [vectorized_fourier_basis, vectorized_fourier_basis]),
    (FourierChebyshevBasis2D, [vectorized_fourier_basis, vectorized_chebyshev_basis]),
    (CosineChebyshevBasis2D, [vectorized_cosine_basis, vectorized_chebyshev_basis])
]

test_degrees = [(2, 3), (3, 2)]
@pytest.mark.parametrize("basis_class, basis_fns", test_setups)
@pytest.mark.parametrize("deg", test_degrees)
def test_2d_bases_against_reference(basis_class, basis_fns, deg):
    basis = basis_class(deg)
    order = 2
    x = jnp.linspace(-1, 1, 10)
    y = jnp.linspace(-1, 1, 10)
    xx, yy = jnp.meshgrid(x, y)
    actual_results = basis(xx, yy, order=order)
    basis_dx = basis(xx, yy, order=(1,0))
    expected_results = get_expected_2d_basis(x, y, deg, basis_fns, order, scattered=False)
    assert len(actual_results) == len(expected_results)
    for actual, expected in zip(actual_results, expected_results):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )
    np.testing.assert_allclose(basis_dx, expected_results[1])

