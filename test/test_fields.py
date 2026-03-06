
import pytest
import jax
import jax.numpy as jnp
from prism.fields import (
    Coeffs,
    StaticCoeffs,
    FactorizedCoeffs,
    BasisField,
    LogBasisField,
    fit_basis_field_from_data,
)

jax.config.update("jax_enable_x64", True)

class MockBasisND:
    """A simple, predictable 2D polynomial basis for testing."""
    def __init__(self, degs):
        if degs != (1, 1):
            raise NotImplementedError("MockBasisND only supports degs=(1, 1)")
        self.degs = degs

    def __call__(self, x, y, order=(0, 0)):
        # This is now just a wrapper for the build method for evaluation
        vander = self.build(x, y, order=order)
        # For a single point, return a 1D array for the matmul
        if jnp.isscalar(x) or x.ndim == 0:
            return vander
        # This part is no longer used by the tests but kept for consistency
        return vander.reshape(1, -1)

    def build(self, x, y, order=(0, 0)):
        """Builds the Vandermonde-like matrix for an array of coordinates."""
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)
        
        dx, dy = order

        if (dx, dy) == (0, 0):       # Evaluate
            # Basis functions are [1, y, x, xy]
            vander = jnp.stack([jnp.ones_like(x), y, x, x * y], axis=-1)
        elif (dx, dy) == (1, 0):     # d/dx
            vander = jnp.stack([jnp.zeros_like(x), jnp.zeros_like(x), jnp.ones_like(x), y], axis=-1)
        elif (dx, dy) == (0, 1):     # d/dy
            vander = jnp.stack([jnp.zeros_like(x), jnp.ones_like(x), jnp.zeros_like(x), x], axis=-1)
        else:
            # Return zeros for other derivatives
            vander = jnp.zeros((x.shape[0], 4))
            
        return vander

@pytest.fixture
def mock_basis():
    """Provides a default MockBasisND instance."""
    return MockBasisND(degs=(1, 1))

@pytest.fixture
def sample_coeffs():
    """Provides a sample Coeffs object."""
    # Corresponds to 1 + 2y + 3x + 4xy
    return Coeffs(jnp.array([1.0, 2.0, 3.0, 4.0]))

@pytest.fixture
def sample_data(sample_coeffs):
    """Generates sample coordinates and values based on sample_coeffs."""
    # Generate a grid of coordinates
    x = jnp.linspace(0, 1, 10)
    y = jnp.linspace(0, 1, 10)
    xx, yy = jnp.meshgrid(x, y)
    coords = (xx.flatten(), yy.flatten())
    
    # Generate the true values from the known coefficients
    # f(x, y) = 1 + 2y + 3x + 4xy
    values = (1.0 + 2.0 * coords[1] + 3.0 * coords[0] + 4.0 * coords[0] * coords[1])
    
    return coords, values

class TestCoeffs:
    def test_coeffs_creation_and_value(self):
        arr = jnp.arange(4, dtype=jnp.float32)
        c = Coeffs(arr)
        assert isinstance(c.value, jax.Array)
        assert jnp.array_equal(c.value, arr)

    def test_coeffs_make_zero(self):
        shape = (2, 3)
        c = Coeffs.make_zero(shape)
        assert c.value.shape == (12,)
        assert jnp.all(c.value == 0)

    def test_static_coeffs_creation_and_value(self):
        data = [1.0, 2.0, 3.0]
        sc = StaticCoeffs(data)
        assert isinstance(sc.value, jax.Array)
        assert jnp.array_equal(sc.value, jnp.array(data))

    def test_static_coeffs_make_zero(self):
        shape = (2, 2)
        sc = StaticCoeffs.make_zero(shape)
        assert isinstance(sc.coeffs, list)
        assert sc.value.shape == (9,)
        assert jnp.all(sc.value == 0)
        
    def test_factorized_coeffs_value(self):
        left = jnp.array([[1.0], [2.0]])
        right = jnp.array([[3.0, 4.0]])
        fc = FactorizedCoeffs(left, right)
        expected = jnp.array([[3.0, 4.0], [6.0, 8.0]])
        assert jnp.allclose(fc.value, expected)
        
    def test_factorized_coeffs_make_zero(self):
        shape = (10, 5, 2) # n, m, r
        fc = FactorizedCoeffs.make_zero(shape)
        assert fc.left_factor.shape == (10, 2)
        assert fc.right_factor.shape == (2, 5)
        assert jnp.all(fc.value == 0)

    def test_factorized_coeffs_factorize(self):
        key = jax.random.PRNGKey(0)
        original_matrix = jax.random.normal(key, (10, 8))
        rank = 8
        
        fc = FactorizedCoeffs.factorize(original_matrix, r=rank)
        
        # Check shapes
        assert fc.left_factor.shape == (10, rank)
        assert fc.right_factor.shape == (rank, 8)
        
        # Check if reconstruction is close to the original
        reconstructed_matrix = fc.value
        assert jnp.allclose(original_matrix, reconstructed_matrix, atol=1e-6)

class TestBasisField:
    def test_initialization(self, mock_basis, sample_coeffs):
        # Default zero initialization
        field_zero = BasisField(mock_basis)
        assert isinstance(field_zero.coeffs, Coeffs)
        assert jnp.all(field_zero.coeffs.value == 0)

        # Initialization with provided coeffs
        field = BasisField(mock_basis, coeffs=sample_coeffs)
        assert field.coeffs is sample_coeffs

    def test_evaluate(self, mock_basis, sample_coeffs):
        field = BasisField(mock_basis, coeffs=sample_coeffs)
        x, y = 0.5, 2.0
        
        # With our mock basis [1, y, x, xy] and coeffs [1, 2, 3, 4],
        # evaluation should be: 1*1 + 2*y + 3*x + 4*xy
        expected = 1*1 + 2*y + 3*x + 4*x*y
        result = field.evaluate(x, y)
        
        assert jnp.isclose(result, expected)

    def test_derivative(self, mock_basis, sample_coeffs):
        field = BasisField(mock_basis, coeffs=sample_coeffs)
        x, y = 0.5, 2.0
        
        # Derivative w.r.t. x: basis is [0, 0, 1, y]
        # Result: 3*1 + 4*y
        expected_dx = 3.0 + 4.0 * y
        result_dx = field.derivative(x, y, order=(1, 0))
        assert jnp.isclose(result_dx, expected_dx)
        
        # Derivative w.r.t. y: basis is [0, 1, 0, x]
        # Result: 2*1 + 4*x
        expected_dy = 2.0 + 4.0 * x
        result_dy = field.derivative(x, y, order=(0, 1))
        assert jnp.isclose(result_dy, expected_dy)

    def test_spectral_filter(self, mock_basis, sample_coeffs):
        field = BasisField(mock_basis, coeffs=sample_coeffs)
        
        # Test frac=1.0 (no change)
        filtered_coeffs_1 = field.spectral_filter(frac=1.0)
        assert jnp.array_equal(filtered_coeffs_1, sample_coeffs.value)

        # Test frac=0.0 (all zero)
        filtered_coeffs_0 = field.spectral_filter(frac=0.0)
        assert jnp.all(filtered_coeffs_0 == 0)

        # Test frac=0.5
        # Shape is (2,2), so cutoff is (1,1). Mask should be [[1,0],[0,0]]
        expected_mask = jnp.array([[1.0, 0.0], [0.0, 0.0]]).flatten()
        expected_coeffs = sample_coeffs.value * expected_mask
        filtered_coeffs_05 = field.spectral_filter(frac=0.5)
        assert jnp.array_equal(filtered_coeffs_05, expected_coeffs)

    def test_evaluate_with_filter(self, mock_basis, sample_coeffs):
        field = BasisField(mock_basis, coeffs=sample_coeffs)
        x, y = 0.5, 2.0

        # With frac=0.5, coeffs become [1, 0, 0, 0].
        # Evaluation should just be 1.0 * 1 = 1.0
        result = field.evaluate(x, y, filter_frac=0.5)
        assert jnp.isclose(result, 1.0)

class TestLogBasisField:
    def test_evaluate(self, mock_basis, sample_coeffs):
        field = LogBasisField(mock_basis, coeffs=sample_coeffs)
        base_field = BasisField(mock_basis, coeffs=sample_coeffs)
        x, y = 0.1, 0.2
        
        base_value = base_field.evaluate(x, y)
        log_value = field.evaluate(x, y)
        
        assert jnp.isclose(log_value, jnp.exp(base_value))

    def test_derivative(self, mock_basis, sample_coeffs):
        field = LogBasisField(mock_basis, coeffs=sample_coeffs)
        x, y = 0.1, 0.2
        
        # Chain rule: d/dx(exp(f(x))) = exp(f(x)) * f'(x)
        val = field.evaluate(x, y) # This is exp(f(x))
        base_field = BasisField(mock_basis, coeffs=sample_coeffs)
        deriv_base = base_field.derivative(x, y, order=(1, 0)) # This is f'(x)
        
        expected_deriv = val * deriv_base
        result_deriv = field.derivative(x, y, order=(1, 0))
        
        assert jnp.isclose(result_deriv, expected_deriv)

    def test_derivative_higher_order_raises_error(self, mock_basis):
        field = LogBasisField(mock_basis)
        x, y = 0.1, 0.2
        
        # First order should work
        field.derivative(x, y, order=(1, 0))
        field.derivative(x, y, order=(0, 1))

        # Higher orders should fail
        with pytest.raises(ValueError, match="only supports first order derivatives"):
            field.derivative(x, y, order=(1, 1))
            
        with pytest.raises(ValueError, match="only supports first order derivatives"):
            field.derivative(x, y, order=(2, 0))

class TestFitFunction:
    def test_fit_basic(self, mock_basis, sample_data, sample_coeffs):
        coords, values = sample_data
        
        field = fit_basis_field_from_data(mock_basis, coords, values, trainable=True)
        
        assert isinstance(field, BasisField)
        assert not isinstance(field, LogBasisField)
        assert isinstance(field.coeffs, Coeffs)
        # Check if the fitted coefficients are very close to the true ones
        assert jnp.allclose(field.coeffs.value, sample_coeffs.value, atol=1e-6)

    def test_fit_log_transform(self, mock_basis, sample_data):
        coords, values = sample_data
        # Ensure values are positive for log transform
        positive_values = values + 2.0 
        
        field = fit_basis_field_from_data(
            mock_basis, coords, positive_values, log_transform=True
        )
        
        assert isinstance(field, LogBasisField)

    def test_fit_static_coeffs(self, mock_basis, sample_data):
        coords, values = sample_data
        
        field = fit_basis_field_from_data(mock_basis, coords, values, trainable=False)
        
        assert isinstance(field.coeffs, StaticCoeffs)
        assert isinstance(field.coeffs.coeffs, list)

    # def test_fit_factorized_coeffs(self, mock_basis, sample_data, sample_coeffs):
    #     coords, values = sample_data
    #     rank = 2
        
    #     field = fit_basis_field_from_data(
    #         mock_basis, coords, values, factorize=True, r=rank
    #     )
        
    #     assert isinstance(field.coeffs, FactorizedCoeffs)
    #     assert field.coeffs.left_factor.shape == (4, rank)
    #     assert field.coeffs.right_factor.shape == (rank, 1) # Coeffs are a vector
        
    #     # Check if the reconstructed coeffs are close to the true ones
    #     reconstructed_coeffs = field.coeffs.value.flatten()
    #     assert jnp.allclose(reconstructed_coeffs, sample_coeffs.value, atol=1e-6)

    def test_fit_with_nan_values(self, mock_basis, sample_data, sample_coeffs):
        coords, values = sample_data
        
        # Introduce NaNs into the data
        values_with_nan = values.at[::5].set(jnp.nan)
        
        field = fit_basis_field_from_data(mock_basis, coords, values_with_nan)
        
        # The fit should still succeed and be close to the original coeffs
        # because it should ignore the NaN values.
        assert isinstance(field.coeffs, Coeffs)
        assert jnp.allclose(field.coeffs.value, sample_coeffs.value, atol=1e-6)
