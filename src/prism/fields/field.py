import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List, Union
import equinox as eqx
from abc import abstractmethod
import string
from ..basis.basis_nd import BasisND

class AbstractField(eqx.Module):
    @abstractmethod
    def evaluate(self, coords: jax.Array) -> jax.Array:
        pass
    
    @abstractmethod
    def derivative(self, coords: jax.Array, order: Tuple[int, ...]) -> jax.Array:
        pass

    @property
    def field_type(self):
        return {"field_type": type(self).__name__, "coeff_type": self.coeffs.coeff_type}

class AbstractCoeffs(eqx.Module):
    @classmethod
    def make_zero(cls, shape: Tuple[int, ...]) -> "AbstractCoeffs":
        pass

    @property
    def value(self):
        pass

    @property
    def coeff_type(self):
        return type(self).__name__

class Coeffs(AbstractCoeffs):
    coeffs: jax.Array

    @classmethod
    def make_zero(cls, shape: Tuple[int, ...]) -> "Coeffs":
        coeff_shape = tuple(d + 1 for d in shape)
        return cls(jnp.zeros(coeff_shape).reshape(-1))

    @property
    def value(self):
        return self.coeffs

class StaticCoeffs(AbstractCoeffs):
    coeffs: List[float]

    def __init__(self, coeffs: Union[List[float], jax.Array]):
        if isinstance(coeffs, jax.Array):
            self.coeffs = coeffs.tolist()
        else:
            self.coeffs = coeffs

    @classmethod
    def make_zero(cls, shape: Tuple[int, ...]) -> "StaticCoeffs":
        coeff_shape = tuple(d + 1 for d in shape)
        return cls(jnp.zeros(coeff_shape).reshape(-1).tolist())

    @property
    def value(self):
        return jnp.array(self.coeffs)


class ExponentialPreconditionedChebyshevCoeffs(AbstractCoeffs):
    """
    A coefficient class that applies a more aggressive exponential decay 
    preconditioner to improve optimization stability for stiff problems.
    This architecture conforms to the user's requested API.
    """
    coeffs: jax.Array          # The optimizable, preconditioned coefficients (flat array)
    preconditioner: list   # Stored as a list to be a static part of the PyTree (flat)

    @classmethod
    def make_zero(cls, degs: Tuple[int, int], decay_rate: float = 4.0) -> "PreconditionedChebyshevCoeffs":
        """Creates a zero-initialized coefficient object."""
        coeff_shape = tuple(d + 1 for d in degs)
        preconditioner = cls.make_preconditioner(coeff_shape, decay_rate)
        # The optimizable coeffs start at zero.
        coeffs = jnp.zeros(coeff_shape).flatten()
        return cls(coeffs, preconditioner.flatten().tolist())

    @staticmethod
    def make_preconditioner(shape: Tuple[int, ...], decay_rate: float) -> jax.Array:
        """Creates an exponentially decaying preconditioner grid."""
        nx, ny = shape
        ix = jnp.arange(nx)
        iy = jnp.arange(ny)
        # The preconditioner applies decay based on the total degree (i+j)
        preconditioner = decay_rate**(ix[:, None] + iy[None, :])
        return preconditioner

    @property
    def value(self) -> jax.Array:
        """Returns the true, physical Chebyshev coefficients as a grid."""
        true_coeffs_flat = self.coeffs * jnp.array(self.preconditioner)
        return true_coeffs_flat

class PreconditionedChebyshevCoeffs(AbstractCoeffs):
    coeffs: jax.Array
    preconditioner: list

    @classmethod
    def make_zero(cls, shape: Tuple[int, ...]) -> "PreconditionedChebyshevCoeffs":
        coeff_shape = tuple(d + 1 for d in shape)
        preconditioner = cls.make_preconditioner(coeff_shape)
        return cls(jnp.zeros(coeff_shape).reshape(-1), preconditioner.tolist())

    @classmethod
    def make_coeffs(cls, shape, coeffs):
        coeffs = jnp.array(coeffs)
        coeff_shape = tuple(d + 1 for d in shape)
        preconditioner = cls.make_preconditioner(coeff_shape)
        coeffs = coeffs/preconditioner
        return cls(coeffs, preconditioner.tolist())

    @staticmethod
    def make_preconditioner(shape: Tuple[int, ...]) -> jax.Array:
        ranges = [jnp.arange(d) for d in shape]
        grids = jnp.meshgrid(*ranges)
        grids = jnp.array(grids)
        preconditioner = 1/(1+jnp.sum((grids)**2, axis=0))
        preconditioner = preconditioner.reshape(-1)
        return preconditioner

    @property
    def value(self):
        return self.coeffs * jnp.array(self.preconditioner)

class PreconditionedNLS2D(eqx.Module):
    """Coefficients stored as 2D (Nx+1, Nt+1); value = D ⊙ θ flattened."""
    coeffs_2d: jax.Array       # trainable θ, shape (Nx+1, Nt+1)
    D_2d: list            # multiplicative factor D = sqrt(P), same shape

    @staticmethod
    def _make_D(shape: Tuple[int, int]) -> jax.Array:

        sigma = 0.5
        c0 = 10.0
        gamma_t = 1.0
        gamma_x = 1.0
        x_scale = 5.0
        t_scale = 0.785398  # pi/4

        Nx1, Nt1 = shape
        kx = jnp.arange(Nx1)[:, None]     # 0..Nx
        kt = jnp.arange(Nt1)[None, :]     # 0..Nt

        alpha_t = gamma_t * (1.0 / (t_scale**2))                 # ~ k_t^2 weight
        alpha_x = gamma_x * ((sigma / (x_scale**2))**2) * (jnp.pi**4 / 16.0)

        denom = c0 + alpha_t * (kt**2) + alpha_x * (kx**4)
        D = 1.0 / jnp.sqrt(denom + 1e-12)

        # Normalize so mean(D) = 1 to preserve LR semantics
        return D / jnp.mean(D)

    @classmethod
    def make_zero(cls, degs: Tuple[int, int], **kw):
        shape = (degs[0] + 1, degs[1] + 1)
        D_2d = cls._make_D(shape).tolist()
        return cls(coeffs_2d=jnp.zeros(shape), D_2d=D_2d)

    @classmethod
    def make_coeffs(cls, degs: Tuple[int, int], coeffs, **kw):
        shape = (degs[0] + 1, degs[1] + 1)
        c2d = jnp.array(coeffs).reshape(shape)
        D_2d = cls._make_D(shape).tolist()
        c2d = c2d/jnp.array(D_2d)
        return cls(coeffs_2d=c2d, D_2d=D_2d)

    @property
    def value(self) -> jax.Array:
        """Flattened, preconditioned coefficients used by BasisField."""
        return (self.coeffs_2d * jnp.array(self.D_2d)).reshape(-1)


class PreconditionedStokesCoeffs(AbstractCoeffs):
    coeffs: jax.Array
    preconditioner: list

    @classmethod
    def make_zero(cls, shape: tuple, x_scale: float = 1.0, y_scale: float = 1.0) -> "PreconditionedChebyshevCoeffs":
        coeff_shape = tuple(d + 1 for d in shape)
        preconditioner = cls.make_preconditioner(coeff_shape, x_scale, y_scale)
        return cls(jnp.zeros(coeff_shape).reshape(-1), preconditioner.tolist())

    @classmethod
    def make_coeffs(cls, shape: tuple, coeffs, x_scale: float = 1.0, y_scale: float = 1.0):
        coeffs = jnp.array(coeffs)
        coeff_shape = tuple(d + 1 for d in shape)
        preconditioner = cls.make_preconditioner(coeff_shape, x_scale, y_scale)
        coeffs = coeffs / preconditioner
        return cls(coeffs, preconditioner.tolist())

    @staticmethod
    def make_preconditioner(shape: tuple, x_scale: float, y_scale: float) -> jax.Array:
        # Create grids of mode numbers (kx, ky)
        ranges = [jnp.arange(d) for d in shape]
        grids = jnp.meshgrid(*ranges, indexing='ij') # Use 'ij' indexing for consistency
        grids = jnp.array(grids)

        # Create a scales array and reshape for broadcasting
        scales = jnp.array([x_scale, y_scale]).reshape(-1, 1, 1)

        # Scale the mode numbers according to the domain scales
        scaled_grids = grids / scales

        # Calculate the preconditioner using the scaled mode numbers
        preconditioner = 1 / (1 + jnp.sum(scaled_grids**2, axis=0))
        return preconditioner.reshape(-1)

    @property
    def value(self):
        return self.coeffs * jnp.array(self.preconditioner)

    
# class PreconditionedFactorizedCoeffs(AbstractCoeffs):
#     left_factor: jax.Array
#     right_factor: jax.Array
#     preconditioner: list
    
#     @classmethod
#     def make_zero(cls, shape: Tuple[int, ...], r: int) -> "PreconditionedFactorizedCoeffs":
#         n, m = shape
#         n += 1
#         m += 1
#         left_factor = jnp.ones((n,r))
#         right_factor = jnp.zeros((r,m))
#         ranges = [jnp.arange(n), jnp.arange(m)]
#         grids = jnp.meshgrid(*ranges)
#         grids = jnp.array(grids)
#         preconditioner = 1/(1+jnp.sum((grids)**2, axis=0))
#         preconditioner = preconditioner.reshape(-1)
#         return cls(left_factor, right_factor, preconditioner.tolist())

#     @property
#     def value(self):
#         coeffs = self.left_factor @ self.right_factor
#         coeffs = coeffs.reshape(-1)
#         return coeffs * jnp.array(self.preconditioner)

class PreconditionedFactorizedCoeffs(eqx.Module):
    """
    An Equinox-compatible factorizer using tuples for static structure.
    """
    factors: Tuple[jax.Array, ...]
    preconditioner: list # Not a trainable parameter
    einsum_str: str

    @classmethod
    def make_zero(cls, shape: Tuple[int, ...], r: int) -> "PreconditionedFactorizedCoeffs":
        temp_factors = []
        adjusted_shape = [s + 1 for s in shape]

        for i, dim_size in enumerate(adjusted_shape):
            if i < len(adjusted_shape) - 1:
                factor = jnp.ones((dim_size, r))
            else:
                factor = jnp.zeros((dim_size, r))
            temp_factors.append(factor)

        ranges = [jnp.arange(s) for s in adjusted_shape]
        grids = jnp.meshgrid(*ranges, indexing='ij')
        grids = jnp.array(grids)
        
        preconditioner = 1 / (1 + jnp.sum(grids**2, axis=0))
        preconditioner = preconditioner.reshape(-1)

        num_dims = len(adjusted_shape)
        input_subs = ','.join([f'{string.ascii_lowercase[i]}r' for i in range(num_dims)])
        output_subs = ''.join([string.ascii_lowercase[i] for i in range(num_dims)])
        einsum_str = f'{input_subs}->{output_subs}'

        # Convert the list of factors to a tuple when creating the class instance
        return cls(tuple(temp_factors), preconditioner.tolist(), einsum_str)

    @property
    def value(self) -> jax.Array:
        # Reconstruct the high-dimensional coefficient tensor
        coeffs = jnp.einsum(self.einsum_str, *self.factors)
        
        # Flatten and apply the preconditioner
        coeffs_flat = coeffs.reshape(-1)
        return coeffs_flat * jnp.array(self.preconditioner)

class FactorizedCoeffs(AbstractCoeffs):
    left_factor: jax.Array
    right_factor: jax.Array

    @classmethod
    def factorize(cls, coeffs: jax.Array, r: int) -> "FactorizedCoeffs":
        u, s, vt = jnp.linalg.svd(coeffs, full_matrices=False)
        U_r = u[:, :r]
        s_r = jnp.diag(s[:r])
        Vt_r = vt[:r, :]
        left_factor = U_r @ jnp.sqrt(s_r)
        right_factor = jnp.sqrt(s_r) @ Vt_r
        return cls(left_factor, right_factor)

    @classmethod
    def make_zero(cls, shape: Tuple[int, ...], r: int) -> "FactorizedCoeffs":
        n,m = shape
        n += 1
        m += 1
        left_factor = jnp.ones((n,r))
        right_factor = jnp.zeros((r,m))
        return cls(left_factor, right_factor)

    @classmethod
    def make_orthonormal(cls, key, shape, r):
        key, subkey, subkey2 = jax.random.split(key,3)
        n,m = shape
        n += 1
        m += 1
        left_factor = jax.random.orthogonal(key, n, m=r)
        right_factor = jax.random.orthogonal(subkey, r, m=m)
        scale = jax.random.uniform(subkey2, (r,), minval=0.0, maxval=0.1)
        scale = jnp.sqrt(jnp.diag(scale))
        left_factor = left_factor @ scale
        right_factor = scale @ right_factor
        return cls(left_factor, right_factor)
    # @property
    # def value(self):
    #     coeffs = self.left_factor @ self.right_factor
    #     return coeffs.reshape(-1)

    @property
    def value(self) -> jax.Array:
        epsilon = 1e-8
        norms = jnp.linalg.norm(self.left_factor, axis=0)
        left = self.left_factor / (norms + epsilon)
        right = self.right_factor * norms.reshape(-1,1)
        
        return (left @ right).reshape(-1)

class SVDCoeffs(AbstractCoeffs):
    left_factor: jax.Array
    scale_vector: jax.Array
    right_factor: jax.Array

    @classmethod
    def initialize(cls, key, shape, r):
        key, subkey, scale_key = jax.random.split(key, 3)
        n, m = shape
        n+=1
        m+=1
        
        left_init = jax.random.normal(key, (n, r))
        right_init = jax.random.normal(subkey, (r, m))
        initial_scales = jax.random.uniform(scale_key, shape=(r,)) * 0.1

        return cls(left_init, initial_scales, right_init)

    @property
    def value(self) -> jax.Array:
        u_l, _, vt_l = jnp.linalg.svd(self.left_factor, full_matrices=False)
        left_ortho = u_l @ vt_l
        u_r, _, vt_r = jnp.linalg.svd(self.right_factor.T, full_matrices=False)
        right_ortho = (u_r @ vt_r).T

        positive_scales = jax.nn.softplus(self.scale_vector)
        scale_matrix = jnp.diag(positive_scales)
        
        coeffs = left_ortho @ scale_matrix @ right_ortho
        return coeffs.reshape(-1)

class BasisField(AbstractField):
    coeffs: AbstractCoeffs
    basis: BasisND

    def __init__(self, basis: BasisND, coeffs: Optional[AbstractCoeffs] = None):
        self.basis = basis
        degs = basis.degs
        shape = tuple(deg + 1 for deg in degs)
        if coeffs is None:
            coeffs = Coeffs.make_zero(shape)
        self.coeffs = coeffs

    @eqx.filter_jit
    def evaluate(self, *coords: jax.Array, filter_frac: Optional[float] = None) -> jax.Array:
        order = tuple(0 for _ in range(len(self.basis.degs)))
        if filter_frac is not None:
            coeffs = self.spectral_filter(filter_frac)
        else:
            coeffs = self.coeffs.value
        return self.basis(*coords, order=order) @ coeffs
    
    @eqx.filter_jit
    def derivative(self, *coords: jax.Array, order: Tuple[int, ...], filter_frac: Optional[float] = None) -> jax.Array:
        if filter_frac is not None:
            coeffs = self.spectral_filter(filter_frac)
        else:
            coeffs = self.coeffs.value
        return self.basis(*coords, order=order) @ coeffs

    def spectral_filter(self, frac: float) -> jax.Array:
        shape = tuple(d + 1 for d in self.basis.degs)
        cutoffs = tuple(int(frac * s) for s in shape)
        
        # 3. Create a tuple of slice objects to define the corner
        # e.g., for cutoffs (2, 2), this creates (slice(2), slice(2))
        # which is equivalent to [:2, :2]
        corner_slice = tuple(slice(c) for c in cutoffs)
        
        # 4. Create a zero-mask and set the corner to one
        mask = jnp.zeros(shape, dtype=jnp.float32)
        mask = mask.at[corner_slice].set(1.0)
        coeffs = self.coeffs.value
        
        return coeffs * mask.flatten()

class LogBasisField(BasisField):
    """Implementation of a basis field that is log-transformed, i.e. field is stored as log(field) and converted back to the original field when evaluated.
    This is useful for fields that are positive but have a wide range of values.
    """
    coeffs: AbstractCoeffs
    basis: BasisND

    def __init__(self, basis: BasisND, coeffs: Optional[AbstractCoeffs] = None):
        super().__init__(basis, coeffs)

    @eqx.filter_jit
    def evaluate(self, *coords: jax.Array, filter_frac: Optional[float] = None) -> jax.Array:
        vals = super().evaluate(*coords, filter_frac=filter_frac)
        return jnp.exp(vals)
    
    @eqx.filter_jit
    def derivative(self, *coords: jax.Array, order: Tuple[int, ...], filter_frac: Optional[float] = None) -> jax.Array:
        if sum(order) == 1:
            derivative = super().derivative(*coords, order=order, filter_frac=filter_frac)
            vals = self.evaluate(*coords, filter_frac=filter_frac)
            return vals * derivative
        elif sum(order) == 2:
            # For second-order derivatives, we need to apply the chain rule:
            # d²/dx²[exp(f)] = exp(f) * (f'² + f'')
            # First, find which coordinate has the second derivative (order=2)
            derivative_coord = order.index(2)
            
            # Create first derivative order tuple (change 2 to 1)
            first_order = tuple(1 if i == derivative_coord else 0 for i in range(len(order)))
            
            # Get first derivative of the log field
            first_derivative = super().derivative(*coords, order=first_order, filter_frac=filter_frac)
            vals = self.evaluate(*coords, filter_frac=filter_frac)
            second_derivative = super().derivative(*coords, order=order, filter_frac=filter_frac)
            return vals * (first_derivative**2 + second_derivative)
        else:
            raise ValueError("LogBasisField only supports first and second order derivatives")

class SoftplusBasisField(BasisField):
    """
    Implements a basis field that is transformed by the softplus function.
    The underlying field can be any real value, but the evaluated field is always positive.
    c = softplus(ψ) = log(1 + exp(ψ))
    """
    coeffs: AbstractCoeffs
    basis: BasisND

    def __init__(self, basis: BasisND, coeffs: Optional[AbstractCoeffs] = None):
        super().__init__(basis, coeffs)

    @eqx.filter_jit
    def evaluate(self, *coords: jax.Array, filter_frac: Optional[float] = None) -> jax.Array:
        # Evaluate the underlying field, which we'll call ψ (psi)
        psi = super().evaluate(*coords, filter_frac=filter_frac)
        # Apply the softplus function to ensure positivity
        return jax.nn.softplus(psi)

    @eqx.filter_jit
    def derivative(self, *coords: jax.Array, order: Tuple[int, ...], filter_frac: Optional[float] = None) -> jax.Array:
        # Get the value of the underlying field, ψ
        psi = super().evaluate(*coords, filter_frac=filter_frac)
        
        if sum(order) == 1:
            # First derivative: c' = sigmoid(ψ) * ψ'
            psi_prime = super().derivative(*coords, order=order, filter_frac=filter_frac)
            return jax.nn.sigmoid(psi) * psi_prime
        
        elif sum(order) == 2:
            # Second derivative: c'' = sigmoid(ψ) * [ (1 - sigmoid(ψ))*(ψ')² + ψ'' ]
            
            # Find which coordinate has the second derivative (order=2)
            derivative_coord = order.index(2)
            
            # Create the order tuple for the corresponding first derivative
            first_order = tuple(1 if i == derivative_coord else 0 for i in range(len(order)))
            
            # Get the first and second derivatives of the underlying field, ψ
            psi_prime = super().derivative(*coords, order=first_order, filter_frac=filter_frac)
            psi_double_prime = super().derivative(*coords, order=order, filter_frac=filter_frac)
            
            # Calculate sigmoid(ψ) once to reuse it
            sig_psi = jax.nn.sigmoid(psi)
            
            # Implement the chain rule formula for the second derivative
            term1 = (1 - sig_psi) * (psi_prime**2)
            term2 = psi_double_prime
            return sig_psi * (term1 + term2)
        else:
            raise ValueError("SoftplusBasisField only supports first and second order derivatives")

class HarmonicBasis2D(eqx.Module):
    degs: Tuple[int, int]
    harmonic_deg: int

    def __init__(self, deg: int):
        if not isinstance(deg, int) or deg < 0:
            raise ValueError("Degree must be a non-negative integer.")
        if deg % 2 != 0:
            raise ValueError(f"For a direct mapping, the grid degree must be even. Got {deg}.")
            
        self.degs = (deg, deg)
        # Calculate the required harmonic degree 'n' such that 2n+1 = (deg+1)^2
        self.harmonic_deg = ((deg + 1)**2 - 1) // 2

    @eqx.filter_jit  
    def __call__(self, *coords: jax.Array, order: Tuple[int, int]) -> jax.Array:
        if order != (0, 0):
            raise NotImplementedError("This basis only supports function evaluation (order=(0,0)).")

        x, y = jnp.asarray(coords[0]).flatten(), jnp.asarray(coords[1]).flatten()
        n_h = self.harmonic_deg

        if n_h < 0:
            num_flat_coeffs = (self.degs[0] + 1)**2
            return jnp.zeros((x.shape[0], num_flat_coeffs))

        # Define the recurrence step function, which operates purely on real numbers
        def recurrence_step(carry, _):
            p_prev, q_prev = carry
            p_curr = x * p_prev - y * q_prev
            q_curr = y * p_prev + x * q_prev
            return (p_curr, q_curr), (p_curr, q_curr)

        # Initial values P_0 = 1, Q_0 = 0
        p0 = jnp.ones_like(x)
        q0 = jnp.zeros_like(x)
        
        # Run the scan to generate all P_n and Q_n up to the harmonic degree
        _, (p_all, q_all) = jax.lax.scan(recurrence_step, (p0, q0), None, length=n_h)

        # Assemble the basis matrix by interleaving P and Q
        num_basis_funcs = 2 * n_h + 1
        basis_matrix = jnp.zeros((x.shape[0], num_basis_funcs))
        
        # Set P_0 (the constant term)
        basis_matrix = basis_matrix.at[:, 0].set(p0)
        if n_h > 0:
            # Set P_n for n > 0
            basis_matrix = basis_matrix.at[:, 1::2].set(p_all.T)
            # Set Q_n for n > 0
            basis_matrix = basis_matrix.at[:, 2::2].set(q_all.T)

        return basis_matrix

class FourierContinuationBasis2D(eqx.Module):
    """
    A harmonic basis using a plane wave expansion.
    
    The basis functions are of the form cos(kx)cosh(ky), sin(kx)sinh(ky), etc.,
    which are guaranteed to be solutions of Laplace's equation.
    """
    degs: Tuple[int, int]
    num_k: int
    num_total_coeffs: int

    def __init__(self, deg: int):
        if not isinstance(deg, int) or deg < 0:
            raise ValueError("Degree must be a non-negative integer.")
        if deg % 2 != 0:
            raise ValueError(f"For a direct mapping, the grid degree must be even. Got {deg}.")
            
        self.degs = (deg, deg)
        # Match the number of coefficients to the equivalent full polynomial grid
        self.num_total_coeffs = (deg + 1)**2
        # Determine the number of wavenumbers needed. We will have 4 functions per wavenumber k > 0.
        # Plus one constant term.
        self.num_k = (self.num_total_coeffs - 1) // 4

    @eqx.filter_jit
    def __call__(self, *coords: jax.Array, order: Tuple[int, int]) -> jax.Array:
        if order != (0, 0):
            raise NotImplementedError("This basis only supports function evaluation (order=(0,0)).")

        x, y = jnp.asarray(coords[0]).flatten(), jnp.asarray(coords[1]).flatten()
        
        if self.num_k < 0:
             return jnp.zeros((x.shape[0], self.num_total_coeffs))

        # Wavenumbers (excluding k=0, which is the constant term)
        # Using pi scales the domain naturally to [-1, 1]
        k = jnp.pi * jnp.arange(1, self.num_k + 1)

        # Reshape for broadcasting
        # k has shape (num_k,)
        # x, y have shape (num_points,)
        # We want outputs of shape (num_points, num_k)
        kx = k[None, :] * x[:, None]
        ky = k[None, :] * y[:, None]

        # Four families of harmonic functions
        basis1 = jnp.cos(kx) * jnp.cosh(ky)
        basis2 = jnp.sin(kx) * jnp.sinh(ky)
        basis3 = jnp.cos(ky) * jnp.cosh(kx) # Swap roles of x and y
        basis4 = jnp.sin(ky) * jnp.sinh(kx)

        # Constant term for k=0
        p0 = jnp.ones_like(x)[:, None]

        # Stack them all together
        # Shape: (num_points, 1 + 4 * num_k)
        full_basis = jnp.hstack([p0, basis1, basis2, basis3, basis4])
        
        # Truncate to the exact number of coefficients requested
        return full_basis[:, :self.num_total_coeffs]