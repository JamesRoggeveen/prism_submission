import jax.numpy as jnp
import jax
from typing import Tuple
import equinox as eqx

@eqx.filter_jit
def vectorized_chebyshev_basis(x: jax.Array, deg: int, order: int = 0) -> Tuple[jax.Array, ...]:
    """
    Computes the value of the Chebyshev polynomial and its derivatives at given points
    in a vectorized manner.
    """
    if deg < 0:
        raise ValueError("Degree must be a non-negative integer.")
    if not 0 <= order <= 2:
        raise ValueError("Order must be 0, 1, or 2.")

    # FIX: Ensure x is always at least a 1D array to handle scalar inputs
    x = jnp.atleast_1d(x)

    # Handle deg=0 case cleanly
    if deg == 0:
        v_x = jnp.ones_like(x)[None, :]
        if order == 0: 
            return (v_x.T,)
        v_dx = jnp.zeros_like(x)[None, :]
        if order == 1: 
            return (v_x.T, v_dx.T)
        v_ddx = jnp.zeros_like(x)[None, :]
        return (v_x.T, v_dx.T, v_ddx.T)

    # --- 1. Vectorized Vandermonde for T_n(x) ---
    def recurrence_step(carry, _):
        p_prev, p_curr = carry
        p_next = 2 * x * p_curr - p_prev
        return (p_curr, p_next), p_next

    t0, t1 = jnp.ones_like(x), x
    
    if deg > 1:
        # We have T_0, T_1, need deg-1 more polynomials (T_2, ..., T_deg)
        _, t_rest = jax.lax.scan(recurrence_step, (t0, t1), None, length=deg - 1)
        v_x = jnp.vstack([t0, t1, t_rest])
    else: # deg == 1
        v_x = jnp.vstack([t0, t1])

    if order == 0:
        return (v_x.T,)

    # --- 2. Vectorized Vandermonde for T'_n(x) ---
    u0, u1 = jnp.ones_like(x), 2 * x
    
    if deg > 1:
        # We need U_0, ..., U_{deg-1}. We have U_0, U_1. Need deg-2 more.
        _, u_rest = jax.lax.scan(recurrence_step, (u0, u1), None, length=max(0, deg - 2))
        u_vals = jnp.vstack([u0, u1, u_rest])
    else: # deg == 1
        u_vals = jnp.atleast_2d(u0)

    degrees = jnp.arange(deg + 1)
    v_dx = jnp.zeros((deg + 1, *x.shape))
    # T'_n = n * U_{n-1} for n>=1
    v_dx = v_dx.at[1:].set(degrees[1:, None] * u_vals)

    if order == 1:
        return v_x.T, v_dx.T

    # --- 3. Vectorized Vandermonde for T''_n(x) ---
    n_sq = degrees**2
    numerator = x * v_dx - n_sq[:, None] * v_x
    denominator = 1 - x**2
    
    v_ddx_interior = numerator / denominator

    v_ddx_boundary = (n_sq * (n_sq - 1) / 3.0)[:, None]
    sign = jnp.power(-1.0, degrees)[:, None]
    v_ddx_minus_one = sign * v_ddx_boundary

    is_plus_one = jnp.isclose(x, 1.0)
    is_minus_one = jnp.isclose(x, -1.0)
    
    v_ddx = jnp.where(is_plus_one, v_ddx_boundary, v_ddx_interior)
    v_ddx = jnp.where(is_minus_one, v_ddx_minus_one, v_ddx)
    
    # T''_0 and T''_1 are always zero.
    v_ddx = v_ddx.at[0:2, :].set(0.0)

    return v_x.T, v_dx.T, v_ddx.T

@eqx.filter_jit
def vectorized_fourier_basis(x: jax.Array, deg: int, order: int = 0) -> Tuple[jax.Array, ...]:
    """
    Computes the values of a Fourier basis and its derivatives in a vectorized manner.

    The basis is [1, cos(pi*x), sin(pi*x), cos(2*pi*x), sin(2*pi*x), ...].

    Args:
        x: jnp.ndarray
            A scalar or array of points at which to evaluate the basis.
        deg: int
            The maximum basis index. The total number of basis functions will be deg + 1.
        order: int, optional
            The order of the derivative to compute (0, 1, or 2). Default is 0.

    Returns:
        A tuple containing the values of the Fourier basis and its requested
        derivatives. Each array has a shape of (num_points, N + 1).
    """
    if deg < 0:
        raise ValueError("N must be a non-negative integer.")
    if not 0 <= order <= 2:
        raise ValueError("Order must be 0, 1, or 2.")

    # Ensure x is always at least a 1D array to handle scalar inputs
    x = jnp.atleast_1d(x)

    # --- 0. Setup ---
    # Handle N=0 case separately
    if deg == 0:
        v_x = jnp.ones_like(x)[None, :]
        if order == 0: 
            return (v_x.T,)
        v_dx = jnp.zeros_like(x)[None, :]
        if order == 1: 
            return (v_x.T, v_dx.T)
        v_ddx = jnp.zeros_like(x)[None, :]
        return (v_x.T, v_dx.T, v_ddx.T)

    # Indices for basis functions, from 1 to N
    i = jnp.arange(1, deg + 1)
    # Corresponding k values for frequency: [1, 1, 2, 2, 3, 3, ...]
    k = (i + 1) // 2
    # Pre-calculate k * pi for broadcasting
    k_pi = k * jnp.pi
    # Argument for trig functions, shape (N, num_points)
    k_pi_x = k_pi[:, None] * x[None, :]

    # --- 1. Basis values (order 0) ---
    v_x = jnp.zeros((deg + 1, *x.shape))
    v_x = v_x.at[0].set(1.0)

    # Calculate cos and sin values for all k
    cos_vals = jnp.cos(k_pi_x)
    sin_vals = jnp.sin(k_pi_x)

    # Use `where` to select based on whether i is odd (cos) or even (sin)
    is_odd = (i % 2 == 1)
    # Broadcast `is_odd` to match the shape of cos_vals/sin_vals
    v_x = v_x.at[1:].set(jnp.where(is_odd[:, None], cos_vals, sin_vals))

    if order == 0:
        return (v_x.T,)

    # --- 2. First derivatives (order 1) ---
    v_dx = jnp.zeros_like(v_x)
    # For cos(k*pi*x), derivative is -k*pi*sin(k*pi*x)
    deriv_cos_basis = -k_pi[:, None] * sin_vals
    # For sin(k*pi*x), derivative is  k*pi*cos(k*pi*x)
    deriv_sin_basis = k_pi[:, None] * cos_vals
    
    v_dx = v_dx.at[1:].set(jnp.where(is_odd[:, None], deriv_cos_basis, deriv_sin_basis))

    if order == 1:
        return v_x.T, v_dx.T

    # --- 3. Second derivatives (order 2) ---
    v_ddx = jnp.zeros_like(v_x)
    # For both cos and sin, the second derivative is -(k*pi)^2 * original_function
    k_pi_sq = k_pi**2
    # We can reuse v_x that we calculated earlier.
    # Note: v_x[0] is 1, its second derivative is 0, which is the default in v_ddx.
    v_ddx = v_ddx.at[1:].set(-k_pi_sq[:, None] * v_x[1:])
    
    return v_x.T, v_dx.T, v_ddx.T

@eqx.filter_jit
def vectorized_cosine_basis(x: jax.Array, deg: int, order: int = 0) -> Tuple[jax.Array, ...]:
    """
    Computes the values of a Cosine basis and its derivatives in a vectorized manner.

    The basis is [1, cos(pi*x), cos(2*pi*x), cos(3*pi*x), ...].

    Args:
        x: jnp.ndarray
            A scalar or array of points at which to evaluate the basis.
        deg: int
            The maximum basis index. The total number of basis functions will be deg + 1.
        order: int, optional
            The order of the derivative to compute (0, 1, or 2). Default is 0.

    Returns:
        A tuple containing the values of the Fourier basis and its requested
        derivatives. Each array has a shape of (num_points, N + 1).
    """
    if deg < 0:
        raise ValueError("N must be a non-negative integer.")
    if not 0 <= order <= 2:
        raise ValueError("Order must be 0, 1, or 2.")

    # Ensure x is always at least a 1D array to handle scalar inputs
    x = jnp.atleast_1d(x)

    # --- 0. Setup ---
    # Handle N=0 case separately
    if deg == 0:
        v_x = jnp.ones_like(x)[None, :]
        if order == 0: 
            return (v_x.T,)
        v_dx = jnp.zeros_like(x)[None, :]
        if order == 1: 
            return (v_x.T, v_dx.T)
        v_ddx = jnp.zeros_like(x)[None, :]
        return (v_x.T, v_dx.T, v_ddx.T)

    # Indices for basis functions, from 1 to N
    i = jnp.arange(1, deg + 1)
    # Corresponding k values for frequency: [1, 2, 3, 4, 5, 6, ...]
    k = i
    # Pre-calculate k * pi for broadcasting
    k_pi = k * jnp.pi
    # Argument for trig functions, shape (N, num_points)
    k_pi_x = k_pi[:, None] * x[None, :]

    # --- 1. Basis values (order 0) ---
    v_x = jnp.zeros((deg + 1, *x.shape))
    v_x = v_x.at[0].set(1.0)

    # Calculate cos and sin values for all k
    cos_vals = jnp.cos(k_pi_x)
    sin_vals = jnp.sin(k_pi_x)

    # Broadcast `is_odd` to match the shape of cos_vals/sin_vals
    v_x = v_x.at[1:].set(cos_vals)

    if order == 0:
        return (v_x.T,)

    # --- 2. First derivatives (order 1) ---
    v_dx = jnp.zeros_like(v_x)
    # For cos(k*pi*x), derivative is -k*pi*sin(k*pi*x)
    deriv_cos_basis = -k_pi[:, None] * sin_vals

    v_dx = v_dx.at[1:].set(deriv_cos_basis)

    if order == 1:
        return v_x.T, v_dx.T

    # --- 3. Second derivatives (order 2) ---
    v_ddx = jnp.zeros_like(v_x)
    # For both cos and sin, the second derivative is -(k*pi)^2 * original_function
    k_pi_sq = k_pi**2
    # We can reuse v_x that we calculated earlier.
    # Note: v_x[0] is 1, its second derivative is 0, which is the default in v_ddx.
    v_ddx = v_ddx.at[1:].set(-k_pi_sq[:, None] * v_x[1:])
    
    return v_x.T, v_dx.T, v_ddx.T

@eqx.filter_jit
def vectorized_legendre_basis(x: jax.Array, deg: int, order: int = 0) -> Tuple[jax.Array, ...]:
    """
    Computes the value of the Legendre polynomial and its derivatives at given points
    in a vectorized manner.

    The basis is [P_0(x), P_1(x), P_2(x), ...].

    Args:
        x: jnp.ndarray
            A scalar or array of points at which to evaluate the basis.
        deg: int
            The maximum degree of the polynomial. The total number of basis
            functions will be deg + 1.
        order: int, optional
            The order of the derivative to compute (0, 1, or 2). Default is 0.

    Returns:
        A tuple containing the values of the Legendre basis and its requested
        derivatives. Each array has a shape of (num_points, deg + 1).
    """
    if deg < 0:
        raise ValueError("Degree must be a non-negative integer.")
    if not 0 <= order <= 2:
        raise ValueError("Order must be 0, 1, or 2.")

    # Ensure x is always at least a 1D array to handle scalar inputs
    x = jnp.atleast_1d(x)

    # Handle deg=0 case cleanly
    if deg == 0:
        v_x = jnp.ones_like(x)[None, :]
        if order == 0:
            return (v_x.T,)
        v_dx = jnp.zeros_like(x)[None, :]
        if order == 1:
            return (v_x.T, v_dx.T)
        v_ddx = jnp.zeros_like(x)[None, :]
        return (v_x.T, v_dx.T, v_ddx.T)

    # --- 1. Vectorized Vandermonde for P_n(x) ---
    # Uses Bonnet's recurrence relation: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
    def recurrence_step(carry, n):
        p_prev, p_curr = carry
        # n in the loop corresponds to the degree of p_curr
        p_next = ((2 * n + 1) * x * p_curr - n * p_prev) / (n + 1)
        return (p_curr, p_next), p_next

    p0, p1 = jnp.ones_like(x), x
    
    if deg > 1:
        # We have P_0, P_1, and need to compute P_2, ..., P_deg
        ns = jnp.arange(1, deg)
        _, p_rest = jax.lax.scan(recurrence_step, (p0, p1), ns)
        v_x = jnp.vstack([p0, p1, p_rest])
    else:  # deg == 1
        v_x = jnp.vstack([p0, p1])

    if order == 0:
        return (v_x.T,)

    # --- 2. Vectorized Vandermonde for P'_n(x) ---
    n = jnp.arange(deg + 1)
    
    # Interior points use the relation: P'_n = n(P_{n-1} - xP_n) / (1-x^2)
    numerator = n[1:, None] * (v_x[:-1] - x[None, :] * v_x[1:])
    denominator = 1 - x**2
    v_dx_interior = numerator / denominator

    v_dx_calc = jnp.zeros((deg + 1, *x.shape))
    v_dx_calc = v_dx_calc.at[1:, :].set(v_dx_interior)
    
    # Boundary points x = +/- 1 have special values
    # P'_n(1) = n(n+1)/2
    # P'_n(-1) = (-1)^{n+1} * n(n+1)/2
    v_dx_boundary_val = (n * (n + 1) / 2.0)[:, None]
    sign = jnp.power(-1.0, n + 1)[:, None]
    v_dx_plus_one = v_dx_boundary_val
    v_dx_minus_one = sign * v_dx_boundary_val

    is_plus_one = jnp.isclose(x, 1.0)
    is_minus_one = jnp.isclose(x, -1.0)
    
    v_dx = jnp.where(is_plus_one, v_dx_plus_one, v_dx_calc)
    v_dx = jnp.where(is_minus_one, v_dx_minus_one, v_dx)

    if order == 1:
        return v_x.T, v_dx.T

    # --- 3. Vectorized Vandermonde for P''_n(x) ---
    # Uses Legendre's diff eq: P''_n = (2xP'_n - n(n+1)P_n) / (1-x^2)
    n_term = (n * (n + 1))[:, None]
    numerator_ddx = 2 * x[None, :] * v_dx - n_term * v_x
    v_ddx_interior = numerator_ddx / denominator
    
    # Boundary points x = +/- 1 have special values
    # P''_n(1) = (n-1)n(n+1)(n+2)/8
    # P''_n(-1) = (-1)^n * P''_n(1)
    v_ddx_boundary_val = ((n - 1) * n * (n + 1) * (n + 2) / 8.0)[:, None]
    sign_ddx = jnp.power(-1.0, n)[:, None]
    v_ddx_plus_one = v_ddx_boundary_val
    v_ddx_minus_one = sign_ddx * v_ddx_boundary_val
    
    v_ddx = jnp.where(is_plus_one, v_ddx_plus_one, v_ddx_interior)
    v_ddx = jnp.where(is_minus_one, v_ddx_minus_one, v_ddx)
    
    # P''_0 and P''_1 are always zero. Set explicitly for robustness.
    v_ddx = v_ddx.at[0:2, :].set(0.0)

    return v_x.T, v_dx.T, v_ddx.T