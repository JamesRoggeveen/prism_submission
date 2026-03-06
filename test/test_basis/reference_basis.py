import numpy as np
import numpy.polynomial.chebyshev as np_cheb
import numpy.polynomial.legendre as np_leg
from typing import Tuple
def get_expected_chebyshev_basis(x, deg, order=0):

    expected_v = np_cheb.chebvander(x, deg)

    expected_results = [expected_v]

    if order > 0:
        expected_vd = np.zeros_like(expected_v)
        for i in range(1, deg + 1):
            # The coefficients for T_i(x) are [0, 0, ..., 1]
            c = np.zeros(i + 1)
            c[i] = 1
            
            # Differentiate and evaluate
            deriv_coeffs = np_cheb.chebder(c)
            expected_vd[:, i] = np_cheb.chebval(x, deriv_coeffs)
        expected_results.append(expected_vd)

    if order > 1:
        # For T''_n(x)
        expected_vdd = np.zeros_like(expected_v)
        for i in range(2, deg + 1):
            c = np.zeros(i + 1)
            c[i] = 1
            
            # Differentiate twice and evaluate
            deriv2_coeffs = np_cheb.chebder(c, m=2)
            expected_vdd[:, i] = np_cheb.chebval(x, deriv2_coeffs)
        expected_results.append(expected_vdd)

    return expected_results

def get_expected_fourier_basis(x, deg, order=0):
    """
    Generates the expected Fourier basis values using a simple loop,
    serving as a ground truth for testing the vectorized version.
    """
    x_vec = np.atleast_1d(x)
    num_x = x_vec.shape[0]
    
    # Initialize arrays for expected values
    expected_v = np.zeros((num_x, deg + 1))
    expected_dx = np.zeros((num_x, deg + 1))
    expected_ddx = np.zeros((num_x, deg + 1))

    # Basis function 0: constant term
    expected_v[:, 0] = 1.0

    # Higher order basis functions
    for i in range(1, deg + 1):
        if i % 2 == 1:  # Odd indices correspond to cos(k * pi * x)
            k = (i + 1) / 2
            expected_v[:, i] = np.cos(k * np.pi * x_vec)
            if order > 0:
                expected_dx[:, i] = -k * np.pi * np.sin(k * np.pi * x_vec)
            if order > 1:
                expected_ddx[:, i] = -(k * np.pi)**2 * expected_v[:, i]
        else:  # Even indices correspond to sin(k * pi * x)
            k = i / 2
            expected_v[:, i] = np.sin(k * np.pi * x_vec)
            if order > 0:
                expected_dx[:, i] = k * np.pi * np.cos(k * np.pi * x_vec)
            if order > 1:
                expected_ddx[:, i] = -(k * np.pi)**2 * expected_v[:, i]
    results = [expected_v]
    if order > 0:
        results.append(expected_dx)
    if order > 1:
        results.append(expected_ddx)
        
    return results

def get_expected_cosine_basis(x, deg, order=0):
    """
    Generates the expected Fourier basis values using a simple loop,
    serving as a ground truth for testing the vectorized version.
    """
    x_vec = np.atleast_1d(x)
    num_x = x_vec.shape[0]
    
    # Initialize arrays for expected values
    expected_v = np.zeros((num_x, deg + 1))
    expected_dx = np.zeros((num_x, deg + 1))
    expected_ddx = np.zeros((num_x, deg + 1))

    # Basis function 0: constant term
    expected_v[:, 0] = 1.0

    # Higher order basis functions
    for i in range(1, deg + 1):
        k = i
        expected_v[:, i] = np.cos(k * np.pi * x_vec)
        if order > 0:
            expected_dx[:, i] = -k * np.pi * np.sin(k * np.pi * x_vec)
        if order > 1:
            expected_ddx[:, i] = -(k * np.pi)**2 * expected_v[:, i]
    
    results = [expected_v]
    if order > 0:
        results.append(expected_dx)
    if order > 1:
        results.append(expected_ddx)
        
    return results

def get_expected_legendre_basis(x, deg, order=0):
    expected_v = np_leg.legvander(x, deg)
    expected_results = [expected_v]

    # Order 1: P'_n(x)
    if order > 0:
        expected_vd = np.zeros_like(expected_v)
        # P'_0(x) is 0, so we start the loop at n=1.
        for i in range(1, deg + 1):
            # The coefficients for a single Legendre polynomial P_i(x) are
            # simply [0, 0, ..., 1] in the Legendre basis.
            c = np.zeros(i + 1)
            c[i] = 1
            
            # Get the coefficients of the derivative polynomial.
            deriv_coeffs = np_leg.legder(c, m=1)
            # Evaluate the derivative polynomial at the points x.
            expected_vd[:, i] = np_leg.legval(x, deriv_coeffs)
        expected_results.append(expected_vd)

    # Order 2: P''_n(x)
    if order > 1:
        expected_vdd = np.zeros_like(expected_v)
        # P''_0(x) and P''_1(x) are 0, so we start at n=2.
        for i in range(2, deg + 1):
            c = np.zeros(i + 1)
            c[i] = 1
            
            # Differentiate twice to get the second derivative's coefficients.
            deriv2_coeffs = np_leg.legder(c, m=2)
            # Evaluate the second derivative polynomial.
            expected_vdd[:, i] = np_leg.legval(x, deriv2_coeffs)
        expected_results.append(expected_vdd)

    return expected_results


def get_expected_2d_basis(x, y, deg, fns, order, scattered) -> Tuple[np.ndarray, ...]:
    """Reference implementation using loops to verify the vectorized functions."""
    deg_x, deg_y = deg

    if scattered:
        num_points = x.shape[0]
        x_coords, y_coords = x, y
    else:
        num_points = x.shape[0] * y.shape[0]
        xx, yy = np.meshgrid(x, y)
        x_coords, y_coords = xx.flatten(), yy.flatten()

    # Pre-compute 1D bases for all required points for the reference calculation
    x_basis_all_pts = fns[0](x_coords, deg_x, order)
    y_basis_all_pts = fns[1](y_coords, deg_y, order)

    total_basis_funcs = (deg_x + 1) * (deg_y + 1)
    
    # Initialize arrays for expected values
    basis_xy = np.zeros((num_points, total_basis_funcs))
    if order > 0:
        basis_dx = np.zeros_like(basis_xy)
        basis_dy = np.zeros_like(basis_xy)
    if order > 1:
        basis_ddx = np.zeros_like(basis_xy)
        basis_ddy = np.zeros_like(basis_xy)
        basis_dxy = np.zeros_like(basis_xy)

    # Loop through each point and basis combination
    for p in range(num_points):
        for i in range(deg_x + 1):
            for j in range(deg_y + 1):
                col_idx = i * (deg_y + 1) + j
                
                # Order 0
                basis_xy[p, col_idx] = x_basis_all_pts[0][p, i] * y_basis_all_pts[0][p, j]
                
                # Order 1
                if order > 0:
                    basis_dx[p, col_idx] = x_basis_all_pts[1][p, i] * y_basis_all_pts[0][p, j]
                    basis_dy[p, col_idx] = x_basis_all_pts[0][p, i] * y_basis_all_pts[1][p, j]
                
                # Order 2
                if order > 1:
                    basis_ddx[p, col_idx] = x_basis_all_pts[2][p, i] * y_basis_all_pts[0][p, j]
                    basis_ddy[p, col_idx] = x_basis_all_pts[0][p, i] * y_basis_all_pts[2][p, j]
                    basis_dxy[p, col_idx] = x_basis_all_pts[1][p, i] * y_basis_all_pts[1][p, j]

    if order == 0: return (basis_xy,)
    if order == 1: return basis_xy, basis_dx, basis_dy
    if order == 2: return basis_xy, basis_dx, basis_dy, basis_ddx, basis_dxy, basis_ddy