import jax.numpy as jnp
import matplotlib.pyplot as plt
import pathlib
import prism as pr
import argparse
import numpy as np
from scipy.integrate import quad
from laplace import domain_from_png, load_config, boundary_condition

RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_laplace"

def find_latest_folder(base_dir: pathlib.Path, name_suffix: str) -> pathlib.Path | None:
    glob_pattern = f"????????{name_suffix}"
    matching_folders = list(base_dir.glob(glob_pattern))
    
    if not matching_folders:
        return None
    latest_folder = sorted(matching_folders, key=lambda p: p.name, reverse=True)[0]
    return latest_folder

def get_target_folder():
    parser = argparse.ArgumentParser(
        description=f"Process data from an experiment folder within '{RESULTS_BASE_DIR}'."
    )
    parser.add_argument(
        "folder_name", 
        nargs='?',
        default=None,
        type=str,
        help=f"Optional: Name of a specific folder in '{RESULTS_BASE_DIR}'. If not provided, finds the latest."
    )
    args = parser.parse_args()

    if args.folder_name:
        target_folder = RESULTS_BASE_DIR / args.folder_name
        
        if not target_folder.is_dir():
            print(f"Error: Folder '{target_folder}' does not exist or is not a directory.")
            return None
            
        print(f"Using provided folder: {target_folder}")
        return target_folder
        
    else:
        print(f"No folder name provided. Searching for the most recent folder in '{RESULTS_BASE_DIR}'...")
        
        latest_folder = find_latest_folder(RESULTS_BASE_DIR, FOLDER_SUFFIX)
        
        if latest_folder:
            print(f"Found latest folder: {latest_folder}")
            return latest_folder
        else:
            print(f"Error: No folders found in '{RESULTS_BASE_DIR}' matching the pattern.")
            return None

def get_laplace_solution_at_point(x, y, boundary_func, radius=1.0):
    # Calculate the radial distance of the point
    r = jnp.sqrt(x**2 + y**2)

    # 1. Check if the point is outside the disk's radius
    if r > radius:
        return jnp.nan

    # 2. Handle the special case at the center (r=0)
    # The solution at the center is the average value of the boundary condition.
    if jnp.isclose(r, 0):
        avg_value, _ = quad(boundary_func, 0, 2 * jnp.pi)
        return avg_value / (2 * jnp.pi)

    # 3. Convert Cartesian coordinates to polar for the formula
    theta = jnp.arctan2(y, x)

    # 4. Define the Poisson Integral Kernel for the integration
    def poisson_integrand(phi):
        """The integrand part of the Poisson formula."""
        f_phi = boundary_func(phi)
        kernel = (radius**2 - r**2) / (radius**2 - 2 * radius * r * jnp.cos(theta - phi) + r**2)
        return f_phi * kernel

    # 5. Perform the numerical integration from 0 to 2*pi
    integral_val, _ = quad(poisson_integrand, 0, 2 * jnp.pi)

    # 6. Return the final solution value
    return integral_val / (2 * jnp.pi)

def boundary_fourier_coeffs(boundary_func, M=2048):
    M = int(M)
    theta = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False, dtype=float)
    f = np.asarray(boundary_func(theta), dtype=float)
    F = np.fft.rfft(f) / M
    a0 = F[0].real
    nmax = F.shape[0] - 1
    if nmax <= 0:
        a = np.zeros((0,), dtype=float)
        b = np.zeros((0,), dtype=float)
        return a0, a, b
    a = 2.0 * F[1:].real
    b = -2.0 * F[1:].imag
    if (M % 2) == 0:
        a[-1] = F[-1].real
        b[-1] = 0.0
    return a0, a, b

def evaluate_on_grid_from_coeffs(X, Y, a0, a, b, radius=1.0):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    R = np.hypot(X, Y)
    TH = np.arctan2(Y, X)
    inside = R <= radius
    r = (R / radius)
    r_flat = r[inside].ravel()
    cos_th = np.cos(TH)[inside].ravel()
    sin_th = np.sin(TH)[inside].ravel()
    s = np.full_like(r_flat, a0, dtype=float)
    cos_n = cos_th.copy()
    sin_n = sin_th.copy()
    r_pow = r_flat.copy()
    for a_n, b_n in zip(a, b):
        s += r_pow * (a_n * cos_n + b_n * sin_n)
        cos_n, sin_n = cos_n * cos_th - sin_n * sin_th, sin_n * cos_th + cos_n * sin_th
        r_pow *= r_flat
    U = np.full_like(R, np.nan, dtype=float)
    U[inside] = s
    return U

def solve_disk_fft_on_grid(X, Y, boundary_func, radius=1.0, M=2048):
    a0, a, b = boundary_fourier_coeffs(boundary_func, M=M)
    return evaluate_on_grid_from_coeffs(X, Y, a0, a, b, radius=radius)

results_path = get_target_folder()
print(results_path)
data = pr.load_dict_from_hdf5(results_path / "full_data.h5")
config = data["config"]
c_coeffs = data["fields"]["c"]
basis = pr.ChebyshevBasis2D((config["basis_Nx"], config["basis_Ny"]))
c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))
config = load_config(results_path / "config.yml")
mask_data, boundary_data = domain_from_png(config)
ny, nx = mask_data.data.shape
# nx = 50
# ny = 50
x_vec = jnp.linspace(-1,1,nx)
y_vec = jnp.linspace(-1,1,ny)
x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
c_value = c_field.evaluate(x_grid, y_grid)
c_value = c_value.reshape(ny, nx)
r_grid = jnp.sqrt(x_grid**2 + y_grid**2)
mask = r_grid <= 1
# c_value = jnp.where(mask == 0, jnp.nan, c_value)
c_value = jnp.where(mask_data.data == 0, jnp.nan, c_value)

bc_eval = boundary_condition(boundary_data.coords[0], boundary_data.coords[1])
bc_eval = bc_eval.reshape(boundary_data.coords[0].shape)

f, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].contourf(x_grid, y_grid, c_value, levels=100)
ax[0].set_aspect("equal")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Laplace equation solution")
ax[1].scatter(boundary_data.coords[0], boundary_data.coords[1], c=bc_eval, s=1)
ax[1].set_aspect("equal")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Boundary points")
plt.savefig(results_path / "laplace_solution.png")
plt.close()
# solution = solve_disk_fft_on_grid(x_grid, y_grid, boundary_condition_theta, 1)


# f, ax = plt.subplots(1,2, figsize=(10,5))
# ax[0].contourf(x_grid, y_grid, c_value, levels=100)
# ax[0].set_aspect("equal")
# ax[0].set_xlabel("x")
# ax[0].set_ylabel("y")
# ax[0].set_title("Laplace equation solution")
# ax[1].contourf(x_grid, y_grid, solution, levels=100)
# ax[1].set_aspect("equal")
# ax[1].set_xlabel("x")
# ax[1].set_ylabel("y")
# ax[1].set_title("Analytical solution")
# plt.savefig("figures/laplace_solution.png")
# plt.close()

# boundary_points = jnp.linspace(0, 2 * jnp.pi, 100)
# boundary_values = boundary_condition_theta(boundary_points)

# boundary_x, boundary_y = jnp.cos(boundary_points), jnp.sin(boundary_points)

# c_field_boundary = c_field.evaluate(boundary_x, boundary_y)


# f, ax = plt.subplots(1,2, figsize=(10,5),sharey=True)
# ax[0].plot(boundary_points, boundary_values)
# ax[0].plot(boundary_points, c_field_boundary, "o")
# ax[0].axvline(x=jnp.pi/2, color='red', linestyle='--', alpha=0.7)
# ax[0].set_xlabel("theta")
# ax[0].set_ylabel("boundary value")
# ax[0].set_title("Boundary condition")
# ax[1].plot(boundary_points, c_field_boundary)
# ax[1].axvline(x=jnp.pi/2, color='red', linestyle='--', alpha=0.7)
# ax[1].set_xlabel("theta")
# ax[1].set_ylabel("c")
# ax[1].set_title("Boundary values")
# plt.savefig("figures/laplace_boundary.png")
# plt.close()