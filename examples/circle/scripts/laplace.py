import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import argparse
from scipy.ndimage import gaussian_filter
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt

class LaplaceConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float

class LaplaceProblem(pr.AbstractProblem):
    c_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: LaplaceConfig) -> jax.Array:
        c_xx = self.c_field.derivative(*problem_data.coords, order=(2,0))
        c_yy = self.c_field.derivative(*problem_data.coords, order=(0,2))
        return c_xx + c_yy
    
    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.ReferenceData, config: LaplaceConfig) -> jax.Array:
        c = self.c_field.evaluate(*problem_data.coords)
        return c - problem_data.data

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config["solver_kwargs"] = {key: float(value) for key, value in config["solver_kwargs"].items()}
    if config["verbose"]:
        config["solver_kwargs"]["verbose"] = frozenset({"loss", "step_size"})
    seed = int(time.time())
    config["seed"] = seed
    return pr.SystemConfig(**config)

def save_config(config, result_path):
    with open(result_path / "config.yml", "w") as f:
        yaml.dump(config_to_dict(config), f)

def config_to_dict(self):
    output_dict = self.data.copy()
    solver_kwargs_copy = {k: v for k, v in output_dict["solver_kwargs"].items() if k != "verbose"}
    output_dict["solver_kwargs"] = solver_kwargs_copy
    return output_dict

def boundary_condition(x,y):
    theta = jnp.arctan2(y, x)
    return boundary_condition_theta(theta)

# def boundary_condition_theta(theta):
#     # A sharp Gaussian peak centered at theta = pi/2
#     peak = 2.0 * jnp.exp(-((theta - jnp.pi / 2)**2) / 0.01)
    
#     # A high-frequency oscillation
#     oscillation = 0.5 * jnp.sin(20 * theta)
    
#     # A smooth, low-frequency background component
#     background = 0.5 * jnp.cos(theta)
    
#     return peak + oscillation + background

def boundary_condition_theta(theta):
    return 2 * jnp.sin(theta) + jnp.cos(3 * theta)

def boundary_from_mask(binary_mask, x_grid, y_grid,config):
    padded_mask = jnp.pad(binary_mask, 1, mode='constant', constant_values=False)

    boundary_mask = (
        (padded_mask[1:-1, 1:-1] != padded_mask[0:-2, 1:-1]) |  # up
        (padded_mask[1:-1, 1:-1] != padded_mask[2:, 1:-1]) |    # down
        (padded_mask[1:-1, 1:-1] != padded_mask[1:-1, 0:-2]) |  # left
        (padded_mask[1:-1, 1:-1] != padded_mask[1:-1, 2:])      # right
    )

    boundary_mask = boundary_mask & binary_mask

    boundary_indices = jnp.nonzero(boundary_mask)
    boundary_x = x_grid[boundary_indices]
    boundary_y = y_grid[boundary_indices]

    # Smooth mask to improve normal vector calculation
    smoothed_mask = gaussian_filter(binary_mask.astype(float), sigma=config.sigma)

    grad_y, grad_x = jnp.gradient(smoothed_mask.astype(float))

    # 2. Extract the gradient vectors at the boundary points
    nx = -grad_x[boundary_indices]
    ny = -grad_y[boundary_indices]

    # 3. Normalize the gradient vectors to get unit normals
    magnitude = jnp.sqrt(nx**2 + ny**2)
    # Avoid division by zero for any potential zero-magnitude vectors
    magnitude = jnp.where(magnitude == 0, 1, magnitude)
    nx_norm = nx / magnitude
    ny_norm = ny / magnitude

    return boundary_x, boundary_y, nx_norm, ny_norm

def domain_from_png(config):
    img = imread(config.domain_path)[:,:,2]
    mask = img > 0
    x_vec = jnp.linspace(-1,1,img.shape[1])
    y_vec = jnp.linspace(-1,1,img.shape[0])
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)

    mask_data = pr.data.ReferenceData(coords=(x_grid, y_grid), data=mask)

    boundary_x, boundary_y, nx_norm, ny_norm = boundary_from_mask(mask, x_grid, y_grid, config)

    boundary_data = pr.data.BoundaryData(coords=(boundary_x, boundary_y), normal_vector=(nx_norm, ny_norm))

    return mask_data, boundary_data

def sample_from_mask(key, x_grid, y_grid, data, ref, M=8000):
    mask = jnp.nonzero(ref)

    xi = x_grid[mask]
    yi = y_grid[mask]

    idx = jax.random.choice(key, len(xi), shape=(M,), replace=False)
    x_samples = jnp.array(xi[idx])
    y_samples = jnp.array(yi[idx])
    data_samples = jnp.array(data[mask][idx])

    return x_samples, y_samples, data_samples

def sample_points_from_png(mask_data, boundary_data, config):
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    mask_x, mask_y = mask_data.coords
    eq_col_x, eq_col_y, _ = sample_from_mask(subkey, mask_x, mask_y, mask_data.data, mask_data.data, config.n_pde)

    collocation_data = pr.data.CollocationPoints(coords=(eq_col_x, eq_col_y))

    boundary_x, boundary_y = boundary_data.coords
    boundary_nx, boundary_ny = boundary_data.normal_vector

    boundary_data = pr.data.ReferenceData(coords=(boundary_x, boundary_y), data=boundary_condition(boundary_x, boundary_y))

    return pr.data.ProblemData(equation=collocation_data, boundary=boundary_data)

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, (n_pde,), minval=0, maxval=1)
    r = jnp.sqrt(u)
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey, (n_pde,), minval=0, maxval=2*jnp.pi)

    x = r*jnp.cos(theta)
    y = r*jnp.sin(theta)
    key, subkey = jax.random.split(key)
    bc_theta = jax.random.uniform(subkey, (n_bc,), minval=0, maxval=2*jnp.pi)

    bc_x = jnp.cos(bc_theta)
    bc_y = jnp.sin(bc_theta)

    boundary_data = boundary_condition(bc_x, bc_y)
    collocation_data = pr.data.CollocationPoints(coords = (x, y))
    boundary_data = pr.data.ReferenceData(coords = (bc_x, bc_y), data = boundary_data)

    return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data)

def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    c_coeffs = optimized_problem.c_field.coeffs.value
    field_dict = {
        "c": c_coeffs
    }
    jnp.save(result_path / "c_coeffs.npy", c_coeffs)
    full_data = {"fields": field_dict, "config": config_to_dict(config)}
    pr.save_dict_to_hdf5(result_path / "full_data.h5", full_data)
    save_config(config, result_path)

def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int)
    args = parser.parse_args()
    timestamp = time.strftime("%m%d%H%M")
    config["timestamp"] = timestamp
    if args.N is not None:
        config["basis_Nx"] = args.N
        config["basis_Ny"] = args.N
        config["script_name"] = f"{args.N}_laplace"
    else:
        config["script_name"] = f"{timestamp}_laplace"
    return config

if __name__ == "__main__":
    config = load_config("configs/laplace.yml")
    config = parse_args(config)
    # problem_data = sample_points(config)
    mask_data, boundary_data = domain_from_png(config)
    problem_data = sample_points_from_png(mask_data, boundary_data, config)
    f, ax = plt.subplots(1,1, figsize=(5,5))
    ax.scatter(problem_data["boundary"].coords[0], problem_data["boundary"].coords[1], c=problem_data["boundary"].data, s=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Boundary points")
    plt.savefig("figures/laplace_boundary.png")
    plt.close()
    basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    # c_field = pr.BasisField(basis, pr.fields.PreconditionedFactorizedCoeffs.make_zero(basis.degs, config.factor))
    c_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    # c_field = pr.BasisField(basis, pr.FactorizedCoeffs.make_orthonormal(jax.random.PRNGKey(config.seed), basis.degs, config.factor))
    # c_field = pr.BasisField(basis, pr.fields.SVDCoeffs.initialize(jax.random.PRNGKey(config.seed), basis.degs, config.factor))
    problem = LaplaceProblem(c_field)
    solver = pr.get_solver(config)
    
    print(f"Script name: {config.script_name}")

    start_time = time.time()
    problem_config = LaplaceConfig.from_config(config)
    optimized_problem, log_data = solver.solve(problem, problem_data, config, LaplaceConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    config["time_taken"] = end_time - start_time
    save_results(config, optimized_problem)

    total_loss = [item["total_loss"] for item in log_data]
    plt.semilogy(total_loss)
    plt.savefig("scripts/laplace_loss.png")
    plt.close()