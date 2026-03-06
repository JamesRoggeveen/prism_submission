import jax
import equinox as eqx
import jax.numpy as jnp
from matplotlib.pyplot import imread
import prism as pr
from typing import Dict
import time
import yaml
import pathlib
from scipy.ndimage import gaussian_filter

class WaveConfig(pr.ProblemConfig):
    filter_frac: float
    residual_weights: Dict[str, float]
    c: float

class WaveProblem(pr.AbstractProblem):
    c_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
            "initial_condition": self.initial_condition_residual,
            "initial_velocity": self.initial_velocity_residual
        }
    
    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: WaveConfig) -> jax.Array:
        c_tt = self.c_field.derivative(*problem_data.coords, order=(0,0,2))
        c_xx = self.c_field.derivative(*problem_data.coords, order=(2,0,0))
        c_yy = self.c_field.derivative(*problem_data.coords, order=(0,2,0))
        return c_tt - config.c * (c_xx + c_yy)

    def boundary_residual(self, problem_data: pr.data.BoundaryData, config: WaveConfig) -> jax.Array:
        c_x = self.c_field.derivative(*problem_data.coords, order=(1,0,0))
        c_y = self.c_field.derivative(*problem_data.coords, order=(0,1,0))
        n_x, n_y = problem_data.normal_vector
        return c_x * n_x + c_y * n_y

    @eqx.filter_jit
    def initial_condition_residual(self, problem_data: pr.data.ReferenceData, config: WaveConfig) -> jax.Array:
        ic_col_x, ic_col_y, ic_col_t = problem_data.coords
        initial_data = problem_data.data
        return self.c_field.evaluate(ic_col_x, ic_col_y, ic_col_t) - initial_data

    @eqx.filter_jit
    def initial_velocity_residual(self, problem_data: pr.data.ReferenceData, config: WaveConfig) -> jax.Array:
        # We want the initial velocity (dc/dt) to be zero
        ic_col_x, ic_col_y, ic_col_t = problem_data.coords
        # Calculate the time derivative at the initial time points
        c_t = self.c_field.derivative(ic_col_x, ic_col_y, ic_col_t, order=(0, 0, 1))
        return c_t # The target is zero, so the residual is just c_t
    
    @eqx.filter_jit
    def loss_function(self, problem_data: pr.ProblemData, config: WaveConfig) -> jax.Array:
        return jnp.sum(self.total_residual(problem_data, config)**2)

    @eqx.filter_jit
    def total_residual(self, problem_data: pr.ProblemData, config: WaveConfig) -> jax.Array:
        equation_weight = jnp.sqrt(config.residual_weights["equation"])
        boundary_weight = jnp.sqrt(config.residual_weights["boundary"])
        ic_weight = jnp.sqrt(config.residual_weights["initial_condition"])
        equation_residual = self.equation_residual(problem_data["equation"], config)
        boundary_residual = self.boundary_residual(problem_data["boundary"], config)
        ic_residual = self.initial_condition_residual(problem_data["initial_condition"], config)
        iv_residual = self.initial_velocity_residual(problem_data["initial_condition"], config)
        n_equation = equation_residual.shape[0]
        n_boundary = boundary_residual.shape[0]
        n_ic = ic_residual.shape[0]
        return jnp.concatenate([equation_weight*equation_residual/jnp.sqrt(n_equation), boundary_weight*boundary_residual/jnp.sqrt(n_boundary), ic_weight*ic_residual/jnp.sqrt(n_ic), ic_weight*iv_residual/jnp.sqrt(n_ic)])

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

def sample_from_mask(key, x_grid, y_grid, data, ref, M=8000):
    mask = jnp.nonzero(ref)

    xi = x_grid[mask]
    yi = y_grid[mask]

    idx = jax.random.choice(key, len(xi), shape=(M,), replace=False)
    x_samples = jnp.array(xi[idx])
    y_samples = jnp.array(yi[idx])
    data_samples = jnp.array(data[mask][idx])

    return x_samples, y_samples, data_samples

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

def smooth_compact_pulse(x, y, center=(0.0, 0.1), radius=0.1, amplitude=1.0):
    x0, y0 = center
    r_squared = (x - x0)**2 + (y - y0)**2

    pulse = amplitude * jnp.exp(-radius * r_squared)
    return pulse

def initial_condition(mask_data, config):
    mask_x, mask_y = mask_data.coords
    radius = config.radius
    amplitude = config.amplitude
    center_x = config.center_x
    center_y = config.center_y
    pulse = smooth_compact_pulse(mask_x, mask_y, center=(center_x, center_y), radius=radius, amplitude=amplitude)
    return pulse

def sample_data(mask_data, boundary_data, config):
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    mask_x, mask_y = mask_data.coords
    eq_col_x, eq_col_y, _ = sample_from_mask(subkey, mask_x, mask_y, mask_data.data, mask_data.data, config.n_pde)
    t_vec = jnp.linspace(-1,1,config.n_t)

    final_col_x = jnp.tile(eq_col_x, config.n_t)
    final_col_y = jnp.tile(eq_col_y, config.n_t)
    final_col_t = jnp.repeat(t_vec, config.n_pde)

    collocation_data = pr.data.CollocationPoints(coords=(final_col_x, final_col_y, final_col_t))

    boundary_x, boundary_y = boundary_data.coords
    boundary_nx, boundary_ny = boundary_data.normal_vector

    n_boundary = config.n_boundary
    key, subkey = jax.random.split(key)
    boundary_indicies = jax.random.choice(subkey, boundary_x.shape[0], (n_boundary,), replace=False)
    sampled_boundary_x = jnp.tile(boundary_x[boundary_indicies], config.n_t)
    sampled_boundary_y = jnp.tile(boundary_y[boundary_indicies], config.n_t)
    sampled_boundary_nx = jnp.tile(boundary_nx[boundary_indicies], config.n_t)
    sampled_boundary_ny = jnp.tile(boundary_ny[boundary_indicies], config.n_t)
    sampled_boundary_t = jnp.repeat(t_vec, n_boundary)

    sampled_boundary_data = pr.data.BoundaryData(coords=(sampled_boundary_x, sampled_boundary_y, sampled_boundary_t), normal_vector=(sampled_boundary_nx, sampled_boundary_ny))

    key, subkey = jax.random.split(key)
    ic_values = initial_condition(mask_data, config)
    ic_col_x, ic_col_y, ic_data = sample_from_mask(subkey, mask_x, mask_y, ic_values, mask_data.data, config.n_ic)
    ic_data = pr.data.ReferenceData(coords=(ic_col_x, ic_col_y, -1*jnp.ones(config.n_ic)), data=ic_data)

    return pr.data.ProblemData(equation=collocation_data, boundary=sampled_boundary_data, initial_condition=ic_data)

def initialize_field(mask_data, config):
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config.basis_Nx, config.basis_Ny, config.basis_Nt))
    # initial_data = initial_condition(mask_data, config)
    # mask_x, mask_y = mask_data.coords
    # mask_x = mask_x[::2,::2].reshape(-1)
    # mask_y = mask_y[::2,::2].reshape(-1)
    # initial_data = initial_data[::2,::2].reshape(-1)
    # t_samples = int(config.n_t/2)
    # t_grid = jnp.linspace(-1,1,t_samples)
    # col_t = jnp.repeat(t_grid, mask_x.shape[0])
    # col_x = jnp.tile(mask_x, t_samples)
    # col_y = jnp.tile(mask_y, t_samples)
    # initial_data = jnp.tile(initial_data, t_samples)
    # basis_field = pr.fit_basis_field_from_data(basis, (col_x, col_y, col_t), initial_data)
    basis_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    # initial_coeffs = jnp.load(f"{config.data_dir}initial_field_coeffs.npy")
    # basis_field = pr.BasisField(basis, pr.Coeffs(initial_coeffs))
    return basis_field

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

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    mask_data, boundary_data = domain_from_png(config)
    problem_data = sample_data(mask_data, boundary_data, config)
    print("Begin field initialization")
    start_time = time.time()
    c_field = initialize_field(mask_data, config)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


    problem = WaveProblem(c_field)
    if config.solver == "Adam":
        solver = pr.AdamSolver()
    elif config.solver == "LevenbergMarquardt":
        solver = pr.LevenbergMarquardtSolver()
    else:
        raise ValueError(f"Solver {config.solver} not supported")

    timestamp = time.strftime("%m%d%H%M")
    script_name = f"{timestamp}_wave"
    print(f"Script name: {script_name}")
    config.script_name = script_name

    start_time = time.time()
    problem_config = WaveConfig.from_config(config)
    optimized_diffusion = solver.solve(problem, problem_data, config, problem_config)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    save_results(config, optimized_diffusion)