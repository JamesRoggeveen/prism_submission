import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

class LaplaceConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    regularization_strength: float

class LaplaceProblem(pr.AbstractProblem):
    u_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: LaplaceConfig) -> jax.Array:
        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0))
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2))
        return u_xx + u_yy
    
    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.ReferenceData, config: LaplaceConfig) -> jax.Array:
        u = self.u_field.evaluate(*problem_data.coords)
        return u - problem_data.data

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

    # Check for NaN values in data_samples
    nan_count = jnp.sum(jnp.isnan(data_samples))
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in data_samples")

    return x_samples, y_samples, data_samples

def outer_boundary_condition(x,y):
    theta = jnp.arctan2(y,x)
    return 2*jnp.sin(theta) + jnp.cos(3*theta)

def inner_boundary_condition(x,y):
    return -2*jnp.ones_like(x)

def sample_points(config):
    def r1(theta):
        return 0.7+0.3*jnp.cos(2*theta)

    circle_x, circle_y, circle_r = .3, .1, 0.15
    # circle_x, circle_y, circle_r = -.014,-.145, 0.15
    x_vec = jnp.linspace(-1,1,500)
    y_vec = jnp.linspace(-1,1,500)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    y_squashed = 0.5*y_grid
    theta_grid = jnp.arctan2(y_squashed, x_grid)
    r_grid = jnp.sqrt(x_grid**2 + (y_squashed)**2)
    mask1 = r_grid <= r1(theta_grid)
    mask2 = (x_grid - circle_x)**2 + (y_grid-circle_y)**2 >= circle_r**2
    mask = mask1 & mask2
    grid_data = pr.data.ReferenceData(coords = (x_grid, y_grid), data = mask)
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x_samples, y_samples, _ = sample_from_mask(subkey, x_grid, y_grid, mask, mask, config.n_pde)
    collocation_data = pr.data.CollocationPoints(coords = (x_samples, y_samples))

    key, subkey = jax.random.split(key)
    reference = np.loadtxt("data/laplace_grid.csv", delimiter=",", skiprows=9)
    ref_x, ref_y, ref_data = reference[:,0], reference[:,1], reference[:,2]
    data_has_nan = True
    while data_has_nan:
        sample_x, sample_y, sample_data = sample_from_mask(subkey, ref_x, ref_y, ref_data, mask.reshape(-1), config.n_bc)
        data_has_nan = jnp.sum(jnp.isnan(sample_data)) > 0
        key, subkey = jax.random.split(key)

    # key, subkey = jax.random.split(key)
    # noise = jax.random.normal(subkey, (config.n_bc,))*0.1 + 1
    # sample_data = sample_data * noise
    boundary_data = pr.data.ReferenceData(coords = (sample_x, sample_y), data = sample_data)
    
    return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data, grid = grid_data)

def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    c_coeffs = optimized_problem.u_field.coeffs.value
    field_dict = {
        "u": c_coeffs
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
        config["script_name"] = f"{timestamp}_laplace_inverse"

    config["results_dir"] = "results"
    return config

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    config = parse_args(config)
    results_path_main = pathlib.Path(config.results_dir) / config.script_name
    results_path_main.mkdir(parents=True, exist_ok=True)

    start_log = np.log10(10)   # log10(10) = 1
    stop_log = np.log10(10000) # log10(10000) = 4

    # Generate 50 points logarithmically spaced between 10 and 10000
    # np.logspace creates numbers from 10**start_log to 10**stop_log
    log_spaced_floats = np.logspace(start_log, stop_log, num=20)

    # Convert the floats to integers and get the unique values
    log_spaced_integers = np.unique(log_spaced_floats.astype(int))
    repeats = 15

    final_list = log_spaced_integers.tolist()
    l2_errors = np.zeros((len(final_list), repeats))
    Linf_errors = np.zeros((len(final_list), repeats))
    times = np.zeros((len(final_list), repeats))
    master_script_name = config.script_name

    for j in range(repeats):
        for i,n_bc in enumerate(final_list):
            print(f"Solving for n_bc: {n_bc} trial {j}", flush=True)
            timestamp = time.strftime("%m%d%H%M")
            config["timestamp"] = timestamp
            results_path = results_path_main / f"n_bc_{n_bc}/trial_{j}"
            results_path.mkdir(parents=True, exist_ok=True)
            config["n_bc"] = n_bc
            config["script_name"] = f"{master_script_name}/n_bc_{n_bc}/trial_{j}"
            problem_data = sample_points(config)
            f, ax = plt.subplots(1,1,figsize=(5,5))
            ax.contourf(problem_data["grid"].coords[0], problem_data["grid"].coords[1], problem_data["grid"].data, levels=100)
            ax.scatter(problem_data["equation"].coords[0], problem_data["equation"].coords[1], c="red", s=1)
            ax.scatter(problem_data["boundary"].coords[0], problem_data["boundary"].coords[1], c="blue", s=1)
            plt.savefig(results_path / "laplace_grid.png")
            plt.close()

            sample_x, sample_y = problem_data["boundary"].coords
            sample_data = problem_data["boundary"].data
            sample_data_stacked = jnp.stack([sample_x, sample_y, sample_data], axis=1)
            jnp.save(results_path / "sample_data.npy", sample_data_stacked)

            reference = np.loadtxt("data/laplace_grid.csv", delimiter=",", skiprows=9)
            ref_x, ref_y, ref_data = reference[:,0], reference[:,1], reference[:,2]
            grid_x, grid_y = problem_data["grid"].coords
            mask = problem_data["grid"].data
            ref_x, ref_y, ref_data = ref_x.reshape(grid_x.shape), ref_y.reshape(grid_y.shape), ref_data.reshape(grid_x.shape)
            gridded_data_mask = jnp.where(mask == 0, jnp.nan, ref_data)
            ref_data = ref_data.reshape(-1)


            grid_data = problem_data["grid"]
            grid_x, grid_y = grid_data.coords
            mask = grid_data.data
            dh = grid_y[1,0] - grid_y[0,0]
            
            basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
            u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
            problem = LaplaceProblem(u_field)
            solver = pr.get_solver(config)
            start_time = time.time()
            problem_config = LaplaceConfig.from_config(config)
            optimized_problem, log_data = solver.solve(problem, problem_data, config, LaplaceConfig.from_config(config))
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds", flush=True)


            u_eval = optimized_problem.u_field.evaluate(grid_x, grid_y) 
            u_eval = u_eval.reshape(grid_x.shape)

            ref_data = ref_data.reshape(grid_x.shape)
            error = jnp.abs(u_eval - ref_data)
            error = jnp.where(mask == 0, jnp.nan, error)
            u_eval = jnp.where(mask == 0, jnp.nan, u_eval)
            ref = gridded_data_mask

            l2_error = jnp.sqrt(jnp.nansum(jnp.power(u_eval - ref_data, 2))*dh**2)
            l2_errors[i,j] = l2_error
            print(f"L2 error: {l2_error}", flush=True)
            Linf_error = jnp.nanmax(jnp.abs(u_eval - ref_data))
            print(f"Linf error: {Linf_error}", flush=True)
            Linf_errors[i,j] = Linf_error
            config["l2_error"] = l2_error
            config["Linf_error"] = Linf_error
            save_results(config, optimized_problem)
            times[i,j] = end_time - start_time

            cmap = "jet"
            f, ax = plt.subplots(1,3,figsize=(15,5))
            im0 = ax[0].contourf(grid_x, grid_y, ref, levels=100, cmap=cmap)
            ax[0].set_aspect("equal")
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            ax[0].set_title("Reference Solution")
            im1 = ax[1].contourf(grid_x, grid_y, u_eval, levels=100, cmap=cmap)
            ax[1].set_aspect("equal")
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("y")
            ax[1].set_title("Optimized Solution")
            im2 = ax[2].contourf(grid_x, grid_y, error, levels=100, cmap=cmap)
            ax[2].set_aspect("equal")
            ax[2].set_xlabel("x")
            ax[2].set_ylabel("y")
            ax[2].set_title("Relative Error")
            f.colorbar(im0,ax=ax[0])
            f.colorbar(im1,ax=ax[1])
            f.colorbar(im2,ax=ax[2])
            plt.tight_layout()
            plt.savefig("figures/laplace_inverse_solution.png")
            plt.savefig(results_path / "laplace_inverse_solution.png")
            plt.close()

            print(f"Data saved to {results_path}", flush=True)
            print(f"Trial {j} complete", flush=True)
        
    error_data = np.vstack((final_list.reshape(-1,1), l2_errors, Linf_errors, times))
    np.save(results_path_main / "errors.npy", error_data)