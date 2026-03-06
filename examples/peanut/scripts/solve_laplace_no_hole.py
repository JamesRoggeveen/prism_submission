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
            # "equation": self.equation_residual,
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

    return x_samples, y_samples, data_samples

def outer_boundary_condition(x,y):
    theta = jnp.arctan2(y,x)
    return 2*jnp.sin(theta) + jnp.cos(3*theta)

def inner_boundary_condition(x,y):
    return -2*jnp.ones_like(x)

def r1(theta):
    return (0.7+0.3*jnp.cos(2*theta))

def sample_points(config):
    circle_x, circle_y, circle_r = .3, .1, 0.15
    # circle_x, circle_y, circle_r = -.014,-.145, 0.15
    x_vec = jnp.linspace(-1,1,500)
    y_vec = jnp.linspace(-1,1,500)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    y_squashed = 0.5*y_grid
    theta_grid = jnp.arctan2(y_squashed, x_grid)
    r_grid = jnp.sqrt(x_grid**2 + (y_squashed)**2)
    mask1 = r_grid <= r1(theta_grid)
    # mask2 = (x_grid - circle_x)**2 + (y_grid-circle_y)**2 >= circle_r**2
    mask = mask1
    grid_data = pr.data.ReferenceData(coords = (x_grid, y_grid), data = mask)
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x_samples, y_samples, _ = sample_from_mask(subkey, x_grid, y_grid, mask, mask, config.n_pde)

    key, subkey = jax.random.split(key)
    # theta_samples = jax.random.uniform(subkey, (config.n_bc,), minval=0, maxval=2*jnp.pi)
    theta_samples = jnp.linspace(0, 2*jnp.pi, config.n_bc)
    r_samples = r1(theta_samples)
    x_bc_1 = r_samples*jnp.cos(theta_samples)
    y_bc_1 = 2*r_samples*jnp.sin(theta_samples)
    boundary_data_1 = outer_boundary_condition(x_bc_1, y_bc_1)
    # x_bc_2 = circle_r*jnp.cos(theta_samples)+circle_x
    # y_bc_2 = circle_r*jnp.sin(theta_samples)+circle_y
    # boundary_data_2 = inner_boundary_condition(x_bc_2, y_bc_2)
    # x_bc = jnp.concatenate([x_bc_1, x_bc_2])
    # y_bc = jnp.concatenate([y_bc_1, y_bc_2])
    # boundary_data = jnp.concatenate([boundary_data_1, boundary_data_2])
    x_bc = x_bc_1
    y_bc = y_bc_1
    boundary_data = boundary_data_1

    collocation_data = pr.data.CollocationPoints(coords = (x_samples, y_samples))
    boundary_data = pr.data.ReferenceData(coords = (x_bc, y_bc), data = boundary_data)
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
        config["script_name"] = f"{timestamp}_laplace"

    config["results_dir"] = f"results/{timestamp}_laplace_no_hole"
    return config

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    config = parse_args(config)
    problem_data = sample_points(config)

    grid_x, grid_y = problem_data["grid"].coords

    basis = pr.fields.HarmonicBasis2D(config.basis_Nx)

    # x_bc, y_bc = problem_data["boundary"].coords
    # boundary_values = problem_data["boundary"].data
    # basis_matrix_bc = basis(x_bc, y_bc, order=(0,0))

    # initial_coeffs_flat, residuals, rank, s = jnp.linalg.lstsq(
    #     basis_matrix_bc, boundary_values, rcond=1e-12
    # )

    # coeff_shape = tuple(d + 1 for d in basis.degs)
    # initial_coeffs_grid = initial_coeffs_flat.reshape(-1)

    # print("Initial fit complete.")
    # u_field = pr.fit_basis_field_from_data(basis, problem_data["boundary"].coords, problem_data["boundary"].data, precondition=False)
    # basis = pr.fields.HarmonicBasis2D(config.basis_Nx)
    # u_field = pr.BasisField(basis, pr.fields.ExponentialPreconditionedChebyshevCoeffs.make_zero(basis.degs))
    u_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs)) 

    # u_field = pr.BasisField(basis, pr.Coeffs(initial_coeffs_grid))

    boundary_theta = jnp.linspace(0, 2*jnp.pi, 1000)
    boudnary_r = r1(boundary_theta)
    boundary_x = boudnary_r*jnp.cos(boundary_theta)
    boundary_y = 2*boudnary_r*jnp.sin(boundary_theta)
    boundary_data = outer_boundary_condition(boundary_x, boundary_y)
    boundary_eval = u_field.evaluate(boundary_x, boundary_y)
    error = jnp.max(jnp.abs(boundary_eval - boundary_data))
    print(f"Boundary error: {error}")
    problem = LaplaceProblem(u_field)
    solver = pr.get_solver(config)
    

    start_time = time.time()
    problem_config = LaplaceConfig.from_config(config)
    optimized_problem, log_data = solver.solve(problem, problem_data, config, LaplaceConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    config["time_taken"] = end_time - start_time
    results_path = pathlib.Path(config.results_dir) / config.script_name
    save_results(config, optimized_problem)

    # f, ax = plt.subplots(2,2,figsize=(12,12))
    # ax = ax.flatten()
    # for key, value in log_data.items():
    #     if "." in key:
    #         key_pre, key_suf = key.split(".")
    #         if key_pre == "unweighted_losses":
    #             ax[0].semilogy(value,label=key_suf)
    #         elif key_pre == "weights":
    #             ax[1].semilogy(value,label=key_suf)
    #         elif key_pre == "grad_mags":
    #             ax[2].semilogy(value,label=key_suf)
    #     else:
    #         ax[3].semilogy(value,label=key)
    # ax[0].set_title("Unweighted Losses")
    # ax[1].set_title("Weights")
    # ax[2].set_title("Grad Magnitudes")
    # ax[3].set_title("Total Loss")
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # plt.tight_layout()
    # plt.savefig("figures/laplace_losses_no_hole.png")
    # plt.savefig(results_path / "losses_no_hole.png")
    # plt.close()

    boundary_theta = jnp.linspace(0, 2*jnp.pi, 1000)
    boudnary_r = r1(boundary_theta)
    boundary_x = boudnary_r*jnp.cos(boundary_theta)
    boundary_y = 2*boudnary_r*jnp.sin(boundary_theta)
    boundary_data = outer_boundary_condition(boundary_x, boundary_y)
    boundary_eval = optimized_problem.u_field.evaluate(boundary_x, boundary_y)
    error = jnp.max(jnp.abs(boundary_eval - boundary_data))
    print(f"Boundary error: {error}")
    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(boundary_x, boundary_y, c=boundary_data, cmap="jet")
    ax.scatter(boundary_x, boundary_y, c=boundary_eval, cmap="jet")
    plt.savefig("figures/laplace_boundary_no_hole.png")
    plt.close()

    save_data = {"time_taken": end_time - start_time, "boundary_error": error}
    with open(results_path / "save_data.yml", "w") as f:
        yaml.dump(save_data, f)

    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(boundary_theta, boundary_data, label="Boundary Data")
    ax.plot(boundary_theta, boundary_eval, label="Boundary Evaluation")
    ax.legend()
    plt.savefig("figures/laplace_boundary_no_hole_plot.png")
    plt.savefig(results_path / "laplace_boundary_no_hole_plot.png")
    plt.close()

    u_eval = optimized_problem.u_field.evaluate(grid_x, grid_y) 
    u_eval = u_eval.reshape(grid_x.shape)
    mask = problem_data["grid"].data

    u_eval = jnp.where(mask == 0, jnp.nan, u_eval)

    cmap = "jet"
    
    
    f, ax = plt.subplots(1,1,figsize=(5,5))
    im1 = ax.contourf(grid_x, grid_y, u_eval, levels=100, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimized Solution")
    f.colorbar(im1,ax=ax)
    plt.tight_layout()
    plt.savefig("figures/laplace_solution_no_hole.png")
    plt.savefig(results_path / "laplace_solution_no_hole.png")
    plt.close()
    