import jax
import jax.numpy as jnp
import equinox as eqx
import time
import yaml
import pathlib
from typing import Dict, Callable
import argparse
import matplotlib.pyplot as plt
from scipy.io import loadmat
from prism import (
    CosineChebyshevBasis2D,
    BasisField,
    fit_basis_field_from_data,
    AbstractProblem,
    ProblemConfig,
    SystemConfig,
    Coeffs,
    save_dict_to_hdf5
)
import prism as pr

# jax.config.update("jax_enable_x64", True)

class ACConfig(ProblemConfig):
    t_scale: float
    filter_frac: float
    residual_weights: Dict[str, float]
    causal_num_chunks: int
    causal_tol: float
    regularization_strength: float

class ACProblem(AbstractProblem):
    u_field: BasisField

    def get_residual_functions(self) -> Dict[str, Callable]:
        return {
            "equation": self.equation_residual,
            "initial_condition": self.initial_condition_residual
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: ProblemConfig) -> jax.Array:
        t_scale = config.t_scale
        filter_frac = config.filter_frac

        eq_col_x, eq_col_y = problem_data.coords

        u_t = self.u_field.derivative(eq_col_x, eq_col_y, order=(0,1))
        u_xx = self.u_field.derivative(eq_col_x, eq_col_y, order=(2,0))
        u_filter = self.u_field.evaluate(eq_col_x, eq_col_y, filter_frac=filter_frac)
        res = 1/t_scale*u_t -0.0001*u_xx + 5*u_filter**3 - 5*u_filter
        return res

    @eqx.filter_jit
    def initial_condition_residual(self, problem_data: pr.data.ReferenceData, config: ProblemConfig) -> jax.Array:
        ic_col_x, ic_col_y = problem_data.coords
        initial_data = problem_data.data
        u_ref = initial_data
        u_initial = self.u_field.evaluate(ic_col_x, ic_col_y)
        return u_initial - u_ref

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config["solver_kwargs"] = {key: float(value) for key, value in config["solver_kwargs"].items()}
    if config["verbose"]:
        config["solver_kwargs"]["verbose"] = frozenset({"loss", "step_size"})
    seed = int(time.time())
    config["seed"] = seed
    return SystemConfig(**config)

def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int)
    args = parser.parse_args()
    timestamp = time.strftime("%m%d%H%M%S")
    config["timestamp"] = timestamp
    if args.N is not None:
        config["basis_Nx"] = args.N
        config["basis_Nt"] = args.N
        script_name = f"{config.basis_Nx}_N_solve_ac"
    else:
        script_name = f"{timestamp}_solve_ac"
    print(f"Script name: {script_name}")
    config.script_name = script_name
    return config

def save_config(config, result_path):
    with open(result_path / "config.yml", "w") as f:
        yaml.dump(config_to_dict(config), f)

def config_to_dict(self):
    output_dict = self.data.copy()
    solver_kwargs_copy = {k: v for k, v in output_dict["solver_kwargs"].items() if k != "verbose"}
    output_dict["solver_kwargs"] = solver_kwargs_copy
    return output_dict

def initial_condition_function(x, t, config: SystemConfig):
    initial_u = x**2 * jnp.cos(jnp.pi*x)
    return initial_u

def initialize_data(config: SystemConfig):
    x_vec = jnp.linspace(-1,1,config.nx)
    t_vec = jnp.linspace(-1,1,config.nt)
    x_grid, t_grid = jnp.meshgrid(x_vec, t_vec)
    print(x_grid.shape)
    grid_data = pr.data.CollocationPoints(coords=(x_grid, t_grid))

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    eq_col_x, eq_col_y = jax.random.uniform(subkey, (2,config.n_pde), minval=-1, maxval=1)
    # Sort the collocation points by time (eq_col_y)
    # eq_col_x, eq_col_y = x_grid.reshape(-1), t_grid.reshape(-1)
    sort_indices = jnp.argsort(eq_col_y)
    eq_col_x = eq_col_x[sort_indices]
    eq_col_y = eq_col_y[sort_indices]
    collocation_data = pr.data.CollocationPoints(coords=(eq_col_x.T, eq_col_y.T))

    key, subkey = jax.random.split(key)
    n_ic = config.n_ic
    ic_col_x, ic_col_y = jax.random.uniform(subkey, (n_ic,), minval=-1, maxval=1), -1*jnp.ones(n_ic)
    initial_u = initial_condition_function(ic_col_x, ic_col_y, config)
    ic_data = pr.data.ReferenceData(coords=(ic_col_x, ic_col_y), data=initial_u)

    problem_data = pr.data.ProblemData(equation=collocation_data, initial_condition=ic_data, grid=grid_data)
    return problem_data

def save_results(config, optimized_ac):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    u_coeffs = optimized_ac.u_field.coeffs.value
    field_dict = {
        "u": u_coeffs
    }
    full_data = {"fields": field_dict, "config": config_to_dict(config)}
    save_dict_to_hdf5(result_path / "full_data.h5", full_data)
    save_config(config, result_path)

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    config = parse_args(config)
    problem_data = initialize_data(config)

    grid_data = problem_data.data["grid"]
    initialize_u = initial_condition_function(*grid_data.coords, config)

    
    if config.fit_initial:
        basis = pr.CosineLegendreBasis2D((config.basis_Nx, config.basis_Nt))
        u_field = fit_basis_field_from_data(basis, grid_data.coords, initialize_u, precondition=config.precondition)
    else:
        if config.precondition:
            basis = pr.CosineChebyshevBasis2D((config.basis_Nx, config.basis_Nt))
            coeffs = jax.random.normal(jax.random.PRNGKey(config.seed), ((basis.degs[0]+1)*(basis.degs[1]+1),))
            u_field = BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_coeffs(basis.degs,coeffs))
        else:
            basis = CosineChebyshevBasis2D((config.basis_Nx, config.basis_Nt))
            coeffs = jax.random.normal(jax.random.PRNGKey(config.seed), ((basis.degs[0]+1)*(basis.degs[1]+1),))
            u_field = BasisField(basis, pr.fields.Coeffs(coeffs))
    initial_u = u_field.evaluate(*grid_data.coords)

    # u_field = BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    problem = ACProblem(u_field)

    solver = pr.get_solver(config)

    start_time = time.time()
    problem_config = ACConfig.from_config(config)
    optimized_ac, log_data = solver.solve(problem, problem_data, config, problem_config)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    config["time_taken"] = end_time - start_time
    save_results(config, optimized_ac)
    if log_data != {}:
        total_loss = log_data["total_loss"]
    else:
        total_loss = jnp.zeros(config.max_steps)
    target_data = loadmat(config.data_path)["usol"]
    u_eval = optimized_ac.u_field.evaluate(*grid_data.coords)
    u_eval = u_eval.reshape(config.nt, config.nx)
    initial_u = initial_u.reshape(config.nt, config.nx)

    error = jnp.linalg.norm(u_eval - target_data)/jnp.linalg.norm(target_data)
    print(f"L2 relative error: {error}")
    config["final_relative_error"] = error

    save_fig_path = pathlib.Path("figures/")
    save_fig_path.mkdir(parents=True, exist_ok=True)
    fig_name = f"{config.timestamp}_ac_solution.png"
    f, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].set_title("Target Solution")
    ax[1].set_title("Optimized Solution")
    solver_type = config.get("solver_type", "")
    optimizer_name = config.get("optimizer_name", "")
    loss_strategy = config.get("loss_strategy", "")
    solver = f"{solver_type}_{optimizer_name}_{loss_strategy}"
    ax[2].set_title(f"{solver} Loss")
    im0 = ax[0].contourf(*grid_data.coords, target_data, cmap="jet", vmin = -1, vmax = 1, levels=100)
    im1 = ax[1].contourf(*grid_data.coords, u_eval, cmap="jet", vmin = -1, vmax = 1, levels=100)
    im2 = ax[2].semilogy(total_loss)
    f.colorbar(im0,ax=ax[0])
    f.colorbar(im1,ax=ax[1])
    plt.tight_layout()
    plt.savefig(save_fig_path / fig_name)
    plt.close()
    
    coeffs = u_field.coeffs.value.reshape(config.basis_Nx+1, config.basis_Nt+1)
    vmax = jnp.max(jnp.abs(coeffs))
    plt.imshow(coeffs,cmap="bwr",vmin=-vmax,vmax=vmax)
    plt.colorbar()
    plt.savefig(save_fig_path / "coeffs.png")
    plt.close()