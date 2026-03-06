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

class ReactionDiffusionConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    alpha: float

class ReactionDiffusionProblem(pr.AbstractProblem):
    u_field: pr.BasisField
    v_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
            "initial": self.initial_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: ReactionDiffusionConfig) -> jax.Array:
        c_xx = self.c_field.derivative(*problem_data.coords, order=(2,0,0))
        c_yy = self.c_field.derivative(*problem_data.coords, order=(0,2,0))
        c_t = self.c_field.derivative(*problem_data.coords, order=(0,0,1))
        alpha = config.alpha
        return c_t - alpha * (c_xx + c_yy)
    
    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.ReferenceData, config: ReactionDiffusionConfig) -> jax.Array:
        c = self.c_field.evaluate(*problem_data.coords)
        return c - problem_data.data

    @eqx.filter_jit
    def initial_residual(self, problem_data: pr.data.ReferenceData, config: ReactionDiffusionConfig) -> jax.Array:
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

def boundary_condition_theta(theta):    
    return jnp.zeros_like(theta)

def initial_condition(x,y):
    """Defines an initial condition shaped like a smiley face."""
    x = np.array(x)
    y = np.array(y)
    left_eye = (x + 0.4)**2 + (y - 0.4)**2 < 0.3**2
    right_eye = (x - 0.4)**2 + (y - 0.4)**2 < 0.3**2
    mouth = (x)**2 + (y+.4)**2 < 0.3**2
    output = np.logical_or.reduce((left_eye, right_eye, mouth)).astype(float)
    return jnp.array(output)

def sample_circle(n_points, key):
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, (n_points,), minval=0, maxval=1)
    r = jnp.sqrt(u)
    theta = jax.random.uniform(key, (n_points,), minval=0, maxval=2*jnp.pi)
    x = r*jnp.cos(theta)
    y = r*jnp.sin(theta)
    return x, y

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_initial = config.n_initial
    n_t = config.n_t

    # t_vec = jnp.linspace(-1,1,n_t)

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x, y = sample_circle(n_pde*n_t, subkey)
    key, subkey = jax.random.split(key)
    final_t = jax.random.uniform(subkey, (n_pde*n_t,), minval=-1, maxval=1)
    final_x = x
    final_y = y
    # final_x = jnp.tile(x, n_t)
    # final_y = jnp.tile(y, n_t)
    # final_t = jnp.repeat(t_vec, n_pde)

    collocation_data = pr.data.CollocationPoints(coords = (final_x, final_y, final_t))

    key, subkey = jax.random.split(key)
    bc_theta = jax.random.uniform(subkey, (n_bc*n_t,), minval=0, maxval=2*jnp.pi)

    bc_x = jnp.cos(bc_theta)
    bc_y = jnp.sin(bc_theta)
    boundary_data = boundary_condition(bc_x, bc_y)
    # boundary_data = jnp.tile(boundary_data, n_t)
    # final_bc_x = jnp.tile(bc_x, n_t)
    # final_bc_y = jnp.tile(bc_y, n_t)
    # final_bc_t = jnp.repeat(t_vec, n_bc)
    final_bc_x = bc_x
    final_bc_y = bc_y
    key, subkey = jax.random.split(key)
    final_bc_t = jax.random.uniform(subkey, (n_bc*n_t,), minval=-1, maxval=1)
    boundary_data = pr.data.ReferenceData(coords = (final_bc_x, final_bc_y, final_bc_t), data = boundary_data)

    x_ic, y_ic = sample_circle(n_initial, key)
    initial_data = initial_condition(x_ic, y_ic)
    t_ic = -1*jnp.ones(n_initial)
    initial_data = pr.data.ReferenceData(coords = (x_ic, y_ic, t_ic), data = initial_data)
    x_vec = jnp.linspace(-1,1,100)
    x_grid, y_grid = jnp.meshgrid(x_vec, x_vec)
    initial_d = initial_condition(x_grid, y_grid)
    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.contourf(x_grid, y_grid, initial_d, levels=100)
    ax.scatter(x_ic, y_ic, c="black", s=1)
    ax.set_aspect('equal', 'box')
    plt.savefig("figures/initial_condition.png")
    plt.close()
    

    return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data, initial = initial_data)

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
        config["basis_Nt"] = args.N
        config["script_name"] = f"{args.N}_heat"
    else:
        config["script_name"] = f"{timestamp}_heat"
    return config

if __name__ == "__main__":
    config = load_config("configs/heat.yml")
    config = parse_args(config)
    problem_data = sample_points(config)
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config.basis_Nx, config.basis_Ny, config.basis_Nt))
    c_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    problem = ReactionDiffusionProblem(c_field)
    if config.solver == "Adam":
        solver = pr.AdamSolver()
    elif config.solver == "LevenbergMarquardt":
        solver = pr.LevenbergMarquardtSolver()
    else:
        raise ValueError(f"Solver {config.solver} not supported")
    
    print(f"Script name: {config.script_name}")

    start_time = time.time()
    problem_config = ReactionDiffusionConfig.from_config(config)
    optimized_problem = solver.solve(problem, problem_data, config, ReactionDiffusionConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    save_results(config, optimized_problem)