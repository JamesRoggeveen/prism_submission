import jax
import jax.numpy as jnp
import equinox as eqx
from jax.random import key_data
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class TransportConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    alpha: float
    gamma: float
    regularization_strength: float

class TransportProblem(pr.AbstractProblem):
    c_field: pr.BasisField
    u_field: pr.BasisField
    v_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
            "target": self.target_residual,
            "continuity": self.continuity_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: TransportConfig) -> jax.Array:
        c_xx = self.c_field.derivative(*problem_data.coords, order=(2,0,0))
        c_yy = self.c_field.derivative(*problem_data.coords, order=(0,2,0))
        c_t = self.c_field.derivative(*problem_data.coords, order=(0,0,1))
        c_x = self.c_field.derivative(*problem_data.coords, order=(1,0,0))
        c_y = self.c_field.derivative(*problem_data.coords, order=(0,1,0))
        u = self.u_field.evaluate(*problem_data.coords)
        v = self.v_field.evaluate(*problem_data.coords)
        diffusion= config.alpha*(c_xx + c_yy)
        advection = u*c_x + v*c_y
        return c_t - config.gamma*(diffusion + advection)

    @eqx.filter_jit
    def continuity_residual(self, problem_data: pr.data.CollocationPoints, config: TransportConfig) -> jax.Array:
        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0,0))
        v_y = self.v_field.derivative(*problem_data.coords, order=(0,1,0))
        return u_x + v_y
    
    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.BoundaryData, config: TransportConfig) -> jax.Array:
        c_x = self.c_field.derivative(*problem_data.coords, order=(1,0,0))
        c_y = self.c_field.derivative(*problem_data.coords, order=(0,1,0))
        nx, ny = problem_data.normal_vector
        u = self.u_field.evaluate(*problem_data.coords)
        v = self.v_field.evaluate(*problem_data.coords)
        return jnp.concatenate([c_x*nx + c_y*ny, u, v], axis=0)

    # @eqx.filter_jit
    # def boundary_residual(self, problem_data: pr.data.BoundaryData, config: TransportConfig) -> jax.Array:
    #     c = self.c_field.evaluate(*problem_data.coords)
    #     u = self.u_field.evaluate(*problem_data.coords)
    #     v = self.v_field.evaluate(*problem_data.coords)
    #     return jnp.concatenate([c, u, v], axis=0)

    @eqx.filter_jit
    def target_residual(self, problem_data: pr.data.ReferenceData, config: TransportConfig) -> jax.Array:
        c = self.c_field.evaluate(*problem_data.coords)
        return c - problem_data.data

    @eqx.filter_jit
    def total_residual(self, problem_data: pr.data.ProblemData, config: TransportConfig) -> jax.Array:
        equation_residual = jnp.sqrt(config.residual_weights["equation"]) * self.equation_residual(problem_data["equation"], config)
        boundary_residual = jnp.sqrt(config.residual_weights["boundary"]) * self.boundary_residual(problem_data["boundary"], config)
        target_residual = jnp.sqrt(config.residual_weights["target"]) * self.target_residual(problem_data["target"], config)
        continuity_residual = jnp.sqrt(config.residual_weights["continuity"]) * self.continuity_residual(problem_data["equation"], config)
        n_equation = equation_residual.shape[0]
        n_boundary = boundary_residual.shape[0]
        n_target = target_residual.shape[0]   
        n_continuity = continuity_residual.shape[0]
        return jnp.concatenate([equation_residual/jnp.sqrt(n_equation), boundary_residual/jnp.sqrt(n_boundary), target_residual/jnp.sqrt(n_target), continuity_residual/jnp.sqrt(n_continuity)], axis=0)

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

def sample_circle(n_points, key):
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, (n_points,), minval=0, maxval=1)
    r = jnp.sqrt(u)
    theta = jax.random.uniform(key, (n_points,), minval=0, maxval=2*jnp.pi)
    x = r*jnp.cos(theta)
    y = r*jnp.sin(theta)
    return x, y

def sample_goal(slice_name,key,n_points):
    if slice_name is None:
        image = plt.imread("data/H.png")
        image = 0.05*jnp.ones_like(image[...,-1])
    else:
        image = plt.imread(f"data/{slice_name}")
        image = image[...,-1]
        image = ~image.astype(bool)
        image = jnp.flipud(image)
    # Apply Gaussian blur to smooth out gradients
    image = gaussian_filter(image.astype(float), sigma=15.0)
    # image = image + jnp.ones_like(image)*0.05
    x_grid, y_grid = jnp.meshgrid(jnp.linspace(-1,1,image.shape[1]), jnp.linspace(-1,1,image.shape[0]))
    r_grid = jnp.sqrt(x_grid**2 + y_grid**2)
    mask = r_grid < 1
    image = image[mask].reshape(-1)
    x_grid = x_grid[mask].reshape(-1)
    y_grid = y_grid[mask].reshape(-1)
    key, subkey = jax.random.split(key)
    inds = jax.random.choice(subkey, jnp.arange(x_grid.size), (n_points,), replace=False)
    x_grid = x_grid[inds]
    y_grid = y_grid[inds]
    image = image[inds].astype(float)
    return x_grid, y_grid, image

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_target = config.n_target
    n_t = config.n_t

    t_vec = jnp.linspace(-1,1,n_t)

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
    boundary_data = pr.data.BoundaryData(coords = (bc_x, bc_y, final_bc_t), normal_vector = (-bc_x, -bc_y))

    # slices = [None, "H.png", None]
    slices = ["H.png", "A.png", "R.png", "V.png", "A.png", "R.png", "D.png"]
    t_sample = jnp.linspace(-1,1,len(slices))
    x_target_list = []
    y_target_list = []
    t_target_list = []
    target_data_list = []
    _, _, h_data = sample_goal("H.png", key, n_target)
    reference_mass = jnp.nansum(h_data)
    print(f"Reference mass: {reference_mass}")
    f, ax = plt.subplots(1, len(slices), figsize=(3*len(slices),3))
    for i, slice in enumerate(slices):
        key, subkey = jax.random.split(key)
        x_target, y_target, target_data = sample_goal(slice,subkey,n_target)
        target_mass = jnp.nansum(target_data)
        print(f"Target mass {slice[:-4]}: {target_mass}")
        target_data = target_data * reference_mass / target_mass
        print(f"New mass {slice[:-4]}: {jnp.nansum(target_data)}")
        t_target = t_sample[i]*jnp.ones_like(x_target)
        ax[i].scatter(x_target, y_target, c=target_data, cmap="jet")
        x_target_list.append(x_target)
        y_target_list.append(y_target)
        t_target_list.append(t_target)
        target_data_list.append(target_data)
    f.tight_layout()
    plt.savefig("data/targets.png")
    plt.close()
    x_target = jnp.concatenate(x_target_list)
    y_target = jnp.concatenate(y_target_list)
    t_target = jnp.concatenate(t_target_list)
    target_data = jnp.concatenate(target_data_list)
    target_data = pr.data.ReferenceData(coords = (x_target, y_target, t_target), data = target_data)
    return pr.data.ProblemData(equation = collocation_data, continuity = collocation_data, boundary = boundary_data, target = target_data)

    # key, subkey = jax.random.split(key)
    # x_ic, y_ic = sample_circle(n_initial, subkey)
    # t_ic = -1*jnp.ones(n_initial)
    # initial_data = pr.data.ReferenceData(coords = (x_ic, y_ic, t_ic), data = initial_condition(x_ic, y_ic))
    
    
    # key, subkey = jax.random.split(key)
    # x_fc, y_fc = sample_circle(n_final, key)
    # t_fc = jnp.ones(n_final)
    # final_data = final_condition(x_fc, y_fc)
    # final_data = pr.data.ReferenceData(coords = (x_fc, y_fc, t_fc), data = final_data)

    # return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data, initial = initial_data, final = final_data)

def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    c_coeffs = optimized_problem.c_field.coeffs.value
    u_coeffs = optimized_problem.u_field.coeffs.value
    v_coeffs = optimized_problem.v_field.coeffs.value
    field_dict = {
        "c": c_coeffs,
        "u": u_coeffs,
        "v": v_coeffs
    }
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
        config["script_name"] = f"{args.N}_transport"
    else:
        config["script_name"] = f"{timestamp}_transport"
    return config

if __name__ == "__main__":
    config = load_config("configs/transport.yml")
    config = parse_args(config)
    problem_data = sample_points(config)
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config.basis_Nx, config.basis_Ny, config.basis_Nt))
    c_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    v_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    problem = TransportProblem(c_field, u_field, v_field)
    solver = pr.get_solver(config)
    
    print(f"Script name: {config.script_name}")

    start_time = time.time()
    problem_config = TransportConfig.from_config(config)
    optimized_problem, log_data = solver.solve(problem, problem_data, config, TransportConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    save_results(config, optimized_problem)
