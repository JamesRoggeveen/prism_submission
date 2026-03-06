import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import argparse
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

class HelmholtzConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    k: float
    regularization_strength: float

class HelmholtzProblem(pr.AbstractProblem):
    u_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: HelmholtzConfig) -> jax.Array:
        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0))
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2))
        forcing = reference_forcing(*problem_data.coords, config)
        return u_xx + u_yy + config.k**2 * self.u_field.evaluate(*problem_data.coords) - forcing
    
    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.ReferenceData, config: HelmholtzConfig) -> jax.Array:
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

def reference_solution(x,y):
    return jnp.sin(3*jnp.pi/2*y)*jnp.cos(jnp.pi*x) + 0.5*(x**2 - y)

def reference_forcing(x,y,config):
    return 1-13/4*jnp.pi**2 * jnp.cos(jnp.pi*x)*jnp.sin(3*jnp.pi/2*y) + config.k**2 * reference_solution(x,y)

def sample_from_mask(key, x_grid, y_grid, data, ref, M=8000):
    mask = jnp.nonzero(ref)

    xi = x_grid[mask]
    yi = y_grid[mask]

    idx = jax.random.choice(key, len(xi), shape=(M,), replace=False)
    x_samples = jnp.array(xi[idx])
    y_samples = jnp.array(yi[idx])
    data_samples = jnp.array(data[mask][idx])

    return x_samples, y_samples, data_samples

def sample_points(config):
    def r1(theta):
        return 0.7+0.3*jnp.cos(2*theta)

    circle_x, circle_y, circle_r = .3, .1, 0.15
    x_vec = jnp.linspace(-1,1,300)
    y_vec = jnp.linspace(-1,1,300)
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

    key, subkey = jax.random.split(key)
    theta_samples = jax.random.uniform(subkey, (config.n_bc,), minval=0, maxval=2*jnp.pi)
    r_samples = r1(theta_samples)
    x_bc_1 = r_samples*jnp.cos(theta_samples)
    y_bc_1 = 2*r_samples*jnp.sin(theta_samples)
    x_bc_2 = circle_r*jnp.cos(theta_samples)+circle_x
    y_bc_2 = circle_r*jnp.sin(theta_samples)+circle_y
    x_bc = jnp.concatenate([x_bc_1, x_bc_2])
    y_bc = jnp.concatenate([y_bc_1, y_bc_2])

    boundary_data = reference_solution(x_bc, y_bc)
    collocation_data = pr.data.CollocationPoints(coords = (x_samples, y_samples))
    boundary_data = pr.data.ReferenceData(coords = (x_bc, y_bc), data = boundary_data)
    return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data, grid = grid_data)

def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    c_coeffs = optimized_problem.u_field.coeffs.value
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
    config = load_config("configs/config.yml")
    config = parse_args(config)
    problem_data = sample_points(config)
    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.contourf(problem_data["grid"].coords[0], problem_data["grid"].coords[1], problem_data["grid"].data, levels=100)
    ax.scatter(problem_data["equation"].coords[0], problem_data["equation"].coords[1], c="red", s=1)
    ax.scatter(problem_data["boundary"].coords[0], problem_data["boundary"].coords[1], c="blue", s=1)
    plt.savefig("figures/helmholtz_grid.png")
    plt.close()

    basis_size = jnp.arange(5,30,3)
    errors = []
    times = []
    basis_N = []

    grid_data = problem_data["grid"]
    grid_x, grid_y = grid_data.coords
    mask = grid_data.data
    ref = reference_solution(grid_x, grid_y)

    for basis_size in basis_size:
        N = int(basis_size)
        basis_N.append(N)
        basis = pr.ChebyshevBasis2D((N, N))
        u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))  
        problem = HelmholtzProblem(u_field)
        solver = pr.get_solver(config)
        
        print(f"Basis size {basis_size}")

        start_time = time.time()
        problem_config = HelmholtzConfig.from_config(config)
        optimized_problem, log_data = solver.solve(problem, problem_data, config, HelmholtzConfig.from_config(config))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        times.append(end_time - start_time)
        u_eval = optimized_problem.u_field.evaluate(grid_x, grid_y)
        u_eval = u_eval.reshape(grid_x.shape)
        error = jnp.linalg.norm(u_eval[mask] - ref[mask])/jnp.linalg.norm(ref[mask])
        errors.append(error)
        print(f"L2 relative error: {error}")


    f, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].semilogy(basis_N, errors,marker="o")
    ax[0].set_xlabel("Basis Size")
    ax[0].set_ylabel("L2 Relative Error")
    ax[1].semilogy(basis_N, times,marker="o")
    ax[1].set_xlabel("Basis Size")
    ax[1].set_ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig("figures/helmholtz_basis_size.png")
    plt.close()

    error = jnp.log10(jnp.abs(u_eval - ref)/jnp.abs(ref))
    ref = jnp.where(mask == 0, jnp.nan, ref)
    u_eval = jnp.where(mask == 0, jnp.nan, u_eval)
    error = jnp.where(mask == 0, jnp.nan, error)
    
    f, ax = plt.subplots(1,3,figsize=(15,5))
    im0 = ax[0].contourf(grid_x, grid_y, ref, levels=100)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Reference Solution")
    im1 = ax[1].contourf(grid_x, grid_y, u_eval, levels=100)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Optimized Solution")
    im2 = ax[2].contourf(grid_x, grid_y, error, levels=100)
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].set_title("Relative Error")
    f.colorbar(im0,ax=ax[0])
    f.colorbar(im1,ax=ax[1])
    f.colorbar(im2,ax=ax[2])
    plt.tight_layout()
    plt.savefig("figures/helmholtz_solution.png")
    plt.close()
    
    
    f, ax = plt.subplots(1,1,figsize=(5,5))
    coeffs = optimized_problem.u_field.coeffs.value
    coeffs = coeffs.reshape(N+1,N+1)
    im = ax.imshow(jnp.log10(jnp.abs(coeffs)),cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x basis index")
    ax.set_ylabel("y basis index")
    ax.set_title("Coefficient Values")
    plt.tight_layout()
    plt.savefig("figures/helmholtz_coeffs.png")
    plt.close()