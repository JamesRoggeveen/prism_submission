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
from scipy.interpolate import griddata

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

    key, subkey = jax.random.split(key)
    theta_samples = jax.random.uniform(subkey, (config.n_bc,), minval=0, maxval=2*jnp.pi)
    r_samples = r1(theta_samples)
    x_bc_1 = r_samples*jnp.cos(theta_samples)
    y_bc_1 = 2*r_samples*jnp.sin(theta_samples)
    boundary_data_1 = outer_boundary_condition(x_bc_1, y_bc_1)
    x_bc_2 = circle_r*jnp.cos(theta_samples)+circle_x
    y_bc_2 = circle_r*jnp.sin(theta_samples)+circle_y
    boundary_data_2 = inner_boundary_condition(x_bc_2, y_bc_2)
    x_bc = jnp.concatenate([x_bc_1, x_bc_2])
    y_bc = jnp.concatenate([y_bc_1, y_bc_2])
    boundary_data = jnp.concatenate([boundary_data_1, boundary_data_2])


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

    config["results_dir"] = f"results/{timestamp}"
    return config

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    config = parse_args(config)
    problem_data = sample_points(config)
    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.contourf(problem_data["grid"].coords[0], problem_data["grid"].coords[1], problem_data["grid"].data, levels=100)
    ax.scatter(problem_data["equation"].coords[0], problem_data["equation"].coords[1], c="red", s=1)
    ax.scatter(problem_data["boundary"].coords[0], problem_data["boundary"].coords[1], c="blue", s=1)
    plt.savefig("figures/laplace_grid.png")
    plt.close()

    basis_size = jnp.arange(30,31,5)
    errors = []
    times = []
    basis_N = []
    Linf_errors = []

    reference = np.loadtxt("data/laplace_grid.csv", delimiter=",", skiprows=9)
    ref_x, ref_y, ref_data = reference[:,0], reference[:,1], reference[:,2]
    grid_x, grid_y = problem_data["grid"].coords
    # gridded_data = griddata(
    # (ref_x, ref_y),
    # ref_data,
    # (grid_x, grid_y),
    # method='cubic')
    mask = problem_data["grid"].data
    # gridded_data_mask = jnp.where(mask == 0, jnp.nan, gridded_data)
    ref_x, ref_y, ref_data = ref_x.reshape(grid_x.shape), ref_y.reshape(grid_y.shape), ref_data.reshape(grid_x.shape)
    gridded_data_mask = jnp.where(mask == 0, jnp.nan, ref_data)
    plt.contourf(grid_x, grid_y, gridded_data_mask, levels=100, cmap="jet")
    plt.savefig("figures/laplace_reference.png")
    plt.close()
    ref_data = ref_data.reshape(-1)


    grid_data = problem_data["grid"]
    grid_x, grid_y = grid_data.coords
    mask = grid_data.data
    dh = grid_y[1,0] - grid_y[0,0]
    print(dh)
    for basis_size in basis_size:
        N = int(basis_size)
        basis_N.append(N)
        basis = pr.ChebyshevBasis2D((N, N))
        u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))  
        problem = LaplaceProblem(u_field)
        solver = pr.get_solver(config)
        
        print(f"Basis size {basis_size}")

        start_time = time.time()
        problem_config = LaplaceConfig.from_config(config)
        optimized_problem, log_data = solver.solve(problem, problem_data, config, LaplaceConfig.from_config(config))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        times.append(end_time - start_time)
        u_eval = optimized_problem.u_field.evaluate(ref_x, ref_y)
        error = jnp.sqrt(jnp.nansum(jnp.power(u_eval - ref_data, 2))*dh**2)
        errors.append(error)
        print(f"L2 error: {error}")
        Linf_error = jnp.nanmax(jnp.abs(u_eval - ref_data))
        Linf_errors.append(Linf_error)
        print(f"Linf error: {Linf_error}")
        config["script_name"] = f"{basis_size}_laplace"
        config["basis_Nx"] = N
        config["basis_Ny"] = N
        save_results(config, optimized_problem)


    f, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].semilogy(basis_N, errors,marker="o",label="L2")
    ax[0].semilogy(basis_N, Linf_errors,marker="o",label="Linf")
    ax[0].set_xlabel("Basis Size")
    ax[0].set_ylabel("L2 Error")
    ax[0].legend()
    ax[1].semilogy(basis_N, times,marker="o")
    ax[1].set_xlabel("Basis Size")
    ax[1].set_ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig("figures/laplace_basis_size.png")
    plt.close()

    basis_N = jnp.array(basis_N)
    errors = jnp.array(errors)
    times = jnp.array(times)
    Linf_errors = jnp.array(Linf_errors)
    error_data = jnp.vstack((basis_N, errors, Linf_errors, times))

    jnp.save(f"{config['results_dir']}/errors.npy", error_data)

    u_eval = optimized_problem.u_field.evaluate(grid_x, grid_y) 
    u_eval = u_eval.reshape(grid_x.shape)

    ref_data = ref_data.reshape(grid_x.shape)
    error = jnp.abs(u_eval - ref_data)
    error = jnp.where(mask == 0, jnp.nan, error)
    u_eval = jnp.where(mask == 0, jnp.nan, u_eval)
    ref = gridded_data_mask

    cmap = "jet"
    results_path = pathlib.Path(config.results_dir) / config.script_name
    
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
    plt.savefig("figures/laplace_solution.png")
    plt.savefig(results_path / "laplace_solution.png")
    plt.close()
    
    
    f, ax = plt.subplots(1,1,figsize=(5,5))
    coeffs = optimized_problem.u_field.coeffs.value
    coeffs = coeffs.reshape(N+1,N+1)
    im = ax.imshow(jnp.log10(jnp.abs(coeffs)),cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x basis index")
    ax.set_ylabel("y basis index")
    ax.set_title("Coefficient Values")
    plt.tight_layout()
    plt.savefig("figures/laplace_coeffs.png")
    plt.close()

    n_grid = 500
    dx = 2/n_grid # Grid spacing in x
    dy = 2/n_grid # Grid spacing in y
        
    # Approximate integral using trapezoidal rule
    total_domain_area = jnp.sum(mask) * dx * dy
    
    # 2. Calculate the mean value of 'u' ONLY over the valid (non-NaN) points.
    mean_u_value = jnp.nanmean(u_eval)
    
    # 3. The integral is the average value multiplied by the total area.
    integral_approx = mean_u_value * total_domain_area
    
    print(f"Approximate integral of solution over domain: {integral_approx:.6f}")