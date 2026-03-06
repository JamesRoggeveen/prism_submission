import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import argparse
import numpy as np # Using numpy for file loading
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import jax.scipy as jsp

def _softmin(values, tau):
    # Smooth approximation of min(values).
    # As tau -> 0, this -> min; for finite tau, gives gradients from many entries.
    return -tau * (jsp.special.logsumexp(-values / tau) - jnp.log(values.size))

jax.config.update("jax_enable_x64", True)

## CHANGED: Added new configuration parameters for the optimization problem
class LaplaceConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    regularization_strength: float
    cyl_radius: float
    x_cyl_initial: float
    y_cyl_initial: float
    n_grid: int

class LaplaceProblem(pr.AbstractProblem):
    u_field: pr.BasisField
    x_cyl: jax.Array
    y_cyl: jax.Array

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "outer_boundary": self.outer_boundary_residual,
            "inner_boundary": self.inner_boundary_residual,
            "integral_objective": self.integral_objective,
            "constraint_penalty": self.constraint_penalty,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: LaplaceConfig) -> jax.Array:
        x, y = problem_data.coords
        mask = (x - self.x_cyl)**2 + (y - self.y_cyl)**2 >= config.cyl_radius**2
        mask = jax.lax.stop_gradient(mask)
        u_xx = self.u_field.derivative(x, y, order=(2,0))
        u_yy = self.u_field.derivative(x, y, order=(0,2))
        
        return (u_xx + u_yy) * mask
    
    @eqx.filter_jit
    def outer_boundary_residual(self, problem_data: pr.data.ReferenceData, config: LaplaceConfig) -> jax.Array:
        u = self.u_field.evaluate(*problem_data.coords)
        return u - problem_data.data
        
    @eqx.filter_jit
    def inner_boundary_residual(self, problem_data: pr.data.ReferenceData, config: LaplaceConfig) -> jax.Array:
        template_x, template_y = problem_data.coords
        
        x_bc = template_x * config.cyl_radius + self.x_cyl
        y_bc = template_y * config.cyl_radius + self.y_cyl
        x_bc = jax.lax.stop_gradient(x_bc)
        y_bc = jax.lax.stop_gradient(y_bc)
        
        u = self.u_field.evaluate(x_bc, y_bc)
        return u - problem_data.data

    @eqx.filter_jit
    def integral_objective(self, problem_data: pr.data.CollocationPoints, config: LaplaceConfig) -> jax.Array:
        x, y = problem_data.coords
        u = jax.lax.stop_gradient(self.u_field.evaluate(x, y))
        distance = jnp.sqrt((x - self.x_cyl)**2 + (y - self.y_cyl)**2) - config.cyl_radius
 
        k = 100.0 
        
        smooth_mask = jax.nn.sigmoid(k * distance)
        
        dA = 2/config.n_grid * 2/config.n_grid
        integral_value = 0.01*jnp.nansum(u * smooth_mask) * dA # Use the smooth_mask here
        
        return jnp.atleast_1d(integral_value)

    @eqx.filter_jit
    def constraint_penalty(self, problem_data: pr.data.CollocationPoints, config: LaplaceConfig) -> jax.Array:
        theta = jnp.linspace(0, 2*jnp.pi, 200)
        r_wall = 0.7 + 0.3 * jnp.cos(2*theta)
        x_wall = r_wall * jnp.cos(theta)
        y_wall = 2 * r_wall * jnp.sin(theta)
        
        sq_dists = (x_wall - self.x_cyl)**2 + (y_wall - self.y_cyl)**2
        min_sq_dist = jnp.min(sq_dists)
        
        required_clearance_sq = (2 * config.cyl_radius)**2
        
        violation = required_clearance_sq - min_sq_dist
        penalty = 1000*jax.nn.relu(violation)
        
        return jnp.atleast_1d(penalty)

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
    return x_samples, y_samples

def outer_boundary_condition(x,y):
    theta = jnp.arctan2(y,x)
    return 2*jnp.sin(theta) + jnp.cos(3*theta) + 4

def inner_boundary_condition(x,y):
    return -2*jnp.ones_like(x)

## CHANGED: Renamed function and completely new logic for data generation
def create_problem_data(config):
    # This function creates the data structures needed for the different residuals.
    
    # 1. Define the outer boundary shape function
    def r1(theta):
        return 0.7+0.3*jnp.cos(2*theta)

    # 2. Create a grid and a mask for the fixed OUTER domain
    n_grid = config.n_grid
    x_vec = jnp.linspace(-1,1,n_grid)
    y_vec = jnp.linspace(-1,1,n_grid)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    y_squashed = 0.5*y_grid
    theta_grid = jnp.arctan2(y_squashed, x_grid)
    r_grid = jnp.sqrt(x_grid**2 + (y_squashed)**2)
    outer_mask = r_grid <= r1(theta_grid)
    
    grid_for_plotting = pr.data.ReferenceData(coords=(x_grid, y_grid), data=outer_mask)
    mask_grid_x = x_grid[outer_mask]
    mask_grid_y = y_grid[outer_mask]
    grid_for_integral = pr.data.CollocationPoints(coords=(mask_grid_x, mask_grid_y))
    
    # 3. Sample a large number of "candidate" collocation points within the outer domain.
    # The dynamic mask inside equation_residual will later exclude points inside the cylinder.
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x_samples, y_samples = sample_from_mask(subkey, x_grid, y_grid, outer_mask, outer_mask, config.n_pde)
    collocation_data = pr.data.CollocationPoints(coords=(x_samples, y_samples))

    # 4. Sample points on the FIXED outer boundary
    key, subkey = jax.random.split(key)
    theta_samples_outer = jax.random.uniform(subkey, (config.n_bc,), minval=0, maxval=2*jnp.pi)
    r_samples_outer = r1(theta_samples_outer)
    x_bc_outer = r_samples_outer*jnp.cos(theta_samples_outer)
    y_bc_outer = 2*r_samples_outer*jnp.sin(theta_samples_outer)
    outer_bc_values = outer_boundary_condition(x_bc_outer, y_bc_outer)
    outer_boundary_data = pr.data.ReferenceData(coords=(x_bc_outer, y_bc_outer), data=outer_bc_values)

    # 5. Create a TEMPLATE for the DYNAMIC inner boundary (points on a unit circle)
    theta_samples_inner = jnp.linspace(0, 2*jnp.pi, config.n_bc)
    x_bc_template = jnp.cos(theta_samples_inner)
    y_bc_template = jnp.sin(theta_samples_inner)
    inner_bc_values = inner_boundary_condition(x_bc_template, y_bc_template) # Value is constant, so this is fine
    inner_boundary_data = pr.data.ReferenceData(coords=(x_bc_template, y_bc_template), data=inner_bc_values)

    # 6. Assemble the final ProblemData object
    # Note how the keys match the keys in get_residual_functions
    return pr.data.ProblemData(
        equation=collocation_data,
        outer_boundary=outer_boundary_data,
        inner_boundary=inner_boundary_data,
        integral_objective=grid_for_integral, # Re-use collocation points for the integral
        constraint_penalty=pr.data.CollocationPoints(coords=()), # This residual doesn't need data
    ), grid_for_plotting


def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    
    field_dict = {
        "u": optimized_problem.u_field.coeffs.value,
        "x_cyl": optimized_problem.x_cyl, ## CHANGED: Save optimized cylinder position
        "y_cyl": optimized_problem.y_cyl,
    }
    
    # Save coeffs as npy for convenience
    jnp.save(result_path / "c_coeffs.npy", field_dict["u"])
    
    full_data = {"fields": field_dict, "config": config_to_dict(config)}
    pr.save_dict_to_hdf5(result_path / "full_data.h5", full_data)
    save_config(config, result_path)

# (This function is unchanged)
def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int)
    args = parser.parse_args()
    timestamp = time.strftime("%m%d%H%M")
    config["timestamp"] = timestamp
    if args.N is not None:
        config["basis_Nx"] = args.N
        config["basis_Ny"] = args.N
        config["script_name"] = f"{args.N}_laplace_optimized"
    else:
        config["script_name"] = f"{timestamp}_laplace_optimized"

    config["results_dir"] = f"results/{timestamp}"
    return config

if __name__ == "__main__":
    # Ensure figures directory exists
    pathlib.Path("figures").mkdir(exist_ok=True)
    
    config = load_config("configs/laplace_optimized.yml")
    config = parse_args(config)
    config["n_grid"] = 800
    
    problem_data, grid_for_plotting = create_problem_data(config)
    
    basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))

    initial_problem = LaplaceProblem(
        u_field=u_field,
        x_cyl=float(config.x_cyl_initial),
        y_cyl=float(config.y_cyl_initial)
    )
    problem_config = LaplaceConfig.from_config(config)
    config_outer = config.copy()
    config_outer["learning_rate"] = 0.1
    config_outer["n_epochs"] = 2000
    config_outer["residual_weights"]["integral_objective"] = 0.0
    config_outer["residual_weights"]["constraint_penalty"] = 0.0
    solver = pr.get_solver(config_outer)
    optimized_initial, log_data = solver.solve(initial_problem, problem_data, config_outer, problem_config)
    
    problem = LaplaceProblem(
        u_field=optimized_initial.u_field,
        x_cyl=jnp.array(config.x_cyl_initial),
        y_cyl=jnp.array(config.y_cyl_initial)
    )
    
    solver = pr.get_solver(config)
    
    print(f"Starting optimization for basis size N={config.basis_Nx}")
    print(f"Initial cylinder position: ({problem.x_cyl:.3f}, {problem.y_cyl:.3f})")

    start_time = time.time()
    problem_config = LaplaceConfig.from_config(config)
    optimized_problem, log_data = solver.solve(problem, problem_data, config, problem_config)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Final optimized cylinder position: ({optimized_problem.x_cyl:.3f}, {optimized_problem.y_cyl:.3f})")
    
    save_results(config, optimized_problem)

    # --- Plotting and Visualization ---
    
    # Get grid coordinates and the fixed outer mask
    grid_x, grid_y = grid_for_plotting.coords
    outer_mask = grid_for_plotting.data
    
    final_x_cyl = optimized_problem.x_cyl
    final_y_cyl = optimized_problem.y_cyl
    final_cyl_mask = (grid_x - final_x_cyl)**2 + (grid_y - final_y_cyl)**2 >= config.cyl_radius**2
    final_mask = outer_mask & final_cyl_mask

    # Evaluate the solution on the grid
    u_eval = optimized_problem.u_field.evaluate(grid_x, grid_y)
    u_eval = u_eval.reshape(grid_x.shape)
    u_eval = jnp.where(final_mask, u_eval, jnp.nan) # Apply final mask for visualization

    n_grid = config.n_grid
    dx = 2/n_grid # Grid spacing in x
    dy = 2/n_grid # Grid spacing in y
        
    # Approximate integral using trapezoidal rule
    # total_domain_area = jnp.sum(final_mask) * dx * dy
    
    # # 2. Calculate the mean value of 'u' ONLY over the valid (non-NaN) points.
    # mean_u_value = jnp.nanmean(u_eval)
    
    # 3. The integral is the average value multiplied by the total area.
    integral_approx = jnp.nansum(u_eval) * dx * dy
    
    print(f"Approximate integral of solution over domain: {integral_approx:.6f}")
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    
    
    # Plot the solution
    cmap = "jet"
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.contourf(grid_x, grid_y, u_eval, levels=100, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Optimized Solution (Cylinder at ({final_x_cyl:.2f}, {final_y_cyl:.2f}))")
    f.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("figures/laplace_optimized_solution.png")
    plt.savefig(result_path / "laplace_optimized_solution.png")
    plt.close()

    # Plot the coefficient spectrum
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    coeffs = optimized_problem.u_field.coeffs.value.reshape(config.basis_Nx+1, config.basis_Ny+1)
    im = ax.imshow(jnp.log10(jnp.abs(coeffs)), cmap=cmap, origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x basis index")
    ax.set_ylabel("y basis index")
    ax.set_title("Coefficient Values")
    plt.tight_layout()
    plt.savefig("figures/laplace_optimized_coeffs.png")
    plt.savefig(result_path / "laplace_optimized_coeffs.png")
    plt.close()

    np.savetxt(result_path / "integral_approx.csv", np.array([integral_approx]))

    f, ax = plt.subplots(1,1,figsize=(5,5))
    for key in log_data.keys():
        ax.semilogy(log_data[key],marker="o",label=key)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/laplace_optimized_loss.png")
    plt.savefig(result_path / "laplace_optimized_loss.png")
    plt.close()