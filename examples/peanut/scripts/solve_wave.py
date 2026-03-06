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

class WaveConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    c: float
    regularization_strength: float

class WaveProblem(pr.AbstractProblem):
    u_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
            "initial_condition": self.initial_condition,
            "initial_velocity": self.initial_velocity,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: WaveConfig) -> jax.Array:
        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0,0))
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2,0))
        u_tt = self.u_field.derivative(*problem_data.coords, order=(0,0,2))
        return u_tt - config.c**2 * (u_xx + u_yy)
    
    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.BoundaryData, config: WaveConfig) -> jax.Array:
        nx,ny = problem_data.normal_vector
        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0,0))
        u_y = self.u_field.derivative(*problem_data.coords, order=(0,1,0))
        return u_x*nx + u_y*ny
    
    @eqx.filter_jit
    def initial_condition(self, problem_data: pr.data.ReferenceData, config: WaveConfig) -> jax.Array:
        initial_u = self.u_field.evaluate(*problem_data.coords)
        reference_u = problem_data.data
        return initial_u - reference_u
    
    @eqx.filter_jit
    def initial_velocity(self, problem_data: pr.data.ReferenceData, config: WaveConfig) -> jax.Array:
        initial_u_t = self.u_field.derivative(*problem_data.coords, order=(0,0,1))
        return initial_u_t

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

def initial_condition(x,y):
    A = 0.5
    sigma = 0.2
    x0 = -.3
    y0 = .1

    return A*jnp.exp(-((x-x0)**2 + (y-y0)**2)/((2*sigma)**2))

def sample_points(config):
    def r1(theta):
        return 0.7+0.3*jnp.cos(2*theta)

    circle_x, circle_y, circle_r = .3, .1, 0.15
    x_vec = jnp.linspace(-1,1,500)
    y_vec = jnp.linspace(-1,1,500)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    y_squashed = 0.5*y_grid
    theta_grid = jnp.arctan2(y_squashed, x_grid)
    r_grid = jnp.sqrt(x_grid**2 + (y_squashed)**2)
    mask1 = r_grid <= r1(theta_grid)
    mask2 = (x_grid - circle_x)**2 + (y_grid-circle_y)**2 >= circle_r**2
    mask = mask1 & mask2
    mask = mask1
    grid_data = pr.data.ReferenceData(coords = (x_grid, y_grid), data = mask)
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x_samples, y_samples, _ = sample_from_mask(subkey, x_grid, y_grid, mask, mask, config.n_pde)
    key, subkey = jax.random.split(key)
    t_samples = jax.random.uniform(subkey, (config.n_pde,), minval=-1, maxval=1)
    collocation_data = pr.data.CollocationPoints(coords = (x_samples, y_samples, t_samples))

    key, subkey = jax.random.split(key)
    # theta_samples = jax.random.uniform(subkey, (config.n_bc,), minval=0, maxval=2*jnp.pi)
    theta_samples = jnp.linspace(0, 2*jnp.pi, config.n_bc)

    def x_func(theta):
        return r1(theta) * jnp.cos(theta)

    def y_func(theta):
        return 2 * r1(theta) * jnp.sin(theta)

    # Calculate boundary points
    x_bc_1 = x_func(theta_samples)
    y_bc_1 = y_func(theta_samples)

    # Use jax.grad and jax.vmap to get the derivatives (tangent vector components)
    # This is more robust than manual differentiation
    dx_dtheta = jax.vmap(jax.grad(x_func))(theta_samples)
    dy_dtheta = jax.vmap(jax.grad(y_func))(theta_samples)

    # The normal vector is perpendicular to the tangent (dy/dtheta, -dx/dtheta)
    # This gives the outward normal for a counter-clockwise curve
    nx_unnormalized = dy_dtheta
    ny_unnormalized = -dx_dtheta

    # Normalize the normal vector to make it a unit vector
    norm_magnitude = jnp.sqrt(nx_unnormalized**2 + ny_unnormalized**2)
    nx_bc_1 = nx_unnormalized / norm_magnitude
    ny_bc_1 = ny_unnormalized / norm_magnitude
    # r_samples = r1(theta_samples)
    # x_bc_1 = r_samples*jnp.cos(theta_samples)
    # y_bc_1 = 2*r_samples*jnp.sin(theta_samples)
    # r_bc_1 = jnp.sqrt(x_bc_1**2 + y_bc_1**2)
    # nx_bc_1 = -x_bc_1/r_bc_1
    # ny_bc_1 = -y_bc_1/r_bc_1
    x_bc_2 = circle_r*jnp.cos(theta_samples)+circle_x
    y_bc_2 = circle_r*jnp.sin(theta_samples)+circle_y
    nx_bc_2 = jnp.cos(theta_samples)
    ny_bc_2 = jnp.sin(theta_samples)
    # x_bc = jnp.concatenate([x_bc_1, x_bc_2])
    # y_bc = jnp.concatenate([y_bc_1, y_bc_2])
    # boundary_nx = jnp.concatenate([nx_bc_1, nx_bc_2])
    # boundary_ny = jnp.concatenate([ny_bc_1, ny_bc_2])
    # boundary_data = jnp.concatenate([boundary_data_1, boundary_data_2])
    # boundary_t = jax.random.uniform(subkey, (2*config.n_bc,), minval=-1, maxval=1)
    boundary_nx = -nx_bc_1
    boundary_ny = -ny_bc_1
    boundary_t = jax.random.uniform(subkey, (config.n_bc,), minval=-1, maxval=1)
    x_bc = x_bc_1
    y_bc = y_bc_1
    boundary_data = pr.data.BoundaryData(coords = (x_bc, y_bc, boundary_t), normal_vector = (boundary_nx, boundary_ny))

    ic_data = initial_condition(x_samples, y_samples)
    ic_data = pr.data.ReferenceData(coords = (x_samples, y_samples, -1*jnp.ones(config.n_pde)), data = ic_data)
    return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data, initial_condition = ic_data,initial_velocity = ic_data, grid = grid_data)

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

    config["results_dir"] = f"results/{timestamp}_wave"
    return config

if __name__ == "__main__":
    config = load_config("configs/wave.yml")
    config = parse_args(config)
    problem_data = sample_points(config)
    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.contourf(problem_data["grid"].coords[0], problem_data["grid"].coords[1], problem_data["grid"].data, levels=100)
    ax.scatter(problem_data["equation"].coords[0], problem_data["equation"].coords[1], c="red", s=1)
    ax.scatter(problem_data["boundary"].coords[0], problem_data["boundary"].coords[1], c="blue", s=1)
    ax.quiver(problem_data["boundary"].coords[0], problem_data["boundary"].coords[1], 
              problem_data["boundary"].normal_vector[0], problem_data["boundary"].normal_vector[1], 
              color="green", scale=20, width=0.002, alpha=0.7)
    plt.savefig("figures/wave_grid.png")
    plt.close()

    basis_size = jnp.arange(15,16,5)
    errors = []
    times = []
    basis_N = []

    # reference = np.loadtxt("data/wave_grid.csv", delimiter=",", skiprows=9)
    # ref_x, ref_y, ref_data = reference[:,0], reference[:,1], reference[:,2]
    # grid_x, grid_y = problem_data["grid"].coords
    # # gridded_data = griddata(
    # # (ref_x, ref_y),
    # # ref_data,
    # # (grid_x, grid_y),
    # # method='cubic')
    # mask = problem_data["grid"].data
    # # gridded_data_mask = jnp.where(mask == 0, jnp.nan, gridded_data)
    # ref_x, ref_y, ref_data = ref_x.reshape(grid_x.shape), ref_y.reshape(grid_y.shape), ref_data.reshape(grid_x.shape)
    # gridded_data_mask = jnp.where(mask == 0, jnp.nan, ref_data)
    # plt.contourf(grid_x, grid_y, gridded_data_mask, levels=100, cmap="jet")
    # plt.savefig("figures/laplace_reference.png")
    # plt.close()
    # ref_data = ref_data.reshape(-1)


    grid_data = problem_data["grid"]
    grid_x, grid_y = grid_data.coords
    mask = grid_data.data
    basis_N = (config.basis_Nx, config.basis_Ny, config.basis_Nt)
    basis = pr.BasisND([pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis], basis_N)
    # u_field = pr.BasisField(basis, pr.fields.PreconditionedFactorizedCoeffs.make_zero(basis.degs,r = 2))  
    u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    problem = WaveProblem(u_field)
    solver = pr.get_solver(config)
    
    print(f"Basis size {config.basis_Nx}")

    start_time = time.time()
    problem_config = WaveConfig.from_config(config)
    optimized_problem, log_data = solver.solve(problem, problem_data, config, WaveConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # times.append(end_time - start_time)
    # u_eval = optimized_problem.u_field.evaluate(ref_x, ref_y)
    # error = jnp.sqrt(jnp.nansum(jnp.power(u_eval - ref_data, 2))*dh**2)
    # errors.append(error)
    # print(f"L2 error: {error}")
    config["script_name"] = f"{config.basis_Nx}_wave"
    save_results(config, optimized_problem)


    u_eval_t0 = optimized_problem.u_field.evaluate(grid_x, grid_y, -1*jnp.ones(grid_x.shape))
    u_eval_t1 = optimized_problem.u_field.evaluate(grid_x, grid_y, -0.4*jnp.ones(grid_x.shape))
    u_eval_t2 = optimized_problem.u_field.evaluate(grid_x, grid_y, 0.0*jnp.ones(grid_x.shape))
    u_eval_t0 = u_eval_t0.reshape(grid_x.shape)
    u_eval_t1 = u_eval_t1.reshape(grid_x.shape)
    u_eval_t2 = u_eval_t2.reshape(grid_x.shape)
    u_eval_t0 = jnp.where(mask == 0, jnp.nan, u_eval_t0)
    u_eval_t1 = jnp.where(mask == 0, jnp.nan, u_eval_t1)
    u_eval_t2 = jnp.where(mask == 0, jnp.nan, u_eval_t2)

    cmap = "jet"
    
    f, ax = plt.subplots(1,3,figsize=(15,5))
    im0 = ax[0].contourf(grid_x, grid_y, u_eval_t0, levels=100, cmap=cmap)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("t0 = -1")
    im1 = ax[1].contourf(grid_x, grid_y, u_eval_t1, levels=100, cmap=cmap)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("t1 = -0.4")
    im2 = ax[2].contourf(grid_x, grid_y, u_eval_t2, levels=100, cmap=cmap)
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].set_title("t2 = 0.0")
    f.colorbar(im0,ax=ax[0])
    f.colorbar(im1,ax=ax[1])
    f.colorbar(im2,ax=ax[2])
    plt.tight_layout()
    plt.savefig("figures/wave_solution.png")
    plt.close()
    
