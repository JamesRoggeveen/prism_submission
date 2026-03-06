import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

class StokesConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float

class StokesProblem(pr.AbstractProblem):
    u_field: pr.BasisField
    v_field: pr.BasisField
    p_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "wall": self.wall_boundary_residual,
            # "immersed": self.immersed_boundary_residual,
            "inlet": self.inlet_residual,
            "outlet": self.outlet_residual,
            # "continuity": self.continuity_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        grad_p_x = self.p_field.derivative(*problem_data.coords, order=(1,0))
        grad_p_y = self.p_field.derivative(*problem_data.coords, order=(0,1))

        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0))
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2))

        v_xx = self.v_field.derivative(*problem_data.coords, order=(2,0))
        v_yy = self.v_field.derivative(*problem_data.coords, order=(0,2))


        x_eq = 0.001*(u_xx + u_yy) - grad_p_x
        y_eq = 0.001*(v_xx + v_yy) - grad_p_y

        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0))
        v_y = self.v_field.derivative(*problem_data.coords, order=(0,1))
        continuity = u_x + v_y

        return jnp.concatenate([x_eq, y_eq, continuity], axis=0)
    
    # @eqx.filter_jit
    # def continuity_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
    #     u_x = self.u_field.derivative(*problem_data.coords, order=(1,0))
    #     v_y = self.v_field.derivative(*problem_data.coords, order=(0,1))
    #     continuity = u_x + v_y
    #     return continuity

    # @eqx.filter_jit
    # def immersed_boundary_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
    #     coords = problem_data.coords
    #     u = self.u_field.evaluate(*coords)
    #     v = self.v_field.evaluate(*coords)
    #     return jnp.concatenate([u,v], axis=0)
 
    @eqx.filter_jit
    def wall_boundary_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        u = self.u_field.evaluate(*problem_data.coords)
        v = self.v_field.evaluate(*problem_data.coords)
        return jnp.concatenate([u, v], axis=0)

    @eqx.filter_jit
    def inlet_residual(self, problem_data: pr.data.ReferenceData, config: StokesConfig) -> jax.Array:
        u = self.u_field.evaluate(*problem_data.coords)
        v = self.v_field.evaluate(*problem_data.coords)
        u_ref, v_ref = problem_data.data
        return jnp.concatenate([u - u_ref, v - v_ref], axis=0)

    @eqx.filter_jit
    def outlet_residual(self, problem_data: pr.data.ReferenceData, config: StokesConfig) -> jax.Array:
        coords = problem_data.coords
        p = self.p_field.evaluate(*coords)

        u_x = self.u_field.derivative(*coords, order=(1, 0))
        v_x = self.v_field.derivative(*coords, order=(1, 0))
        return jnp.concatenate([p, v_x], axis=0)

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

def config_to_dict(config):
    output_dict = config.data.copy()
    solver_kwargs_copy = {k: v for k, v in output_dict["solver_kwargs"].items() if k != "verbose"}
    output_dict["solver_kwargs"] = solver_kwargs_copy
    return output_dict

def inlet_condition(x,y):
    return -0.3*(y-1)*(y+1),jnp.zeros_like(y)

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_immersed = config.n_immersed
    key = jax.random.PRNGKey(config.seed)

    key, subkey = jax.random.split(key)
    eq_col_x = jax.random.uniform(subkey, (n_pde,), minval=-1, maxval=1)
    key, subkey = jax.random.split(key)
    eq_col_y = jax.random.uniform(subkey, (n_pde,), minval=-1, maxval=1)
    cylinder_x, cylinder_y = config.cylinder_x, config.cylinder_y
    cylinder_radius = config.cylinder_radius
    mask = (eq_col_x - cylinder_x)**2 + (eq_col_y - cylinder_y)**2 <= cylinder_radius**2
    # eq_col_data = pr.data.CollocationPoints(coords = (eq_col_x[~mask], eq_col_y[~mask]))
    n_pde_near_cylinder = config.n_pde # Or some other number
    key, subkey = jax.random.split(key)
    
    # Sample radius from cylinder edge outwards
    r_samples = jax.random.uniform(subkey, (n_pde_near_cylinder,)) * (2*config.cylinder_radius) + config.cylinder_radius
    key, subkey = jax.random.split(key)
    
    # Sample angle
    theta_samples = jax.random.uniform(subkey, (n_pde_near_cylinder,)) * 2 * jnp.pi

    # Convert to Cartesian
    annulus_x = config.cylinder_x + r_samples * jnp.cos(theta_samples)
    annulus_y = config.cylinder_y + r_samples * jnp.sin(theta_samples)

    # Sample near walls
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    near_wall_x = jax.random.uniform(subkey1, (n_pde_near_cylinder,), minval=-1, maxval=1)
    near_wall_y = jax.random.uniform(subkey2, (n_pde_near_cylinder,), minval=0.9, maxval=1)*jax.random.choice(subkey3, jnp.array([1,-1]), shape=(n_pde_near_cylinder,))

    # Combine with the existing uniform points
    combined_eq_x = jnp.concatenate([eq_col_x[~mask], annulus_x, near_wall_x])
    combined_eq_y = jnp.concatenate([eq_col_y[~mask], annulus_y, near_wall_y])
    
    eq_col_data = pr.data.CollocationPoints(coords=(combined_eq_x, combined_eq_y))

    key, subkey = jax.random.split(key)
    wall_x = jax.random.uniform(subkey, (2*n_bc,), minval=-1, maxval=1)
    wall_y = jnp.concatenate([jnp.ones((n_bc,))*1,jnp.ones((n_bc,))*-1])
    wall_data = pr.data.CollocationPoints(coords = (wall_x, wall_y))

    key, subkey = jax.random.split(key)
    theta_immersed = jnp.linspace(0, 2*jnp.pi, n_immersed)
    x_immersed = cylinder_x + cylinder_radius*jnp.cos(theta_immersed)
    y_immersed = cylinder_y + cylinder_radius*jnp.sin(theta_immersed)
    immersed_data = pr.data.CollocationPoints(coords=(x_immersed, y_immersed))

    key, subkey = jax.random.split(key)
    inlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    inlet_x = -1*jnp.ones_like(inlet_y)
    inlet_data = pr.data.ReferenceData(coords=(inlet_x, inlet_y), data=inlet_condition(inlet_x, inlet_y))

    key, subkey = jax.random.split(key)
    outlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    outlet_x = jnp.ones_like(outlet_y)
    outlet_data = pr.data.ReferenceData(coords=(outlet_x, outlet_y), data=inlet_condition(outlet_x, outlet_y))

    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(*eq_col_data.coords, s = 1, zorder = 1)
    ax.scatter(*immersed_data.coords, s = 1, c = "black", zorder = 2)
    ax.scatter(*wall_data.coords, s = 1, c = "red", zorder = 2)
    ax.scatter(*inlet_data.coords, s = 1, c = "blue", zorder = 3)
    ax.scatter(*outlet_data.coords, s = 1, c = "green", zorder = 4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1,1)
    ax.set_aspect("equal")
    plt.savefig("scripts/sphere_flow_points.png")
    plt.close()

    combined_wall_data = pr.data.CollocationPoints(coords = (jnp.concatenate([wall_x, x_immersed]), jnp.concatenate([wall_y, y_immersed])))

    return pr.data.ProblemData(equation = eq_col_data, continuity = eq_col_data, immersed = immersed_data, wall = combined_wall_data, inlet = inlet_data, outlet = outlet_data)

def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    field_dict = {
        "u": optimized_problem.u_field.coeffs.value,
        "v": optimized_problem.v_field.coeffs.value,
        "p": optimized_problem.p_field.coeffs.value,
    }
    full_data = {"fields": field_dict, "config": config_to_dict(config)}
    pr.save_dict_to_hdf5(result_path / "full_data.h5", full_data)
    save_config(config, result_path)

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    problem_data = sample_points(config)
    basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))

    if config.init == "fitted":
        fit_path = "results/09011613_sphere_flow/full_data.h5"
        fit_data = pr.load_dict_from_hdf5(fit_path)
        u_field = pr.BasisField(basis, pr.fields.PreconditionedStokesCoeffs.make_coeffs(basis.degs, fit_data["fields"]["u"].reshape(-1),x_scale=config.x_scale,y_scale=config.y_scale))
        v_field = pr.BasisField(basis, pr.fields.PreconditionedStokesCoeffs.make_coeffs(basis.degs, fit_data["fields"]["v"].reshape(-1),x_scale=config.x_scale,y_scale=config.y_scale))
        p_field = pr.BasisField(basis, pr.fields.PreconditionedStokesCoeffs.make_coeffs(basis.degs, fit_data["fields"]["p"].reshape(-1),x_scale=config.x_scale,y_scale=config.y_scale))
    elif config.precondition:
        u_field = pr.BasisField(basis, pr.fields.PreconditionedStokesCoeffs.make_zero(basis.degs,x_scale=config.x_scale,y_scale=config.y_scale))
        v_field = pr.BasisField(basis, pr.fields.PreconditionedStokesCoeffs.make_zero(basis.degs,x_scale=config.x_scale,y_scale=config.y_scale))
        p_field = pr.BasisField(basis, pr.fields.PreconditionedStokesCoeffs.make_zero(basis.degs,x_scale=config.x_scale,y_scale=config.y_scale))
    else:
        u_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
        v_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
        p_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    # r_basis = pr.BasisND([pr.basis.vectorized_cosine_basis],(1,))
    # r_field = pr.BasisField(r_basis, pr.StaticCoeffs(jnp.array([.1,0])))

    problem = StokesProblem(u_field, v_field, p_field)
    solver = pr.get_solver(config)
    
    timestamp = time.strftime("%m%d%H%M")
    config.script_name = f"{timestamp}_sphere_flow"

    print(f"Script name: {config.script_name}")

    start_time = time.time()
    problem_config = StokesConfig.from_config(config)
    optimized_problem, log_data = solver.solve(problem, problem_data, config, StokesConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    config.total_time = end_time - start_time
    save_results(config, optimized_problem)

    if log_data != {}:
        total_loss = log_data["total_loss"]
    else:
        total_loss = jnp.zeros(config.max_steps)
    wall = .205/config.x_scale
    f, ax = plt.subplots(1,3,figsize=(15,5))
    x_grid = jnp.linspace(-1, 1, 100)
    y_grid = jnp.linspace(-1, 1, 100)
    data = jnp.load("data/stokes.npy",allow_pickle=True).item()
    u = data["u"]
    v = data["v"]
    vel_mag = jnp.sqrt(u**2 + v**2)
    vel_mag_max = jnp.max(vel_mag)
    x, y = data["coords"].T
    print(log_data.keys())

    x_grid, y_grid = jnp.meshgrid(x_grid, y_grid)
    # x_phys = (x_grid+1)*config.x_scale
    # y_phys = (y_grid+1)*config.y_scale
    mask = (x_grid - config.cylinder_x)**2 + (y_grid - config.cylinder_y)**2 > config.cylinder_radius**2
    u_eval = optimized_problem.u_field.evaluate(x_grid, y_grid).reshape(100,100)
    v_eval = optimized_problem.v_field.evaluate(x_grid, y_grid).reshape(100,100)
    vel_eval_mag = jnp.sqrt(u_eval**2 + v_eval**2)
    ax[0].contourf(x_grid, y_grid, jnp.sqrt(u_eval**2 + v_eval**2), levels=100)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Velocity Magnitude")
    # ax[1].tricontourf(x, y, vel_mag, levels=100, vmax=vel_mag_max, vmin=0)
    ax[1].streamplot(np.asarray(x_grid), np.asarray(y_grid), np.asarray(u_eval), np.asarray(v_eval), density=2, color='k', linewidth=0.5)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Velocity Magnitude")
    ax[2].semilogy(total_loss)
    for key, value in log_data.items():
        split_key = key.split(".")
        if split_key[0] == "unweighted_losses":
            ax[2].semilogy(value, label=split_key[1])
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(f"figures/{config.script_name}_results.png")
    plt.savefig(f"results/{config.script_name}/loss.png")
    plt.close()

