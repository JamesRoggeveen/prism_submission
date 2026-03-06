import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import matplotlib.pyplot as plt

class StokesConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    x_scale: float
    y_scale: float
    nu: float
    regularization_strength: float

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
            "continuity": self.continuity_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        x_scale = config.x_scale
        y_scale = config.y_scale
        nu = config.nu

        grad_p_x = self.p_field.derivative(*problem_data.coords, order=(1,0))/x_scale
        grad_p_y = self.p_field.derivative(*problem_data.coords, order=(0,1))/y_scale

        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0))/x_scale**2
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2))/y_scale**2

        v_xx = self.v_field.derivative(*problem_data.coords, order=(2,0))/x_scale**2
        v_yy = self.v_field.derivative(*problem_data.coords, order=(0,2))/y_scale**2


        x_eq = nu*(u_xx + u_yy) - grad_p_x
        y_eq = nu*(v_xx + v_yy) - grad_p_y
        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0))/x_scale
        v_y = self.v_field.derivative(*problem_data.coords, order=(0,1))/y_scale
        continuity = u_x + v_y
        
        return jnp.concatenate([x_eq, y_eq], axis=0)
    
    @eqx.filter_jit
    def continuity_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        x_scale = config.x_scale
        y_scale = config.y_scale
        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0))/x_scale
        v_y = self.v_field.derivative(*problem_data.coords, order=(0,1))/y_scale
        continuity = u_x + v_y
        return continuity

    @eqx.filter_jit
    def immersed_boundary_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        coords = problem_data.coords
        u = self.u_field.evaluate(*coords)
        v = self.v_field.evaluate(*coords)
        return jnp.concatenate([u,v], axis=0)
 
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
        x_scale = config.x_scale
        nu = config.nu
        # Evaluate pressure at the outlet
        p = self.p_field.evaluate(*coords)
        u_x = self.u_field.derivative(*coords, order=(1, 0))/x_scale
        v_x = self.v_field.derivative(*coords, order=(1, 0))/x_scale
        return jnp.concatenate([nu*u_x-p, nu*v_x], axis=0)

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
    # return -(1-y**2)**2,jnp.zeros_like(y)
    # return -jnp.cos(jnp.pi*y/2),jnp.zeros_like(y)
    return -0.3*(y-1)*(y+1),jnp.zeros_like(y)

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_immersed = config.n_immersed
    key = jax.random.PRNGKey(config.seed)

    x_scale = config.x_scale
    y_scale = config.y_scale
    data = jnp.load("data/stokes.npy",allow_pickle=True).item()
    eq_col_x, eq_col_y = data["coords"].T
    key, subkey = jax.random.split(key)
    eq_col_x, eq_col_y = jax.random.uniform(subkey, (2, n_pde), minval=0, maxval=1)
    eq_col_x = eq_col_x*x_scale*2
    eq_col_y = eq_col_y*y_scale*2
    cylinder_x, cylinder_y = 0.2, 0.2
    cylinder_radius = 0.05
    mask = (eq_col_x - cylinder_x)**2 + (eq_col_y - cylinder_y)**2 <= cylinder_radius**2
    eq_col_x = eq_col_x/x_scale - 1
    eq_col_y = eq_col_y/y_scale - 1
    eq_col_x = eq_col_x[~mask]
    eq_col_y = eq_col_y[~mask]

    # data = jnp.load("data/stokes.npy",allow_pickle=True).item()
    # x, y = data["coords"].T
    # x = x/x_scale - 1
    # y = y/y_scale - 1
    # eq_col_x = x
    # eq_col_y = y

    eq_col_data = pr.data.CollocationPoints(coords = (eq_col_x, eq_col_y))

    # eq_col_x, eq_col_y = jax.random.uniform(subkey, (2, n_pde), minval=-1, maxval=1)
    # eq_col_data = pr.data.CollocationPoints(coords = (eq_col_x, eq_col_y))

    key, subkey = jax.random.split(key)
    wall_x = jax.random.uniform(subkey, (2*n_bc,), minval=-1, maxval=1)
    wall_y = jnp.concatenate([jnp.ones((n_bc,))*1,jnp.ones((n_bc,))*-1])
    wall_data = pr.data.CollocationPoints(coords = (wall_x, wall_y))

    key, subkey = jax.random.split(key)
    # theta_immersed = jax.random.uniform(subkey, (n_immersed,), minval=0, maxval=2*jnp.pi)
    theta_immersed = jnp.linspace(0, 2*jnp.pi, n_immersed)
    x_immersed = cylinder_x + cylinder_radius*jnp.cos(theta_immersed)
    y_immersed = cylinder_y + cylinder_radius*jnp.sin(theta_immersed)
    x_immersed = x_immersed/x_scale - 1
    y_immersed = y_immersed/y_scale - 1
    immersed_data = pr.data.CollocationPoints(coords=(x_immersed, y_immersed))

    wall_data = pr.data.CollocationPoints(coords = (jnp.concatenate([wall_x, x_immersed]), jnp.concatenate([wall_y, y_immersed])))

    key, subkey = jax.random.split(key)
    inlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    inlet_x = -1*jnp.ones_like(inlet_y)
    inlet_data = pr.data.ReferenceData(coords=(inlet_x, inlet_y), data=inlet_condition(inlet_x, inlet_y))

    key, subkey = jax.random.split(key)
    outlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    outlet_x = jnp.ones_like(outlet_y)
    outlet_data = pr.data.ReferenceData(coords=(outlet_x, outlet_y), data=inlet_condition(outlet_x, outlet_y))

    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(*eq_col_data.coords, s = 1.0, marker=".", zorder = 1)
    ax.scatter(*immersed_data.coords, s = 1.0, c = "black", marker=".", zorder = 2)
    ax.scatter(*wall_data.coords, s = 1.0, c = "red", marker=".", zorder = 2)
    ax.scatter(*inlet_data.coords, s = 1.0, c = "blue", marker=".", zorder = 3)
    ax.scatter(*outlet_data.coords, s = 1.0, c = "green", marker=".", zorder = 4)
    plt.savefig("scripts/sphere_flow_points.png",dpi=300)
    plt.close()

    return pr.data.ProblemData(equation = eq_col_data, continuity = eq_col_data, immersed = immersed_data, wall = wall_data, inlet = inlet_data, outlet = outlet_data)

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
    config = load_config("configs/sphere_flow.yml")
    problem_data = sample_points(config)
    data = jnp.load("data/stokes.npy",allow_pickle=True).item()
    x, y = data["coords"].T
    x = x/config.x_scale - 1
    y = y/config.y_scale - 1
    # problem_data["equation"] = pr.data.CollocationPoints(coords=(x, y))
    # problem_data["continuity"] = pr.data.CollocationPoints(coords=(x, y))

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
    
    x_grid = jnp.linspace(-1, 1, 1000)
    y_grid = jnp.linspace(-1, 1, 1000)
    data = jnp.load("data/stokes.npy",allow_pickle=True).item()
    u = data["u"]
    v = data["v"]
    p = data["p"]
    vel_mag = jnp.sqrt(u**2 + v**2)
    vel_mag_max = jnp.max(vel_mag)
    x, y = data["coords"].T

    col_x, col_y = problem_data["equation"].coords

    x_grid, y_grid = jnp.meshgrid(x_grid, y_grid)
    x_phys = (x_grid+1)*config.x_scale
    y_phys = (y_grid+1)*config.y_scale
    col_x_phys = (col_x+1)*config.x_scale
    col_y_phys = (col_y+1)*config.y_scale
    cylinder_x, cylinder_y = 0.2, 0.2
    cylinder_radius = 0.05
    mask = (x_grid - cylinder_x)**2 + (y_grid - cylinder_y)**2 > cylinder_radius**2
    x_comp = x/config.x_scale - 1
    y_comp = y/config.y_scale - 1
    u_eval = optimized_problem.u_field.evaluate(x_comp, y_comp)
    v_eval = optimized_problem.v_field.evaluate(x_comp, y_comp)
    p_eval = optimized_problem.p_field.evaluate(x_comp, y_comp)
    # u_eval = jnp.where(mask, u_eval, jnp.nan)
    # v_eval = jnp.where(mask, v_eval, jnp.nan)
    # u = jnp.where(mask, u, jnp.nan)
    # v = jnp.where(mask, v, jnp.nan)
    vel_eval_mag = jnp.sqrt(u_eval**2 + v_eval**2)

    error = jnp.linalg.norm(vel_eval_mag - vel_mag)/jnp.linalg.norm(vel_mag)
    print(f"L2 relative error: {error}")
    u_min, u_max = jnp.min(u), jnp.max(u)
    v_min, v_max = jnp.min(v), jnp.max(v)
    p_min, p_max = jnp.min(p), jnp.max(p)
    f, ax = plt.subplots(2,3,figsize=(15,5))
    ax = ax.flatten()
    circle = [plt.Circle((cylinder_x, cylinder_y), cylinder_radius, color='white', fill=True) for _ in range(6)]
    names = ["u_eval", "v_eval", "p_eval", "u", "v", "p"]
    ranges = [(jnp.min(u_eval), jnp.max(u_eval)), (jnp.min(v_eval), jnp.max(v_eval)), (jnp.min(p_eval), jnp.max(p_eval)), (jnp.min(u), jnp.max(u)), (jnp.min(v), jnp.max(v)), (jnp.min(p), jnp.max(p))]
    ax[0].tricontourf(x, y, u_eval, cmap="jet", levels=100, vmax=u_max, vmin=u_min)
    # ax[0].scatter(col_x_phys, col_y_phys, marker=".", c="red", s=.1)
    ax[1].tricontourf(x, y, v_eval, cmap="jet", levels=100, vmax=v_max, vmin=v_min)
    ax[2].tricontourf(x, y, p_eval, cmap="jet", levels=100)
    ax[3].tricontourf(x, y, u, cmap="jet", levels=100, vmax=u_max, vmin=u_min)
    ax[4].tricontourf(x, y, v, cmap="jet", levels=100, vmax=v_max, vmin=v_min)
    ax[5].tricontourf(x, y, p, cmap="jet", levels=100, vmax=p_max, vmin=p_min)
    for i in range(6):
        ax[i].add_patch(circle[i])
        ax[i].set_aspect("equal")
        ax[i].set_xlabel("x")
        ax[i].set_ylabel("y")
        ax[i].set_title(f"{names[i]}, min: {ranges[i][0]:.2f}, max: {ranges[i][1]:.2f}")
    plt.tight_layout()
    plt.savefig(f"figures/{config.script_name}_results_full.png")
    plt.savefig(f"results/{config.script_name}/results_full.png")
    plt.close()

    f, ax = plt.subplots(1,2,figsize=(10,2.5))
    circle = plt.Circle((cylinder_x, cylinder_y), cylinder_radius, color='white', fill=True)
    ax[0].tricontourf(x, y, vel_eval_mag, levels=100, vmax=vel_mag_max, vmin=0)
    ax[0].add_patch(circle)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title(f"Velocity Magnitude, error: {error:.2f}")
    ax[1].tricontourf(x, y, vel_mag, levels=100, vmax=vel_mag_max, vmin=0)
    circle2 = plt.Circle((cylinder_x, cylinder_y), cylinder_radius, color='white', fill=True)
    ax[1].add_patch(circle2)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Reference Velocity Magnitude")
    plt.tight_layout()
    plt.savefig(f"figures/{config.script_name}_results.png")
    plt.savefig(f"results/{config.script_name}/loss.png")
    plt.close()

