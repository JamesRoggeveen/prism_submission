import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import matplotlib.pyplot as plt
# jax.config.update("jax_enable_x64", True)

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
            "inlet": self.inlet_residual,
            "outlet": self.outlet_residual,
            "continuity": self.continuity_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        grad_p_x = self.p_field.derivative(*problem_data.coords, order=(1,0))
        grad_p_y = self.p_field.derivative(*problem_data.coords, order=(0,1))

        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0))
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2))

        v_xx = self.v_field.derivative(*problem_data.coords, order=(2,0))
        v_yy = self.v_field.derivative(*problem_data.coords, order=(0,2))


        x_eq = (u_xx + u_yy) - grad_p_x
        y_eq = (v_xx + v_yy) - grad_p_y

        full_residual = jnp.concatenate([x_eq, y_eq], axis=0)
        return full_residual
    
    @eqx.filter_jit
    def continuity_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0))
        v_y = self.v_field.derivative(*problem_data.coords, order=(0,1))
        continuity = u_x + v_y
        return continuity

    @eqx.filter_jit
    def wall_boundary_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        u = self.u_field.evaluate(*problem_data.coords)
        v = self.v_field.evaluate(*problem_data.coords)
        full_residual = jnp.concatenate([u, v ], axis=0)
        return full_residual

    @eqx.filter_jit
    def inlet_residual(self, problem_data: pr.data.ReferenceData, config: StokesConfig) -> jax.Array:
        u = self.u_field.evaluate(*problem_data.coords)
        v = self.v_field.evaluate(*problem_data.coords)
        u_ref, v_ref = problem_data.data
        full_residual = jnp.concatenate([u - u_ref, v - v_ref], axis=0)
        return full_residual

    @eqx.filter_jit
    def outlet_residual(self, problem_data: pr.data.ReferenceData, config: StokesConfig) -> jax.Array:
        coords = problem_data.coords
        p = self.p_field.evaluate(*coords)
        v_x = self.v_field.derivative(*coords, order=(1, 0))
        u_y = self.u_field.derivative(*coords, order=(0, 1))
        full_residual = jnp.concatenate([p, (v_x + u_y)], axis=0)
        return full_residual

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config["solver_kwargs"] = {key: float(value) for key, value in config["solver_kwargs"].items()}
    if config["verbose"]:
        config["solver_kwargs"]["verbose"] = frozenset({"loss", "step_size"})
    seed = int(time.time())
    config["seed"] = seed
    config["x_scale"] = 1.0
    config["y_scale"] = 1.0
    return pr.SystemConfig(**config)

def save_config(config, result_path):
    with open(result_path / "config.yml", "w") as f:
        yaml.dump(config_to_dict(config), f)

def config_to_dict(config):
    output_dict = config.data.copy()
    solver_kwargs_copy = {k: v for k, v in output_dict["solver_kwargs"].items() if k != "verbose"}
    output_dict["solver_kwargs"] = solver_kwargs_copy
    return output_dict

def inlet_condition(x,y,wall_y_val,vel_scale):
    return -vel_scale/(wall_y_val**2)*(y-wall_y_val)*(y+wall_y_val),jnp.zeros_like(y)

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_immersed = config.n_immersed
    key = jax.random.PRNGKey(config.seed)

    wall_y_val = .5
    vel_scale = 1.0
    n_grid = 300
    grid_x, grid_y = jnp.meshgrid(jnp.linspace(-1, 1, n_grid), jnp.linspace(-wall_y_val, wall_y_val, n_grid))

    cylinder_x, cylinder_y = 0.0, 0.0
    cylinder_radius = 0.1
    cylinder_2_x, cylinder_2_y = -.5,.2
    cylinder_2_radius = 0.075

    mask = (grid_x - cylinder_x)**2 + (grid_y - cylinder_y)**2 <= cylinder_radius**2
    mask2 = (grid_x - cylinder_2_x)**2 + (grid_y - cylinder_2_y)**2 <= cylinder_2_radius**2
    mask = mask | mask2
    grid_data = pr.data.ReferenceData(coords = (grid_x, grid_y),data=mask)

    key, subkey = jax.random.split(key)
    eq_col_x = jax.random.uniform(subkey, (n_pde,), minval=-1, maxval=1)
    key, subkey = jax.random.split(key)
    eq_col_y = jax.random.uniform(subkey, (n_pde,), minval=-wall_y_val, maxval=wall_y_val)
    mask_col_1 = (eq_col_x - cylinder_x)**2 + (eq_col_y - cylinder_y)**2 <= cylinder_radius**2
    mask_col_2 = (eq_col_x - cylinder_2_x)**2 + (eq_col_y - cylinder_2_y)**2 <= cylinder_2_radius**2
    mask_col = mask_col_1 | mask_col_2
    eq_col_x = eq_col_x[~mask_col]
    eq_col_y = eq_col_y[~mask_col]

    n_annulus = config.n_annulus
    annulus_r = config.annulus_r
    key, subkey = jax.random.split(key)
    r_samples_1 = jax.random.uniform(subkey, (n_annulus,), minval=cylinder_radius, maxval=annulus_r*cylinder_radius)
    key, subkey = jax.random.split(key)
    theta_samples = jax.random.uniform(subkey, (n_annulus,), minval=0, maxval=2*jnp.pi)
    key, subkey = jax.random.split(key)
    r_samples_2 = jax.random.uniform(subkey, (n_annulus,), minval=cylinder_2_radius, maxval=annulus_r*cylinder_2_radius)
    annulus_x_1 = cylinder_x + r_samples_1*jnp.cos(theta_samples)
    annulus_y_1 = cylinder_y + r_samples_1*jnp.sin(theta_samples)
    annulus_x_2 = cylinder_2_x + r_samples_2*jnp.cos(theta_samples)
    annulus_y_2 = cylinder_2_y + r_samples_2*jnp.sin(theta_samples)
    annulus_x = jnp.concatenate([annulus_x_1, annulus_x_2])
    annulus_y = jnp.concatenate([annulus_y_1, annulus_y_2])
    # eq_col_x = jnp.concatenate([eq_col_x, annulus_x])
    # eq_col_y = jnp.concatenate([eq_col_y, annulus_y])

    eq_col_data = pr.data.CollocationPoints(coords = (eq_col_x, eq_col_y))

    key, subkey = jax.random.split(key)
    wall_x = jax.random.uniform(subkey, (2*n_bc,), minval=-1, maxval=1)
    wall_y = jnp.concatenate([jnp.ones((n_bc,))*wall_y_val,jnp.ones((n_bc,))*-wall_y_val])
    wall_data = pr.data.CollocationPoints(coords = (wall_x, wall_y))

    key, subkey = jax.random.split(key)
    theta_immersed = jnp.linspace(0, 2*jnp.pi, n_immersed)
    x_immersed = cylinder_x + cylinder_radius*jnp.cos(theta_immersed)
    y_immersed = cylinder_y + cylinder_radius*jnp.sin(theta_immersed)
    x_immersed_2 = cylinder_2_x + cylinder_2_radius*jnp.cos(theta_immersed)
    y_immersed_2 = cylinder_2_y + cylinder_2_radius*jnp.sin(theta_immersed)
    x_immersed = jnp.concatenate([x_immersed, x_immersed_2])
    y_immersed = jnp.concatenate([y_immersed, y_immersed_2])
    wall_data = pr.data.CollocationPoints(coords = (jnp.concatenate([wall_x, x_immersed]), jnp.concatenate([wall_y, y_immersed])))

    key, subkey = jax.random.split(key)
    inlet_y = jax.random.uniform(subkey, (n_bc,), minval=-wall_y_val, maxval=wall_y_val)
    inlet_x = -1*jnp.ones_like(inlet_y)
    inlet_vals = inlet_condition(inlet_x, inlet_y, wall_y_val, vel_scale)
    inlet_data = pr.data.ReferenceData(coords=(inlet_x, inlet_y), data=inlet_vals)


    key, subkey = jax.random.split(key)
    outlet_y = jax.random.uniform(subkey, (n_bc,), minval=-wall_y_val, maxval=wall_y_val)
    outlet_x = jnp.ones_like(outlet_y)
    outlet_data = pr.data.CollocationPoints(coords=(outlet_x, outlet_y))

    f, ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(*eq_col_data.coords, s = 1.0, marker=".", zorder = 1)
    ax.scatter(*wall_data.coords, s = 1.0, c = "red", marker=".", zorder = 2)
    ax.scatter(*inlet_data.coords, s = 1.0, c = "blue", marker=".", zorder = 3)
    ax.scatter(*outlet_data.coords, s = 1.0, c = "green", marker=".", zorder = 4)
    plt.savefig("scripts/sphere_flow_points.png",dpi=300)
    plt.close()

    return pr.data.ProblemData(equation = eq_col_data, continuity = eq_col_data, wall = wall_data, inlet = inlet_data, outlet = outlet_data, grid = grid_data)

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
    config = load_config("configs/two_sphere.yml")
    problem_data = sample_points(config)
    basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    config["basis_type"] = "chebyshev"

    if config.init == "fitted":
        fit_path = "data/reference_flow.csv"
        fit_data = np.loadtxt(fit_path, delimiter=",", skiprows=9)
        x = fit_data[:,0]
        y = fit_data[:,1]
        u = fit_data[:,2]
        v = fit_data[:,3]
        p = fit_data[:,4]
        u_field = pr.fit_basis_field_from_data(basis, (x, y), u, precondition=config.precondition)
        v_field = pr.fit_basis_field_from_data(basis, (x, y), v, precondition=config.precondition)
        p_field = pr.fit_basis_field_from_data(basis, (x, y), p, precondition=config.precondition)
        grid_x, grid_y = problem_data["grid"].coords
        u_eval = u_field.evaluate(grid_x, grid_y)
        v_eval = v_field.evaluate(grid_x, grid_y)
        p_eval = p_field.evaluate(grid_x, grid_y)
        u_eval = u_eval.reshape(grid_x.shape)
        v_eval = v_eval.reshape(grid_x.shape)
        p_eval = p_eval.reshape(grid_x.shape)
        u_eval = jnp.where(problem_data["grid"].data == 0, u_eval, jnp.nan)
        v_eval = jnp.where(problem_data["grid"].data == 0, v_eval, jnp.nan)
        p_eval = jnp.where(problem_data["grid"].data == 0, p_eval, jnp.nan)
        f, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].contourf(grid_x, grid_y, u_eval, levels=100)
        ax[1].contourf(grid_x, grid_y, v_eval, levels=100)
        ax[2].contourf(grid_x, grid_y, p_eval, levels=100)
        plt.show()
    elif config.precondition:
        u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
        v_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
        p_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    else:
        u_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
        v_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
        p_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))

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
    f, ax = plt.subplots(2,2,figsize=(12,12))
    ax = ax.flatten()
    for key, value in log_data.items():
        if "." in key:
            key_pre, key_suf = key.split(".")
            if key_pre == "unweighted_losses":
                ax[0].semilogy(value,label=key_suf)
            elif key_pre == "weights":
                ax[1].semilogy(value,label=key_suf)
            elif key_pre == "grad_mags":
                ax[2].semilogy(value,label=key_suf)
        else:
            ax[3].semilogy(value,label=key)
    ax[0].set_title("Unweighted Losses")
    ax[1].set_title("Weights")
    ax[2].set_title("Grad Magnitudes")
    ax[3].set_title("Total Loss")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(f"results/{config.script_name}/losses.png")
    plt.close()

    col_x, col_y = problem_data["equation"].coords

    x_grid, y_grid = problem_data["grid"].coords
    cmap = plt.get_cmap('rainbow')
    u_eval = optimized_problem.u_field.evaluate(x_grid, y_grid)
    v_eval = optimized_problem.v_field.evaluate(x_grid, y_grid)
    p_eval = optimized_problem.p_field.evaluate(x_grid, y_grid)
    print(u_eval.shape, v_eval.shape, p_eval.shape)
    print(x_grid.shape, y_grid.shape)

    f, ax = plt.subplots(1,4,figsize=(15,5))
    ax = ax.flatten()
    names = ["u_eval", "v_eval", "p_eval", "u", "v", "p"]
    u_eval = u_eval.reshape(x_grid.shape)
    v_eval = v_eval.reshape(x_grid.shape)
    p_eval = p_eval.reshape(x_grid.shape)
    mask = problem_data["grid"].data
    u_eval = jnp.where(mask, jnp.nan, u_eval)
    v_eval = jnp.where(mask, jnp.nan, v_eval)
    p_eval = jnp.where(mask, jnp.nan, p_eval)
    v_eval_mag = jnp.sqrt(u_eval**2 + v_eval**2)
    im1 = ax[0].contourf(x_grid, y_grid, u_eval, levels=100,cmap=cmap)
    im2 = ax[1].contourf(x_grid, y_grid, v_eval, levels=100,cmap=cmap)
    im3 = ax[2].contourf(x_grid, y_grid, p_eval, levels=100,cmap=cmap)
    im4 = ax[3].contourf(x_grid, y_grid, v_eval_mag, levels=100,cmap="jet")
    cbar = f.colorbar(im1, ax=ax[0])
    cbar = f.colorbar(im2, ax=ax[1])
    cbar = f.colorbar(im3, ax=ax[2])
    cbar = f.colorbar(im4, ax=ax[3])
    for i in range(4):
        ax[i].set_aspect("equal")
        ax[i].set_xlabel("x")
        ax[i].set_ylabel("y")
    plt.tight_layout()
    plt.savefig(f"figures/{config.script_name}_results_full.png")
    plt.savefig(f"results/{config.script_name}/results_full.png")
    plt.close()