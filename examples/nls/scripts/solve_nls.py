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
import prism as pr
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

jax.config.update("jax_enable_x64", True)

class NLSConfig(ProblemConfig):
    t_scale: float
    x_scale: float
    filter_frac: float
    residual_weights: Dict[str, float]
    causal_num_chunks: int
    causal_tol: float

class NLSProblem(AbstractProblem):
    u_field: BasisField
    v_field: BasisField

    def get_residual_functions(self) -> Dict[str, Callable]:
        return {
            "equation": self.equation_residual,
            "initial_condition": self.initial_condition_residual
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: ProblemConfig) -> jax.Array:
        t_scale = config.t_scale
        x_scale = config.x_scale
        filter_frac = config.filter_frac

        eq_col_x, eq_col_y = problem_data.coords

        u_t = self.u_field.derivative(eq_col_x, eq_col_y, order=(0,1))
        u_xx = self.u_field.derivative(eq_col_x, eq_col_y, order=(2,0))
        v_t = self.v_field.derivative(eq_col_x, eq_col_y, order=(0,1))
        v_xx = self.v_field.derivative(eq_col_x, eq_col_y, order=(2,0))
        u = self.u_field.evaluate(eq_col_x, eq_col_y, filter_frac=filter_frac)
        v = self.v_field.evaluate(eq_col_x, eq_col_y, filter_frac=filter_frac)
        res_real = 1/t_scale*v_t - 0.5/x_scale**2 * u_xx - (u**2 + v**2) * u
        res_imag = 1/t_scale*u_t + 0.5/x_scale**2 * v_xx + (u**2 + v**2) * v
        return jnp.concatenate([res_real, res_imag])

    @eqx.filter_jit
    def initial_condition_residual(self, problem_data: pr.data.ReferenceData, config: ProblemConfig) -> jax.Array:
        ic_col_x, ic_col_y = problem_data.coords
        initial_data = problem_data.data
        u_ref = initial_data[:,0]
        v_ref = initial_data[:,1]
        u_initial = self.u_field.evaluate(ic_col_x, ic_col_y)
        v_initial = self.v_field.evaluate(ic_col_x, ic_col_y)
        return jnp.concatenate([u_initial - u_ref, v_initial - v_ref])

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
    timestamp = time.strftime("%m%d%H%M")
    config["timestamp"] = timestamp   
    if args.N is not None:
        config["basis_Nx"] = args.N
        config["basis_Nt"] = args.N
        script_name = f"{config.basis_Nx}_N_solve_nls"
    else:
        script_name = f"{timestamp}_solve_nls"
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
    x_scale = config.x_scale
    initial_u = 2/jnp.cosh(x_scale*x)
    initial_v = jnp.zeros_like(x)
    return initial_u, initial_v

def initialize_data(config: SystemConfig):
    x_vec = jnp.linspace(-1,1,config.nx)
    t_vec = jnp.linspace(-1,1,config.nt)
    x_grid, t_grid = jnp.meshgrid(x_vec, t_vec)

    grid_data = pr.data.CollocationPoints(coords=(x_grid, t_grid))

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    eq_col_x, eq_col_y = jax.random.uniform(subkey, (2,config.n_pde), minval=-1, maxval=1)
    # eq_col_x = 2*jax.random.beta(subkey, a=2, b=2, shape=(config.n_pde,)) - 1
    # key, subkey = jax.random.split(key)
    # eq_col_y = jax.random.uniform(subkey, (config.n_pde,), minval=-1, maxval=1)
    # k_x = jnp.arange(config.basis_Nx + 1)
    # x_nodes = jnp.cos(k_x * jnp.pi / config.basis_Nx)

    # # 2. Generate the 1D points for the y-axis (Ny + 1 points)
    # k_t = jnp.arange(config.basis_Nt + 1)
    # t_nodes = jnp.cos(k_t * jnp.pi / config.basis_Nt)

    # # 3. Create the 2D grid using a tensor product
    # X, Y = jnp.meshgrid(x_nodes, t_nodes)
    # eq_col_x, eq_col_y = X.reshape(-1), Y.reshape(-1)

    sort_indices = jnp.argsort(eq_col_y)
    eq_col_x = eq_col_x[sort_indices]
    eq_col_y = eq_col_y[sort_indices]
    collocation_data = pr.data.CollocationPoints(coords=(eq_col_x.T, eq_col_y.T))

    key, subkey = jax.random.split(key)
    n_ic = config.n_ic
    ic_col_x, ic_col_y = jax.random.uniform(subkey, (n_ic,), minval=-1, maxval=1), -1*jnp.ones(n_ic)
    initial_u, initial_v = initial_condition_function(ic_col_x, ic_col_y, config)
    initial_data = jnp.stack([initial_u, initial_v], axis=1)
    ic_data = pr.data.ReferenceData(coords=(ic_col_x, ic_col_y), data=initial_data)

    problem_data = pr.data.ProblemData(equation=collocation_data, initial_condition=ic_data, grid=grid_data)
    return problem_data

def save_results(config, optimized_nls):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    u_coeffs, v_coeffs = optimized_nls.u_field.coeffs.value, optimized_nls.v_field.coeffs.value
    field_dict = {
        "u": u_coeffs,
        "v": v_coeffs
    }
    full_data = {"fields": field_dict, "config": config_to_dict(config)}
    save_dict_to_hdf5(result_path / "full_data.h5", full_data)
    save_config(config, result_path)

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    config = parse_args(config)
    problem_data = initialize_data(config)

    grid_data = problem_data.data["grid"]
    initialize_u, initialize_v = initial_condition_function(*grid_data.coords, config)

    basis = CosineChebyshevBasis2D((config.basis_Nx, config.basis_Nt))
    u_field = fit_basis_field_from_data(basis, grid_data.coords, initialize_u, trainable=config.trainable, precondition=config.precondition)
    u_coeffs = u_field.coeffs.value
    u_coeffs = u_coeffs.reshape(config.basis_Nx+1, config.basis_Nt+1)
    u_field = BasisField(basis, pr.fields.PreconditionedNLS2D.make_coeffs(basis.degs, u_coeffs))
    if config.precondition:
        v_field = BasisField(basis, pr.fields.PreconditionedNLS2D.make_zero(basis.degs))
    else:
        v_field = BasisField(basis, Coeffs.make_zero(basis.degs))
    problem = NLSProblem(u_field, v_field)


    initial_u = u_field.evaluate(*grid_data.coords).reshape(config.nt, config.nx)
    initial_v = v_field.evaluate(*grid_data.coords).reshape(config.nt, config.nx)
    initial_h = jnp.sqrt(initial_u**2 + initial_v**2)

    solver = pr.get_solver(config)

    

    start_time = time.time()
    problem_config = NLSConfig.from_config(config)
    optimized_nls, log_data = solver.solve(problem, problem_data, config, problem_config)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    config["time_taken"] = end_time - start_time
    save_results(config, optimized_nls)

    total_loss = jnp.array(log_data.get("total_loss", []))
    final_points = log_data.get("final_collocation_points", None)
    if final_points is None:
        final_points = problem_data["equation"].coords
    u_eval = optimized_nls.u_field.evaluate(*grid_data.coords)
    u_eval = u_eval.reshape(config.nt, config.nx)
    v_eval = optimized_nls.v_field.evaluate(*grid_data.coords)
    v_eval = v_eval.reshape(config.nt, config.nx)
    h_eval = jnp.sqrt(u_eval**2 + v_eval**2)

    NLS_data = loadmat(config["data_path"])
    Exact = NLS_data['uu']
    Exact_u = jnp.real(Exact).T
    Exact_v = jnp.imag(Exact).T
    Exact_h = jnp.sqrt(Exact_u**2 + Exact_v**2)
    h_max = jnp.max(jnp.abs(Exact_h))

    dx =(2*config.x_scale)/config.nx
    dt =(2*config.t_scale)/config.nt
    da = dx*dt
    h_l2_error = jnp.sqrt(jnp.nansum(jnp.power(h_eval - Exact_h, 2))*da)
    h_linf_error = jnp.nanmax(jnp.abs(h_eval - Exact_h))
    u_l2_error = jnp.sqrt(jnp.nansum(jnp.power(u_eval - Exact_u, 2))*da)
    u_linf_error = jnp.nanmax(jnp.abs(u_eval - Exact_u))
    v_l2_error = jnp.sqrt(jnp.nansum(jnp.power(v_eval - Exact_v, 2))*da)
    v_linf_error = jnp.nanmax(jnp.abs(v_eval - Exact_v))
    print(f"L2 error: h: {h_l2_error}, u: {u_l2_error}, v: {v_l2_error}")
    print(f"Linf error: h: {h_linf_error}, u: {u_linf_error}, v: {v_linf_error}")

    save_fig_path = pathlib.Path("figures/")
    save_fig_path.mkdir(parents=True, exist_ok=True)
    fig_name = f"{config.timestamp}_nls_solution.png"
    f, ax = plt.subplots(1,4,figsize=(20,5))
    ax[0].set_title("Target Solution")
    ax[1].set_title("Optimized Solution")
    solver_type = config.get("solver_type", "")
    optimizer_name = config.get("optimizer_name", "")
    loss_strategy = config.get("loss_strategy", "")
    solver = f"{solver_type}_{optimizer_name}_{loss_strategy}"
    ax[3].set_title(f"{solver} Loss")
    im0 = ax[0].contourf(*grid_data.coords, Exact_h, cmap="jet", vmin = 0, vmax = h_max, levels=100)
    im1 = ax[1].contourf(*grid_data.coords, h_eval, cmap="jet", vmin = 0, vmax = h_max, levels=100)
    ax[2].scatter(*final_points, marker=".", color="red",zorder=10)
    im2 =ax[2].contourf(*grid_data.coords, initial_h, cmap="jet", vmin = 0, vmax = h_max, levels=100,zorder=5)
    im3 = ax[3].semilogy(total_loss)
    f.colorbar(im0,ax=ax[0])
    f.colorbar(im1,ax=ax[1])
    f.colorbar(im2,ax=ax[2])
    plt.tight_layout()
    plt.savefig(save_fig_path / fig_name)
    plt.close()