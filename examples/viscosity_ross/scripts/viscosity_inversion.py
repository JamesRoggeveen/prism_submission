import jax
import jax.numpy as jnp
import equinox as eqx
import time
import yaml
import pathlib
from typing import Dict, Callable

from prism import (
    ChebyshevBasis2D,
    BasisField,
    fit_basis_field_from_data,
    sample_from_mask,
    load_dict_from_hdf5,
    save_dict_to_hdf5,
    ProblemData,
    SystemConfig,
    AbstractProblem,
    Coeffs,
    LogBasisField,
    ProblemConfig,
)

import prism as pr

class SSAConfig(ProblemConfig):
    filter_frac: float
    aspect_ratio: float
    residual_weights: Dict[str, float]

class SSAProblem(AbstractProblem):
    h_field: BasisField
    u_field: BasisField
    v_field: BasisField
    mu_field: BasisField

    def get_residual_functions(self) -> Dict[str, Callable]:
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
            "reference_data": self.data_residual
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: ProblemConfig) -> jax.Array:
        filter_frac = config.filter_frac
        r = config.aspect_ratio

        eq_col_x, eq_col_y = problem_data.coords

        mu = self.mu_field.evaluate(eq_col_x, eq_col_y, filter_frac=filter_frac)
        h = self.h_field.evaluate(eq_col_x, eq_col_y, filter_frac=filter_frac)
        mu_x = self.mu_field.derivative(eq_col_x, eq_col_y, order=(1,0), filter_frac=filter_frac)
        mu_y = self.mu_field.derivative(eq_col_x, eq_col_y, order=(0,1), filter_frac=filter_frac)

        h_x = self.h_field.derivative(eq_col_x, eq_col_y, order=(1,0), filter_frac=filter_frac)
        h_y = self.h_field.derivative(eq_col_x, eq_col_y, order=(0,1), filter_frac=filter_frac)
        u_x = self.u_field.derivative(eq_col_x, eq_col_y, order=(1,0), filter_frac=filter_frac)
        u_y = self.u_field.derivative(eq_col_x, eq_col_y, order=(0,1), filter_frac=filter_frac)
        v_x = self.v_field.derivative(eq_col_x, eq_col_y, order=(1,0), filter_frac=filter_frac)
        v_y = self.v_field.derivative(eq_col_x, eq_col_y, order=(0,1), filter_frac=filter_frac)
        u_xx = self.u_field.derivative(eq_col_x, eq_col_y, order=(2,0), filter_frac=filter_frac)
        u_yy = self.u_field.derivative(eq_col_x, eq_col_y, order=(0,2), filter_frac=filter_frac)
        u_xy = self.u_field.derivative(eq_col_x, eq_col_y, order=(1,1), filter_frac=filter_frac)
        v_xx = self.v_field.derivative(eq_col_x, eq_col_y, order=(2,0), filter_frac=filter_frac)
        v_yy = self.v_field.derivative(eq_col_x, eq_col_y, order=(0,2), filter_frac=filter_frac)
        v_xy = self.v_field.derivative(eq_col_x, eq_col_y, order=(1,1), filter_frac=filter_frac)
        T_xx_x = 2*((mu_x*h + mu*h_x)*(2*u_x + r*v_y) + mu*h*(2*u_xx + r*v_xy))
        T_yy_y = 2*r*((mu_y*h + mu*h_y)*(2*r*v_y + u_x) + mu*h*(2*r*v_yy + u_xy))
        T_xy_x = (mu_x*h + mu*h_x)*(r*u_y + v_x) + mu*h*(r*u_xy + v_xx)
        T_xy_y = r*((mu_y*h + mu*h_y)*(r*u_y + v_x) + mu*h*(r*u_yy + v_xy))
        rhs_1 = h*h_x
        rhs_2 = r*h*h_y

        f1 = T_xx_x + T_xy_y - rhs_1
        f2 = T_xy_x + T_yy_y - rhs_2
        return jnp.concatenate([f1, f2])

    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.BoundaryData, config: ProblemConfig) -> jax.Array:
        filter_frac = config.filter_frac
        r = config.aspect_ratio
        bc_col_x, bc_col_y = problem_data.coords
        n_x, n_y = problem_data.normal_vector
        u_x = self.u_field.derivative(bc_col_x, bc_col_y, order=(1,0), filter_frac=filter_frac)
        u_y = self.u_field.derivative(bc_col_x, bc_col_y, order=(0,1), filter_frac=filter_frac)
        v_x = self.v_field.derivative(bc_col_x, bc_col_y, order=(1,0), filter_frac=filter_frac)
        v_y = self.v_field.derivative(bc_col_x, bc_col_y, order=(0,1), filter_frac=filter_frac)
        h = self.h_field.evaluate(bc_col_x, bc_col_y, filter_frac=filter_frac)
        mu = self.mu_field.evaluate(bc_col_x, bc_col_y, filter_frac=filter_frac)
        # Add factor of r to account for coordinate transformation of normal vector, only ratio of normal vector components is relevant
        f3 = 2*mu*(2*u_x + r*v_y)*n_x + mu*(r*u_y + v_x)*n_y - 1/2 * h * n_x
        f4 = 2*mu*(2*r*v_y + u_x)*n_y + mu*(r*u_y + v_x)*n_x - 1/2 * h * n_y
        return jnp.concatenate([f3, f4])

    @eqx.filter_jit
    def data_residual(self, reference_data, config: ProblemConfig) -> jax.Array:
        h_data = reference_data["h"]
        u_data = reference_data["u"]
        v_data = reference_data["v"]
        h_data_x, h_data_y = h_data.coords
        u_data_x, u_data_y = u_data.coords
        v_data_x, v_data_y = v_data.coords
        h = self.h_field.evaluate(h_data_x, h_data_y)
        u = self.u_field.evaluate(u_data_x, u_data_y)
        v = self.v_field.evaluate(v_data_x, v_data_y)
        f5 = h - h_data.data
        f6 = u - u_data.data
        f7 = v - v_data.data
        return jnp.concatenate([f5, f6, f7])

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config["solver_kwargs"] = {key: float(value) for key, value in config["solver_kwargs"].items()}
    if config["verbose"]:
        config["solver_kwargs"]["verbose"] = frozenset({"loss", "step_size"})
    seed = int(time.time())
    config["seed"] = seed
    return SystemConfig(**config)

def save_config(config, result_path):
    with open(result_path / "config.yml", "w") as f:
        yaml.dump(config_to_dict(config), f)

def config_to_dict(self):
    output_dict = self.data.copy()
    solver_kwargs_copy = {k: v for k, v in output_dict["solver_kwargs"].items() if k != "verbose"}
    output_dict["solver_kwargs"] = solver_kwargs_copy
    output_dict["aspect_ratio"] = output_dict.get("aspect_ratio", 0.0)
    output_dict["aspect_ratio"] = float(output_dict["aspect_ratio"])
    output_dict["data_shape"] = list(output_dict["data_shape"])
    return output_dict

def sample_from_data_dict(data_dict, N, key):
    vals = data_dict.values()
    mask = jnp.ones(data_dict["x"].shape, dtype=bool)
    mask = mask.squeeze()
    for val in vals:
        val = val.squeeze()
        mask = mask & ~jnp.isnan(val)
    mask = mask.reshape(-1)
    n_total = jnp.sum(mask)
    indices = jax.random.choice(key, n_total, (N,), replace=False)
    sampled_data_dict = {}
    for key, val in data_dict.items():
        flat_val = val.reshape(-1)
        flat_val_masked = flat_val[mask]
        sampled_data_dict[key] = flat_val_masked[indices]
    sampled_data = pr.data.ReferenceData(coords=(sampled_data_dict["x"],sampled_data_dict["y"]), data=sampled_data_dict["data"])
    return sampled_data

def load_and_sample_data(config):
    full_data_dict = load_dict_from_hdf5(config.data_path)
    field_data = full_data_dict["fields"]
    sampled_reference_data = {}
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    for key, data_dict in field_data.items():
        subkey, subsubkey = jax.random.split(subkey)
        sampled_reference_data[key] = sample_from_data_dict(data_dict, config.n_data, subsubkey)

    u_data = field_data["u"]
    eq_col_x, eq_col_y = sample_from_mask(subkey,u_data["x"],u_data["y"],u_data["data"],config.n_pde)
    collocation_data = pr.data.CollocationPoints(coords=(eq_col_x, eq_col_y))

    bc_data = full_data_dict["bc"]
    bc_data = pr.data.BoundaryData(coords=(bc_data["x"], bc_data["y"]), normal_vector=(bc_data["nx"], bc_data["ny"]))

    scaling_dict = full_data_dict["scaling_dict"]
    config.aspect_ratio = scaling_dict["aspect_ratio"]
    config.data_shape = field_data["u"]["x"].shape
    config.scaling_dict = scaling_dict
    problem_data = ProblemData(equation=collocation_data, boundary=bc_data, reference_data=sampled_reference_data)
    return problem_data, config

def initialize_fields(problem_data, basis, config):
    reference_data = problem_data["reference_data"]
    fields = {}
    for key, data in reference_data.items():
        field = fit_basis_field_from_data(basis, data.coords, data.data, trainable=config.trainable)
        fields[f"{key}_field"] = field
    return fields

def save_results(config, optimized_ssa):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    mu_coeffs = optimized_ssa.mu_field.coeffs.value
    u_coeffs = optimized_ssa.u_field.coeffs.value
    v_coeffs = optimized_ssa.v_field.coeffs.value
    h_coeffs = optimized_ssa.h_field.coeffs.value
    field_dict = {
        "mu": mu_coeffs,
        "u": u_coeffs,
        "v": v_coeffs,
        "h": h_coeffs
    }
    full_data = {"fields": field_dict, "config": config_to_dict(config)}
    save_dict_to_hdf5(result_path / "full_data.h5", full_data)
    save_config(config, result_path)

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    problem_data, config = load_and_sample_data(config)

    basis = ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    fields = initialize_fields(problem_data, basis, config)
    fields["mu_field"] = LogBasisField(basis, Coeffs.make_zero(basis.degs))

    problem = SSAProblem(**fields)
    if config.solver == "Adam":
        solver = pr.AdamSolver()
    elif config.solver == "LevenbergMarquardt":
        solver = pr.LevenbergMarquardtSolver()
    else:
        raise ValueError(f"Solver {config.solver} not supported")

    timestamp = time.strftime("%m%d%H%M")
    script_name = f"{timestamp}_{config["experiment_name"]}"
    print(f"Script name: {script_name}")
    config.script_name = script_name

    start_time = time.time()
    problem_config = SSAConfig.from_config(config)
    optimized_ssa = solver.solve(problem, problem_data, config, problem_config)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    config["solve_time"] = end_time - start_time
    save_results(config, optimized_ssa)