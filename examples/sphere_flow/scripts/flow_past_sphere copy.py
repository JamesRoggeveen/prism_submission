import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib

class StokesConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float

class StokesProblem(pr.AbstractProblem):
    u_field: pr.BasisField
    v_field: pr.BasisField
    p_field: pr.BasisField
    r_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "wall": self.wall_boundary_residual,
            # "immersed": self.immersed_boundary_residual,
            "inlet": self.inlet_residual,
            "outlet": self.outlet_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        grad_p_x = self.p_field.derivative(*problem_data.coords, order=(1,0))
        grad_p_y = self.p_field.derivative(*problem_data.coords, order=(0,1))

        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0))
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2))

        v_xx = self.v_field.derivative(*problem_data.coords, order=(2,0))
        v_yy = self.v_field.derivative(*problem_data.coords, order=(0,2))


        x_eq = u_xx + u_yy - grad_p_x
        y_eq = v_xx + v_yy - grad_p_y

        # return jnp.concatenate([x_eq, y_eq], axis=0)

        obstacle_radius = self.r_field.coeffs.value[0]
        x, y = problem_data.coords
        point_radius = jnp.sqrt(x**2 + y**2)
        mask = jnp.where(point_radius > obstacle_radius, 1.0, 0.0)
        
        x_eq_masked = x_eq * mask
        y_eq_masked = y_eq * mask
        

        return jnp.concatenate([x_eq_masked, y_eq_masked], axis=0)
    
    @eqx.filter_jit
    def continuity_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0))
        v_y = self.v_field.derivative(*problem_data.coords, order=(0,1))
        return u_x + v_y

    @eqx.filter_jit
    def immersed_boundary_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        coords, N_normalized, T_normalized = self.immersed_boundary_geometry(problem_data, config)
        u = self.u_field.evaluate(*coords)
        v = self.v_field.evaluate(*coords)
        u_vec = jnp.stack([u, v], axis=1)
        u_n = (u_vec*N_normalized).sum(axis=1)
        u_t = (u_vec*T_normalized).sum(axis=1)

        return jnp.concatenate([u_n, u_t], axis=0)
    
    @eqx.filter_jit
    def immersed_boundary_geometry(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        theta, = problem_data.coords
        t = theta/jnp.pi -1 
        r = self.r_field.evaluate(t)
        dr_dtheta = self.r_field.derivative(t, order=(1,))/jnp.pi

        x = r * jnp.cos(theta)
        y = (r * jnp.sin(theta))

        T_x = dr_dtheta * jnp.cos(theta) - r * jnp.sin(theta)
        T_y = (dr_dtheta * jnp.sin(theta) + r * jnp.cos(theta))
        T = jnp.stack([T_x, T_y], axis=1)

        N_x = T_y
        N_y = -T_x
        N = jnp.stack([N_x, N_y], axis=1)
        N_normalized = N / jnp.linalg.norm(N, axis=1, keepdims=True)
        T_normalized = T / jnp.linalg.norm(T, axis=1, keepdims=True)
                
        return (x, y), N_normalized, T_normalized

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
    def outlet_residual(self, problem_data: pr.data.BoundaryData, config: StokesConfig) -> jax.Array:
        # Get the coordinates for the outlet boundary
        coords = problem_data.coords

        # Evaluate pressure at the outlet
        p = self.p_field.evaluate(*coords)

        # Calculate the required velocity derivatives
        u_x = self.u_field.derivative(*coords, order=(1, 0))
        v_x = self.v_field.derivative(*coords, order=(1, 0))

        return jnp.concatenate([u_x+p, v_x], axis=0)

        # We assume mu=1 for Stokes flow
        # First component of the zero-traction condition: p - 2*mu*du/dx = 0
        # Note: The normal vector n_x is -1. The stress component is sigma_xx = -p + 2*u_x.
        # The traction t_x = sigma_xx * n_x + sigma_xy * n_y = (-p + 2*u_x)*(-1) = p - 2*u_x
        # residual_x = p - 2 * u_x
        residual_x = p + u_x
        
        # Second component of the zero-traction condition: du/dy + dv/dx = 0
        # Note: The stress component is sigma_yx = u_y + v_x.
        # The traction t_y = sigma_yx * n_x + sigma_yy * n_y = (u_y + v_x)*(-1) = -(u_y + v_x)
        # residual_y = u_y + v_x
        residual_y = v_x

        return jnp.concatenate([residual_x, residual_y], axis=0)

    @eqx.filter_jit
    def total_residual(self, problem_data: pr.data.ProblemData, config: StokesConfig) -> jax.Array:
        equation_residual = jnp.sqrt(config.residual_weights["equation"]) * self.equation_residual(problem_data["equation"], config)
        continuity_residual = jnp.sqrt(config.residual_weights["continuity"]) * self.continuity_residual(problem_data["equation"], config)
        wall_residual = jnp.sqrt(config.residual_weights["wall"]) * self.wall_boundary_residual(problem_data["wall"], config)
        # immersed_residual = jnp.sqrt(config.residual_weights["immersed"]) * self.immersed_boundary_residual(problem_data["immersed"], config)
        inlet_residual = jnp.sqrt(config.residual_weights["inlet"]) * self.inlet_residual(problem_data["inlet"], config)
        outlet_residual = jnp.sqrt(config.residual_weights["outlet"]) * self.outlet_residual(problem_data["outlet"], config)
        n_equation = equation_residual.shape[0]
        n_wall = wall_residual.shape[0]
        # n_immersed = immersed_residual.shape[0]
        n_inlet = inlet_residual.shape[0]
        n_outlet = outlet_residual.shape[0]
        return jnp.concatenate([equation_residual/jnp.sqrt(n_equation),continuity_residual/jnp.sqrt(n_equation), wall_residual/jnp.sqrt(n_wall), inlet_residual/jnp.sqrt(n_inlet), outlet_residual/jnp.sqrt(n_outlet)], axis=0)
        # return jnp.concatenate([equation_residual/jnp.sqrt(n_equation),continuity_residual/jnp.sqrt(n_equation), wall_residual/jnp.sqrt(n_wall), immersed_residual/jnp.sqrt(n_immersed), inlet_residual/jnp.sqrt(n_inlet), outlet_residual/jnp.sqrt(n_outlet)], axis=0)

def compute_mean_abs_grad(grad: eqx.Module) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(grad)
    if not leaves:
        return 0.0
    return jnp.mean(jnp.concatenate([jnp.abs(leaf.ravel()) for leaf in leaves]))

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
    return 0.5*(y-1)*(y+1),jnp.zeros_like(y)

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_immersed = config.n_immersed

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    eq_col_x, eq_col_y = jax.random.uniform(subkey, (2, n_pde), minval=-1, maxval=1)
    eq_col_data = pr.data.CollocationPoints(coords = (eq_col_x, eq_col_y))

    key, subkey = jax.random.split(key)
    wall_x = jax.random.uniform(subkey, (2*n_bc,), minval=-.95, maxval=1)
    wall_y = jnp.concatenate([jnp.ones((n_bc,))*1,jnp.ones((n_bc,))*-1])
    wall_data = pr.data.CollocationPoints(coords = (wall_x, wall_y))

    key, subkey = jax.random.split(key)
    theta_immersed = jax.random.uniform(subkey, (n_immersed,), minval=0, maxval=2*jnp.pi)
    immersed_data = pr.data.CollocationPoints(coords=(theta_immersed,))

    key, subkey = jax.random.split(key)
    inlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    inlet_x = jnp.ones_like(inlet_y)
    inlet_data = pr.data.ReferenceData(coords=(inlet_x, inlet_y), data=inlet_condition(inlet_x, inlet_y))

    key, subkey = jax.random.split(key)
    outlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    outlet_x = -jnp.ones_like(outlet_y)
    outlet_data = pr.data.ReferenceData(coords=(outlet_x, outlet_y), data=inlet_condition(outlet_x, outlet_y))
    # outlet_data = pr.data.BoundaryData(coords=(outlet_x, outlet_y), normal_vector=(-1*jnp.ones_like(outlet_x), jnp.zeros_like(outlet_y)))


    return pr.data.ProblemData(equation = eq_col_data, immersed = immersed_data, wall = wall_data, inlet = inlet_data, outlet = outlet_data)

def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    field_dict = {
        "u": optimized_problem.u_field.coeffs.value,
        "v": optimized_problem.v_field.coeffs.value,
        "p": optimized_problem.p_field.coeffs.value,
        "r": optimized_problem.r_field.coeffs.value
    }
    full_data = {"fields": field_dict, "config": config_to_dict(config)}
    pr.save_dict_to_hdf5(result_path / "full_data.h5", full_data)
    save_config(config, result_path)

if __name__ == "__main__":
    config = load_config("configs/sphere_flow.yml")
    problem_data = sample_points(config)
    basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    # basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    v_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    p_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    # u_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    # v_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    # p_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    r_basis = pr.BasisND([pr.basis.vectorized_cosine_basis],(1,))
    r_field = pr.BasisField(r_basis, pr.StaticCoeffs(jnp.array([.1,0])))

    problem = StokesProblem(u_field, v_field, p_field, r_field)
    if config.solver == "Adam":
        solver = pr.AdamSolver()
    elif config.solver == "LevenbergMarquardt":
        solver = pr.LevenbergMarquardtSolver()
    else:
        raise ValueError(f"Solver {config.solver} not supported")
    
    timestamp = time.strftime("%m%d%H%M")
    config.script_name = f"{timestamp}_sphere_flow"

    print(f"Script name: {config.script_name}")

    start_time = time.time()
    problem_config = StokesConfig.from_config(config)
    optimized_problem = solver.solve(problem, problem_data, config, StokesConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    config.total_time = end_time - start_time
    save_results(config, optimized_problem)