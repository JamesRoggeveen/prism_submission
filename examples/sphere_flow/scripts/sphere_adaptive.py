import jax
import jax.numpy as jnp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

class StokesConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    x_scale: float
    y_scale: float

class StokesProblem(pr.AbstractProblem):
    u_field: pr.BasisField
    v_field: pr.BasisField
    p_field: pr.BasisField
    r_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "wall": self.wall_boundary_residual,
            "inlet": self.inlet_residual,
            "outlet": self.outlet_residual,
            # "immersed": self.immersed_boundary_residual,
            "continuity": self.continuity_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        x_scale = config.x_scale
        y_scale = config.y_scale
        grad_p_x = self.p_field.derivative(*problem_data.coords, order=(1,0))/x_scale
        grad_p_y = self.p_field.derivative(*problem_data.coords, order=(0,1))/y_scale

        u_xx = self.u_field.derivative(*problem_data.coords, order=(2,0))/x_scale**2
        u_yy = self.u_field.derivative(*problem_data.coords, order=(0,2))/y_scale**2

        v_xx = self.v_field.derivative(*problem_data.coords, order=(2,0))/x_scale**2
        v_yy = self.v_field.derivative(*problem_data.coords, order=(0,2))/y_scale**2


        x_eq = u_xx + u_yy - grad_p_x
        y_eq = v_xx + v_yy - grad_p_y


        return jnp.concatenate([x_eq, y_eq], axis=0)

        obstacle_radius = self.r_field.coeffs.value[0]
        x, y = problem_data.coords
        point_radius = jnp.sqrt(x**2 + y**2)
        mask = jnp.where(point_radius > obstacle_radius, 1.0, 0.0)
        
        x_eq_masked = x_eq * mask
        y_eq_masked = y_eq * mask

        return jnp.concatenate([x_eq_masked, y_eq_masked], axis=0)

    @eqx.filter_jit
    def continuity_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        x_scale = config.x_scale
        y_scale = config.y_scale
        u_x = self.u_field.derivative(*problem_data.coords, order=(1,0))/x_scale
        v_y = self.v_field.derivative(*problem_data.coords, order=(0,1))/y_scale
        continuity_residual = u_x + v_y

        return continuity_residual

        obstacle_radius = self.r_field.coeffs.value[0]
        x, y = problem_data.coords
        point_radius = jnp.sqrt(x**2 + y**2)
        mask = jnp.where(point_radius > obstacle_radius, 1.0, 0.0)
        
        continuity_residual_masked = continuity_residual * mask

        return continuity_residual_masked

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
    def outlet_residual(self, problem_data: pr.data.CollocationPoints, config: StokesConfig) -> jax.Array:
        # Get the coordinates for the outlet boundary
        coords = problem_data.coords
        x_scale = config.x_scale
        y_scale = config.y_scale
        # Evaluate pressure at the outlet
        p = self.p_field.evaluate(*coords)

        # Calculate the required velocity derivatives
        u_x = self.u_field.derivative(*coords, order=(1, 0))/x_scale
        v_x = self.v_field.derivative(*coords, order=(1, 0))/x_scale

        return jnp.concatenate([u_x-p, v_x], axis=0)

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
    return -0.5*(y-1)*(y+1),jnp.zeros_like(y)

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_immersed = config.n_immersed

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    eq_col_x, eq_col_y = jax.random.uniform(subkey, (2, n_pde), minval=-1, maxval=1)
    eq_col_data = pr.data.CollocationPoints(coords = (eq_col_x, eq_col_y))

    key, subkey = jax.random.split(key)
    wall_x = jax.random.uniform(subkey, (2*n_bc,), minval=-1, maxval=1)
    wall_y = jnp.concatenate([jnp.ones((n_bc,))*1,jnp.ones((n_bc,))*-1])
    wall_data = pr.data.CollocationPoints(coords = (wall_x, wall_y))

    key, subkey = jax.random.split(key)
    theta_immersed = jax.random.uniform(subkey, (n_immersed,), minval=0, maxval=2*jnp.pi)
    immersed_data = pr.data.CollocationPoints(coords=(theta_immersed,))

    key, subkey = jax.random.split(key)
    inlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    inlet_x = -1*jnp.ones_like(inlet_y)
    inlet_data = pr.data.ReferenceData(coords=(inlet_x, inlet_y), data=inlet_condition(inlet_x, inlet_y))

    key, subkey = jax.random.split(key)
    outlet_y = jax.random.uniform(subkey, (n_bc,), minval=-1, maxval=1)
    outlet_x = jnp.ones_like(outlet_y)
    outlet_data = pr.data.CollocationPoints(coords=(outlet_x, outlet_y))
    # outlet_data = pr.data.BoundaryData(coords=(outlet_x, outlet_y), normal_vector=(-1*jnp.ones_like(outlet_x), jnp.zeros_like(outlet_y)))


    return pr.data.ProblemData(equation = eq_col_data, continuity = eq_col_data, immersed = immersed_data, wall = wall_data, inlet = inlet_data, outlet = outlet_data)

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

@eqx.filter_jit
def adaptive_loss_fn(problem: StokesProblem, problem_data: pr.data.ProblemData, problem_config: StokesConfig):
    
    # Get the trainable parameters (the coefficients)
    params, static = eqx.partition(problem, eqx.is_array)

    # Dictionaries to store results
    losses = {}
    grads = {}
    grad_mags = {}

    # Get all the individual residual functions from the problem
    residual_fns = problem.get_residual_functions()

    # --- Step 1: Calculate loss and gradient for EACH residual ---
    for name, res_fn in residual_fns.items():
        # Get the string name of the residual function (e.g., "equation_residual")
        method_name = res_fn.func.__name__

        def per_residual_loss_fn(p):
            prob_i = eqx.combine(p, static)
            
            correct_method_to_call = getattr(prob_i, method_name)
            
            residual_values = correct_method_to_call(problem_data[name], problem_config)
            return jnp.mean(residual_values**2)

        loss_val, grad_val = eqx.filter_value_and_grad(per_residual_loss_fn)(params)
        
        losses[name] = loss_val
        grads[name] = grad_val

    # --- Step 2: Compute adaptive weights from the gradients ---
    for name, grad_val in grads.items():
        grad_mags[name] = compute_mean_abs_grad(grad_val)
    
    # Calculate the mean of all the gradient magnitudes
    mean_of_grad_mags = jnp.mean(jnp.stack(list(grad_mags.values())))
    
    # The weight is the mean magnitude divided by the individual magnitude.
    # jax.lax.stop_gradient is crucial: the weights affect the loss value,
    # but we don't backpropagate through the weight calculation itself.
    lambdas = {
        name: jax.lax.stop_gradient(mean_of_grad_mags / (mag + 1e-8)) # Add epsilon for stability
        for name, mag in grad_mags.items()
    }

    # --- Step 3: Calculate the final adaptively weighted total loss ---
    total_loss = sum(lambdas[name] * losses[name] for name in losses.keys())

    # We also return the dictionary of individual losses and weights for logging
    log_data = {"total_loss": total_loss, "losses": losses, "weights": lambdas}
    
    return total_loss, log_data

if __name__ == "__main__":
    config = load_config("configs/config.yml")
    problem_data = sample_points(config)
    basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    # p_basis = pr.ChebyshevBasis2D((config.basis_Nx-2, config.basis_Ny-2))
    # basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    v_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    # p_field = pr.BasisField(p_basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(p_basis.degs))
    p_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    # u_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    # v_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    # p_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    r_basis = pr.BasisND([pr.basis.vectorized_cosine_basis],(1,))
    r_field = pr.BasisField(r_basis, pr.StaticCoeffs(jnp.array([.2,0])))

    problem = StokesProblem(u_field, v_field, p_field, r_field)
    
    timestamp = time.strftime("%m%d%H%M")
    config.script_name = f"{timestamp}_sphere_flow"

    print(f"Script name: {config.script_name}")

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    problem_config = StokesConfig.from_config(config)
    # 2. Partition the initial problem into trainable parameters and static data
    params, static = eqx.partition(problem, eqx.is_array)
    opt_state = optimizer.init(params)

    # 3. Define the function that performs one step of optimization
    @eqx.filter_jit
    def make_step(params, static, opt_state, problem_data):
        # Differentiate the adaptive_loss_fn to get the gradient of the *total* loss
        (total_loss, log_data), total_grad = eqx.filter_value_and_grad(
            lambda p, s: adaptive_loss_fn(eqx.combine(p, s), problem_data, problem_config), has_aux=True
        )(params, static)
        
        updates, opt_state = optimizer.update(total_grad, opt_state, params)
        params = eqx.apply_updates(params, updates)
        
        return params, opt_state, log_data

    # 4. The main training loop
    print("Starting training with adaptive weights...")
    start_time = time.time()
    total_loss = []
    loss_values = []
    weights = []
    for step in (pbar := tqdm(range(config.n_epochs))): # Use max_steps from your config
        params, opt_state, log_data = make_step(params, static, opt_state, problem_data)
        pbar.set_description(f"Loss: {log_data['total_loss']:.4e}")
        total_loss.append(log_data['total_loss'])
        loss_values.append(log_data['losses'])
        weights.append(log_data['weights'])

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # 5. Get the final optimized problem
    optimized_problem = eqx.combine(params, static)

    weight_dict = {key: [] for key in weights[0].keys()}
    for weight in weights:
        for key, value in weight.items():
            weight_dict[key].append(value)
    loss_dict = {key: [] for key in loss_values[0].keys()}
    for loss in loss_values:
        for key, value in loss.items():
            loss_dict[key].append(value)

    # --- (Your existing code to save results) ---
    config.total_time = end_time - start_time
    save_results(config, optimized_problem)

    total_loss = jnp.array(total_loss)
    f, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].semilogy(total_loss)
    ax[0].set_title("Total Loss")
    for key, value in weight_dict.items():
        ax[1].semilogy(value, label=key)
    ax[1].set_title("Weights")
    ax[1].legend()
    for key, value in loss_dict.items():
        ax[2].semilogy(value, label=key)
    ax[2].set_title("Individual Losses")
    ax[2].legend()
    plt.savefig("scripts/losses.png")
    plt.savefig(f"results/{config.script_name}/losses.png")
    plt.close()