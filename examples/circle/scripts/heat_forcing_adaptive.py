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
import optax
from tqdm import tqdm

class HeatConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    alpha: float

class HeatProblem(pr.AbstractProblem):
    c_field: pr.BasisField
    f_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "boundary": self.boundary_residual,
            "target": self.target_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.CollocationPoints, config: HeatConfig) -> jax.Array:
        c_xx = self.c_field.derivative(*problem_data.coords, order=(2,0,0))
        c_yy = self.c_field.derivative(*problem_data.coords, order=(0,2,0))
        c_t = self.c_field.derivative(*problem_data.coords, order=(0,0,1))
        alpha = config.alpha
        return c_t - alpha * (c_xx + c_yy)
    
    @eqx.filter_jit
    def boundary_residual(self, problem_data: pr.data.ReferenceData, config: HeatConfig) -> jax.Array:
        c = self.c_field.evaluate(*problem_data.coords)
        problem_x, problem_y, problem_t = problem_data.coords
        problem_theta = jnp.arctan2(problem_y, problem_x)/jnp.pi
        f = self.f_field.evaluate(problem_theta, problem_t)
        return c - f

    @eqx.filter_jit
    def target_residual(self, problem_data: pr.data.ReferenceData, config: HeatConfig) -> jax.Array:
        c = self.c_field.evaluate(*problem_data.coords)
        return c - problem_data.data

    # @eqx.filter_jit
    # def final_residual(self, problem_data: pr.data.ReferenceData, config: HeatConfig) -> jax.Array:
    #     c = self.c_field.evaluate(*problem_data.coords)
    #     return c - problem_data.data

    @eqx.filter_jit
    def total_residual(self, problem_data: pr.data.ProblemData, config: HeatConfig) -> jax.Array:
        equation_residual = jnp.sqrt(config.residual_weights["equation"]) * self.equation_residual(problem_data["equation"], config)
        boundary_residual = jnp.sqrt(config.residual_weights["boundary"]) * self.boundary_residual(problem_data["boundary"], config)
        target_residual = jnp.sqrt(config.residual_weights["target"]) * self.target_residual(problem_data["target"], config)
        n_equation = equation_residual.shape[0]
        n_boundary = boundary_residual.shape[0]
        n_target = target_residual.shape[0]   
        return jnp.concatenate([equation_residual/jnp.sqrt(n_equation), boundary_residual/jnp.sqrt(n_boundary), target_residual/jnp.sqrt(n_target)], axis=0)

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

def boundary_condition(x,y):
    theta = jnp.arctan2(y, x)
    return boundary_condition_theta(theta)

def boundary_condition_theta(theta):    
    return jnp.zeros_like(theta)

# def initial_condition(x,y):
#     x = np.array(x)
#     y = np.array(y)
#     left_eye = (x + 0.4)**2 + (y - 0.4)**2 < 0.3**2
#     right_eye = (x - 0.4)**2 + (y - 0.4)**2 < 0.3**2
#     mouth = (x)**2 + (y+.4)**2 < 0.3**2
#     output = np.logical_or.reduce((left_eye, right_eye, mouth)).astype(float)
#     return jnp.array(output)

# def final_condition(x,y):
#     x = np.array(x)
#     y = np.array(y)
#     left_eye = (x + 0.4)**2 + (y + 0.4)**2 < 0.3**2
#     right_eye = (x - 0.4)**2 + (y + 0.4)**2 < 0.3**2
#     mouth = (x)**2 + (y-.4)**2 < 0.3**2
#     output = np.logical_or.reduce((left_eye, right_eye, mouth)).astype(float)
#     return jnp.array(output)

def sample_circle(n_points, key):
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, (n_points,), minval=0, maxval=1)
    r = jnp.sqrt(u)
    theta = jax.random.uniform(key, (n_points,), minval=0, maxval=2*jnp.pi)
    x = r*jnp.cos(theta)
    y = r*jnp.sin(theta)
    return x, y

def sample_goal(slice_name,key,n_points):
    if slice_name is None:
        image = plt.imread("data/H.png")
        image = jnp.zeros_like(image[...,-1])
    else:
        image = plt.imread(f"data/{slice_name}")
        image = image[...,-1]
        image = ~image.astype(bool)
    x_grid, y_grid = jnp.meshgrid(jnp.linspace(-1,1,image.shape[1]), jnp.linspace(-1,1,image.shape[0]))
    r_grid = jnp.sqrt(x_grid**2 + y_grid**2)
    mask = r_grid < 1
    image = image[mask].reshape(-1)
    x_grid = x_grid[mask].reshape(-1)
    y_grid = y_grid[mask].reshape(-1)
    key, subkey = jax.random.split(key)
    inds = jax.random.choice(subkey, jnp.arange(x_grid.size), (n_points,), replace=False)
    x_grid = x_grid[inds]
    y_grid = y_grid[inds]
    image = image[inds].astype(float)
    return x_grid, y_grid, image

def sample_points(config):
    n_pde = config.n_pde
    n_bc = config.n_bc
    n_target = config.n_target
    n_t = config.n_t

    t_vec = jnp.linspace(-1,1,n_t)

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x, y = sample_circle(n_pde*n_t, subkey)
    key, subkey = jax.random.split(key)
    final_t = jax.random.uniform(subkey, (n_pde*n_t,), minval=-1, maxval=1)
    final_x = x
    final_y = y
    # final_x = jnp.tile(x, n_t)
    # final_y = jnp.tile(y, n_t)
    # final_t = jnp.repeat(t_vec, n_pde)

    collocation_data = pr.data.CollocationPoints(coords = (final_x, final_y, final_t))

    key, subkey = jax.random.split(key)
    bc_theta = jax.random.uniform(subkey, (n_bc*n_t,), minval=0, maxval=2*jnp.pi)

    bc_x = jnp.cos(bc_theta)
    bc_y = jnp.sin(bc_theta)
    boundary_data = boundary_condition(bc_x, bc_y)
    # boundary_data = jnp.tile(boundary_data, n_t)
    # final_bc_x = jnp.tile(bc_x, n_t)
    # final_bc_y = jnp.tile(bc_y, n_t)
    # final_bc_t = jnp.repeat(t_vec, n_bc)
    final_bc_x = bc_x
    final_bc_y = bc_y
    key, subkey = jax.random.split(key)
    final_bc_t = jax.random.uniform(subkey, (n_bc*n_t,), minval=-1, maxval=1)
    boundary_data = pr.data.CollocationPoints(coords = (bc_x, bc_y, final_bc_t))

    # slices = [None, "H.png", None]
    slices = ["H.png", "A.png", "R.png"]
    t_sample = jnp.linspace(-1,1,len(slices))
    x_target_list = []
    y_target_list = []
    t_target_list = []
    target_data_list = []
    f, ax = plt.subplots(1, len(slices), figsize=(3*len(slices),3))
    for i, slice in enumerate(slices):
        key, subkey = jax.random.split(key)
        x_target, y_target, target_data = sample_goal(slice,subkey,n_target)
        t_target = t_sample[i]*jnp.ones_like(x_target)
        ax[i].scatter(x_target, y_target, c=target_data, cmap="jet")
        x_target_list.append(x_target)
        y_target_list.append(y_target)
        t_target_list.append(t_target)
        target_data_list.append(target_data)
    f.tight_layout()
    plt.savefig("data/targets.png")
    plt.close()
    x_target = jnp.concatenate(x_target_list)
    y_target = jnp.concatenate(y_target_list)
    t_target = jnp.concatenate(t_target_list)
    target_data = jnp.concatenate(target_data_list)
    target_data = pr.data.ReferenceData(coords = (x_target, y_target, t_target), data = target_data)
    return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data, target = target_data)

    # key, subkey = jax.random.split(key)
    # x_ic, y_ic = sample_circle(n_initial, subkey)
    # t_ic = -1*jnp.ones(n_initial)
    # initial_data = pr.data.ReferenceData(coords = (x_ic, y_ic, t_ic), data = initial_condition(x_ic, y_ic))
    
    
    # key, subkey = jax.random.split(key)
    # x_fc, y_fc = sample_circle(n_final, key)
    # t_fc = jnp.ones(n_final)
    # final_data = final_condition(x_fc, y_fc)
    # final_data = pr.data.ReferenceData(coords = (x_fc, y_fc, t_fc), data = final_data)

    # return pr.data.ProblemData(equation = collocation_data, boundary = boundary_data, initial = initial_data, final = final_data)

def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    c_coeffs = optimized_problem.c_field.coeffs.value
    f_coeffs = optimized_problem.f_field.coeffs.value
    field_dict = {
        "c": c_coeffs,
        "f": f_coeffs
    }
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
        config["basis_Nt"] = args.N
        config["script_name"] = f"{args.N}_heat"
    else:
        config["script_name"] = f"{timestamp}_heat"
    return config

def compute_mean_abs_grad(grad: eqx.Module) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(grad)
    if not leaves:
        return 0.0
    return jnp.mean(jnp.concatenate([jnp.abs(leaf.ravel()) for leaf in leaves]))

@eqx.filter_jit
def adaptive_loss_fn(problem: HeatProblem, problem_data: pr.data.ProblemData, problem_config: HeatConfig):
    
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
    config = load_config("configs/heat_forcing.yml")
    config = parse_args(config)
    problem_data = sample_points(config)
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config.basis_Nx, config.basis_Ny, config.basis_Nt))
    f_basis = pr.basis.FourierChebyshevBasis2D((config.basis_bc, config.basis_Nt))
    # c_field = pr.BasisField(basis, pr.Coeffs.make_zero(basis.degs))
    # f_field = pr.BasisField(f_basis, pr.Coeffs.make_zero(f_basis.degs))
    c_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    f_field = pr.BasisField(f_basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(f_basis.degs))
    problem = HeatProblem(c_field, f_field)
    timestamp = time.strftime("%m%d%H%M")
    config.script_name = f"{timestamp}_heat"

    print(f"Script name: {config.script_name}")

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    problem_config = HeatConfig.from_config(config)
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
    optimized_problem = eqx.combine(params, static)

    print(f"Time taken: {end_time - start_time} seconds")
    save_results(config, optimized_problem)

    weight_dict = {key: [] for key in weights[0].keys()}
    for weight in weights:
        for key, value in weight.items():
            weight_dict[key].append(value)
    loss_dict = {key: [] for key in loss_values[0].keys()}
    for loss in loss_values:
        for key, value in loss.items():
            loss_dict[key].append(value)

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