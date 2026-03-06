# solve_laplace_bilevel.py

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import yaml
import time
from typing import Dict
import prism as pr
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Imports for the new solver logic
import optimistix as optx
import lineax as lx
import optax

jax.config.update("jax_enable_x64", True)

# A numerically stable softmin implementation
def _softmin(x, tau=1e-3):
    return -tau * jsp.special.logsumexp(-x / tau)

# --- Configuration and Problem Definition ---

class LaplaceConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    regularization_strength: float
    cyl_radius: float
    x_cyl_initial: float
    y_cyl_initial: float
    n_outer_steps: int # For the bilevel loop
    n_grid: int
    inner_solver_rtol: float
    inner_solver_atol: float
    outer_solver_lr: float

class LaplaceProblem(pr.AbstractProblem):
    u_field: pr.BasisField
    x_cyl: jax.Array
    y_cyl: jax.Array

    # These are helper methods to define the physics loss
    def equation_residual(self, problem_data, config):
        x, y = problem_data.coords
        mask = (x - self.x_cyl)**2 + (y - self.y_cyl)**2 >= config.cyl_radius**2
        u_xx = self.u_field.derivative(x, y, order=(2,0))
        u_yy = self.u_field.derivative(x, y, order=(0,2))
        return (u_xx + u_yy) * mask

    def outer_boundary_residual(self, problem_data, config):
        u = self.u_field.evaluate(*problem_data.coords)
        return u - problem_data.data

    def inner_boundary_residual(self, problem_data, config):
        template_x, template_y = problem_data.coords
        x_bc = template_x * config.cyl_radius + self.x_cyl
        y_bc = template_y * config.cyl_radius + self.y_cyl
        u = self.u_field.evaluate(x_bc, y_bc)
        return u - problem_data.data

# --- Bilevel Optimization Core ---

def physics_loss_fn(coeffs_trial, static_field, problem_data, config, x_cyl, y_cyl):
    """Calculates the loss for the physics ONLY (PDE + BCs). This is the 'inner' objective."""
    problem = LaplaceProblem(
        u_field=eqx.combine(coeffs_trial, static_field), x_cyl=x_cyl, y_cyl=y_cyl
    )
    res_eq = problem.equation_residual(problem_data['equation'], config)
    res_ob = problem.outer_boundary_residual(problem_data['outer_boundary'], config)
    res_ib = problem.inner_boundary_residual(problem_data['inner_boundary'], config)
    
    # Using weights from config to balance the physics terms
    w = config.residual_weights
    loss = (w['equation'] * jnp.mean(res_eq**2) + 
            w['outer_boundary'] * jnp.mean(res_ob**2) + 
            w['inner_boundary'] * jnp.mean(res_ib**2))
    return loss

@jax.custom_vjp
def solve_field_implicit(geom_params, initial_coeffs, static_field, problem_data, config):
    """
    'Black-box' solver that finds field coefficients for a given geometry.
    We define a custom VJP to differentiate through this function implicitly.
    """
    x_cyl, y_cyl = geom_params['x_cyl'], geom_params['y_cyl']
    loss_for_solver = lambda c, _: physics_loss_fn(c, static_field, problem_data, config, x_cyl, y_cyl)
    
    # Use a robust solver like L-BFGS to find the optimal coefficients c*
    solver = optx.LBFGS(rtol=config.inner_solver_rtol, atol=config.inner_solver_atol)
    sol = optx.minimise(loss_for_solver, solver, initial_coeffs)
    
    return sol.value # Return the converged coefficients c*

# 1. Define the forward pass for the custom VJP
def solve_field_fwd(geom_params, initial_coeffs, static_field, problem_data, config):
    optimal_coeffs = solve_field_implicit(geom_params, initial_coeffs, static_field, problem_data, config)
    # Save all necessary variables for the backward pass
    saved_for_backward = (geom_params, optimal_coeffs, static_field, problem_data, config)
    return optimal_coeffs, saved_for_backward

# 2. Define the backward pass (the implicit derivative logic)
def solve_field_bwd(saved_for_backward, grad_coeffs_out):
    """
    Calculates the VJP for the geometry parameters using the Implicit Function Theorem.
    'grad_coeffs_out' is the upstream gradient from the outer objective, i.e., ∂(Integral)/∂c
    """
    geom_params, optimal_coeffs, static_field, problem_data, config = saved_for_backward
    x_cyl, y_cyl = geom_params['x_cyl'], geom_params['y_cyl']

    # We need the cross-derivative: ∂(PhysicsLoss)/∂p, where p are the geometry params
    # `jax.grad` is with respect to the first arg, so we define a wrapper.
    def loss_wrapper_for_geom(p, c):
        return physics_loss_fn(c, static_field, problem_data, config, p['x_cyl'], p['y_cyl'])
    
    # This is (∂(PhysicsLoss)/∂x_cyl, ∂(PhysicsLoss)/∂y_cyl)
    grad_loss_wrt_geom = jax.grad(loss_wrapper_for_geom)(geom_params, optimal_coeffs)

    # We need to solve the linear system: H_cc * z = g
    # where H_cc is the Hessian of the physics loss w.r.t. the coefficients.
    def loss_wrapper_for_coeffs(c, p):
        return physics_loss_fn(c, static_field, problem_data, config, p['x_cyl'], p['y_cyl'])

    # Using Lineax to represent the Hessian H_cc as a linear operator is efficient
    H_cc_operator = lx.JacobianLinearOperator(jax.grad(loss_wrapper_for_coeffs), optimal_coeffs, tags=lx.symmetric_tag)

    # Solve H_cc * z = grad_coeffs_out using a conjugate gradient solver
    solver = lx.CG(rtol=1e-6, atol=1e-6)
    z = lx.linear_solve(H_cc_operator, grad_coeffs_out).value

    # The final gradient for the geometry is -z^T * (∂(PhysicsLoss)/∂p)
    # We use tree_map and vdot for robust PyTree dot products
    implicit_grad_geom = jax.tree_util.tree_map(lambda g: -jnp.vdot(z, g), grad_loss_wrt_geom)
    
    # Return gradients for geom_params (and None for the other static args)
    return (implicit_grad_geom, None, None, None, None)

# 3. Link the forward and backward passes to the main function
solve_field_implicit.defvjp(solve_field_fwd, solve_field_bwd)

# --- Outer Objective Functions ---

def integral_objective_fn(problem, problem_data, config):
    """More accurate integral calculation using polar quadrature."""
    x_out, y_out = problem_data['integral_objective'].coords
    I_outer = jnp.sum(problem.u_field.evaluate(x_out, y_out)) * (2/config.n_grid)**2

    R = config.cyl_radius
    M_theta, M_r = 128, 32
    theta = jnp.linspace(0, 2*jnp.pi, M_theta, endpoint=False)
    r = jnp.linspace(0, R, M_r, endpoint=False) + 0.5 * (R/M_r)
    c, s = jnp.cos(theta), jnp.sin(theta)
    X = problem.x_cyl + r[:,None] * c[None,:]
    Y = problem.y_cyl + r[:,None] * s[None,:]
    U = problem.u_field.evaluate(X, Y)
    weights = r[:,None] * (R/M_r) * (2*jnp.pi/M_theta)
    I_disk = jnp.sum(U * weights)
    
    return I_outer - I_disk

def constraint_penalty_fn(problem, config):
    """Smooth penalty using softmin to avoid wall collision."""
    theta = jnp.linspace(0, 2*jnp.pi, 360, endpoint=False)
    r_wall = 0.7 + 0.3 * jnp.cos(2*theta)
    x_wall = r_wall * jnp.cos(theta)
    y_wall = 2 * r_wall * jnp.sin(theta)
    sq_dists = (x_wall - problem.x_cyl)**2 + (y_wall - problem.y_cyl)**2
    soft_min_sq = _softmin(sq_dists, tau=(0.1 * config.cyl_radius)**2)
    
    # Penalty for center getting closer than 2*R to the wall
    required_clearance_sq = (2 * config.cyl_radius)**2
    violation = required_clearance_sq - soft_min_sq
    return jax.nn.relu(violation)**2

# --- Utility and Main Script Functions ---

def load_config(config_path):
    with open(config_path, 'r') as file: config = yaml.safe_load(file)
    config["solver_kwargs"] = {key: float(value) for key, value in config.get("solver_kwargs", {}).items()}
    config["seed"] = int(time.time())
    return pr.SystemConfig(**config)

def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int)
    args = parser.parse_args()
    timestamp = time.strftime("%m%d%H%M")
    config["timestamp"] = timestamp
    if args.N is not None:
        config["basis_Nx"] = args.N; config["basis_Ny"] = args.N
        config["script_name"] = f"{args.N}_laplace_bilevel"
    else:
        config["script_name"] = f"{timestamp}_laplace_bilevel"
    config["results_dir"] = f"results/{config['script_name']}"
    return config

# (create_problem_data, save_results, etc. can be copied from your original script)
def create_problem_data(config):
    def r1(theta): return 0.7+0.3*jnp.cos(2*theta)
    n_grid = config.n_grid
    x_vec = jnp.linspace(-1,1,n_grid); y_vec = jnp.linspace(-1,1,n_grid)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    y_squashed = 0.5*y_grid
    theta_grid = jnp.arctan2(y_squashed, x_grid)
    r_grid = jnp.sqrt(x_grid**2 + (y_squashed)**2)
    outer_mask = r_grid <= r1(theta_grid)
    grid_for_plotting = pr.data.ReferenceData(coords=(x_grid, y_grid), data=outer_mask)
    mask_grid_x = x_grid[outer_mask]; mask_grid_y = y_grid[outer_mask]
    grid_for_integral = pr.data.CollocationPoints(coords=(mask_grid_x, mask_grid_y))
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x_samples, y_samples = sample_from_mask(subkey, x_grid, y_grid, outer_mask, outer_mask, config.n_pde)
    collocation_data = pr.data.CollocationPoints(coords=(x_samples, y_samples))
    key, subkey = jax.random.split(key)
    theta_samples_outer = jax.random.uniform(subkey, (config.n_bc,), minval=0, maxval=2*jnp.pi)
    r_samples_outer = r1(theta_samples_outer)
    x_bc_outer = r_samples_outer*jnp.cos(theta_samples_outer)
    y_bc_outer = 2*r_samples_outer*jnp.sin(theta_samples_outer)
    outer_bc_values = outer_boundary_condition(x_bc_outer, y_bc_outer)
    outer_boundary_data = pr.data.ReferenceData(coords=(x_bc_outer, y_bc_outer), data=outer_bc_values)
    theta_samples_inner = jnp.linspace(0, 2*jnp.pi, config.n_bc)
    x_bc_template = jnp.cos(theta_samples_inner); y_bc_template = jnp.sin(theta_samples_inner)
    inner_bc_values = inner_boundary_condition(x_bc_template, y_bc_template)
    inner_boundary_data = pr.data.ReferenceData(coords=(x_bc_template, y_bc_template), data=inner_bc_values)
    return pr.data.ProblemData(
        equation=collocation_data, outer_boundary=outer_boundary_data,
        inner_boundary=inner_boundary_data, integral_objective=grid_for_integral,
        constraint_penalty=pr.data.CollocationPoints(coords=()),
    ), grid_for_plotting
def outer_boundary_condition(x,y):
    theta = jnp.arctan2(y,x)
    return 2*jnp.sin(theta) + jnp.cos(3*theta) + 4
def inner_boundary_condition(x,y):
    return -2*jnp.ones_like(x)
def sample_from_mask(key, x_grid, y_grid, data, ref, M=8000):
    mask = jnp.nonzero(ref)
    xi = x_grid[mask]; yi = y_grid[mask]
    idx = jax.random.choice(key, len(xi), shape=(M,), replace=False)
    return jnp.array(xi[idx]), jnp.array(yi[idx])
    
if __name__ == "__main__":
    pathlib.Path("figures").mkdir(exist_ok=True)
    
    # --- 1. Initial Setup ---
    config = load_config("configs/laplace_bilevel.yml")
    config = parse_args(config)
    problem_config = LaplaceConfig.from_config(config)
    
    problem_data, grid_for_plotting = create_problem_data(config)
    
    basis = pr.ChebyshevBasis2D((config.basis_Nx, config.basis_Ny))
    u_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    
    coeffs, static_field = eqx.partition(u_field, eqx.is_array)
    geom_params = {'x_cyl': jnp.array(config.x_cyl_initial), 'y_cyl': jnp.array(config.y_cyl_initial)}
    
    # --- 2. Bilevel Optimization Loop ---
    print("--- Starting Bilevel Optimization ---")
    optimizer = optax.adam(learning_rate=problem_config.outer_solver_lr)
    opt_state = optimizer.init(geom_params)
    
    start_time = time.time()
    pbar = tqdm(range(problem_config.n_outer_steps))
    for step in pbar:
        # Define the objective for the current outer step
        def outer_objective_fn(p):
            # INNER LOOP: Solve the PDE for the current geometry `p`
            optimal_coeffs = solve_field_implicit(p, coeffs, static_field, problem_data, problem_config)
            
            # Reconstruct the problem with the converged field
            problem = LaplaceProblem(
                u_field=eqx.combine(optimal_coeffs, static_field), x_cyl=p['x_cyl'], y_cyl=p['y_cyl']
            )

            # OUTER OBJECTIVE: Calculate integral and penalty
            integral_val = integral_objective_fn(problem, problem_data, problem_config)
            penalty_val = constraint_penalty_fn(problem, problem_config)
            
            w = problem_config.residual_weights
            return w['integral_objective'] * integral_val + w['constraint_penalty'] * penalty_val

        # Get the implicit gradient and update geometry
        loss_val, grad_geom = jax.value_and_grad(outer_objective_fn)(geom_params)
        updates, opt_state = optimizer.update(grad_geom, opt_state)
        geom_params = optax.apply_updates(geom_params, updates)

        # WARM START: Re-solve for the coefficients at the new position to use as the next initial guess
        coeffs = solve_field_implicit(geom_params, coeffs, static_field, problem_data, problem_config)
        
        pbar.set_description(f"Loss: {loss_val:.4e} | Position: ({geom_params['x_cyl']:.3f}, {geom_params['y_cyl']:.3f})")

    end_time = time.time()
    print(f"Bilevel optimization finished in {end_time - start_time:.2f} seconds.")
    
    # --- 3. Finalization and Plotting ---
    final_problem = LaplaceProblem(
        u_field=eqx.combine(coeffs, static_field),
        x_cyl=geom_params['x_cyl'],
        y_cyl=geom_params['y_cyl']
    )
    print(f"Final optimized cylinder position: ({final_problem.x_cyl:.3f}, {final_problem.y_cyl:.3f})")

    # (Your original plotting and saving logic can go here)
    grid_x, grid_y = grid_for_plotting.coords
    outer_mask = grid_for_plotting.data
    final_cyl_mask = (grid_x - final_problem.x_cyl)**2 + (grid_y - final_problem.y_cyl)**2 >= config.cyl_radius**2
    final_mask = outer_mask & final_cyl_mask
    u_eval = final_problem.u_field.evaluate(grid_x, grid_y).reshape(grid_x.shape)
    u_eval = jnp.where(final_mask, u_eval, jnp.nan)
    
    integral_approx = jnp.nansum(u_eval) * (2/config.n_grid)**2
    print(f"Approximate integral of final solution: {integral_approx:.6f}")

    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x, grid_y, u_eval, levels=100, cmap="jet")
    plt.colorbar(label="Solution u")
    plt.title(f"Optimized Solution (Cylinder at ({final_problem.x_cyl:.2f}, {final_problem.y_cyl:.2f}))")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("figures/laplace_bilevel_solution.png")
    plt.close()