import equinox as eqx
from abc import abstractmethod
import optimistix as optx
import lineax as lx
import optax
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.flatten_util
from collections import defaultdict

# Assume these are defined in your project's data and problem modules
from .data import ProblemData, SystemConfig, CollocationPoints
from ._problem import AbstractProblem, ProblemConfig


## ------------------------------------------------------------------
## │ A. CORE ABSTRACTIONS │
## ------------------------------------------------------------------

class AbstractSolver(eqx.Module):
    """Abstract interface for all solvers."""
    @abstractmethod
    def solve(self, problem: AbstractProblem, problem_data: ProblemData, config: SystemConfig, problem_config: ProblemConfig):
        pass

class AbstractLoss(eqx.Module):
    """Abstract interface for loss calculation strategies."""
    @abstractmethod
    def get_loss_and_logs(self, problem, problem_data, problem_config, **kwargs):
        """
        Calculates the scalar loss and auxiliary data for logging.
        This method does NOT compute gradients.

        Returns:
            A tuple: `(scalar_loss, log_dictionary)`.
        """
        pass


## ------------------------------------------------------------------
## │ B. OPTIMISTIX SOLVERS │
## ------------------------------------------------------------------

class OptimistixSolver(AbstractSolver):
    """Base class for Optimistix-based solvers."""
    solver: optx.AbstractLeastSquaresSolver

    def solve(self, problem: AbstractProblem, problem_data: ProblemData, config: SystemConfig, problem_config: ProblemConfig):
        @eqx.filter_jit
        def loss_fn(params, static_problem):
            problem_for_loss = eqx.combine(params, static_problem[0])
            return problem_for_loss.total_residual(problem_data, problem_config)

        params, static = eqx.partition(problem, eqx.is_array)
        sol = optx.least_squares(
            loss_fn,
            self.solver,
            params,
            args=(static,),
            max_steps=config.max_steps
        )
        final_model = eqx.combine(sol.value, static)
        final_log = _process_log_data({})
        return final_model, final_log

class LevenbergMarquardtSolver(OptimistixSolver):
    def __init__(self, **kwargs):
        self.solver = optx.LevenbergMarquardt(**kwargs)

class DoglegSolver(OptimistixSolver):
    def __init__(self, well_posed=True, **kwargs):
        if not well_posed:
            kwargs["linear_solver"] = lx.AutoLinearSolver(well_posed=False)
        self.solver = optx.Dogleg(**kwargs)


## ------------------------------------------------------------------
## │ C. OPTAX-BASED SOLVERS │
## ------------------------------------------------------------------

class OptaxSolver(AbstractSolver):
    """
    A generic solver for any Optax optimizer and any loss strategy.
    It computes gradients with respect to the standard 'problem' state.
    """
    optimizer: optax.GradientTransformation
    loss_strategy: AbstractLoss

    @eqx.filter_jit
    def _make_step(self, problem, problem_data, problem_config, opt_state, key):
        # Define the function that computes the loss; this is what we differentiate.
        def loss_fn(p):
            # The 'requires_key' attribute is checked to decide how to call.
            # This logic is handled inside the get_loss_and_logs methods.
            loss, logs = self.loss_strategy.get_loss_and_logs(
                p, problem_data, problem_config, key=key
            )
            return loss, logs

        # The solver is responsible for computing the gradient.
        (loss, log_data), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(problem)

        params, static = eqx.partition(problem, eqx.is_array)
        updates, opt_state = self.optimizer.update(grad, opt_state, params)
        problem = eqx.apply_updates(problem, updates)
        
        return problem, opt_state, log_data

    def solve(self, problem: AbstractProblem, problem_data: ProblemData, config: SystemConfig, problem_config: ProblemConfig):
        params, _ = eqx.partition(problem, eqx.is_array)
        opt_state = self.optimizer.init(params)
        key = jax.random.PRNGKey(config.seed)
        
        log_data = []
        pbar = tqdm(range(config.n_epochs), disable=not config.verbose, desc="Initializing...")
        
        for _ in pbar:
            key, step_key = jax.random.split(key)
            problem, opt_state, epoch_log = self._make_step(
                problem, problem_data, problem_config, opt_state, step_key
            )
            pbar.set_description(f"Loss: {epoch_log['total_loss']:.4e}")
            log_data.append(epoch_log)
            
        final_log = _process_log_data(log_data)
        return problem, final_log

class LBFGSSolver(OptaxSolver):
    """
    A specialized solver for L-BFGS that correctly handles static model data
    during its line search updates.
    """
    @eqx.filter_jit
    def _make_step(self, problem, problem_data, problem_config, opt_state, key):
        # 1. Separate the trainable parameters from the static structure.
        params, static = eqx.partition(problem, eqx.is_array)

        # 2. Define a value_fn for the optimizer. It MUST accept only parameters
        #    and know how to recombine them with the static data.
        def value_fn_for_optimizer(p_params):
            # Reconstruct the full problem object for this evaluation.
            p_problem = eqx.combine(p_params, static)
            loss, _ = self.loss_strategy.get_loss_and_logs(
                p_problem, problem_data, problem_config, key=key
            )
            return loss

        # 3. Define a function to get the initial value and gradient.
        #    This also needs to perform the combine step.
        def value_and_grad_fn(p_params):
            p_problem = eqx.combine(p_params, static)
            loss, logs = self.loss_strategy.get_loss_and_logs(
                p_problem, problem_data, problem_config, key=key
            )
            return loss, logs

        (loss, log_data), grad = eqx.filter_value_and_grad(
            value_and_grad_fn, has_aux=True
        )(params)

        # 4. Call the optimizer with the correctly defined value_fn.
        updates, opt_state = self.optimizer.update(
            grad,
            opt_state,
            params,
            value=loss,
            grad=grad,
            value_fn=value_fn_for_optimizer
        )
        
        # apply_updates is the standard way to handle the output of optimizer.update
        new_params = optax.apply_updates(params, updates)
        problem = eqx.combine(new_params, static)
        
        return problem, opt_state, log_data

class _CollocationTrainableState(eqx.Module):
    """Helper to group model params and latent coords for the optimizer."""
    params: eqx.Module
    latent_pde_coords: tuple

class AdaptiveCollocationSolver(AbstractSolver):
    """
    A specialized solver that trains both model parameters and collocation points.
    It computes gradients with respect to the '_CollocationTrainableState'.
    """
    optimizer: optax.GradientTransformation
    loss_strategy: AbstractLoss

    @eqx.filter_jit
    def _make_step(self, trainable_state, static_problem, pde_points, static_data, problem_config, opt_state, key):
        # Define the function that computes the loss for the combined state.
        def loss_fn_for_strategy(state):
            prob = eqx.combine(state.params, static_problem)
            phys_coords = jax.tree_util.tree_map(jnp.tanh, state.latent_pde_coords)
            
            current_data_dict = {**static_data, 'equation': CollocationPoints(coords=phys_coords)}
            current_problem_data = ProblemData(**current_data_dict)

            loss, logs = self.loss_strategy.get_loss_and_logs(
                prob, current_problem_data, problem_config, key=key
            )
            return loss, logs

        # The solver computes the gradient for its unique trainable state.
        (loss, logs), grad_state = eqx.filter_value_and_grad(
            loss_fn_for_strategy, has_aux=True
        )(trainable_state)
        
        # Perform gradient ASCENT for coordinates by negating their gradients
        neg_grad_coords = jax.tree_util.tree_map(lambda x: -x, grad_state.latent_pde_coords)
        combined_grad = eqx.tree_at(lambda s: s.latent_pde_coords, grad_state, neg_grad_coords)

        updates, opt_state = self.optimizer.update(combined_grad, opt_state)
        new_state = eqx.apply_updates(trainable_state, updates)
        return new_state, opt_state, logs

    def solve(self, problem: AbstractProblem, problem_data: ProblemData, config: SystemConfig, problem_config: ProblemConfig):
        # --- SETUP ---
        pde_points = problem_data['equation']
        static_data = {k: v for k, v in problem_data.data.items() if k != 'equation'}
        params, static_problem = eqx.partition(problem, eqx.is_array)
        epsilon = 1e-6
        initial_latents = jax.tree_util.tree_map(lambda c: jnp.arctanh(jnp.clip(c, -1 + epsilon, 1 - epsilon)), pde_points.coords)
        trainable_state = _CollocationTrainableState(params=params, latent_pde_coords=initial_latents)
        opt_state = self.optimizer.init(eqx.filter(trainable_state, eqx.is_array))
        key = jax.random.PRNGKey(config.seed)
        
        log_data = {"per_epoch": []}
        pbar = tqdm(range(config.n_epochs), disable=not config.verbose, desc="Initializing...")

        # --- TRAINING LOOP ---
        for _ in pbar:
            key, step_key = jax.random.split(key)
            trainable_state, opt_state, epoch_log = self._make_step(
                trainable_state, static_problem, pde_points, static_data, problem_config, opt_state, step_key
            )
            pbar.set_description(f"Loss: {epoch_log['total_loss']:.4e}")
            log_data["per_epoch"].append(epoch_log)

        # --- TEARDOWN ---
        final_problem = eqx.combine(trainable_state.params, static_problem)
        final_coords = jax.tree_util.tree_map(jnp.tanh, trainable_state.latent_pde_coords)
        log_data["final_collocation_points"] = final_coords
        
        final_log = _process_log_data(log_data)
        return final_problem, final_log

class StaticPreconditionedSolver(OptaxSolver):
    """
    Implements a solver that uses a fixed Hessian (Gauss-Newton approximation)
    computed at the beginning of training to precondition the gradients.
    This is conceptually similar to Natural Gradient Descent.
    """
    damping: float = 1e-6  # Small value to ensure the matrix is invertible

    @eqx.filter_jit
    def _make_step(self, problem, problem_data, problem_config, opt_state, key, G_inv_flat):
        # This step is similar to the standard OptaxSolver...
        def loss_fn(p):
            loss, logs = self.loss_strategy.get_loss_and_logs(
                p, problem_data, problem_config, key=key
            )
            return loss, logs
        
        (loss, log_data), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(problem)

        # --- PRECONDITIONING STEP ---
        # 1. Flatten the gradient PyTree into a vector
        grad_flat, unflatten_fn = jax.flatten_util.ravel_pytree(grad)
        
        # 2. Apply the preconditioning by multiplying with the inverse Hessian
        preconditioned_grad_flat = G_inv_flat @ grad_flat
        
        # 3. Unflatten the preconditioned vector back into a PyTree
        preconditioned_grad = unflatten_fn(preconditioned_grad_flat)
        
        # 4. Use the preconditioned gradient for the optimizer update
        params, static = eqx.partition(problem, eqx.is_array)
        updates, opt_state = self.optimizer.update(preconditioned_grad, opt_state, params)
        problem = eqx.apply_updates(problem, updates)
        
        return problem, opt_state, log_data

    def solve(self, problem: AbstractProblem, problem_data: ProblemData, config: SystemConfig, problem_config: ProblemConfig):
        print("StaticPreconditionedSolver: Performing initial Hessian calculation...")
        # --- INITIAL HESSIAN CALCULATION ---
        params, static = eqx.partition(problem, eqx.is_array)

        # Define a function that returns the residual vector for the equation term
        def residual_fn(p_params):
            p_problem = eqx.combine(p_params, static)
            return p_problem.equation_residual(problem_data['equation'], problem_config)

        # Compute the Jacobian (J) of the residual function
        J = jax.jacfwd(residual_fn)(params)
        
        # The Jacobian PyTree needs to be flattened into a matrix
        J_flat, _ = jax.flatten_util.ravel_pytree(J)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        # For a PyTree of residuals, reshape the flattened Jacobian
        J_matrix = J_flat.reshape(-1, num_params)

        # Form the Gauss-Newton Hessian approximation: G = J^T J
        G = J_matrix.T @ J_matrix
        
        # Add damping for numerical stability and compute the inverse
        num_params_from_shape = G.shape[0]
        G_inv = jnp.linalg.inv(G + self.damping * jnp.eye(int(num_params_from_shape)))
        # --- END FIX ---
        print("StaticPreconditionedSolver: Initial calculation complete.")

        # --- STANDARD TRAINING LOOP ---
        opt_state = self.optimizer.init(params)
        key = jax.random.PRNGKey(config.seed)
        log_data = []
        pbar = tqdm(range(config.n_epochs), disable=not config.verbose, desc="Initializing...")
        
        for _ in pbar:
            key, step_key = jax.random.split(key)
            # Pass the inverse Hessian to the JIT-compiled step function
            problem, opt_state, epoch_log = self._make_step(
                problem, problem_data, problem_config, opt_state, step_key, G_inv
            )
            pbar.set_description(f"Loss: {epoch_log['total_loss']:.4e}")
            log_data.append(epoch_log)
            
        final_log = _process_log_data(log_data)
        return problem, final_log

class StaticHessianSolver(StaticPreconditionedSolver):
    """
    Implements a solver that uses the exact, full Hessian of the loss,
    computed once at the beginning of training, to precondition the gradients.
    """
    def solve(self, problem: AbstractProblem, problem_data: ProblemData, config: SystemConfig, problem_config: ProblemConfig):
        print("StaticHessianSolver: Performing initial FULL Hessian calculation...")
        # --- INITIAL HESSIAN CALCULATION ---
        params, static = eqx.partition(problem, eqx.is_array)

        # 1. Define a SCALAR loss function for the Hessian calculation.
        #    We only use the equation residual as per your idea.
        def scalar_equation_loss_fn(p_params):
            p_problem = eqx.combine(p_params, static)
            residuals = p_problem.equation_residual(problem_data['equation'], problem_config)
            return jnp.sum(residuals**2)

        # 2. Compute the full Hessian using eqx.filter_hessian.
        #    The result is a PyTree of PyTrees.
        H_pytree = eqx.filter_hessian(scalar_equation_loss_fn)(params)
        
        # 3. Flatten the Hessian PyTree into a 2D matrix
        H_flat, _ = jax.flatten_util.ravel_pytree(H_pytree)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        H_matrix = H_flat.reshape(num_params, num_params)

        # 4. Add damping and compute the inverse. Damping is CRITICAL here
        #    to handle potential negative eigenvalues and ensure stability.
        num_params_from_shape = H_matrix.shape[0]
        H_inv = jnp.linalg.inv(H_matrix + self.damping * jnp.eye(int(num_params_from_shape)))
        print("StaticHessianSolver: Initial calculation complete.")

        # --- STANDARD TRAINING LOOP (identical to parent class) ---
        opt_state = self.optimizer.init(params)
        key = jax.random.PRNGKey(config.seed)
        log_data = []
        pbar = tqdm(range(config.n_epochs), disable=not config.verbose, desc="Initializing...")
        
        for _ in pbar:
            key, step_key = jax.random.split(key)
            # The _make_step function is inherited and works perfectly with H_inv
            problem, opt_state, epoch_log = self._make_step(
                problem, problem_data, problem_config, opt_state, step_key, H_inv
            )
            pbar.set_description(f"Loss: {epoch_log['total_loss']:.4e}")
            log_data.append(epoch_log)
            
        final_log = _process_log_data(log_data)
        return problem, final_log
## ------------------------------------------------------------------
## │ D. LOSS CALCULATION STRATEGIES │
## ------------------------------------------------------------------

class StandardLoss(AbstractLoss):
    """The simplest loss: the mean squared error of all residuals."""
    def get_loss_and_logs(self, problem, problem_data, problem_config, **kwargs):
        loss = problem.loss_function(problem_data, problem_config)
        logs = {"total_loss": loss}
        return loss, logs

class RegularizedStandardLoss(AbstractLoss):
    def get_loss_and_logs(self, problem, problem_data, problem_config, **kwargs):
        # Extract all AbstractField attributes and their coefficients for regularization
        field_coeffs = []
        for attr_name in dir(problem):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr_value = getattr(problem, attr_name)
                # Check if the attribute is an AbstractField (duck typing approach)
                if hasattr(attr_value, 'coeffs') and hasattr(attr_value.coeffs, 'value'):
                    field_coeffs.append(attr_value.coeffs.value)
        
        # Combine all field coefficients for regularization
        if field_coeffs:
            all_coeffs = jnp.concatenate([coeff.flatten() for coeff in field_coeffs])
        else:
            all_coeffs = jnp.array([])
        loss = problem.loss_function(problem_data, problem_config) + problem_config.regularization_strength * jnp.sum(jnp.abs(all_coeffs))
        logs = {"total_loss": loss}
        return loss, logs

class AdaptiveLoss(AbstractLoss):
    """Implements adaptive re-weighting based on gradient magnitudes."""
    max_weight_clip: float = 100.0
    
    def get_loss_and_logs(self, problem, problem_data, problem_config, **kwargs):
        params, static = eqx.partition(problem, eqx.is_array)
        losses, grads = {}, {}
        residual_fns = problem.get_residual_functions()

        for name, res_fn in residual_fns.items():
            def per_term_loss_fn(p_inner):
                prob_i = eqx.combine(p_inner, static)
                res_vals = getattr(prob_i, res_fn.func.__name__)(problem_data[name], problem_config)
                return jnp.mean(res_vals**2)
            loss_val, grad_val = eqx.filter_value_and_grad(per_term_loss_fn)(params)
            losses[name], grads[name] = loss_val, grad_val

        grad_mags = {name: compute_mean_abs_grad(grad) for name, grad in grads.items()}
        mean_of_mags = jnp.mean(jnp.stack(list(grad_mags.values())))
        lambdas = {name: jax.lax.stop_gradient(jnp.clip(mean_of_mags / (mag + 1e-8), a_max=self.max_weight_clip))
                   for name, mag in grad_mags.items()}
        # lambdas = {name: jax.lax.stop_gradient(mean_of_mags / (mag + 1e-8))
        #            for name, mag in grad_mags.items()}

        total_loss = sum(lambdas[name] * losses[name] for name in losses)
        logs = {"total_loss": total_loss, "unweighted_losses": losses, "weights": lambdas, "grad_mags": grad_mags}
        return total_loss, logs

class RegularizedAdaptiveLoss(AdaptiveLoss):
    def get_loss_and_logs(self, problem, problem_data, problem_config, **kwargs):
        # Call parent class method to get adaptive loss behavior
        total_loss, logs = super().get_loss_and_logs(problem, problem_data, problem_config, **kwargs)
        
        # Extract all AbstractField attributes and their coefficients for regularization
        field_coeffs = []
        for attr_name in dir(problem):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr_value = getattr(problem, attr_name)
                # Check if the attribute is an AbstractField (duck typing approach)
                if hasattr(attr_value, 'coeffs') and hasattr(attr_value.coeffs, 'value'):
                    field_coeffs.append(attr_value.coeffs.value)
        
        # Combine all field coefficients for regularization
        if field_coeffs:
            all_coeffs = jnp.concatenate([coeff.flatten() for coeff in field_coeffs])
            regularization_term = problem_config.regularization_strength * jnp.sum(jnp.abs(all_coeffs))
            total_loss = total_loss + regularization_term
        
        logs["total_loss"] = total_loss
        return total_loss, logs

class CausalAdaptiveLoss(AbstractLoss):
    """
    Combines Causal curriculum learning with Adaptive re-weighting.
    This works as a loss strategy because the final scalar loss depends
    on the magnitude of intermediate gradients.
    """
    max_weight_clip: float = 100.0
    
    def get_loss_and_logs(self, problem, problem_data, problem_config, **kwargs):
        params, static = eqx.partition(problem, eqx.is_array)
        losses, grads = {}, {}
        residual_fns = problem.get_residual_functions()
        pde_residual_name = "equation"

        for name, res_fn in residual_fns.items():
            method_name = res_fn.func.__name__
            if name == pde_residual_name:
                def causal_pde_loss_fn(p):
                    prob_i = eqx.combine(p, static)
                    data = problem_data[name]
                    residuals = getattr(prob_i, method_name)(data, problem_config)
                    num_chunks, tol = problem_config.causal_num_chunks, problem_config.causal_tol
                    
                    # Causal weighting logic
                    num_pts = residuals.shape[0]
                    pts_per_chunk = num_pts // num_chunks
                    reshaped_res = residuals[:pts_per_chunk * num_chunks].reshape(num_chunks, -1)
                    chunk_loss = jnp.mean(reshaped_res**2, axis=1)
                    cum_loss = jnp.pad(jax.lax.cumsum(chunk_loss)[:-1], (1, 0))
                    w = jax.lax.stop_gradient(jnp.exp(-tol * cum_loss))
                    return jnp.mean(chunk_loss * w)
                loss_fn = causal_pde_loss_fn
            else:
                def standard_loss_fn(p):
                    prob_i = eqx.combine(p, static)
                    data = problem_data[name]
                    res = getattr(prob_i, method_name)(data, problem_config)
                    return jnp.mean(res**2)
                loss_fn = standard_loss_fn

            loss_val, grad_val = eqx.filter_value_and_grad(loss_fn)(params)
            losses[name], grads[name] = loss_val, grad_val
        
        # Adaptive weighting logic based on gradient magnitudes
        grad_mags = {n: compute_mean_abs_grad(g) for n, g in grads.items()}
        mean_mag = jnp.mean(jnp.stack(list(grad_mags.values())))
        lambdas = {n: jax.lax.stop_gradient(jnp.clip(mean_mag / (mag + 1e-8), a_max=self.max_weight_clip)) 
                   for n, mag in grad_mags.items()}
        
        total_loss = sum(lambdas[n] * losses[n] for n in losses)
        logs = {"total_loss": total_loss, "unweighted_losses": losses, "weights": lambdas}
        
        return total_loss, logs

class CausalLoss(AbstractLoss):
    """
    Implements a causal training curriculum for the PDE loss term.

    This strategy sequentially introduces regions of the PDE problem,
    weighting them based on the accumulated loss of previously introduced
    regions. All other loss terms (e.g., boundary conditions) are
    weighted equally.
    """
    def get_loss_and_logs(self, problem, problem_data, problem_config, **kwargs):
        total_loss = 0.0
        logs = {"unweighted_losses": {}, "causal_weights": None}
        residual_fns = problem.get_residual_functions()
        pde_residual_name = "equation"

        for name, res_fn in residual_fns.items():
            data = problem_data[name]
            method_to_call = getattr(problem, res_fn.func.__name__)
            residuals = method_to_call(data, problem_config)

            if name == pde_residual_name:
                # --- Causal Logic for the PDE loss ---
                num_chunks = problem_config.causal_num_chunks
                tol = problem_config.causal_tol
                
                num_pts = residuals.shape[0]
                pts_per_chunk = num_pts // num_chunks
                reshaped_res = residuals[:pts_per_chunk * num_chunks].reshape(num_chunks, -1)
                
                per_chunk_loss = jnp.mean(reshaped_res**2, axis=1)

                # Calculate weights based on accumulated loss of previous chunks
                cumulative_loss = jax.lax.cumsum(per_chunk_loss)
                accumulated_loss_up_to_prev = jnp.pad(cumulative_loss[:-1], (1, 0))
                w = jax.lax.stop_gradient(jnp.exp(-tol * accumulated_loss_up_to_prev))
                
                # Apply weights to get the final PDE loss
                term_loss = jnp.mean(per_chunk_loss * w)
                logs["causal_weights"] = w
            else:
                # --- Standard MSE for all other loss terms ---
                term_loss = jnp.mean(residuals**2)

            total_loss += term_loss
            logs["unweighted_losses"][name] = term_loss

        logs["total_loss"] = total_loss
        return total_loss, logs

class ConflictFreeSolver(OptaxSolver):
    """
    A specialized solver that implements Gradient Surgery (PCGrad) to prevent
    conflicting gradients from different loss terms.
    """
    @eqx.filter_jit
    def _make_step(self, problem, problem_data, problem_config, opt_state, key):
        params, static = eqx.partition(problem, eqx.is_array)

        # 1. Calculate all individual gradients and losses
        losses, grads = {}, {}
        residual_fns = problem.get_residual_functions()
        
        for name, res_fn in residual_fns.items():
            # For each loss term, calculate its value and gradient separately
            def per_residual_loss_fn(p_params):
                prob_i = eqx.combine(p_params, static)
                method_to_call = getattr(prob_i, res_fn.func.__name__)
                residual_values = method_to_call(problem_data[name], problem_config)
                return jnp.mean(residual_values**2)

            loss_val, grad_val = eqx.filter_value_and_grad(per_residual_loss_fn)(params)
            losses[name], grads[name] = loss_val, grad_val

        # 2. Perform Gradient Surgery (PCGrad)
        grad_names = list(grads.keys())
        _, unflatten_fn = jax.flatten_util.ravel_pytree(grads[grad_names[0]])
        flat_grads = {name: jax.flatten_util.ravel_pytree(g)[0] for name, g in grads.items()}

        modified_flat_grads = {}
        for name_i in grad_names:
            g_i = flat_grads[name_i]
            for name_j in grad_names:
                if name_i == name_j:
                    continue
                
                g_j = flat_grads[name_j]
                dot_product = jnp.dot(g_i, g_j)
                
                def project_gradient():
                    return g_i - (dot_product / (jnp.dot(g_j, g_j) + 1e-8)) * g_j

                g_i = jax.lax.cond(dot_product < 0.0, project_gradient, lambda: g_i)
                
            modified_flat_grads[name_i] = g_i

        # 3. Sum the modified gradients to get the final update direction
        final_flat_grad = sum(modified_flat_grads.values())
        final_grad_pytree = unflatten_fn(final_flat_grad)

        # 4. Apply the updates using the modified gradient
        updates, opt_state = self.optimizer.update(final_grad_pytree, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        problem = eqx.combine(new_params, static)

        # For logging, report the simple unweighted sum of losses
        total_unweighted_loss = sum(losses.values())
        logs = {"total_loss": total_unweighted_loss, "unweighted_losses": losses}
        
        return problem, opt_state, logs

def get_optimizer(config: SystemConfig):
    """Factory for optax optimizers."""
    
    if config.optimizer_name.lower() == "adam":
        # # 1. Define the learning rate schedule
        # lr_schedule = optax.exponential_decay(
        #     init_value=config.learning_rate,  # The starting learning rate
        #     transition_steps=5000,            # How many steps for one decay cycle
        #     decay_rate=0.9                    # The rate of decay (e.g., 0.9 = 10% reduction)
        # )

        # # 2. Chain the components together
        # return optax.chain(
        #     # First, clip the gradients to prevent explosions
        #     optax.clip_by_global_norm(1.0),
        #     # Then, apply the Adam update rule using the scheduled learning rate
        #     optax.adam(learning_rate=lr_schedule)
        # )
        return optax.adam(learning_rate=config.learning_rate)
    
    elif config.optimizer_name.lower() == "lbfgs":
        return optax.lbfgs(learning_rate=config.learning_rate)

    elif config.optimizer_name.lower() == "nadam":
        return optax.nadam(learning_rate=config.learning_rate)
    
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported.")

def get_loss_strategy(config: SystemConfig):
    """Factory for loss calculation strategies."""
    strategy_name = config.loss_strategy
    clipping = config.get("grad_clipping", 100.0)
    if strategy_name == "Standard":
        return StandardLoss()
    elif strategy_name == "Adaptive":
        return AdaptiveLoss(max_weight_clip=clipping)
    elif strategy_name == "RegularizedStandard":
        return RegularizedStandardLoss()
    elif strategy_name == "RegularizedAdaptive":
        return RegularizedAdaptiveLoss(max_weight_clip=clipping)
    elif strategy_name == "Causal":
        return CausalLoss()
    elif strategy_name == "CausalAdaptive":
        return CausalAdaptiveLoss()
    else:
        raise ValueError(f"Loss strategy {strategy_name} not supported.")

def get_solver(config: SystemConfig):
    """Main factory to compose and return the final solver object."""
    solver_type = config.solver_type

    if solver_type == "ConflictFree": # <-- New solver type for PCGrad
        optimizer = get_optimizer(config)
        # Note: ConflictFree doesn't use a loss_strategy, its logic is self-contained.
        return ConflictFreeSolver(optimizer=optimizer, loss_strategy=None)

    elif solver_type == "StaticPreconditioned": # <-- Add this new option
        optimizer = get_optimizer(config)
        loss_strategy = get_loss_strategy(config) # Still needed for logging
        return StaticPreconditionedSolver(optimizer=optimizer, loss_strategy=loss_strategy, damping=float(config.damping))
    elif solver_type == "StaticHessian": # <-- Add this new option
        optimizer = get_optimizer(config)
        loss_strategy = get_loss_strategy(config)
        damping = config.solver_kwargs.get("damping", 1e-4) # Damping is very important
        return StaticHessianSolver(
            optimizer=optimizer, 
            loss_strategy=loss_strategy, 
            damping=damping
        )

    if solver_type in ["LevenbergMarquardt", "Dogleg"]:
        if solver_type == "LevenbergMarquardt":
            return LevenbergMarquardtSolver(**config.solver_kwargs)
        elif solver_type == "Dogleg":
            return DoglegSolver(**config.solver_kwargs)
    
    elif solver_type in ["FirstOrder", "AdaptiveCollocation"]:
        optimizer = get_optimizer(config)
        loss_strategy = get_loss_strategy(config)
        
        # **THE FIX**: Select the correct solver based on the optimizer type
        if config.optimizer_name == "lbfgs":
            SolverClass = LBFGSSolver
        else:
            SolverClass = OptaxSolver

        if solver_type == "FirstOrder":
            return SolverClass(optimizer=optimizer, loss_strategy=loss_strategy)
        elif solver_type == "AdaptiveCollocation":
            # You would need to create an AdaptiveCollocationLBFGSSolver if you
            # want to combine these two advanced techniques.
            if config.optimizer_name == "lbfgs":
                raise NotImplementedError("AdaptiveCollocation with L-BFGS requires a dedicated solver class.")
            return AdaptiveCollocationSolver(optimizer=optimizer, loss_strategy=loss_strategy)
            
    else:
        raise ValueError(f"Solver type {config.solver_type} not supported.")


## ------------------------------------------------------------------
## │ F. HELPER FUNCTIONS │
## ------------------------------------------------------------------
        
# def compute_mean_abs_grad(grad):
#     """Calculates the mean absolute value of a PyTree of gradients."""
#     leaves = jax.tree_util.tree_leaves(grad)
#     if not leaves: return jnp.array(0.0)
#     total_abs = sum(jnp.sum(jnp.abs(x)) for x in leaves)
#     total_elems = sum(x.size for x in leaves)
#     return total_abs / total_elems

def compute_mean_abs_grad(grad: eqx.Module) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(grad)
    if not leaves:
        return 0.0
    return jnp.mean(jnp.concatenate([jnp.abs(leaf.ravel()) for leaf in leaves]))

def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.'):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _process_log_data(raw_log_data) -> dict:
    """Transforms raw solver logs into a plot-friendly format."""
    if not raw_log_data:
        return {}

    if isinstance(raw_log_data, list):
        epoch_logs = raw_log_data
        final_data = {}
    elif isinstance(raw_log_data, dict) and "per_epoch" in raw_log_data:
        epoch_logs = raw_log_data.get("per_epoch", [])
        final_data = {k: v for k, v in raw_log_data.items() if k != "per_epoch"}
    else:
        return raw_log_data
    
    if not epoch_logs:
        return final_data

    processed_dict = defaultdict(list)
    for epoch_dict in epoch_logs:
        flat_dict = _flatten_dict(epoch_dict)
        for key, value in flat_dict.items():
            processed_dict[key].append(value)

    for key, value_list in processed_dict.items():
        try:
            final_data[key] = jnp.array(value_list)
        except Exception:
            final_data[key] = value_list
            
    return final_data