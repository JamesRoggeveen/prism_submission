import equinox as eqx
import jax
import jax.numpy as jnp
from .data import ProblemData, SystemConfig
from typing import Dict, Callable
from abc import abstractmethod

class ProblemConfig(eqx.Module):
    residual_weights: Dict[str, float]

    @classmethod
    def from_config(cls, config: SystemConfig):
        # Get all fields from the config that match the class fields
        kwargs = {}
        for field in cls.__annotations__:
            if field in config:
                kwargs[field] = config[field]
        return cls(**kwargs)


class AbstractProblem(eqx.Module):
    @abstractmethod
    def get_residual_functions(self) -> Dict[str, Callable]:
        """
        Returns a dictionary mapping residual names to their functions.

        Each function must accept two arguments: a `AbstractData` object and a `SolverConfig` object.
        
        Example:
            return {
                "equation": self.equation_residual,
                "boundary": self.boundary_residual,
                "observation": self.observation_residual
            }
        """
        raise NotImplementedError

    def total_residual(self, problem_data: ProblemData, config: ProblemConfig) -> jax.Array:
        """
        Calculates the combined, weighted residual for the optimizer.
        This method is generic and does not need to be overridden.
        """
        all_residuals = []
        residual_fns = self.get_residual_functions()

        for name, func in residual_fns.items():
            # Check if data and a weight are provided for this residual
            if name in problem_data.data.keys() and name in config.residual_weights:
                residual_data = problem_data[name]
                weight = config.residual_weights[name]
                
                # Calculate the raw residual
                res = func(residual_data, config)
                
                n_pts = res.shape[0] if res.shape[0] > 0 else 1
                weighted_res = jnp.sqrt(weight) * res / jnp.sqrt(n_pts)
                all_residuals.append(weighted_res)

        if not all_residuals:
            raise ValueError("No residuals were calculated. Check residual_data and residual_weights in your config.")

        return jnp.concatenate(all_residuals)
    
    @eqx.filter_jit
    def loss_function(self, problem_data: ProblemData, config: ProblemConfig) -> jax.Array:
        """
        Calculates the loss function for the optimizer.
        """
        return jnp.sum(self.total_residual(problem_data, config)**2)
