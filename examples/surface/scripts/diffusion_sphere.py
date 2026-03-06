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
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

class DiffusionConfig(pr.ProblemConfig):
    residual_weights: Dict[str, float]
    filter_frac: float
    alpha: float
    regularization_strength: float

class DiffusionProblem(pr.AbstractProblem):
    c_field: pr.BasisField

    def get_residual_functions(self):
        return {
            "equation": self.equation_residual,
            "initial": self.initial_residual,
        }

    @eqx.filter_jit
    def equation_residual(self, problem_data: pr.data.BoundaryData, config: DiffusionConfig) -> jax.Array:
        # This function's logic remains exactly as you wanted.
        c_t = self.c_field.derivative(*problem_data.coords, order=(0,0,0,1))
        alpha = config.alpha
        laplacian = self.surface_laplacian(problem_data, self.c_field)
        return c_t - alpha * laplacian

    def surface_laplacian(self, problem_data: pr.data.BoundaryData, field: pr.BasisField):
        c_xx = field.derivative(*problem_data.coords, order=(2,0,0,0))
        c_yy = field.derivative(*problem_data.coords, order=(0,2,0,0))
        c_zz = field.derivative(*problem_data.coords, order=(0,0,2,0))
        c_xy = field.derivative(*problem_data.coords, order=(1,1,0,0))
        c_xz = field.derivative(*problem_data.coords, order=(1,0,1,0))
        c_yz = field.derivative(*problem_data.coords, order=(0,1,1,0))
        full_hessian = jnp.array([[c_xx, c_xy, c_xz], [c_xy, c_yy, c_yz], [c_xz, c_yz, c_zz]])
        normal_vector = jnp.stack(problem_data.normal_vector, axis=0)
        second_normal_deriv = jnp.einsum('ik,ijk,jk->k', normal_vector, full_hessian, normal_vector)
        laplacian = c_xx + c_yy + c_zz
        c_x = field.derivative(*problem_data.coords, order=(1,0,0,0))
        c_y = field.derivative(*problem_data.coords, order=(0,1,0,0))
        c_z = field.derivative(*problem_data.coords, order=(0,0,1,0))

        grad_c = jnp.stack((c_x, c_y, c_z), axis=0)
        grad_c_dot_normal = jnp.einsum('ik,ik->k', grad_c, normal_vector)
        # NOTE: THE MEAN CURVATURE OF A SPHERE IS 1. We do not calculate it explicitly here. It should be passed in as reference data for a more general surface.
        mean_curvature = 1
        return laplacian - second_normal_deriv - 2*mean_curvature* grad_c_dot_normal

    @staticmethod
    def _surface_gradient(spatial_coords_tuple, t, field, normal):
        grad_x = field.derivative(*spatial_coords_tuple, t, order=(1,0,0,0))
        grad_y = field.derivative(*spatial_coords_tuple, t, order=(0,1,0,0))
        grad_z = field.derivative(*spatial_coords_tuple, t, order=(0,0,1,0))

        identity = jnp.eye(len(spatial_coords_tuple))
        nnT = normal[:, None] @ normal[None, :]
        grad = jnp.stack((grad_x, grad_y, grad_z)).reshape(-1)

        return (identity - nnT) @ grad

    @eqx.filter_jit
    def initial_residual(self, problem_data: pr.data.ReferenceData, config: DiffusionConfig) -> jax.Array:
        c = self.c_field.evaluate(*problem_data.coords)
        return c - problem_data.data

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

def create_smooth_spot(points, center, radius=0.95, softness=50.0):
    """Creates a smooth, round spot on the sphere."""
    # The dot product is 1 at the center and decreases as you move away.
    dot_product = jnp.dot(points, center)
    
    # Use a sigmoid-like function (specifically, a scaled and shifted tanh)
    # for a smooth transition from 0 to 1.
    # The 'softness' parameter controls how sharp the edge is.
    arg = softness * (dot_product - radius)
    return 0.5 * (jnp.tanh(arg) + 1.0)

def initial_condition_sphere(x, y, z):
    """Defines a smooth initial condition shaped like a smiley face."""
    points = jnp.stack([x, y, z], axis=-1)

    # Define feature centers (already normalized)
    left_eye_center = jnp.array([-0.4, 0.4, 0.82])
    left_eye_center /= jnp.linalg.norm(left_eye_center)

    right_eye_center = jnp.array([0.4, 0.4, 0.82])
    right_eye_center /= jnp.linalg.norm(right_eye_center)

    mouth_center = jnp.array([0.0, -0.4, 0.916])
    mouth_center /= jnp.linalg.norm(mouth_center)

    print(left_eye_center, right_eye_center, mouth_center)
    
    # Create each feature as a smooth spot
    left_eye = create_smooth_spot(points, left_eye_center,softness=10.0)
    right_eye = create_smooth_spot(points, right_eye_center,softness=10.0)
    # You can make the mouth wider by adjusting the radius
    mouth = create_smooth_spot(points, mouth_center,softness=10.0)

    # Combine the features
    return left_eye + right_eye + mouth

def sample_sphere(n_points, key):
    """Samples points uniformly on the surface of a unit sphere."""
    key, subkey = jax.random.split(key)
    # Generate 3D points from a standard normal distribution
    points = jax.random.normal(key, (n_points, 3))
    # Normalize the points to project them onto the sphere's surface
    norm = jnp.linalg.norm(points, axis=1, keepdims=True)
    points = points / norm
    # Unpack into x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return x, y, z

def sample_points(config):
    n_pde = config.n_pde
    n_initial = config.n_initial
    n_t = config.n_t

    # t_vec = jnp.linspace(-1,1,n_t)

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    x, y, z = sample_sphere(n_pde*n_t, subkey)
    key, subkey = jax.random.split(key)
    t = jax.random.uniform(subkey, (n_pde*n_t,), minval=-1, maxval=1)
    key, subkey = jax.random.split(key)
    final_sample = 500
    t_final = jnp.ones(final_sample)
    x_final,y_final,z_final = sample_sphere(final_sample, subkey)
    x = jnp.concatenate((x, x_final))
    y = jnp.concatenate((y, y_final))
    z = jnp.concatenate((z, z_final))
    t = jnp.concatenate((t, t_final))
    
    collocation_data = pr.data.BoundaryData(coords = (x, y, z, t), normal_vector = (x,y,z))

    key, subkey = jax.random.split(key)
    x_ic, y_ic, z_ic = sample_sphere(n_initial, subkey)
    initial_values = initial_condition_sphere(x_ic, y_ic, z_ic)
    t_ic = -1 * jnp.ones(n_initial)
    
    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot(111, projection='3d')
    im = ax.scatter(x_ic, y_ic, z_ic, c=initial_values, cmap="viridis")
    f.colorbar(im, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("figures/initial_condition.png")
    plt.close()

    initial_data = pr.data.ReferenceData(coords=(x_ic, y_ic, z_ic, t_ic), data=initial_values)

    return pr.data.ProblemData(equation=collocation_data, initial=initial_data)


def save_results(config, optimized_problem):
    result_path = pathlib.Path(config.results_dir) / config.script_name
    result_path.mkdir(parents=True, exist_ok=True)
    c_coeffs = optimized_problem.c_field.coeffs.value
    field_dict = {
        "c": c_coeffs
    }
    jnp.save(result_path / "c_coeffs.npy", c_coeffs)
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
        config["script_name"] = f"{args.N}_diffusion_sphere"
    else:
        config["script_name"] = f"{timestamp}_diffusion_sphere"
    return config

if __name__ == "__main__":
    config = load_config("configs/diffusion_sphere.yml")
    config = parse_args(config)
    problem_data = sample_points(config)
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config.basis_Nx, config.basis_Ny, config.basis_Nz, config.basis_Nt))
    c_field = pr.BasisField(basis, pr.fields.PreconditionedChebyshevCoeffs.make_zero(basis.degs))
    problem = DiffusionProblem(c_field)
    problem_config = DiffusionConfig.from_config(config)
    solver = pr.get_solver(config)
    
    print(f"Script name: {config.script_name}", flush=True)

    start_time = time.time()
    problem_config = DiffusionConfig.from_config(config)
    optimized_problem, log_data = solver.solve(problem, problem_data, config, DiffusionConfig.from_config(config))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds", flush=True)
    save_results(config, optimized_problem)