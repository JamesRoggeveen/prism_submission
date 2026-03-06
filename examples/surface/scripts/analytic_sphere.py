import numpy as np
import jax.numpy as jnp
from scipy.special import sph_harm, roots_legendre
import matplotlib.pyplot as plt

# --- Your JAX initial condition function ---
def create_smooth_spot(points, center, radius=0.95, softness=50.0):
    dot_product = jnp.dot(points, center)
    arg = softness * (dot_product - radius)
    return 0.5 * (jnp.tanh(arg) + 1.0)

def initial_condition_sphere(x, y, z):
    points = jnp.stack([x, y, z], axis=-1)
    left_eye_center = jnp.array([-0.4, 0.4, 0.82])
    left_eye_center /= jnp.linalg.norm(left_eye_center)
    right_eye_center = jnp.array([0.4, 0.4, 0.82])
    right_eye_center /= jnp.linalg.norm(right_eye_center)
    mouth_center = jnp.array([0.0, -0.4, 0.916])
    mouth_center /= jnp.linalg.norm(mouth_center)
    left_eye = create_smooth_spot(points, left_eye_center, softness=10.0)
    right_eye = create_smooth_spot(points, right_eye_center, softness=10.0)
    mouth = create_smooth_spot(points, mouth_center, softness=10.0)
    return left_eye + right_eye + mouth

# --- Function to compute the coefficients ---
def get_spherical_harmonic_coeffs(initial_cond_func, n_max, n_points=100):
    cos_theta, w_theta = roots_legendre(n_points)
    theta = np.arccos(cos_theta)
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    w_phi = 2 * np.pi / n_points
    PHI, THETA = np.meshgrid(phi, theta)

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    print("Evaluating initial condition on the grid...")
    C0_grid = initial_cond_func(X.flatten(), Y.flatten(), Z.flatten())
    C0_grid = C0_grid.reshape(n_points, n_points)

    coeffs = {}
    print(f"Computing coefficients up to n_max = {n_max}...")
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            Ynm_conj = np.conj(sph_harm(m, n, PHI, THETA))
            integrand = C0_grid * Ynm_conj
            integral = np.sum(integrand * w_theta[:, np.newaxis] * w_phi)
            coeffs[(n, m)] = integral
    return coeffs

def reconstruct_solution(coeffs, theta, phi, t, D, R):
    """
    Reconstructs the concentration field from spherical harmonic coefficients
    at a given time.
    """
    # Ensure inputs are numpy arrays for broadcasting
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    
    C_reconstructed = np.zeros_like(theta, dtype=complex)
    
    for (n, m), Anm_val in coeffs.items():
        if np.abs(Anm_val) > 1e-9: # Optimization for sparse coefficients
            time_decay = np.exp(-n * (n + 1) * D * t / R**2)
            Ynm = sph_harm(m, n, phi, theta)
            C_reconstructed += Anm_val * Ynm * time_decay
            
    return np.real(C_reconstructed)


# --- Main script for computation and plotting ---
if __name__ == '__main__':
    # --- 1. Define Parameters ---
    N_MAX = 20          # Max harmonic degree (higher is more accurate)
    D = 0.5             # Diffusion coefficient
    R = 1.0             # Sphere radius
    
    # --- 2. Compute Coefficients for the Initial Condition ---
    Anm_coeffs = get_spherical_harmonic_coeffs(initial_condition_sphere, N_MAX)

    # --- 3. Set up the line for plotting ---
    # NOTE: Your code uses 'phi' for the meridional angle (latitude), 
    # which is conventionally 'theta'. We will follow your naming.
    # This line runs from the North Pole (0) to the South Pole (pi) along a single meridian.
    meridional_angle = np.linspace(0, np.pi, 400) # This is theta
    longitude_angle = np.zeros_like(meridional_angle)   # This is phi
    
    times_to_plot = np.linspace(0,2,12)

    # --- 4. Set up Matplotlib plots ---
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 0.9, len(times_to_plot)))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    fig.suptitle("Diffusion of Initial Condition on a Sphere", fontsize=16)

    # --- 5. Loop through time, reconstruct solution, and plot ---
    print("\nReconstructing solution at different times and plotting...")
    for t_eval, color in zip(times_to_plot, colors):
        # Reconstruct the solution along the defined line
        c_values_line = reconstruct_solution(
            Anm_coeffs, meridional_angle, longitude_angle, t_eval, D, R
        )
        
        # Plot 1: Concentration in physical space
        ax1.plot(meridional_angle, c_values_line, label=f't = {t_eval:.3f}', color=color)
        
        # Plot 2: Rescaled profile (skip t=0 to avoid division by zero)
        if t_eval > 1e-9:
            eta_scaled_x = meridional_angle / np.sqrt(D * t_eval)
            c_scaled_y = c_values_line * np.sqrt(t_eval)
            ax2.plot(eta_scaled_x, c_scaled_y, color=color)

    ax1.set_title("Concentration Profile Along Meridian")
    ax1.set_xlabel("Meridional Angle (radians)")
    ax1.set_ylabel("Concentration $c$")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title("Rescaled Profile (No Collapse)")
    ax2.set_xlabel(r"Similarity Variable $\eta = \theta / \sqrt{Dt}$")
    ax2.set_ylabel(r"Rescaled Concentration $c \cdot \sqrt{t}$")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(0, 15) # Adjust x-limit for better visualization
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("spherical_diffusion_profile.png", dpi=150)
    print("\nPlot saved as 'spherical_diffusion_profile.png'")
    plt.show()