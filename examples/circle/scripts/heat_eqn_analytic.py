import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import jn, jn_zeros
from scipy.integrate import dblquad
import pathlib # Import pathlib to handle file paths

def calculate_fourier_bessel_coeffs(initial_condition_func, radius, n_radial_terms, n_angular_terms):
    """Calculates the Fourier-Bessel coefficients for a given initial condition."""
    print("Pre-calculating Bessel function roots...")
    bessel_roots = [jn_zeros(n, n_radial_terms) for n in range(n_angular_terms)]

    print("Calculating Fourier-Bessel coefficients... (This is the slow part)")
    A = np.zeros((n_angular_terms, n_radial_terms))
    B = np.zeros((n_angular_terms, n_radial_terms))

    for n in range(n_angular_terms):
        for m in range(n_radial_terms):
            j_nm = bessel_roots[n][m]
            norm_factor = (np.pi * radius**2 * jn(n + 1, j_nm)**2) / 2
            
            integrand_A = lambda r, theta: initial_condition_func(r, theta) * \
                                           jn(n, j_nm * r / radius) * np.cos(n * theta) * r
            integral_A, _ = dblquad(integrand_A, 0, 2 * np.pi, 0, radius)
            A[n, m] = integral_A / norm_factor
            
            if n > 0:
                integrand_B = lambda r, theta: initial_condition_func(r, theta) * \
                                               jn(n, j_nm * r / radius) * np.sin(n * theta) * r
                integral_B, _ = dblquad(integrand_B, 0, 2 * np.pi, 0, radius)
                B[n, m] = integral_B / norm_factor
        
        print(f"  -> Finished coefficients for angular mode n={n}")
    
    return A, B, bessel_roots

def create_solution_function(A, B, bessel_roots, radius, diffusivity, n_radial_terms, n_angular_terms):
    """Constructs a function that evaluates the series solution at any (r, theta, t)."""
    def get_solution(r, theta, t):
        solution = np.zeros_like(r, dtype=float)
        for n in range(n_angular_terms):
            for m in range(n_radial_terms):
                j_nm = bessel_roots[n][m]
                lambda_nm = j_nm / radius
                
                temporal_decay = np.exp(-diffusivity * (lambda_nm**2) * t)
                
                spatial_mode = jn(n, lambda_nm * r) * \
                               (A[n, m] * np.cos(n * theta) + B[n, m] * np.sin(n * theta))
                
                solution += spatial_mode * temporal_decay
        return solution
    return get_solution

def generate_solution_grid(solution_func, radius, grid_size=200, time_points=200, t_max=1.0):
    """Generates a 3D grid [t, y, x] of the solution."""
    print("\nGenerating 3D solution grid...")
    x_vals = np.linspace(-radius, radius, grid_size)
    y_vals = np.linspace(-radius, radius, grid_size)
    t_vals = np.linspace(0, t_max, time_points)
    
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Convert Cartesian grid to polar coordinates for the solver
    R_grid = np.sqrt(X**2 + Y**2)
    Theta_grid = np.arctan2(Y, X)
    
    # Initialize the 3D data cube
    solution_grid = np.full((time_points, grid_size, grid_size), np.nan)
    
    # Mask for points inside the disk
    disk_mask = R_grid <= radius

    for i, t in enumerate(t_vals):
        print(f"  -> Calculating time step {i+1}/{time_points} (t={t:.3f})")
        # Calculate solution for all points at time t
        solution_at_t = solution_func(R_grid, Theta_grid, t)
        # Apply the disk mask
        solution_grid[i, disk_mask] = solution_at_t[disk_mask]
        
    print("✅ 3D solution grid generated.")
    return solution_grid, t_vals, x_vals, y_vals

def animate_heat_solution(solution_func, radius):
    """Creates and saves an animation of the heat diffusion."""
    print("\nSetting up visualization for animation...")
    grid_res = 100
    r_vals = np.linspace(0, radius, grid_res)
    theta_vals = np.linspace(0, 2 * np.pi, grid_res)
    r_grid, theta_grid = np.meshgrid(r_vals, theta_vals)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    initial_solution = solution_func(r_grid, theta_grid, 0)
    vmax = np.max(np.abs(initial_solution))
    vmin = -vmax if np.min(initial_solution) < 0 else 0

    c = ax.pcolormesh(theta_grid, r_grid, initial_solution, cmap='hot', vmin=vmin, vmax=vmax)
    fig.colorbar(c, ax=ax, label="Temperature")
    time_text = ax.text(0.05, 1.02, '', transform=ax.transAxes)

    def update(frame):
        t = frame * 0.005
        solution_t = solution_func(r_grid, theta_grid, t)
        c.set_array(solution_t.flatten())
        time_text.set_text(f'Time = {t:.3f} s')
        return c, time_text

    print("Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    
    try:
        ani.save("heat_equation_on_disk.gif", writer='pillow', fps=20)
        print("✅ Animation saved to 'heat_equation_on_disk.gif'")
    except Exception as e:
        print(f"\nCould not save animation. Error: {e}")
    
    plt.show()

def initial_condition(x,y):
    """Defines an initial condition shaped like a smiley face."""
    x = np.array(x)
    y = np.array(y)
    left_eye = (x + 0.4)**2 + (y - 0.4)**2 < 0.3**2
    right_eye = (x - 0.4)**2 + (y - 0.4)**2 < 0.3**2
    mouth = (x)**2 + (y+.4)**2 < 0.3**2
    output = np.logical_or.reduce((left_eye, right_eye, mouth)).astype(float)
    return output

if __name__ == '__main__':
    # --- Configuration ---
    RADIUS = 1.0
    DIFFUSIVITY = 0.125
    N_RADIAL = 15
    N_ANGULAR = 15
    
    # --- 1. Calculate the coefficients (this is the slow part) ---
    A_coeffs, B_coeffs, roots = calculate_fourier_bessel_coeffs(
        initial_condition_func=initial_condition,
        radius=RADIUS,
        n_radial_terms=N_RADIAL,
        n_angular_terms=N_ANGULAR
    )

    # --- 2. Create the solver function ---
    solver_func = create_solution_function(
        A_coeffs, B_coeffs, roots, RADIUS, DIFFUSIVITY, N_RADIAL, N_ANGULAR
    )

    # --- 3. Generate the 200x200x200 data cube ---
    grid, t_grid, x_grid, y_grid = generate_solution_grid(
        solution_func=solver_func,
        radius=RADIUS,
        grid_size=200,
        time_points=200,
        t_max=2.0 # Max time for the grid
    )

    print(f"\nShape of the final data grid: {grid.shape}")

    # --- 4. Save the generated grid to a file ---
    output_path = pathlib.Path("data/ground_truth.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True) # Create 'data' directory
    np.save(output_path, grid)
    print(f"✅ Data grid saved to '{output_path}'")


    # --- 5. Visualize a few time slices from the generated grid as a check ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    time_indices_to_plot = [0, 100, 199] # Start, middle, end
    
    for i, ax in enumerate(axes):
        time_idx = time_indices_to_plot[i]
        time_val = t_grid[time_idx]
        
        # pcolormesh will ignore NaN values, creating the circular plot
        im = ax.pcolormesh(x_grid, y_grid, grid[time_idx, :, :], cmap='hot', shading='auto')
        ax.set_title(f'Time Slice at t = {time_val:.3f} s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.savefig("figures/heat_equation_on_disk.png")
    plt.close()

