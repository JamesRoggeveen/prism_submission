import jax.numpy as jnp
import matplotlib.pyplot as plt
import pathlib
import prism as pr
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import shutil

# --- Unchanged Setup Code ---
RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_diffusion_sphere"

def find_latest_folder(base_dir: pathlib.Path, name_suffix: str) -> pathlib.Path | None:
    glob_pattern = f"????????{name_suffix}"
    matching_folders = list(base_dir.glob(glob_pattern))
    
    if not matching_folders:
        return None
    latest_folder = sorted(matching_folders, key=lambda p: p.name, reverse=True)[0]
    return latest_folder

# --- MODIFICATION START: Update function to handle and return all arguments ---
def get_args_and_target_folder():
    parser = argparse.ArgumentParser(
        description=f"Process data from an experiment folder within '{RESULTS_BASE_DIR}'."
    )
    parser.add_argument(
        "folder_name", 
        nargs='?',
        default=None,
        type=str,
        help=f"Optional: Name of a specific folder in '{RESULTS_BASE_DIR}'. If not provided, finds the latest."
    )
    # Add the new flag
    parser.add_argument(
        '--no-animation',
        action='store_true',
        help='If set, disables rendering the mp4 animation.'
    )
    args = parser.parse_args()

    if args.folder_name:
        target_folder = RESULTS_BASE_DIR / args.folder_name
        
        if not target_folder.is_dir():
            print(f"Error: Folder '{target_folder}' does not exist or is not a directory.")
            return None, args
            
        print(f"Using provided folder: {target_folder}")
        return target_folder, args
        
    else:
        print(f"No folder name provided. Searching for the most recent folder in '{RESULTS_BASE_DIR}'...")
        
        latest_folder = find_latest_folder(RESULTS_BASE_DIR, FOLDER_SUFFIX)
        
        if latest_folder:
            print(f"Found latest folder: {latest_folder}")
            return latest_folder, args
        else:
            print(f"Error: No folders found in '{RESULTS_BASE_DIR}' matching the pattern.")
            return None, args
# --- MODIFICATION END ---

if __name__ == "__main__":
    # Update how we get the folder and args
    target_folder, args = get_args_and_target_folder()
    
    if target_folder is None:
        raise ValueError("No target folder found")
    
    pathlib.Path("figures").mkdir(exist_ok=True)
    
    data = pr.load_dict_from_hdf5(target_folder / "full_data.h5")
    config = data["config"]
    c_coeffs = data["fields"]["c"]
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nz"], config["basis_Nt"]))
    c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))
    
    # ==============================================================================
    # 2. MERIDIAN PLOT: SIDE-BY-SIDE, SQUARE, AND COLORBLIND-FRIENDLY
    # ==============================================================================
    print("\nCreating meridian diffusion plot with similarity rescaling...")

    radius = 1.0
    n_line_points = 400
    D = 1/config["alpha"]
    
    phi_line = np.linspace(0, np.pi, n_line_points)
    theta_line = np.zeros_like(phi_line)
    x_line = radius * np.sin(phi_line) * np.cos(theta_line)
    y_line = radius * np.sin(phi_line) * np.sin(theta_line)
    z_line = radius * np.cos(phi_line)
    times_to_plot = np.linspace(-1.0, 1.0, 12)

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 0.9, len(times_to_plot)))

    fig_meridian, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(14, 7)
    )
    fig_meridian.suptitle("Verification of Diffusion Along a Meridian", fontsize=16)
    
    for t_eval, color in zip(times_to_plot, colors):
        T_line = t_eval * jnp.ones_like(x_line)
        c_values_line = np.asarray(c_field.evaluate(x_line, y_line, z_line, T_line))
        
        ax1.plot(phi_line, c_values_line, label=f't = {t_eval:.2f}', color=color)
        
        elapsed_time = t_eval + 1.0
        if elapsed_time > 1e-6:
            eta_scaled_x = phi_line / np.sqrt(D * elapsed_time)
            c_scaled_y = c_values_line * np.sqrt(elapsed_time)
            ax2.plot(eta_scaled_x, c_scaled_y, label=f't = {t_eval:.2f}', color=color)

    ax1.set_title("Concentration Profile in Physical Space")
    ax1.set_xlabel("Angle along Meridian (φ, radians)")
    ax1.set_ylabel("Concentration $c$")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_box_aspect(1)

    ax2.set_title("Rescaled Profile in Similarity Coordinates")
    ax2.set_xlabel(r"Similarity Variable $\eta = \phi / \sqrt{Dt}$")
    ax2.set_ylabel(r"Rescaled Concentration $c \cdot \sqrt{t}$")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_box_aspect(1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    meridian_plot_filename = "figures/meridian_diffusion_profile_side_by_side.png"
    plt.savefig(meridian_plot_filename, dpi=150, bbox_inches='tight')
    plt.savefig(target_folder / "meridian_diffusion_profile_side_by_side.png")
    print(f"Meridian plot saved as '{meridian_plot_filename}'")
    plt.close(fig_meridian)

    # --- MODIFICATION START: Conditionally run the animation section ---
    if not args.no_animation:
        # ==============================================================================
        # 3. ANIMATION WITH FIXED COLOR SCALE
        # ==============================================================================
        n_frames = 100
        t_min, t_max = -1.0, 1.0
        time_points = np.linspace(t_min, t_max, n_frames)

        n_points = 150
        theta = np.linspace(0, 2 * np.pi, n_points)
        phi = np.linspace(0, np.pi, n_points)
        theta, phi = np.meshgrid(theta, phi)
        X = radius * np.sin(phi) * np.cos(theta)
        Y = radius * np.sin(phi) * np.sin(theta)
        Z = radius * np.cos(phi)

        print("\nCalculating global min/max for fixed color scale... (this may take a moment)")
        global_vmin, global_vmax = np.inf, -np.inf
        
        # for i, t_eval in enumerate(time_points):
        #     print(f"Scanning time step {i+1}/{n_frames}...", end='\r')
        #     T_grid = t_eval * jnp.ones_like(X)
        #     c_values_at_t = c_field.evaluate(X, Y, Z, T_grid)
        #     current_min, current_max = jnp.min(c_values_at_t), jnp.max(c_values_at_t)
        #     if current_min < global_vmin: global_vmin = current_min
        #     if current_max > global_vmax: global_vmax = current_max

        global_vmin, global_vmax = 0, .75
                
        print(f"\nGlobal range found: [{global_vmin:.4f}, {global_vmax:.4f}]")
        norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])

        def update(frame):
            ax.clear()
            t_eval = time_points[frame]
            T = t_eval * jnp.ones_like(X)
            c_values = np.asarray(c_field.evaluate(X, Y, Z, T).reshape(X.shape))
            colors = cm.jet(norm(c_values))
            ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=False, shade=False,cmap='jet')
            ax.set_xlabel("X-axis"); ax.set_ylabel("Y-axis"); ax.set_zlabel("Z-axis")
            ax.set_box_aspect([1, 1, 1])
            ax.set_title(f"c_field on a Sphere (t = {t_eval:.2f})")
            ax.view_init(elev=20., azim=30 + frame*0.5)
            print(f"Rendering frame {frame + 1}/{n_frames} (t={t_eval:.2f})")
            return ax,

        mappable = cm.ScalarMappable(cmap=cm.jet, norm=norm)
        fig.colorbar(mappable, cax=cax, label="c_field Value")

        ani = FuncAnimation(fig, update, frames=n_frames, blit=False)
        output_filename = "figures/c_field_sphere_animation_fixed_scale.mp4"
        print(f"\nSaving animation to '{output_filename}'. This may take a while...")
        ani.save(output_filename, writer='ffmpeg', fps=15, dpi=150)
        shutil.copy2(output_filename, target_folder / "c_field_sphere_animation_fixed_scale.mp4")
        print("Animation saved successfully.")
        plt.close(fig)
    else:
        print("\n--no-animation flag detected. Skipping animation rendering.")
    # --- MODIFICATION END ---