import numpy as np
import prism as pr
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
import matplotlib.animation as animation
import argparse
import shutil

def find_latest_folder(base_dir: pathlib.Path, name_suffix: str) -> pathlib.Path | None:
    glob_pattern = f"????????{name_suffix}"
    matching_folders = list(base_dir.glob(glob_pattern))
    
    if not matching_folders:
        return None
    latest_folder = sorted(matching_folders, key=lambda p: p.name, reverse=True)[0]
    return latest_folder

def get_target_folder():
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
    args = parser.parse_args()

    if args.folder_name:
        target_folder = RESULTS_BASE_DIR / args.folder_name
        
        if not target_folder.is_dir():
            print(f"Error: Folder '{target_folder}' does not exist or is not a directory.")
            return None
            
        print(f"Using provided folder: {target_folder}")
        return target_folder
        
    else:
        print(f"No folder name provided. Searching for the most recent folder in '{RESULTS_BASE_DIR}'...")
        
        latest_folder = find_latest_folder(RESULTS_BASE_DIR, FOLDER_SUFFIX)
        
        if latest_folder:
            print(f"Found latest folder: {latest_folder}")
            return latest_folder
        else:
            print(f"Error: No folders found in '{RESULTS_BASE_DIR}' matching the pattern.")
            return None
RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_wave"
def find_latest_folder(base_dir: pathlib.Path, name_suffix: str) -> pathlib.Path | None:
    glob_pattern = f"????????{name_suffix}"
    matching_folders = list(base_dir.glob(glob_pattern))
    
    if not matching_folders:
        return None
    latest_folder = sorted(matching_folders, key=lambda p: p.name, reverse=True)[0]
    return latest_folder

def get_target_folder():
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
    args = parser.parse_args()

    if args.folder_name:
        target_folder = RESULTS_BASE_DIR / args.folder_name
        
        if not target_folder.is_dir():
            print(f"Error: Folder '{target_folder}' does not exist or is not a directory.")
            return None
            
        print(f"Using provided folder: {target_folder}")
        
        # Find the child folder (there should only be one directory inside)
        child_folders = [p for p in target_folder.iterdir() if p.is_dir()]
        if len(child_folders) != 1:
            print(f"Error: Expected exactly one child directory in '{target_folder}', found {len(child_folders)}")
            return None
        
        return child_folders[0]
        
    else:
        print(f"No folder name provided. Searching for the most recent folder in '{RESULTS_BASE_DIR}'...")
        
        latest_folder = find_latest_folder(RESULTS_BASE_DIR, FOLDER_SUFFIX)
        
        if latest_folder:
            print(f"Found latest folder: {latest_folder}")
            
            # Find the child folder (there should only be one directory inside)
            child_folders = [p for p in latest_folder.iterdir() if p.is_dir()]
            if len(child_folders) != 1:
                print(f"Error: Expected exactly one child directory in '{latest_folder}', found {len(child_folders)}")
                return None
            
            return child_folders[0]
        else:
            print(f"Error: No folders found in '{RESULTS_BASE_DIR}' matching the pattern.")
            return None

def define_geometry(n_points):
    def r1(theta):
        return 0.7+0.3*jnp.cos(2*theta)
    circle_x, circle_y, circle_r = .3, .1, 0.15
    x_vec = jnp.linspace(-1,1,n_points)
    y_vec = jnp.linspace(-1,1,n_points)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    y_squashed = 0.5*y_grid
    theta_grid = jnp.arctan2(y_squashed, x_grid)
    r_grid = jnp.sqrt(x_grid**2 + (y_squashed)**2)
    mask1 = r_grid <= r1(theta_grid)
    mask2 = (x_grid - circle_x)**2 + (y_grid-circle_y)**2 >= circle_r**2
    mask = mask1 & mask2
    mask = mask1

    return x_grid, y_grid, mask

def import_results(results_path):
    data = pr.load_dict_from_hdf5(results_path / "full_data.h5")
    config = data["config"]
    u_coeffs = data["fields"]["u"]
    basis_N = (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"])
    basis = pr.basis.BasisND([pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis], basis_N)
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
    true_basis_N = (config["basis_Nx"]+1, config["basis_Ny"]+1, config["basis_Nt"]+1)
    f, axs = plt.subplots(3,5,figsize=(15,6))
    u_coeffs_abs = jnp.abs(u_coeffs).reshape(true_basis_N)
    print(u_coeffs_abs.shape)
    vmax = jnp.max(u_coeffs_abs)
    axs = axs.flatten()
    for i in range(15):
        axs[i].imshow(u_coeffs.reshape(true_basis_N)[:,:,i],cmap="bwr",vmin=-vmax,vmax=vmax)
        axs[i].set_title(f"Abs(Coefficient {i})")
    plt.tight_layout()
    plt.savefig("figures/wave_coeffs.png")
    plt.close()
    return u_field

if __name__ == "__main__":
    # results_path = pathlib.Path("results/09171028_wave/15_wave/")
    results_path = get_target_folder()
    if not results_path:
        raise ValueError("No target folder found")
    print(f"\n✅ Proceeding with folder: {results_path}")
    u_field = import_results(results_path)
    n_grid = 100
    x_grid, y_grid, mask = define_geometry(n_grid)
    t_vec = jnp.arange(-1,1.01,0.01)
    slice_data = []
    print("Begin evaluating field")
    for t in tqdm(t_vec):
        u_eval = u_field.evaluate(x_grid, y_grid, t*jnp.ones(x_grid.shape))
        u_eval = u_eval.reshape(n_grid,n_grid)
        u_eval = jnp.where(mask == 0, jnp.nan, u_eval)
        slice_data.append(u_eval)
    slice_data = jnp.array(slice_data)
    print(slice_data.shape)
    
    reference_data = np.loadtxt("data/wave5.csv", delimiter=",", skiprows=9)
    ref_x = reference_data[:,0]
    ref_y = reference_data[:,1]
    ref_x.reshape(n_grid,n_grid)
    ref_y.reshape(n_grid,n_grid)
    ref_slice_data = []
    print("Begin processing reference")
    for i in tqdm(range(len(t_vec))):
        ref_data = reference_data[:,i+2]
        ref_data = ref_data.reshape(n_grid,n_grid)
        ref_slice_data.append(ref_data)
    ref_slice_data = jnp.array(ref_slice_data)
    f, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].contourf(x_grid, y_grid, slice_data[0], levels=100)
    ax[1].contourf(x_grid, y_grid, ref_slice_data[0], levels=100)
    plt.savefig("figures/wave_comparison_t0.png")
    plt.savefig(results_path / "wave_comparison_t0.png")
    plt.close()

    dx = x_grid[0,1] - x_grid[0,0]
    dy = y_grid[1,0] - y_grid[0,0]
    dt = t_vec[1] - t_vec[0]
    dh = dx*dy*dt
    error = jnp.sqrt(jnp.nansum(jnp.power(slice_data - ref_slice_data, 2))*dh)
    print(f"L2 error: {error}")
    Linf_error = jnp.nanmax(jnp.abs(slice_data - ref_slice_data))
    print(f"Linf error: {Linf_error}")
    error = np.array([error, Linf_error])
    np.savetxt(results_path / "error.csv", error, delimiter=",", header="L2 error, Linf error")

    fig, ax = plt.subplots(1,2,figsize=(10,5))

    cmap = "jet"

    vmax = jnp.nanmax(jnp.abs(ref_slice_data))
    vmin = jnp.nanmin(jnp.abs(ref_slice_data))
    print(f"vmax: {vmax}, vmin: {vmin}")
    slice_data = slice_data
    cont = ax[0].contourf(x_grid, y_grid, slice_data[0], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    cont1 = ax[1].contourf(x_grid, y_grid, ref_slice_data[0], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    cbar1 = fig.colorbar(cont1, ax=ax[1])
    cbar = fig.colorbar(cont, ax=ax[0])

    def update(frame):
    
        ax[0].clear()
        ax[1].clear()
        cont = ax[0].contourf(x_grid, y_grid, slice_data[frame], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
        cont1 = ax[1].contourf(x_grid, y_grid, ref_slice_data[frame], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[0].set_title(f'Polynomial Fit')
        ax[1].set_title(f'Reference (COMSOL)')
        
        print(f"Processing frame {frame+1}/{len(t_vec)}")

        return cont, cont1

    # --- Create and save the animation ---
    # FuncAnimation will call the 'update' function for each frame.
    ani = animation.FuncAnimation(fig, update, frames=len(t_vec), blit=False)

    # Save the animation as an MP4 file.
    # This requires having ffmpeg installed on your system.
    output_file = "figures/wave_comparison.gif"
    ani.save(output_file, writer='ffmpeg', fps=20, dpi=150)
    
    # Copy the file instead of re-rendering
    shutil.copy2(output_file, results_path / "wave_comparison.gif")

    print(f"\nAnimation successfully saved to {output_file}")
    plt.close()