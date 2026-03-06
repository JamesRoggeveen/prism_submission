import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib
import prism as pr
from matplotlib.image import imread
import argparse
import numpy as np

RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_sphere_flow"

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

if __name__ == "__main__":  
    results_path = get_target_folder()
    print(results_path)
    data = pr.load_dict_from_hdf5(results_path / "full_data.h5")
    config = data["config"]

    u_coeffs = data["fields"]["u"]
    v_coeffs = data["fields"]["v"]
    p_coeffs = data["fields"]["p"]

    basis = pr.ChebyshevBasis2D((config["basis_Nx"], config["basis_Ny"]))
    p_basis = pr.ChebyshevBasis2D((config["basis_Nx"]-2, config["basis_Ny"]-2))
    r_basis = pr.BasisND([pr.basis.vectorized_cosine_basis],(1,))
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
    v_field = pr.BasisField(basis, pr.Coeffs(v_coeffs))
    if len(p_coeffs) == len(v_coeffs):
        p_field = pr.BasisField(basis, pr.Coeffs(p_coeffs))
    else:
        p_field = pr.BasisField(p_basis, pr.Coeffs(p_coeffs))

    nx, ny = 100,100
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, nx), jnp.linspace(-1, 1, ny))
    u = u_field.evaluate(x, y).reshape(ny,nx)
    v = v_field.evaluate(x, y).reshape(ny,nx)
    p = p_field.evaluate(x, y).reshape(ny,nx)
    u_x = u_field.derivative(x, y, order=(1,0)).reshape(ny,nx)/config["x_scale"]
    v_y = v_field.derivative(x, y, order=(0,1)).reshape(ny,nx)/config["y_scale"]
    continuity = u_x + v_y
    theta = jnp.linspace(0, 2*jnp.pi, 100)
    t = theta/jnp.pi -1 
    # x = (x+1)*config["x_scale"]
    # y = (y+1)*config["y_scale"]
    # mask = (x - 0.2)**2 + (y - 0.2)**2 > 0.05**2
    mask = (x - config["cylinder_x"])**2 + (y - config["cylinder_y"])**2 > config["cylinder_radius"]**2
    u = jnp.where(mask, u, jnp.nan)
    v = jnp.where(mask, v, jnp.nan)
    p = jnp.where(mask, p, jnp.nan)
    u_x = jnp.where(mask, u_x, jnp.nan)
    v_y = jnp.where(mask, v_y, jnp.nan)
    continuity = jnp.where(mask, continuity, jnp.nan)

    fig, axs = plt.subplots(2,3, figsize=(15,10))
    ax = axs.flatten()
    im0 = ax[0].contourf(x, y, u, levels=100)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    im1 = ax[1].contourf(x, y, v, levels=100)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    im2 = ax[2].contourf(x, y, p, levels=100)
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    
    # Add vector field
    skip = 5  # Skip every 5th point for cleaner visualization
    # ax[3].quiver(x[::skip, ::skip], y[::skip, ::skip], 
    #              u[::skip, ::skip], v[::skip, ::skip], 
    #              alpha=0.7, scale=20, width=0.003)
    ax[3].contourf(x, y, jnp.sqrt(u**2 + v**2), levels=100,cmap="jet")
    ax[3].set_aspect("equal")
    ax[3].set_xlabel("x")
    ax[3].set_ylabel("y")
    ax[3].set_aspect("equal")
    ax[3].set_xlabel("x")
    ax[3].set_ylabel("y")
    im3 = ax[4].contourf(x, y, continuity, levels=100)
    ax[4].set_aspect("equal")
    ax[4].set_xlabel("x")
    ax[4].set_ylabel("y")
    ax[5].streamplot(np.asarray(x), np.asarray(y), np.asarray(u), np.asarray(v), density=2, color='k', linewidth=0.5)
    ax[5].axis('off')
    ax[5].set_aspect("equal")
    ax[5].set_xlabel("x")
    ax[5].set_ylabel("y")
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar1 = fig.colorbar(im1, ax=ax[1])
    cbar2 = fig.colorbar(im2, ax=ax[2])
    cbar3 = fig.colorbar(im3, ax=ax[4])
    cbar0.set_label("u")
    cbar1.set_label("v")
    cbar2.set_label("p")
    cbar3.set_label("continuity")
    plt.tight_layout()
    plt.savefig("figures/sphere_flow.png")
    plt.close()
