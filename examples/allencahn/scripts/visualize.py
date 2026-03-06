import pathlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import yaml
import argparse
from scipy.io import loadmat
from prism import (CosineChebyshevBasis2D, BasisField, Coeffs, load_dict_from_hdf5)

RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_solve_ac"

def find_latest_folder(base_dir: pathlib.Path, name_suffix: str) -> pathlib.Path | None:
    glob_pattern = f"??????????{name_suffix}"
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
    target = get_target_folder()
    if not target:
        raise ValueError("No target folder found")
    print(f"\n✅ Proceeding with folder: {target}")

    full_data = load_dict_from_hdf5(target / "full_data.h5")
    config = full_data["config"]

    u_coeffs = full_data["fields"]["u"]
    
    basis = CosineChebyshevBasis2D((config["basis_Nx"], config["basis_Nt"]))
    print(basis.degs)
    u_field = BasisField(basis, Coeffs(u_coeffs))

    nx, nt = config["nx"], config["nt"]
    x_vec = jnp.linspace(-1,1,nx)
    t_vec = jnp.linspace(-1,1,nt)
    x_grid, t_grid = jnp.meshgrid(x_vec, t_vec)

    u_eval = u_field.evaluate(x_grid, t_grid)
    u_eval = u_eval.reshape(nt,nx)

    target_data = loadmat(config["data_path"])["usol"]

    error = jnp.linalg.norm(u_eval - target_data)/jnp.linalg.norm(target_data)
    print(f"L2 relative error: {error}")
    config["final_relative_error"] = error

    save_fig_path = pathlib.Path("figures/")
    save_fig_path.mkdir(parents=True, exist_ok=True)
    fig_name = "solution.png"
    f, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].set_title("Target Solution")
    ax[1].set_title("Optimized Solution")
    im0 = ax[0].contourf(x_grid, t_grid, target_data, cmap="jet", vmin = -1, vmax = 1, levels=100)
    im1 = ax[1].contourf(x_grid, t_grid, u_eval, cmap="jet", vmin = -1, vmax = 1, levels=100)
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[0].set_title("Target Solution")
    ax[1].set_title("Optimized Solution")
    for axis in ax:
        axis.set_xlabel("x")
        axis.set_ylabel("t")
    f.colorbar(im0,ax=ax[0])
    f.colorbar(im1,ax=ax[1])
    plt.tight_layout()
    plt.savefig(save_fig_path / fig_name)
    plt.close()

    plot_info = {
        "name": target.name,
        "basis_Nx": config["basis_Nx"],
        "basis_Nt": config["basis_Nt"],
        "data_path": config["data_path"],
        "filter_frac": config["filter_frac"],
        "residual_weights": config["residual_weights"],
        "learning_rate": config["learning_rate"],
        "n_pde": config["n_pde"],
        "n_ic": config["n_ic"],
        "solve_time": config.get("solve_time", 0.0),
        "error": error
    }

    with open(save_fig_path / "plot_info.yml", "w") as f:
        yaml.dump(plot_info, f)

    print(f"Plot info saved to {save_fig_path / 'plot_info.yml'}")