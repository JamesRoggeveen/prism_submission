import pathlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import yaml
import argparse
from scipy.io import loadmat
from prism import (CosineChebyshevBasis2D, BasisField, Coeffs, load_dict_from_hdf5)

RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_solve_nls"

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
    target = get_target_folder()
    if not target:
        raise ValueError("No target folder found")
    print(f"\n✅ Proceeding with folder: {target}")

    full_data = load_dict_from_hdf5(target / "full_data.h5")
    config = full_data["config"]

    u_coeffs = full_data["fields"]["u"]
    v_coeffs = full_data["fields"]["v"]
    
    basis = CosineChebyshevBasis2D((config["basis_Nx"], config["basis_Nt"]))
    u_field = BasisField(basis, Coeffs(u_coeffs))
    v_field = BasisField(basis, Coeffs(v_coeffs))

    nx, nt = config["nx"], config["nt"]
    x_vec = jnp.linspace(-1,1,nx)
    t_vec = jnp.linspace(-1,1,nt)
    x_grid, t_grid = jnp.meshgrid(x_vec, t_vec)

    u_eval = u_field.evaluate(x_grid, t_grid)
    v_eval = v_field.evaluate(x_grid, t_grid)
    u_eval = u_eval.reshape(nt,nx)
    v_eval = v_eval.reshape(nt,nx)
    h = jnp.sqrt(u_eval**2 + v_eval**2)
    
    NLS_data = loadmat(config["data_path"])
    Exact = NLS_data['uu']
    Exact_u = jnp.real(Exact).T
    Exact_v = jnp.imag(Exact).T
    Exact_h = jnp.sqrt(Exact_u**2 + Exact_v**2)

    u_max = jnp.max(jnp.abs(Exact_u))
    v_max = jnp.max(jnp.abs(Exact_v))
    h_max = jnp.max(jnp.abs(Exact_h))
    figure_path = pathlib.Path("figures/")
    fig_name = "solution.png"
    f, ax = plt.subplots(2,3,figsize=(10,5))
    im0 = ax[0,0].imshow(u_eval,vmin=-u_max,vmax=u_max)
    im1 = ax[0,1].imshow(v_eval,vmin=-v_max,vmax=v_max)
    im2 = ax[0,2].imshow(h,vmin=0,vmax=h_max)
    im3 = ax[1,0].imshow(Exact_u,vmin=-u_max,vmax=u_max)
    im4 = ax[1,1].imshow(Exact_v,vmin=-v_max,vmax=v_max)
    im5 = ax[1,2].imshow(Exact_h,vmin=0,vmax=h_max)
    f.colorbar(im0,ax=ax[0,0])
    f.colorbar(im1,ax=ax[0,1])
    f.colorbar(im2,ax=ax[0,2])
    f.colorbar(im3,ax=ax[1,0])
    f.colorbar(im4,ax=ax[1,1])
    f.colorbar(im5,ax=ax[1,2])
    ax[0,0].set_title("u_opt")
    ax[0,1].set_title("v_opt")
    ax[0,2].set_title("h_opt")
    ax[1,0].set_title("Exact u")
    ax[1,1].set_title("Exact v")
    ax[1,2].set_title("Exact h")
    plt.tight_layout()
    f.savefig(figure_path / fig_name)
    f.savefig(target / fig_name)
    plt.close()

    error = jnp.linalg.norm(h - Exact_h)/jnp.linalg.norm(Exact_h)
    print(f"L2 relative error: {error}")
    config["final_relative_error"] = error

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
        "solve_time": config.get("solve_time", 0.0)
    }

    with open(figure_path / "plot_info.yml", "w") as f:
        yaml.dump(plot_info, f)

    print(f"Plot info saved to {figure_path / 'plot_info.yml'}")