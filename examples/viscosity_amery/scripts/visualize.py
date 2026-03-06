import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml
import pathlib
import argparse

from prism import (
    ChebyshevBasis2D,
    BasisField,
    LogBasisField, Coeffs, load_dict_from_hdf5
)

RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_visc_inv_Amery"

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

def compute_rate_of_strain(u, v, x, y, scaling_dict):
    r = scaling_dict["aspect_ratio"]
    u_x = u.derivative(x, y, order=(1,0))
    u_y = r*u.derivative(x, y, order=(0,1))
    v_x = v.derivative(x, y, order=(1,0))
    v_y = r*v.derivative(x, y, order=(0,1))
    rate_of_strain = jnp.sqrt(u_x**2 + v_y**2 + 0.25*(u_y + v_x)**2 + u_x*v_y)
    rate_of_strain = rate_of_strain.reshape(x.shape)
    rate_of_strain = rate_of_strain[::-1, :]
    return rate_of_strain

def evaluate_field(field, x_grid, y_grid, mask):
    field_eval = field.evaluate(x_grid, y_grid)
    field_eval = field_eval.reshape(x_grid.shape)
    field_eval = field_eval[::-1, :]
    field_eval = jnp.where(mask, field_eval, jnp.nan)
    return field_eval

if __name__ == "__main__":
    target = get_target_folder()
    
    if not target:
        raise ValueError("No target folder found")
    
    print(f"\n✅ Proceeding with folder: {target}")

    full_data = load_dict_from_hdf5(target / "full_data.h5")
    config = full_data["config"]
    problem_data = full_data["fields"]

    h_coeffs = problem_data["h"]
    u_coeffs = problem_data["u"]
    v_coeffs = problem_data["v"]
    mu_coeffs = problem_data["mu"]

    basis = ChebyshevBasis2D((config["basis_Nx"], config["basis_Ny"]))

    h_field = BasisField(basis, Coeffs(h_coeffs))
    u_field = BasisField(basis, Coeffs(u_coeffs))
    v_field = BasisField(basis, Coeffs(v_coeffs))
    mu_field = LogBasisField(basis, Coeffs(mu_coeffs))

    reference_data = load_dict_from_hdf5(config["data_path"])

    mask = reference_data["fields"]["u"]["mask"]

    ny, nx = reference_data["fields"]["u"]["x"].shape
    x_vec, y_vec = jnp.linspace(-1,1,nx), jnp.linspace(-1,1,ny)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)

    rate_of_strain = compute_rate_of_strain(u_field, v_field, x_grid, y_grid, config["scaling_dict"])
    rate_of_strain = jnp.where(mask, rate_of_strain, jnp.nan)

    mu_eval = evaluate_field(mu_field, x_grid, y_grid, mask)
    mu_eval = mu_eval*config["scaling_dict"]["mu"]
    tau_eval = 2*mu_eval*rate_of_strain


    h_eval = evaluate_field(h_field, x_grid, y_grid, mask)
    u_eval = evaluate_field(u_field, x_grid, y_grid, mask)
    v_eval = evaluate_field(v_field, x_grid, y_grid, mask)


    figure_path = pathlib.Path("figures/")
    fig_name = "inversion_results.png"
    f, ax = plt.subplots(1,3,figsize=(12,3))
    im0 = ax[0].imshow(jnp.log10(rate_of_strain),cmap="jet")
    ax[0].set_title("Log(Rate of Strain)")
    im1 = ax[1].imshow(jnp.log10(tau_eval),cmap="jet")
    ax[1].set_title("Log(tau)")
    im2 = ax[2].imshow(jnp.log10(mu_eval),cmap="jet",vmin=13,vmax=16)
    ax[2].set_title("Log(Viscosity)")
    
    # Add colorbars with reduced height using aspect parameter
    f.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, aspect=20)
    f.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, aspect=20)
    f.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, aspect=20)
    
    plt.tight_layout()
    plt.savefig(figure_path / fig_name)
    plt.close()

    f, ax = plt.subplots(2,4,figsize=(12,4))
    ax = ax.flatten()
    im0 = ax[0].imshow(h_eval,cmap="jet")
    ax[0].set_title("h")
    im1 = ax[1].imshow(u_eval,cmap="jet")
    ax[1].set_title("u")
    im2 = ax[2].imshow(v_eval,cmap="jet")
    ax[2].set_title("v")
    im3 = ax[3].imshow(jnp.log10(mu_eval),cmap="jet",vmin=13,vmax=16)
    ax[3].set_title("Log(Viscosity)")
    # f.colorbar(im0, ax=ax[0])
    # f.colorbar(im1, ax=ax[1])
    # f.colorbar(im2, ax=ax[2])
    f.colorbar(im3, ax=ax[3])
    ax[4].imshow(reference_data["fields"]["h"]["data"],cmap="jet")
    ax[4].set_title("h_ref")
    ax[5].imshow(reference_data["fields"]["u"]["data"],cmap="jet")
    ax[5].set_title("u_ref")
    ax[6].imshow(reference_data["fields"]["v"]["data"],cmap="jet")
    ax[6].set_title("v_ref")
    # Hide the 8th subplot (ax[7])
    ax[7].axis('off')
    plt.tight_layout()
    plt.savefig(figure_path / "inversion_results_comparison.png")
    plt.savefig(target / "inversion_results_comparison.png")
    plt.close()


    # Get the absolute maximum value for symmetric color scaling
    f, ax = plt.subplots(1,4,figsize=(16,4))
    vmax = jnp.max(jnp.abs(mu_coeffs))
    ax[0].imshow(mu_coeffs.reshape(config["basis_Nx"]+1, config["basis_Ny"]+1), cmap="bwr", vmin=-vmax, vmax=vmax)
    ax[0].set_title("Viscosity Coefficients")
    vmax = jnp.max(jnp.abs(h_coeffs))
    ax[1].imshow(h_coeffs.reshape(config["basis_Nx"]+1, config["basis_Ny"]+1), cmap="bwr", vmin=-vmax, vmax=vmax)
    ax[1].set_title("Height Coefficients")
    vmax = jnp.max(jnp.abs(u_coeffs))
    ax[2].imshow(u_coeffs.reshape(config["basis_Nx"]+1, config["basis_Ny"]+1), cmap="bwr", vmin=-vmax, vmax=vmax)
    ax[2].set_title("Velocity X Coefficients")
    vmax = jnp.max(jnp.abs(v_coeffs))
    ax[3].imshow(v_coeffs.reshape(config["basis_Nx"]+1, config["basis_Ny"]+1), cmap="bwr", vmin=-vmax, vmax=vmax)
    ax[3].set_title("Velocity Y Coefficients")
    plt.tight_layout()
    plt.savefig(figure_path / "inversion_results_coefficients.png")
    plt.close()

    plot_info = {
        "name": target.name,
        "basis_Nx": config["basis_Nx"],
        "basis_Ny": config["basis_Ny"],
        "data_path": config["data_path"],
        "filter_frac": config["filter_frac"],
        "residual_weights": config["residual_weights"],
        "learning_rate": config["learning_rate"],
        "n_epochs": config["n_epochs"],
        "n_pde": config["n_pde"],
        "n_data": config["n_data"],
        "n_bc": config["n_bc"],
        "solve_time": config.get("solve_time", 0.0)
    }

    with open(figure_path / "plot_info.yml", "w") as f:
        yaml.dump(plot_info, f)
    
    print(f"Plot info saved to {figure_path / 'plot_info.yml'}")

# f, ax = plt.subplots(3,4,figsize=(12,8))

# coeff_matricies = [mu_coeffs, h_coeffs, u_coeffs, v_coeffs]
# names = ["Viscosity", "Height", "Velocity X", "Velocity Y"]

# for i in range(4):
#     coeffs = coeff_matricies[i]
#     coeffs = coeffs.reshape(config["basis_Nx"]+1, config["basis_Ny"]+1)
#     vmax = jnp.max(jnp.abs(coeffs))
#     ax0 = ax[0,i]
#     ax0.imshow(coeffs, cmap="bwr", vmin=-vmax, vmax=vmax)
#     ax0.set_title(f"{names[i]} Coefficients")
#     # Compute the singular values. We don't need U and V, so this is efficient.
#     s = jnp.linalg.svd(coeffs, compute_uv=False)
#     ax1 = ax[1,i]
#     # --- 3. Plot the Scree Plot (Singular Values) ---
#     # The y-axis is logarithmic to better visualize the drop-off
#     ax1.semilogy(s, marker='o', linestyle='-')
    
#     # --- 4. Plot the Cumulative Explained Variance ---
#     # Variance is proportional to the square of the singular values
#     variance_explained = s**2 / jnp.sum(s**2)
#     cumulative_variance = jnp.cumsum(variance_explained)
#     ax2 = ax[2,i]
#     ax2.plot(cumulative_variance, marker='.', linestyle='-')
# plt.tight_layout()
# plt.show()