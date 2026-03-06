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
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
    v_field = pr.BasisField(basis, pr.Coeffs(v_coeffs))
    p_field = pr.BasisField(basis, pr.Coeffs(p_coeffs))

    nx, ny = 300,300
    wall_y_val = .5
    vel_scale = 1.0
    nu = config["nu"]

    grid_x, grid_y = jnp.meshgrid(jnp.linspace(-1, 1, nx), jnp.linspace(-wall_y_val, wall_y_val, ny))

    cylinder_x, cylinder_y = 0.0, 0.0
    cylinder_radius = 0.1
    cylinder_2_x, cylinder_2_y = -.5,.2
    cylinder_2_radius = 0.075

    mask = (grid_x - cylinder_x)**2 + (grid_y - cylinder_y)**2 <= cylinder_radius**2
    mask2 = (grid_x - cylinder_2_x)**2 + (grid_y - cylinder_2_y)**2 <= cylinder_2_radius**2
    mask = mask | mask2

    u = u_field.evaluate(grid_x, grid_y).reshape(ny,nx)
    v = v_field.evaluate(grid_x, grid_y).reshape(ny,nx)
    p = p_field.evaluate(grid_x, grid_y).reshape(ny,nx)
    u_x = u_field.derivative(grid_x, grid_y, order=(1,0)).reshape(ny,nx)
    v_y = v_field.derivative(grid_x, grid_y, order=(0,1)).reshape(ny,nx)
    u_xx = u_field.derivative(grid_x, grid_y, order=(2,0)).reshape(ny,nx)
    u_yy = u_field.derivative(grid_x, grid_y, order=(0,2)).reshape(ny,nx)
    v_xx = v_field.derivative(grid_x, grid_y, order=(2,0)).reshape(ny,nx)
    v_yy = v_field.derivative(grid_x, grid_y, order=(0,2)).reshape(ny,nx)
    p_x = p_field.derivative(grid_x, grid_y, order=(1,0)).reshape(ny,nx)
    p_y = p_field.derivative(grid_x, grid_y, order=(0,1)).reshape(ny,nx)
    
    y_inlet = jnp.linspace(-wall_y_val, wall_y_val, 100)
    inlet_u = u_field.evaluate(-jnp.ones(100),y_inlet)
    inlet_v = v_field.evaluate(-jnp.ones(100),y_inlet)
    outlet_p = p_field.evaluate(jnp.ones(100),y_inlet)
    outlet_u_x = u_field.derivative(jnp.ones(100),y_inlet, order=(1,0))
    outlet_v_x = v_field.derivative(jnp.ones(100),y_inlet, order=(1,0))
    outlet_u_y = u_field.derivative(jnp.ones(100),y_inlet, order=(0,1))
    u = jnp.where(mask==0, u, jnp.nan)
    v = jnp.where(mask==0, v, jnp.nan)
    p = jnp.where(mask==0, p, jnp.nan)
    u_x = jnp.where(mask==0, u_x, jnp.nan)
    v_y = jnp.where(mask==0, v_y, jnp.nan)
    u_xx = jnp.where(mask==0, u_xx, jnp.nan)
    u_yy = jnp.where(mask==0, u_yy, jnp.nan)
    v_xx = jnp.where(mask==0, v_xx, jnp.nan)
    v_yy = jnp.where(mask==0, v_yy, jnp.nan)
    p_x = jnp.where(mask==0, p_x, jnp.nan)
    p_y = jnp.where(mask==0, p_y, jnp.nan)


    u_ref = -vel_scale/(wall_y_val**2)*(y_inlet-wall_y_val)*(y_inlet+wall_y_val)
    continuity = u_x + v_y
    x_eq = (u_xx + u_yy) - p_x
    y_eq = (v_xx + v_yy) - p_y
    continuity = jnp.log10(jnp.abs(continuity))
    x_eq = jnp.log10(jnp.abs(x_eq))
    y_eq = jnp.log10(jnp.abs(y_eq))
    inlet_u_res = inlet_u - u_ref
    inlet_v_res = inlet_v
    outlet_x_res = outlet_p
    outlet_y_res = outlet_v_x + outlet_u_y
    f, ax = plt.subplots(2,3, figsize=(15,5))
    ax = ax.flatten()
    im0 = ax[0].contourf(grid_x, grid_y, u_xx, levels=100)
    ax[0].set_aspect("equal")
    ax[0].set_title("u_xx")
    im1 = ax[1].contourf(grid_x, grid_y, u_yy, levels=100)
    ax[1].set_aspect("equal")
    ax[1].set_title("u_yy")
    im2 = ax[2].contourf(grid_x, grid_y, p_x, levels=100)
    ax[2].set_aspect("equal")
    ax[2].set_title("p_x")
    im3 = ax[3].contourf(grid_x, grid_y, v_xx, levels=100, cmap="bwr")
    ax[3].set_aspect("equal")
    ax[3].set_title("v_xx")
    im4 = ax[4].contourf(grid_x, grid_y, v_yy, levels=100, cmap="bwr")
    ax[4].set_aspect("equal")
    ax[4].set_title("v_yy")
    im5 = ax[5].contourf(grid_x, grid_y, p_y, levels=100, cmap="bwr")
    ax[5].set_aspect("equal")
    ax[5].set_title("p_y")
    cbar0 = f.colorbar(im0, ax=ax[0])
    cbar1 = f.colorbar(im1, ax=ax[1])
    cbar2 = f.colorbar(im2, ax=ax[2])
    cbar3 = f.colorbar(im3, ax=ax[3])
    cbar4 = f.colorbar(im4, ax=ax[4])
    cbar5 = f.colorbar(im5, ax=ax[5])
    plt.tight_layout()
    plt.savefig("figures/two_sphere_vis_equation.png")
    plt.savefig(results_path / "two_sphere_vis_equation.png")
    plt.close()


    f,ax = plt.subplots(3,3, figsize=(15,15))
    ax = ax.flatten()
    im0 = ax[0].contourf(grid_x, grid_y, u, levels=100)
    ax[0].set_aspect("equal")
    ax[0].set_title("u")
    im1 = ax[1].contourf(grid_x, grid_y, v, levels=100)
    ax[1].set_aspect("equal")
    ax[1].set_title("v")
    im2 = ax[2].contourf(grid_x, grid_y, p, levels=100)
    ax[2].set_aspect("equal")
    ax[2].set_title("p")
    im3 = ax[3].contourf(grid_x, grid_y, x_eq, levels=100, cmap="bwr")
    ax[3].set_aspect("equal")
    ax[3].set_title("log x_eq residual")
    im4 = ax[4].contourf(grid_x, grid_y, y_eq, levels=100, cmap="bwr")
    ax[4].set_aspect("equal")
    ax[4].set_title("log y_eq residual")
    im5 = ax[5].contourf(grid_x, grid_y, continuity, levels=100, cmap="bwr")
    ax[5].set_aspect("equal")
    ax[5].set_title("log continuity residual")
    im6 = ax[6].plot(y_inlet, inlet_u_res)
    ax[6].plot(y_inlet, inlet_v_res)
    ax[6].set_title("inlet residual")
    im7 = ax[7].plot(y_inlet, outlet_x_res)
    ax[7].set_title("outlet residual x")
    ax[8].plot(y_inlet, outlet_y_res)
    ax[8].set_title("outlet residual y")
    cbar0 = f.colorbar(im0, ax=ax[0])
    cbar1 = f.colorbar(im1, ax=ax[1])
    cbar2 = f.colorbar(im2, ax=ax[2])
    cbar3 = f.colorbar(im3, ax=ax[3])
    cbar4 = f.colorbar(im4, ax=ax[4])
    cbar5 = f.colorbar(im5, ax=ax[5])
    plt.tight_layout()
    plt.savefig("figures/two_sphere_vis.png")
    plt.savefig(results_path / "two_sphere_vis.png")
    plt.close()
