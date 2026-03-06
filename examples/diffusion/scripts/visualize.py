import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib
import prism as pr
from matplotlib.image import imread
import argparse

RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_diffusion"

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

results_path = get_target_folder()
print(results_path)
data = pr.load_dict_from_hdf5(results_path / "full_data.h5")
config = data["config"]
c_coeffs = data["fields"]["c"]
basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"]))
c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))

mask = imread(config["domain_path"])[:,:,3]

ny, nx = mask.shape

x_vec = jnp.linspace(-1,1,nx)
y_vec = jnp.linspace(-1,1,ny)
t_vec = jnp.linspace(-1,-.8,100)
x_grid_2d, y_grid_2d = jnp.meshgrid(x_vec, y_vec)
c_initial = c_field.evaluate(x_grid_2d, y_grid_2d, jnp.ones((ny,nx))*-1)
c_initial = c_initial.reshape(ny,nx)
c_initial = jnp.where(mask == 0, jnp.nan, c_initial)
plt.contourf(x_grid_2d, y_grid_2d, c_initial, levels=100)
plt.colorbar()
plt.savefig("figures/c_initial.png")
plt.close()


fig, ax = plt.subplots()

t_grid_slice = jnp.full_like(x_grid_2d, t_vec[0])
c_slice = c_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny,nx)
c_slice = jnp.where(mask == 0, jnp.nan, c_slice)


vmin, vmax = 0, 1
cont = ax.contourf(x_grid_2d, y_grid_2d, c_slice, levels=100, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cont, ax=ax)
cbar.set_label('c value')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame):
    t = t_vec[frame]
    
    ax.clear()

    t_grid_slice = jnp.full_like(x_grid_2d, t)
    c_slice_new = c_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice)
    c_slice_new = c_slice_new.reshape(ny, nx)
    c_slice_new = jnp.where(mask == 0, jnp.nan, c_slice_new)

    cont = ax.contourf(x_grid_2d, y_grid_2d, c_slice_new, levels=100, vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Time step t = {t:.2f}')
    
    print(f"Processing frame {frame+1}/{len(t_vec)}")

    return cont,

# --- Create and save the animation ---
# FuncAnimation will call the 'update' function for each frame.
ani = animation.FuncAnimation(fig, update, frames=len(t_vec), blit=False)

# Save the animation as an MP4 file.
# This requires having ffmpeg installed on your system.
output_file = "figures/c_field_animation.gif"
ani.save(output_file, writer='ffmpeg', fps=10, dpi=150)

print(f"\nAnimation successfully saved to {output_file}")
plt.close()

x_point = jnp.zeros(100)
y_point = jnp.zeros(100)
t_point = jnp.linspace(-1,1,100)

c_point = c_field.evaluate(x_point, y_point, t_point)

plt.plot(t_point, c_point)
plt.savefig("figures/c_point.png")
plt.close()