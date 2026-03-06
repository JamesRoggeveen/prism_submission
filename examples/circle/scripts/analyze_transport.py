import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib
import prism as pr
import argparse

RESULTS_BASE_DIR = pathlib.Path("results")
FOLDER_SUFFIX = "_transport"

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
u_coeffs = data["fields"]["u"]
v_coeffs = data["fields"]["v"]
basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"]))
c_field = pr.fields.BasisField(basis, pr.Coeffs(c_coeffs))
u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
v_field = pr.BasisField(basis, pr.Coeffs(v_coeffs))

true_solution = jnp.load("data/ground_truth.npy")
ny, nx, nt = true_solution.shape
nt = 200
x_vec = jnp.linspace(-1,1,nx)
y_vec = jnp.linspace(-1,1,ny)
t_vec = jnp.linspace(-1,1,nt)
x_grid_2d, y_grid_2d = jnp.meshgrid(x_vec, y_vec)
c_initial = c_field.evaluate(x_grid_2d, y_grid_2d, jnp.ones((ny,nx))*-1)
c_initial = c_initial.reshape(ny,nx)

r_grid = jnp.sqrt(x_grid_2d**2 + y_grid_2d**2)
mask = r_grid <= 1

c_initial = jnp.where(mask == 0, jnp.nan, c_initial)

c_mid = c_field.evaluate(x_grid_2d, y_grid_2d, jnp.zeros((ny,nx)))
c_mid = c_mid.reshape(ny,nx)
c_mid = jnp.where(mask == 0, jnp.nan, c_mid)


c_final = c_field.evaluate(x_grid_2d, y_grid_2d, jnp.ones((ny,nx))*1)
c_final = c_final.reshape(ny,nx)
c_final = jnp.where(mask == 0, jnp.nan, c_final)

f, ax = plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
im0 = ax[0].contourf(x_grid_2d, y_grid_2d, c_initial, levels=100,cmap="hot")
im1 = ax[1].contourf(x_grid_2d, y_grid_2d, c_mid, levels=100,cmap="hot")
im2 = ax[2].contourf(x_grid_2d, y_grid_2d, c_final, levels=100,cmap="hot")
ax[0].set_title("Initial")
ax[1].set_title("Mid")
ax[2].set_title("Final")
ax[0].set_aspect('equal', 'box')
ax[1].set_aspect('equal', 'box')
ax[2].set_aspect('equal', 'box')
cbar = f.colorbar(im0, ax=ax[0])
cbar = f.colorbar(im1, ax=ax[1])
cbar = f.colorbar(im2, ax=ax[2])
plt.tight_layout()
plt.savefig("figures/c_comparison.png")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 5)) # Adjusted size for colorbar

# --- Create a dedicated axis for the color bar ---
# This creates a new small axis on the right side of the figure.
# The arguments are [left, bottom, width, height] in figure-fraction coordinates.
# cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# --- Initial Plot Setup ---
# We still need to draw the first frame to establish the mappable for the colorbar
t_initial = t_vec[0]
t_grid_slice = jnp.full_like(x_grid_2d, t_initial)
c_slice_initial = c_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx)
c_slice_initial = jnp.where(mask == 0, jnp.nan, c_slice_initial)
u_slice_initial = u_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx)
u_slice_initial = jnp.where(mask == 0, jnp.nan, u_slice_initial)
v_slice_initial = v_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx)
v_slice_initial = jnp.where(mask == 0, jnp.nan, v_slice_initial)
# vmin, vmax = jnp.nanmin(c_slice_initial), jnp.nanmax(c_slice_initial)
vmin, vmax = 0, jnp.nanmax(c_slice_initial)
cont = ax[0].contourf(x_grid_2d, y_grid_2d, c_slice_initial, levels=100, cmap="jet", vmin=vmin, vmax=vmax)
# cbar = fig.colorbar(cont, cax=cax) # Tell the colorbar to use our dedicated axis
cbar = fig.colorbar(cont, ax=ax[0])
ax[0].set_aspect('equal', 'box')
ax[0].set_title(f'Time step t = {t_initial:.2f}')

cont1 = ax[1].quiver(x_grid_2d[::10,::10], y_grid_2d[::10,::10], u_slice_initial[::10,::10], v_slice_initial[::10,::10])
ax[1].set_title(f'Time step t = {t_initial:.2f}')
ax[1].set_aspect('equal', 'box')

total_c = []
total_c.append(jnp.nansum(c_slice_initial))

def update(frame):
    # --- 1. Clear the contents of both axes ---
    ax[0].clear()
    ax[1].clear()
    # cax.clear()

    # --- 2. Calculate the new data for the current time step ---
    t = t_vec[frame]
    t_grid_slice = jnp.full_like(x_grid_2d, t)
    c_slice_new = c_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice)
    c_slice_new = c_slice_new.reshape(ny, nx)
    c_slice_new = jnp.where(mask == 0, jnp.nan, c_slice_new)

    # --- 3. Draw the new plot and the new color bar ---
    # vmin/vmax are determined automatically for this frame
    # vmin, vmax = jnp.nanmin(c_slice_new), jnp.nanmax(c_slice_new)

    cont_new = ax[0].contourf(x_grid_2d, y_grid_2d, c_slice_new, levels=100, cmap="jet", vmin=vmin, vmax=vmax)
    u_slice_new = u_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx)
    u_slice_new = jnp.where(mask == 0, jnp.nan, u_slice_new)
    v_slice_new = v_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx)
    v_slice_new = jnp.where(mask == 0, jnp.nan, v_slice_new)
    cont1 = ax[1].quiver(x_grid_2d[::10,::10], y_grid_2d[::10,::10], u_slice_new[::10,::10], v_slice_new[::10,::10])
    # Create the new color bar on the same, now-cleared, dedicated axis
    # fig.colorbar(cont_new, cax=cax) # Tell the colorbar to use our dedicated axis
    #cbar = fig.colorbar(cont_new, ax=ax)
    total_c.append(jnp.nansum(c_slice_new))
    # --- 4. Set titles and aspect ratio for the new plot ---
    ax[0].set_title(f'Time step t = {t:.2f}')
    ax[0].set_aspect('equal', 'box')
    ax[1].set_title(f'Time step t = {t:.2f}')
    ax[1].set_aspect('equal', 'box')
    print(f"Processing frame {frame+1}/{len(t_vec)}")

ani = animation.FuncAnimation(fig, update, frames=len(t_vec), blit=False)

# Save the animation as an MP4 file.
# This requires having ffmpeg installed on your system.
output_file = "figures/c_field_animation.mp4"
ani.save(output_file, writer='ffmpeg', fps=10, dpi=150)
print(f"\nAnimation successfully saved to {output_file}")
plt.close()
# Copy the animation to the results directory
import shutil
results_animation_path = results_path / "c_field_animation.mp4"
shutil.copy2(output_file, results_animation_path)
print(f"Animation also copied to {results_animation_path}")



plt.plot(total_c)
plt.savefig("figures/transport_total_c.png")
plt.savefig(results_path / "transport_total_c.png")
plt.close()