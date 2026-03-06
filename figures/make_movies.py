import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pathlib
import prism as pr
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

jax.config.update("jax_enable_x64", True)
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 16
})

def sphere_animation(plot_config):
    text_color = "white" if plot_config["dark_mode"] else "black"
    box_color = "black" if plot_config["dark_mode"] else "white"
    target_folder = pathlib.Path("data/fig_sphere_diffusion")
    data = pr.load_dict_from_hdf5(target_folder / "full_data.h5")
    config = data["config"]
    c_coeffs = data["fields"]["c"]
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nz"], config["basis_Nt"]))
    c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))

    radius = 1.0

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

    global_vmin, global_vmax = 0, .75
            
    print(f"\nGlobal range found: [{global_vmin:.4f}, {global_vmax:.4f}]")
    norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    cmap = plot_config["cmap"]
    def update(frame):
        ax.clear()
        t_eval = time_points[frame]
        T = t_eval * jnp.ones_like(X)
        c_values = np.asarray(c_field.evaluate(X, Y, Z, T).reshape(X.shape))
        colors = cm.jet(norm(c_values))
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=False, shade=False,cmap=cmap)
        ax.set_xlabel("X-axis", color=text_color)
        ax.set_ylabel("Y-axis", color=text_color)
        ax.set_zlabel("Z-axis", color=text_color)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.tick_params(colors=text_color)
        ax.set_facecolor(box_color)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f"Diffusion on a Sphere (t = {t_eval:.2f})", color=text_color)
        ax.tick_params(colors=text_color)
        ax.view_init(elev=20., azim=30 + frame*0.5)
        print(f"Rendering frame {frame + 1}/{n_frames} (t={t_eval:.2f})")
        return ax,

    mappable = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    fig.colorbar(mappable, cax=cax, label="C Value")

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)
    output_filename = f"{plot_config['save_dir']}/sphere_diffusion_dark.mp4" if plot_config["dark_mode"] else f"{plot_config['save_dir']}/sphere_diffusion.mp4"
    print(f"\nSaving animation to '{output_filename}'. This may take a while...")
    ani.save(output_filename, writer='ffmpeg', fps=15, dpi=150)
    print("Animation saved successfully.")
    plt.close(fig)

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

def wave_animation(plot_config):
    results_path = pathlib.Path("data/fig_wave")
    data = pr.load_dict_from_hdf5(results_path / "full_data.h5")
    text_color = "white" if plot_config["dark_mode"] else "black"
    box_color = "black" if plot_config["dark_mode"] else "white"
    config = data["config"]
    u_coeffs = data["fields"]["u"]
    basis_N = (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"])
    basis = pr.basis.BasisND([pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis], basis_N)
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
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
    
    reference_data = np.loadtxt(results_path / "wave_grid.csv", delimiter=",", skiprows=9)
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

    diff = slice_data.reshape(-1) - ref_slice_data.reshape(-1)
    dA = 0.01**3
    error = jnp.sqrt(jnp.nansum(diff**2)*dA)
    print(f"L2 error: {error}")

    fig, ax = plt.subplots(1,2,figsize=(12,5))

    cmap = plot_config["cmap"]

    vmax = jnp.nanmax(jnp.abs(ref_slice_data))
    vmin = jnp.nanmin(jnp.abs(ref_slice_data))
    print(f"vmax: {vmax}, vmin: {vmin}")
    slice_data = slice_data
    cont = ax[0].contourf(x_grid, y_grid, slice_data[0], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
    ax[0].set_xlabel('x', color=text_color)
    ax[0].set_ylabel('y', color=text_color)
    ax[0].tick_params(colors=text_color)
    cont1 = ax[1].contourf(x_grid, y_grid, ref_slice_data[0], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
    ax[1].set_xlabel('x', color=text_color)
    ax[1].set_ylabel('y', color=text_color)
    ax[1].tick_params(colors=text_color)
    fig.colorbar(cont1, ax=ax[1])
    fig.colorbar(cont1, ax=ax[0])
    ax[0].set_facecolor(box_color)
    ax[1].set_facecolor(box_color)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    def update(frame):
        ax[0].clear()
        ax[1].clear()
        cont = ax[0].contourf(x_grid, y_grid, slice_data[frame], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
        cont1 = ax[1].contourf(x_grid, y_grid, ref_slice_data[frame], levels=100, cmap=cmap,vmin=vmin,vmax=vmax)
        ax[0].set_xlabel('x', color=text_color)
        ax[0].set_ylabel('y', color=text_color)
        ax[0].tick_params(colors=text_color)
        ax[1].set_xlabel('x', color=text_color)
        ax[1].set_ylabel('y', color=text_color)
        ax[1].tick_params(colors=text_color)
        ax[0].set_title('Optimized Fit', color=text_color)
        ax[1].set_title('Reference (COMSOL)', color=text_color)
        ax[0].set_facecolor(box_color)
        ax[1].set_facecolor(box_color)
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        
        print(f"Processing frame {frame+1}/{len(t_vec)}")

        return cont, cont1

    ani = FuncAnimation(fig, update, frames=len(t_vec), blit=False)

    output_file = f"{plot_config['save_dir']}/wave_comparison_dark.mp4" if plot_config["dark_mode"] else f"{plot_config['save_dir']}/wave_comparison.mp4"
    ani.save(output_file, writer='ffmpeg', fps=20, dpi=150)
    print(f"\nAnimation successfully saved to {output_file}")
    plt.close()

def heat_forcing_animation(plot_config):
    results_path = pathlib.Path("data/fig_heat_forcing")
    data = pr.load_dict_from_hdf5(results_path / "full_data.h5")
    text_color = "white" if plot_config["dark_mode"] else "black"
    box_color = "black" if plot_config["dark_mode"] else "white"
    config = data["config"]
    c_coeffs = data["fields"]["c"]
    f_coeffs = data["fields"]["f"]
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"]))
    c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))
    f_basis = pr.basis.FourierChebyshevBasis2D((config["basis_bc"], config["basis_Nt"]))
    f_field = pr.BasisField(f_basis, pr.Coeffs(f_coeffs))

    ny, nx, nt = 200, 200, 200
    x_vec = jnp.linspace(-1, 1, nx)
    y_vec = jnp.linspace(-1, 1, ny)
    t_vec = jnp.linspace(-1, 1, nt)
    x_grid_2d, y_grid_2d = jnp.meshgrid(x_vec, y_vec)
    r_grid = jnp.sqrt(x_grid_2d**2 + y_grid_2d**2)
    mask = r_grid <= 1
    
    theta_true = jnp.linspace(-np.pi, np.pi, 300)
    theta_eval = jnp.linspace(-1, 1, 300)

    # --- 2. Pre-computation Loop ---
    print("Pre-computing all frames...")
    c_frames = []
    f_frames = []
    for t in t_vec:
        # Evaluate concentration field
        t_grid_slice = jnp.full_like(x_grid_2d, t)
        c_slice = c_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx)
        c_slice = jnp.where(mask == 0, jnp.nan, c_slice)
        c_frames.append(c_slice)
        
        # Evaluate forcing field
        t_f_slice = jnp.full_like(theta_eval, t)
        f_slice = f_field.evaluate(theta_eval, t_f_slice)
        f_frames.append(f_slice)

    # Stack into single arrays for easy access
    c_frames_arr = jnp.stack(c_frames)
    f_frames_arr = jnp.stack(f_frames)
    print("Pre-computation complete.")

    # --- 3. Determine Global Min/Max and Normalization ---
    global_vmin = jnp.nanmin(c_frames_arr)
    global_vmax = jnp.nanmax(c_frames_arr)
    global_bound = np.max([jnp.abs(global_vmin), jnp.abs(global_vmax)])
    norm = Normalize(vmin=-global_bound, vmax=global_bound)
    cmap = plt.get_cmap("bwr")

    # --- 4. Initial Plot Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 1, 0.05]})
    ax0, ax1, cax = axes[0], axes[1], axes[2]

    # Plot initial contour
    cont = ax0.contourf(x_grid_2d, y_grid_2d, c_frames_arr[0], levels=100, cmap=cmap, norm=norm)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig.colorbar(sm, cax=cax)
    ax0.set_aspect('equal', 'box')
    ax0.set_title('Optimized Solution', color=text_color)
    ax0.set_facecolor('black')
    ax0.tick_params(colors=text_color)

    # <<< CHANGE 1: Setup for LineCollection >>>
    # Create segments for the line [(x1, y1), (x2, y2)], [(x2, y2), (x3, y3)], ...
    points = np.array([theta_true, f_frames_arr[0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the LineCollection object
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the color of each segment based on the y-value at the start of the segment
    lc.set_array(f_frames_arr[0][:-1])
    ax1.add_collection(lc)
    
    # Configure the forcing plot
    ax1.set_title('Optimized Forcing', color=text_color)
    ax1.set_xlim(theta_true.min(), theta_true.max())
    ax1.set_ylim(-global_bound, global_bound)
    ax1.set_xlabel('$\\theta$', color=text_color)
    ax1.axhline(y=0,color='grey',linestyle='--')
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'], color=text_color)
    ax1.tick_params(colors=text_color)
    # --- 5. Animation Update Function ---
    def update(frame):
        ax0.clear()
        ax1.clear()
        
        # <<< CHANGE 2: Use pre-computed data >>>
        c_slice_new = c_frames_arr[frame]
        f_slice_new = f_frames_arr[frame]

        # Update contour plot
        ax0.contourf(x_grid_2d, y_grid_2d, c_slice_new, levels=100, cmap=cmap, norm=norm)
        ax0.set_title('Optimized Solution')
        ax0.set_aspect('equal', 'box')
        ax0.set_facecolor('black')
        ax0.tick_params(colors=text_color)

        # <<< CHANGE 3: Update LineCollection >>>
        points_new = np.array([theta_true, f_slice_new]).T.reshape(-1, 1, 2)
        segments_new = np.concatenate([points_new[:-1], points_new[1:]], axis=1)
        lc.set_segments(segments_new)
        lc.set_array(f_slice_new[:-1]) # Update colors
        ax1.add_collection(lc)

        # Re-apply plot settings since clear() removes them
        ax1.set_title('Optimized Forcing', color=text_color)
        ax1.set_xlim(theta_true.min(), theta_true.max())
        ax1.set_ylim(-global_bound, global_bound) # Use pre-computed limits
        ax1.set_xlabel('$\\theta$', color=text_color)
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'], color=text_color)
        ax1.tick_params(colors=text_color)
        ax1.axhline(y=0,color='grey',linestyle='--')
        
        print(f"Processing frame {frame+1}/{len(t_vec)}")
        return cont, lc

    # --- 6. Save Animation ---
    ani = FuncAnimation(fig, update, frames=len(t_vec), blit=False)
    
    save_dir = pathlib.Path(plot_config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    output_file = f"{save_dir}/heat_forcing_dark.mp4" if plot_config["dark_mode"] else f"{save_dir}/heat_forcing.mp4"
    ani.save(output_file, writer='ffmpeg', fps=10, dpi=150)
    print(f"\nAnimation successfully saved to {output_file}")
    plt.close()


def transport_animation(plot_config):
    results_path = pathlib.Path("data/fig_transport")
    data = pr.load_dict_from_hdf5(results_path / "full_data.h5")
    text_color = "white" if plot_config["dark_mode"] else "black"
    box_color = "black" if plot_config["dark_mode"] else "white"
    config = data["config"]
    c_coeffs = data["fields"]["c"]
    u_coeffs = data["fields"]["u"]
    v_coeffs = data["fields"]["v"]
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"]))
    c_field = pr.fields.BasisField(basis, pr.Coeffs(c_coeffs))
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
    v_field = pr.BasisField(basis, pr.Coeffs(v_coeffs))

    ny, nx, nt = 200, 200, 200
    x_vec = np.linspace(-1, 1, nx)
    y_vec = np.linspace(-1, 1, ny)
    t_vec = np.linspace(-1, 1, nt)
    x_grid_2d, y_grid_2d = np.meshgrid(x_vec, y_vec)
    r_grid = np.sqrt(x_grid_2d**2 + y_grid_2d**2)
    mask = r_grid > 1 # Mask areas outside the circle

    # --- 2. Pre-computation Loop ---
    print("Pre-computing all frames...")
    c_frames, u_frames, v_frames = [], [], []
    for t in t_vec:
        t_grid_slice = jnp.full_like(x_grid_2d, t)
        
        # Evaluate and mask each field
        c_slice = np.where(mask, np.nan, c_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx))
        u_slice = np.where(mask, np.nan, u_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx))
        v_slice = np.where(mask, np.nan, v_field.evaluate(x_grid_2d, y_grid_2d, t_grid_slice).reshape(ny, nx))
        
        c_frames.append(c_slice)
        u_frames.append(u_slice)
        v_frames.append(v_slice)
    
    # <<< Stack lists into 3D tensors for easy indexing >>>
    c_frames_arr = np.stack(c_frames)
    u_frames_arr = np.stack(u_frames)
    v_frames_arr = np.stack(v_frames)
    print("Pre-computation complete.")

    vmin, vmax = 0, np.nanmax(c_frames_arr)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("jet")
    
    magnitudes_arr = np.sqrt(u_frames_arr**2 + v_frames_arr**2)
    global_max_mag = np.nanmax(magnitudes_arr)

    mag_norm = Normalize(vmin=0, vmax=global_max_mag)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    
    ax[0].contourf(x_grid_2d, y_grid_2d, c_frames_arr[0], levels=100, cmap=cmap, norm=norm)
    ax[0].set_aspect('equal', 'box')
    ax[0].set_facecolor(box_color)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar0 = fig.colorbar(sm, ax=ax[0], shrink=0.8)
    cbar0.set_label('Concentration', color=text_color)
    cbar0.ax.yaxis.set_tick_params(color=text_color)
    cbar0.ax.yaxis.label.set_color(text_color)

    # 2. Use contourf to draw the black and white background.
    #    zorder=0 ensures it's drawn behind everything else.
    # ax[1].contourf(x_grid_2d, y_grid_2d, background_data, 
    #                levels=[-0.5, 0.5, 1.5], 
    #                colors=['black', 'white'], 
    #                zorder=0)
    ax[1].contourf(x_grid_2d, y_grid_2d, magnitudes_arr[0], levels=100, cmap=cmap, norm=mag_norm,zorder=0)
    sm_mag = plt.cm.ScalarMappable(cmap=cmap, norm=mag_norm)
    cbar1 = fig.colorbar(sm_mag, ax=ax[1], shrink=0.8)
    cbar1.set_label('Velocity Magnitude', color=text_color)
    cbar1.ax.yaxis.set_tick_params(color=text_color)
    cbar1.ax.yaxis.label.set_color(text_color)
    
    skip = 5 
    quiver_plot = ax[1].quiver(x_grid_2d[::skip, ::skip], y_grid_2d[::skip, ::skip], 
                               u_frames_arr[0, ::skip, ::skip], v_frames_arr[0, ::skip, ::skip],
                               scale_units='xy', scale=10*global_max_mag,zorder=1,color='white') 
    ax[1].set_aspect('equal', 'box')
    ax[1].set_facecolor(box_color)

    # Set initial titles
    ax[0].set_title('Concentration', color=text_color)
    ax[1].set_title('Velocity', color=text_color)
    ax[0].tick_params(colors=text_color)
    ax[1].tick_params(colors=text_color)

    # --- 4. Animation Update Function ---
    def update(frame):
        ax[0].clear()
        ax[0].contourf(x_grid_2d, y_grid_2d, c_frames_arr[frame], levels=100, cmap=cmap, norm=norm)
        
        u_slice = u_frames_arr[frame, ::skip, ::skip]
        v_slice = v_frames_arr[frame, ::skip, ::skip]
        quiver_plot.set_UVC(u_slice, v_slice)
        ax[1].contourf(x_grid_2d, y_grid_2d, magnitudes_arr[frame], levels=100, cmap=cmap, norm=mag_norm,zorder=0)

        ax[0].set_title('Concentration', color=text_color)
        ax[0].set_aspect('equal', 'box')
        ax[1].set_title('Velocity', color=text_color)
        ax[0].tick_params(colors=text_color)
        ax[1].tick_params(colors=text_color)
        ax[0].set_facecolor(box_color)
        ax[1].set_facecolor(box_color)
        print(f"Processing frame {frame+1}/{len(t_vec)}")
        
    # --- 5. Save Animation ---
    ani = FuncAnimation(fig, update, frames=len(t_vec), blit=False)
    
    save_dir = pathlib.Path(plot_config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    output_file = f"{save_dir}/transport_dark.mp4" if plot_config["dark_mode"] else f"{save_dir}/transport.mp4"
    ani.save(output_file, writer='ffmpeg', fps=10, dpi=150)
    print(f"\nAnimation successfully saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    plot_config = {"cmap":"jet", "save_dir":"movies","colors":["#64CC33","#33BACC","#FF405E"],"cbar_size":(1.25, 4),"transparent":False,"figsize":(4,4),"file_type":"png","dark_mode":True}

    if plot_config["dark_mode"]:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    # sphere_animation(plot_config)
    wave_animation(plot_config)
    # heat_forcing_animation(plot_config)
    # transport_animation(plot_config)