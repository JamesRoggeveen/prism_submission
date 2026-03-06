import prism as pr
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pathlib
import numpy as np
import yaml
import jax
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from matplotlib.pyplot import imread
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec # <-- New import
from scipy.special import roots_legendre
from scipy.special import sph_harm
from mpl_toolkits.axes_grid1 import make_axes_locatable # <-- New import

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

def standalone_log_colorbar(plot_config, log_range, label, name):
    min_val, max_val = log_range
    cmap = plot_config["cmap"]
    size = plot_config["cbar_size"]
    fig, cax = plt.subplots(figsize=size)
    norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
    cbar.set_label(label, size=12, weight='bold')
    fig.tight_layout()
    transparent = plot_config.get("transparent", False)
    save_dir = plot_config['save_dir']
    plt.savefig(f"{save_dir}/{name}_log_colorbar.{plot_config["file_type"]}", transparent=transparent, dpi=300,format=plot_config["file_type"])
    plt.close(fig)

def standalone_colorbar(plot_config,range,label,name):
    min_val, max_val = range
    cmap = plot_config["cmap"]
    size = plot_config["cbar_size"]
    fig, cax = plt.subplots(figsize=size)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
    cbar.set_label(label, size=12, weight='bold')
    fig.tight_layout()
    transparent = plot_config["transparent"]
    plt.savefig(f"{plot_config['save_dir']}/{name}_colorbar.{plot_config["file_type"]}", transparent=transparent,dpi=300,format=plot_config["file_type"])
    plt.close()

def boundary_from_mask(binary_mask, x_grid, y_grid):
    padded_mask = jnp.pad(binary_mask, 1, mode='constant', constant_values=False)

    boundary_mask = (
        (padded_mask[1:-1, 1:-1] != padded_mask[0:-2, 1:-1]) |  # up
        (padded_mask[1:-1, 1:-1] != padded_mask[2:, 1:-1]) |    # down
        (padded_mask[1:-1, 1:-1] != padded_mask[1:-1, 0:-2]) |  # left
        (padded_mask[1:-1, 1:-1] != padded_mask[1:-1, 2:])      # right
    )

    boundary_mask = boundary_mask & binary_mask

    boundary_indices = jnp.nonzero(boundary_mask)
    boundary_x = x_grid[boundary_indices]
    boundary_y = y_grid[boundary_indices]

    # Smooth mask to improve normal vector calculation
    sigma = 2.0
    smoothed_mask = gaussian_filter(binary_mask.astype(float), sigma=sigma)

    grad_y, grad_x = jnp.gradient(smoothed_mask.astype(float))

    # 2. Extract the gradient vectors at the boundary points
    nx = -grad_x[boundary_indices]
    ny = -grad_y[boundary_indices]

    # 3. Normalize the gradient vectors to get unit normals
    magnitude = jnp.sqrt(nx**2 + ny**2)
    # Avoid division by zero for any potential zero-magnitude vectors
    magnitude = jnp.where(magnitude == 0, 1, magnitude)
    nx_norm = nx / magnitude
    ny_norm = ny / magnitude

    return boundary_x, boundary_y, nx_norm, ny_norm

def domain_from_png(path):
    img = imread(path)[:,:,2]
    mask = img > 0
    x_vec = jnp.linspace(-1,1,img.shape[1])
    y_vec = jnp.linspace(-1,1,img.shape[0])
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)

    mask_data = pr.data.ReferenceData(coords=(x_grid, y_grid), data=mask)

    boundary_x, boundary_y, nx_norm, ny_norm = boundary_from_mask(mask, x_grid, y_grid)

    boundary_data = pr.data.BoundaryData(coords=(boundary_x, boundary_y), normal_vector=(nx_norm, ny_norm))

    return mask_data, boundary_data

def define_peanut_geometry(n_points, circle = False):
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
    if circle:
        mask = mask1 & mask2
    else:
        mask = mask1
    return x_grid, y_grid, mask

def plot_laplace_figure(plot_config):
    def boundary_condition_theta(x,y):
        theta = jnp.arctan2(y, x)
        return 2 * jnp.sin(theta) + jnp.cos(3 * theta)
    data_path = pathlib.Path("data/fig_laplace")
    full_data = pr.load_dict_from_hdf5(data_path / "full_data.h5")
    config = full_data["config"]
    u_coeffs = full_data["fields"]["u"]
    errors = np.load(data_path / "errors.npy")
    basis_N = errors[0,:]
    L2_errors = errors[1,:]
    Linf_errors = errors[2,:]
    times = errors[3,:]
    N = config["basis_Nx"]
    N=60
    basis = pr.ChebyshevBasis2D((N, N))
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
    n_grid = 500
    x_grid, y_grid, mask = define_peanut_geometry(n_grid, circle=True)
    u_eval = u_field.evaluate(x_grid, y_grid)
    u_eval = u_eval.reshape(x_grid.shape)
    u_eval = jnp.where(mask == 0, jnp.nan, u_eval)

    basis_inverse = pr.ChebyshevBasis2D((45,45))
    data_inverse = pr.load_dict_from_hdf5(data_path / "full_data_inverse.h5")
    print(data_inverse["config"])
    inverse_coeffs = data_inverse["fields"]["u"]
    u_field_inverse = pr.BasisField(basis_inverse, pr.Coeffs(inverse_coeffs))
    u_eval_inverse = u_field_inverse.evaluate(x_grid, y_grid)
    u_eval_inverse = u_eval_inverse.reshape(x_grid.shape)
    u_eval_inverse = jnp.where(mask == 0, jnp.nan, u_eval_inverse)

    reference_data = np.loadtxt(data_path / "laplace_grid.csv", delimiter=",", skiprows=9)
    ref_x, ref_y, ref_data = reference_data[:,0], reference_data[:,1], reference_data[:,2]
    ref_x, ref_y, ref_data = ref_x.reshape(n_grid,n_grid), ref_y.reshape(n_grid,n_grid), ref_data.reshape(n_grid,n_grid)
    ref_data = jnp.where(mask == 0, jnp.nan, ref_data)

    lake_data = pr.load_dict_from_hdf5(data_path / "laplace_taal_100.h5")
    lake_mask = data_path/"taal_lake_mask_small.png"
    lake_mask_data, lake_mask_boundary = domain_from_png(lake_mask)

    bc_x, bc_y = lake_mask_boundary.coords
    lake_x, lake_y = lake_mask_data.coords
    lake_mask = lake_mask_data.data
    lake_N = lake_data["config"]["basis_Nx"]
    basis = pr.ChebyshevBasis2D((lake_N, lake_N))
    lake_field = pr.BasisField(basis, pr.Coeffs(lake_data["fields"]["c"]))
    lake_eval = lake_field.evaluate(lake_x, lake_y)
    lake_eval = lake_eval.reshape(lake_x.shape)
    lake_eval = jnp.where(lake_mask == 0, jnp.nan, lake_eval)
    lake_bc = boundary_condition_theta(bc_x, bc_y)

    cmap = plot_config["cmap"]
    labelx, labely = 0.02, 0.1
    label_pad = 0.25
    # Plot A: Optimized Solution
    f, ax = plt.subplots(1,1,figsize=(4,4))
    im0 = ax.contourf(x_grid, y_grid, u_eval, levels=100,cmap = cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$",rotation=0)
    ax.set_ylabel("$y$",rotation=0)
    ax.set_title("Optimized Solution")
    ax.text(labelx, labely, "$(a)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/laplace_plot_a.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()
    
    # Plot A Colorbar
    fig_cb, ax_cb = plt.subplots(figsize=(1.25, 4)) # A new figure with a tall, narrow aspect ratio

    # Create the colorbar, telling it to use the new axes (ax_cb)
    # and the color information from our original plot's mappable.
    cb = fig_cb.colorbar(im0, cax=ax_cb)
    cb.set_label("Solution Value (u)") # Optional: add a label to the colorbar

    fig_cb.tight_layout()
    fig_cb.savefig(f"{plot_config['save_dir']}/laplace_plot_a_colorbar.png")
    plt.close(fig_cb) # Close the colorbar figure
    
    # Plot B: Reference Solution
    f, ax = plt.subplots(1,1,figsize=(4,4))
    im1 = ax.contourf(x_grid, y_grid, ref_data, levels=100,cmap = cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$",rotation=0)
    ax.set_ylabel("$y$",rotation=0)
    ax.set_title("Reference Solution")
    ax.text(labelx, labely, "$(b)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/laplace_plot_b.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()
    
    # Plot C: Error Analysis
    f, ax = plt.subplots(1,1,figsize=(4,4))
    ax.set_xlabel("Basis Size")
    ax.set_ylabel("Error")
    ax.set_title("Relative Error")
    ax.semilogy(basis_N, L2_errors,marker="o",label="$L_2$ Error",color=plot_config["colors"][0])
    ax.semilogy(basis_N, Linf_errors,marker="o",label="$L_\\infty$ Error",color=plot_config["colors"][1])
    ax.legend()
    ax.text(labelx, labely, "$(c)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/laplace_plot_c.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    f, ax = plt.subplots(1,1,figsize=(4,4))
    ax.scatter(bc_x, bc_y, c=lake_bc, cmap=cmap, s=.3)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$",rotation=0)
    ax.set_ylabel("$y$",rotation=0)
    ax.set_title("Lake Taal Boundary Conditions")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.text(labelx, labely, "$(e)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/laplace_plot_e.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    f,ax = plt.subplots(1,1,figsize=(4,4))
    ax.contourf(lake_x, lake_y, lake_eval, levels=100,cmap = cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$",rotation=0)
    ax.set_ylabel("$y$",rotation=0)
    ax.set_title("Lake Taal Solution")
    ax.text(labelx, labely, "$(f)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/laplace_plot_f.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()
    
    if config["solver_type"] == "FirstOrder":
        solver = f"{config['solver_type']}_{config['optimizer_name']}_{config['loss_strategy']}"
        steps = config["n_epochs"]
    else:
        solver = config["solver_type"]
        steps = config["max_steps"]


    plot_data = {
        "basis_Nx": config["basis_Nx"],
        "basis_Ny": config["basis_Ny"],
        "n_pde": config["n_pde"],
        "n_bc": config["n_bc"],
        "solver": solver,
        "steps": steps,
        "basis": "chebyshev",
        "grid": "n_grid",
        "results_dir": config["results_dir"],
        "time_taken": float(times[-1]),
        "lake_n_pde": lake_data["config"]["n_pde"],
        "lake_n_bc": lake_data["config"]["n_bc"],
        "lake_Nx": lake_data["config"]["basis_Nx"],
        "lake_Ny": lake_data["config"]["basis_Ny"],
        "lake_basis": "chebyshev",
        "lake_solver": lake_data["config"]["solver"],
    }

    sample_data = np.load(data_path / "sample_data.npy")
    sample_x, sample_y, sample_data = sample_data[:,0], sample_data[:,1], sample_data[:,2]

    f, ax = plt.subplots(1,1,figsize=(4,4))
    im0 = ax.contourf(x_grid, y_grid, u_eval_inverse, levels=100,cmap = cmap)
    ax.scatter(sample_x, sample_y, c='black', s=.3)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$",rotation=0)
    ax.set_ylabel("$y$",rotation=0)
    ax.set_title(f"Inverse Solution ({data_inverse['config']['n_bc']} Data Points)")
    ax.text(labelx, labely, "$(d)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/laplace_plot_d.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    with open("figure_info/laplace_plot_info.yml", "w") as f:
        yaml.dump(plot_data, f)
    


    error_data = np.loadtxt(data_path / "errors.csv", delimiter=',')
    repeats = 15
    n_samples = error_data[:,0]
    l2_errors = error_data[:,1:repeats+1]
    Linf_errors = error_data[:,repeats+1:2*repeats+1]

    reference_error = np.load("data/fig_laplace_coeffs_error/adam_01_20_10/errors.npy")
    basis_size = reference_error[0,:]
    ref_l2_errors = reference_error[1,:]
    ref_Linf_errors = reference_error[2,:]
    # Find the index where basis_size equals 30
    basis_30_idx = jnp.where(basis_size == 30)[0][0]
    ref_l2_error = ref_l2_errors[basis_30_idx]
    ref_Linf_error = ref_Linf_errors[basis_30_idx]

    mean_l2_errors = np.mean(l2_errors, axis=1)
    mean_Linf_errors = np.mean(Linf_errors, axis=1)
    std_l2_errors = np.std(l2_errors, axis=1)
    std_Linf_errors = np.std(Linf_errors, axis=1)

    f, ax = plt.subplots(1,1,figsize=(4,4))
    ax.errorbar(n_samples, mean_l2_errors, yerr=std_l2_errors, label='$L_2$ Error', capsize=3,color=plot_config["colors"][0])
    ax.errorbar(n_samples, mean_Linf_errors, yerr=std_Linf_errors, label='$L_\\infty$ Error', capsize=3,color=plot_config["colors"][1])
    ax.set_xlabel('Number of Samples')
    ax.legend(numpoints=1, markerscale=1.5, handlelength=0, handletextpad=0.5)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.axhline(y=ref_l2_error, color='black', linestyle='--')
    ax.axhline(y=ref_Linf_error, color='grey', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/laplace_inverse_error.png")
    plt.close()



def plot_wave_figure(plot_config):
    data_path = pathlib.Path("data/fig_wave")
    t_vec = jnp.linspace(-1,1,201)
    plot_indicies = [0,50,100,150,200]
    n_grid=100
    full_data = pr.load_dict_from_hdf5(data_path / "full_data.h5")
    config = full_data["config"]
    u_coeffs = full_data["fields"]["u"]
    basis = pr.basis.BasisND([pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis, pr.basis.vectorized_legendre_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"]))
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
    x_grid, y_grid, mask = define_peanut_geometry(n_grid, circle=False)

    reference_data = np.loadtxt(data_path / "wave_grid.csv", delimiter=",", skiprows=9)
    ref_x, ref_y = reference_data[:,0], reference_data[:,1]
    ref_x, ref_y = ref_x.reshape(n_grid,n_grid), ref_y.reshape(n_grid,n_grid)
    ref_data = reference_data[:,2:]
    ref_data = ref_data[:,plot_indicies]
    eval_data = []
    for i in range(len(plot_indicies)):
        eval_data.append(u_field.evaluate(x_grid, y_grid, t_vec[plot_indicies[i]]*jnp.ones((n_grid,n_grid))))
        eval_data[i] = eval_data[i].reshape(n_grid,n_grid)
        eval_data[i] = jnp.where(mask == 0, jnp.nan, eval_data[i])
    eval_data = jnp.array(eval_data)

    cmap = plot_config["cmap"]
    labelx, labely = 0.02, 0.1
    label_pad = 0.25
    f, ax = plt.subplots(2,len(plot_indicies),figsize=(10,4),sharex=True,sharey=True)
    labels = [("a","f"),("b","g"),("c","h"),("d","i"),("e","j")]
    vmin, vmax = jnp.nanmin(ref_data[:,0]), jnp.nanmax(ref_data[:,0])
    for i in range(len(plot_indicies)):
        f,ax = plt.subplots(1,1,figsize=(4,4))
        im0 = ax.contourf(x_grid, y_grid, eval_data[i,:,:], levels=100,cmap = cmap,vmin=vmin,vmax=vmax)
        ax.text(labelx, labely, f"$({labels[i][0]})$", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y", rotation=0)
        ax.set_title(f"t = {t_vec[plot_indicies[i]]+1:.2f}")
        plt.tight_layout()
        plt.savefig(f"{plot_config['save_dir']}/wave_plot_{labels[i][0]}.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
        plt.close()
        if i == 0:
            f_cb, ax_cb = plt.subplots(figsize=(1.25, 4))
            cb = f_cb.colorbar(im0, cax=ax_cb)
            cb.set_label("Solution Value (u)")
            f_cb.tight_layout()
            f_cb.savefig(f"{plot_config['save_dir']}/wave_plot_a_colorbar.png")
            plt.close(f_cb)

        f,ax = plt.subplots(1,1,figsize=(4,4))
        im1 =ax.contourf(x_grid, y_grid, ref_data[:,i].reshape(n_grid,n_grid), levels=100,cmap = cmap,vmin=vmin,vmax=vmax)
        ax.text(labelx, labely, f"$({labels[i][1]})$", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
        ax.set_xlabel("x")
        ax.set_ylabel("y", rotation=0)
        ax.set_title(f"t = {t_vec[plot_indicies[i]]+1:.2f}")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(f"{plot_config['save_dir']}/wave_plot_{labels[i][1]}.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
        plt.close()
        if i == 0:
            f_cb, ax_cb = plt.subplots(figsize=(1.25, 4))
            cb = f_cb.colorbar(im1, cax=ax_cb)
            cb.set_label("Solution Value (u)")
            f_cb.tight_layout()
            f_cb.savefig(f"{plot_config['save_dir']}/wave_plot_f_colorbar.png")
            plt.close(f_cb)

    if config["solver_type"] == "FirstOrder":
        solver = f"{config['solver_type']}_{config['optimizer_name']}_{config['loss_strategy']}"
        steps = config["n_epochs"]
    else:
        solver = config["solver_type"]
        steps = config["max_steps"]
    plot_data = {
        "basis_Nx": config["basis_Nx"],
        "basis_Ny": config["basis_Ny"],
        "basis_Nt": config["basis_Nt"],
        "learning_rate": config["learning_rate"],
        "c": config["c"],
        "n_pde": config["n_pde"],
        "n_bc": config["n_bc"],
        "solver": solver,
        "steps": steps,
        "basis": "legendre",
        "grid": "n_grid"
    }
    with open("figure_info/wave_plot_info.yml", "w") as f:
        yaml.dump(plot_data, f)

def plot_schematic(plot_config):
    data_path = pathlib.Path("data/fig_wave")
    full_data = pr.load_dict_from_hdf5(data_path / "full_data.h5")
    config = full_data["config"]
    coeffs = full_data["fields"]["u"]
    N = config["basis_Nx"]+1
    coeffs = coeffs.reshape((N,N,N))
    size = (N,N)
    coeffs_slice = coeffs[::2,::2,4]

    random_coeffs = jax.random.normal(jax.random.PRNGKey(0), coeffs_slice.shape)*.5
    cmap = "viridis"
    f, ax = plt.subplots(1,1,figsize=(3,3))
    im0 = ax.imshow(jnp.log10(jnp.abs(coeffs_slice)),cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/schematic_a.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()
    f, ax = plt.subplots(1,1,figsize=(3,3))
    im0 = ax.imshow(jnp.log10(jnp.abs(random_coeffs)),cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/schematic_b.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()

    _,_,mask = define_peanut_geometry(100, circle=True)
    f, ax = plt.subplots(1,1,figsize=(3,3))
    im0 = ax.imshow(~mask, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/schematic_c.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()

def plot_stiff_figure(plot_config):
    data_path = pathlib.Path("data/fig_stiff")
    full_data_ac = pr.load_dict_from_hdf5(data_path / "full_data_ac.h5")
    full_data_nls = pr.load_dict_from_hdf5(data_path / "full_data_nls.h5")
    config_ac = full_data_ac["config"]
    config_nls = full_data_nls["config"]
    u_coeffs_ac = full_data_ac["fields"]["u"]
    u_coeffs_nls = full_data_nls["fields"]["u"]
    v_coeffs_nls = full_data_nls["fields"]["v"]
    ac_basis = pr.basis.CosineLegendreBasis2D((config_ac["basis_Nx"], config_ac["basis_Nt"]))
    nls_basis = pr.basis.CosineChebyshevBasis2D((config_nls["basis_Nx"], config_nls["basis_Nt"]))
    u_field_ac = pr.BasisField(ac_basis, pr.Coeffs(u_coeffs_ac))
    u_field_nls = pr.BasisField(nls_basis, pr.Coeffs(u_coeffs_nls))
    v_field_nls = pr.BasisField(nls_basis, pr.Coeffs(v_coeffs_nls))

    x_vec_ac = jnp.linspace(-1,1,config_ac["nx"])
    t_vec_ac = jnp.linspace(-1,1,config_ac["nt"])
    x_vec_nls = jnp.linspace(-1,1,config_nls["nx"])
    t_vec_nls = jnp.linspace(-1,1,config_nls["nt"])
    x_grid_ac, t_grid_ac = jnp.meshgrid(x_vec_ac, t_vec_ac)
    x_grid_nls, t_grid_nls = jnp.meshgrid(x_vec_nls, t_vec_nls)
    u_eval_ac = u_field_ac.evaluate(x_grid_ac, t_grid_ac)
    u_eval_nls = u_field_nls.evaluate(x_grid_nls, t_grid_nls)
    v_eval_nls = v_field_nls.evaluate(x_grid_nls, t_grid_nls)
    u_eval_ac = u_eval_ac.reshape(config_ac["nt"], config_ac["nx"])
    u_eval_nls = u_eval_nls.reshape(config_nls["nt"], config_nls["nx"])
    v_eval_nls = v_eval_nls.reshape(config_nls["nt"], config_nls["nx"])
    h_eval_nls = jnp.sqrt(u_eval_nls**2 + v_eval_nls**2)
    
    nls_data = loadmat(data_path / "NLS.mat")
    Exact = nls_data['uu']
    Exact_u = jnp.real(Exact).T
    Exact_v = jnp.imag(Exact).T
    Exact_h = jnp.sqrt(Exact_u**2 + Exact_v**2)

    ac_data = loadmat(data_path / "allen_cahn.mat")
    ac_exact = ac_data['usol']

    error_ac = jnp.load(data_path / "errors_ac.npy")
    error_nls = jnp.load(data_path / "errors_nls.npy")

    cmap = plot_config["cmap"]
    colors = plot_config["colors"]
    t_grid_ac = (t_grid_ac + 1)*config_ac["t_scale"]
    t_grid_nls = (t_grid_nls + 1)*config_nls["t_scale"]
    x_grid_nls = (x_grid_nls)*config_nls["x_scale"]

    labelx, labely = 0.02, 0.1
    label_pad = 2.0
    f, ax = plt.subplots(1,1,figsize=(4,4))
    im0 = ax.contourf(t_grid_ac.T, x_grid_ac.T, u_eval_ac.T, levels=100,cmap = cmap)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("t",rotation=0)
    ax.set_ylabel("x",rotation=0)
    ax.set_title("Allen-Cahn Optimized Solution")
    ax.text(labelx, labely, "$(a)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/stiff_plot_a.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()

    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 4)) # A new figure with a tall, narrow aspect ratio
    cb = fig_cb.colorbar(im0, cax=ax_cb)
    cb.set_label("Solution Value (u)")
    fig_cb.tight_layout()
    fig_cb.savefig(f"{plot_config['save_dir']}/stiff_plot_a_colorbar.png")
    plt.close(fig_cb)

    f,ax = plt.subplots(1,1,figsize=(4,4))
    im1 = ax.contourf(t_grid_ac.T, x_grid_ac.T, ac_exact.T, levels=100,cmap = cmap)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("t",rotation=0)
    ax.set_ylabel("x",rotation=0)
    ax.set_title("Allen-Cahn Reference Solution")
    ax.text(labelx, labely, "$(b)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/stiff_plot_b.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()


    f,ax = plt.subplots(1,1,figsize=(4,4))
    ax.semilogy(error_ac[0,:],error_ac[1,:],marker="o",label="$L_2$ Error",color=colors[0])
    ax.semilogy(error_ac[0,:],error_ac[2,:],marker="o",label="$L_\\infty$ Error",color=colors[1])
    ax.set_xlabel("$N$")
    ax.legend()
    ax.set_title("Allen-Cahn Error")
    ax.text(labelx, labely, "$(c)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/stiff_plot_c.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()

    h_max = jnp.max(jnp.abs(Exact_h))

    f,ax = plt.subplots(1,1,figsize=(4,4))
    im2 = ax.contourf(t_grid_nls.T, x_grid_nls.T, h_eval_nls.T, levels=100,cmap = cmap,vmin=0,vmax=h_max)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("t",rotation=0)
    ax.set_ylabel("x",rotation=0)
    ax.set_title("Nonlinear Schrödinger Optimized Solution")
    ax.set_xticks([0, jnp.pi/8, jnp.pi/4, 3*jnp.pi/8, jnp.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.text(labelx, labely, "$(d)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/stiff_plot_d.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()
    fig_cb, ax_cb = plt.subplots(figsize=(1.15, 4))
    cb = fig_cb.colorbar(im2, cax=ax_cb)
    cb.set_label("Solution Value (h)")
    fig_cb.tight_layout()
    fig_cb.savefig(f"{plot_config['save_dir']}/stiff_plot_d_colorbar.png")
    plt.close(fig_cb)

    f,ax = plt.subplots(1,1,figsize=(4,4))
    im3 = ax.contourf(t_grid_nls.T, x_grid_nls.T, Exact_h.T, levels=100,cmap = cmap,vmin=0,vmax=h_max)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("t",rotation=0)
    ax.set_ylabel("x",rotation=0)
    ax.set_xticks([0, jnp.pi/8, jnp.pi/4, 3*jnp.pi/8, jnp.pi/2])
    ax.set_xticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.set_title("Nonlinear Schrödinger Reference Solution")
    ax.text(labelx, labely, "$(e)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/stiff_plot_e.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()

    f,ax = plt.subplots(1,1,figsize=(4,4))
    ax.semilogy(error_nls[:,0],error_nls[:,1],marker="o",label="$L_2$ Error",color=colors[0])
    ax.semilogy(error_nls[:,0],error_nls[:,4],marker="o",label="$L_\\infty$ Error",color=colors[1])
    # ax.semilogy(error_nls[:,0],error_nls[:,2],marker="o",label="$L_2$ Error $u$",color=colors[0])
    # ax.semilogy(error_nls[:,0],error_nls[:,3],marker="s",label="$L_2$ Error $v$",color=colors[0])
    # ax.semilogy(error_nls[:,0],error_nls[:,5],marker="o",label="$L_\\infty$ Error $u$",color=colors[1])
    # ax.semilogy(error_nls[:,0],error_nls[:,6],marker="s",label="$L_\\infty$ Error $v$",color=colors[1])
    ax.set_xlabel("$N$")
    ax.legend()
    ax.set_title("Nonlinear Schrödinger Error")
    ax.text(labelx, labely, "$(f)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/stiff_plot_f.{plot_config["file_type"]}", dpi=300,format=plot_config["file_type"])
    plt.close()

    if config_ac["solver_type"] == "FirstOrder":
        solver_ac = f"{config_ac['solver_type']}_{config_ac['optimizer_name']}_{config_ac['loss_strategy']}"
        steps_ac = config_ac["n_epochs"]
    else:
        solver_ac = config_ac["solver_type"]
        steps_ac = config_ac["max_steps"]
    if config_nls["solver_type"] == "FirstOrder":
        solver_nls = f"{config_nls['solver_type']}_{config_nls['optimizer_name']}_{config_nls['loss_strategy']}"
        steps_nls = config_nls["n_epochs"]
    else:
        solver_nls = config_nls["solver_type"]
        steps_nls = config_nls["max_steps"]
    plot_data = {
        "basis_Nx_ac": config_ac["basis_Nx"],
        "basis_Nt_ac": config_ac["basis_Nt"],
        "basis_Nx_nls": config_nls["basis_Nx"],
        "basis_Nt_nls": config_nls["basis_Nt"],
        "solver_ac": solver_ac,
        "solver_nls": solver_nls,
        "steps_ac": steps_ac,
        "steps_nls": steps_nls,
        "time_taken_ac": float(config_ac["time_taken"]),
        "time_taken_nls": float(config_nls.get("time_taken", -1)),
        "x_scale_nls": config_nls["x_scale"],
        "t_scale_nls": config_nls["t_scale"],
        "x_scale_ac": config_ac.get("x_scale", 1),
        "t_scale_ac": config_ac["t_scale"],
        "n_pde_ac": config_ac["n_pde"],
        "n_pde_nls": config_nls["n_pde"],
        "n_ic_ac": config_ac["n_ic"],
        "n_ic_nls": config_nls["n_ic"],
        "nx_ac": config_ac["nx"],
        "nx_nls": config_nls["nx"],
        "nt_ac": config_ac["nt"],
        "nt_nls": config_nls["nt"],
        "basis_ac": "cosine_legendre",
        "basis_nls": "cosine_chebyshev",
    }
    with open("figure_info/stiff_plot_info.yml", "w") as f:
        yaml.dump(plot_data, f)

def create_smooth_spot(points, center, radius=0.95, softness=50.0):
    dot_product = jnp.dot(points, center)
    arg = softness * (dot_product - radius)
    return 0.5 * (jnp.tanh(arg) + 1.0)

def initial_condition_sphere(x, y, z):
    points = jnp.stack([x, y, z], axis=-1)
    left_eye_center = jnp.array([-0.4, 0.4, 0.82])
    left_eye_center /= jnp.linalg.norm(left_eye_center)
    right_eye_center = jnp.array([0.4, 0.4, 0.82])
    right_eye_center /= jnp.linalg.norm(right_eye_center)
    mouth_center = jnp.array([0.0, -0.4, 0.916])
    mouth_center /= jnp.linalg.norm(mouth_center)
    left_eye = create_smooth_spot(points, left_eye_center, softness=10.0)
    right_eye = create_smooth_spot(points, right_eye_center, softness=10.0)
    mouth = create_smooth_spot(points, mouth_center, softness=10.0)
    return left_eye + right_eye + mouth

def get_spherical_harmonic_coeffs(initial_cond_func, n_max, n_points=100):
    cos_theta, w_theta = roots_legendre(n_points)
    theta = np.arccos(cos_theta)
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    w_phi = 2 * np.pi / n_points
    PHI, THETA = np.meshgrid(phi, theta)

    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    print("Evaluating initial condition on the grid...")
    C0_grid = initial_cond_func(X.flatten(), Y.flatten(), Z.flatten())
    C0_grid = C0_grid.reshape(n_points, n_points)

    coeffs = {}
    print(f"Computing coefficients up to n_max = {n_max}...")
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            Ynm_conj = np.conj(sph_harm(m, n, PHI, THETA))
            integrand = C0_grid * Ynm_conj
            integral = np.sum(integrand * w_theta[:, np.newaxis] * w_phi)
            coeffs[(n, m)] = integral
    return coeffs

def reconstruct_solution(coeffs, theta, phi, t, D, R):
    """
    Reconstructs the concentration field from spherical harmonic coefficients
    at a given time.
    """
    # Ensure inputs are numpy arrays for broadcasting
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    
    C_reconstructed = np.zeros_like(theta, dtype=complex)
    
    for (n, m), Anm_val in coeffs.items():
        if np.abs(Anm_val) > 1e-9: # Optimization for sparse coefficients
            time_decay = np.exp(-n * (n + 1) * D * t / R**2)
            Ynm = sph_harm(m, n, phi, theta)
            C_reconstructed += Anm_val * Ynm * time_decay
            
    return np.real(C_reconstructed)

def plot_sphere_diffusion_figure(plot_config):
    data_path = pathlib.Path("data/fig_sphere_diffusion")
    full_data = pr.load_dict_from_hdf5(data_path / "full_data.h5")
    config = full_data["config"]
    c_coeffs = full_data["fields"]["c"]
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis,pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nz"], config["basis_Nt"]))
    c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))
    N_MAX = 20          # Max harmonic degree (higher is more accurate)
    D = config["alpha"]             # Diffusion coefficient
    R = 1.0             # Sphere radius
    
    Anm_coeffs = get_spherical_harmonic_coeffs(initial_condition_sphere, N_MAX)

    meridional_angle = np.linspace(0, np.pi, 400)
    longitude_angle = np.zeros_like(meridional_angle)
    x_line = R * np.sin(meridional_angle) * np.cos(longitude_angle)
    y_line = R * np.sin(meridional_angle) * np.sin(longitude_angle)
    z_line = R * np.cos(meridional_angle)
    times_to_plot = np.linspace(-1,1,5)
    u_eval_list = []
    u_analytic_list = []
    rescaled_u_eval_list = []
    rescaled_u_analytic_list = []
    scaled_x = []
    for t in times_to_plot:
        T_line = t * jnp.ones_like(x_line)
        c_values_line = np.asarray(c_field.evaluate(x_line, y_line, z_line, T_line))
        u_eval_list.append(c_values_line)
        u_analytic_line = reconstruct_solution(Anm_coeffs, meridional_angle, longitude_angle, t+1.0, D, R)
        u_analytic_list.append(u_analytic_line)
        elapsed_time = t + 1.0
        if elapsed_time > 1e-6:
            eta_scaled_x = meridional_angle / np.sqrt(D * elapsed_time)
            scaled_x.append(eta_scaled_x)
            c_scaled_y = c_values_line * np.sqrt(elapsed_time)
            rescaled_u_eval_list.append(c_scaled_y)
            rescaled_u_analytic_line = u_analytic_line * np.sqrt(elapsed_time)
            rescaled_u_analytic_list.append(rescaled_u_analytic_line)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.9, len(times_to_plot)))
    f, ax = plt.subplots(1,1,figsize=plot_config["figsize"])
    for i in range(len(times_to_plot)):
        ax.scatter(meridional_angle[::10], u_eval_list[i][::10],zorder=10,marker='o',facecolors='none',edgecolors=colors[i])
        ax.plot(meridional_angle, u_analytic_list[i],color=colors[i],zorder=1,label=f't = {times_to_plot[i]+1:.2f}')

    ax.legend()
    ax.set_xlabel("Meridional Angle")
    ax.set_ylabel("Concentration")
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/sphere_diffusion_plot_c.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()
    f, ax = plt.subplots(1,1,figsize=plot_config["figsize"])
    for i in range(1,len(times_to_plot)):
        # ax.scatter(scaled_x[i][::20], rescaled_u_eval_list[i][::20], label=f't = {times_to_plot[i]+1:.2f}',zorder=10,marker='o',facecolors='none',edgecolors=colors[i])
        ax.plot(scaled_x[i-1], rescaled_u_eval_list[i-1],color=colors[i],zorder=1,label=f't = {times_to_plot[i]+1:.2f}')
    ax.legend()
    ax.set_xlabel("Similarity Variable $\\eta = \\phi / \\sqrt{Dt}$")
    ax.set_ylabel("Concentration")
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/sphere_diffusion_plot_d.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    time_points = [-1,-.5]

    n_points = 150
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)
    theta, phi = np.meshgrid(theta, phi)
    X = R * np.sin(phi) * np.cos(theta)
    Y = R * np.sin(phi) * np.sin(theta)
    Z = R * np.cos(phi)

    c_values_grid_list = []

    for t in time_points:
        T_grid = t * jnp.ones_like(X)
        c_values_grid = np.asarray(c_field.evaluate(X, Y, Z, T_grid)).reshape(X.shape)
        c_values_grid_list.append(c_values_grid)
    cmap = plt.get_cmap(plot_config["cmap"])
    f, ax = plt.subplots(1,1,figsize=plot_config["figsize"])
    vmax = jnp.max(jnp.abs(jnp.array(c_values_grid_list)))
    vmin = 0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    elev = 30.
    azim = 30.
    f = plt.figure(figsize=plot_config["figsize"])
    ax = f.add_subplot(111, projection='3d')
    colors = cmap(norm(c_values_grid_list[0]))
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=False, shade=False,cmap='jet')
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"t = {time_points[0]+1:.2f}")
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(f"{plot_config['save_dir']}/sphere_diffusion_plot_a.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    f = plt.figure(figsize=plot_config["figsize"])
    ax = f.add_subplot(111, projection='3d')
    colors = cmap(norm(c_values_grid_list[1]))
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=False, shade=False,cmap='jet')
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"t = {time_points[1]+1:.2f}")
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(f"{plot_config['save_dir']}/sphere_diffusion_plot_b.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()
    
    standalone_colorbar(plot_config,(vmin,vmax),"Concentration","sphere_diffusion_plot_a")

    if config["solver_type"] == "FirstOrder":
        solver = f"{config['solver_type']}_{config['optimizer_name']}_{config['loss_strategy']}"
        steps = config["n_epochs"]
    else:
        solver = config["solver_type"]
        steps = config["max_steps"]
    plot_data = {
        "basis_Nx": config["basis_Nx"],
        "basis_Ny": config["basis_Ny"],
        "basis_Nz": config["basis_Nz"],
        "basis_Nt": config["basis_Nt"],
        "learning_rate": config["learning_rate"],
        "solver": solver,
        "steps": steps,
        "n_pde": config["n_pde"],
        "n_initial": config["n_initial"],
        "n_t": config["n_t"],
        "alpha": config["alpha"],
        "basis_fn": "Chebyshev",
    }
    with open("figure_info/sphere_diffusion_plot_info.yml", "w") as f:
        yaml.dump(plot_data, f)
    
def plot_heat_forcing_figure(plot_config):
    data_path = pathlib.Path("data/fig_heat_forcing")
    full_data = pr.load_dict_from_hdf5(data_path / "full_data.h5")
    config = full_data["config"]
    c_coeffs = full_data["fields"]["c"]
    f_coeffs = full_data["fields"]["f"]
    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"]))
    c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))
    f_basis = pr.basis.FourierChebyshevBasis2D((config["basis_bc"], config["basis_Nt"]))
    f_field = pr.BasisField(f_basis, pr.Coeffs(f_coeffs))
    theta = jnp.linspace(-1,1,300)
    time = jnp.linspace(-1,1,300)
    theta_grid, time_grid = jnp.meshgrid(theta, time)
    f_grid = f_field.evaluate(theta_grid, time_grid)
    f_grid = f_grid.reshape(300,300)
    theta_grid = (theta_grid)*jnp.pi
    time_grid = (time_grid+1)/2

    x_vec = jnp.linspace(-1,1,300)
    y_vec = jnp.linspace(-1,1,300)
    x_grid_2d, y_grid_2d = jnp.meshgrid(x_vec, y_vec)
    r_grid = jnp.sqrt(x_grid_2d**2 + y_grid_2d**2)
    mask = r_grid <= 1
    c_initial = c_field.evaluate(x_grid_2d, y_grid_2d, jnp.ones((300,300))*-1)
    c_initial = c_initial.reshape(300,300)
    t_mid = 0.8
    t_mid_scaled = (t_mid*2-1)
    c_mid = c_field.evaluate(x_grid_2d, y_grid_2d, jnp.ones((300,300))*t_mid_scaled)
    c_mid = c_mid.reshape(300,300)
    c_final = c_field.evaluate(x_grid_2d, y_grid_2d, jnp.ones((300,300))*1)
    c_final = c_final.reshape(300,300)
    c_initial = jnp.where(mask == 0, jnp.nan, c_initial)
    c_mid = jnp.where(mask == 0, jnp.nan, c_mid)
    c_final = jnp.where(mask == 0, jnp.nan, c_final)

    max_val = jnp.max(jnp.abs(f_grid))
    vmin, vmax = -max_val, max_val
    cmap = plot_config["cmap"]
    cmap = "bwr"
    labelx, labely = 0.02, 0.1
    label_pad = 0.25
    f, ax = plt.subplots(1,1,figsize=(4,4))
    im0 = ax.contourf(x_grid_2d, y_grid_2d, c_initial, levels=100,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_facecolor('black')
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y",rotation=0)
    ax.set_title("Initial t = 0")
    ax.text(labelx, labely, "$(a)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/heat_forcing_plot_a.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    f, ax = plt.subplots(1,1,figsize=(4,4))
    im1 = ax.contourf(x_grid_2d, y_grid_2d, c_mid, levels=100,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_facecolor('black')
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y",rotation=0)
    ax.set_title(f"t = {t_mid:.2f}")
    ax.text(labelx, labely, "$(b)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/heat_forcing_plot_b.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    f, ax = plt.subplots(1,1,figsize=(4,4))
    im2 = ax.contourf(x_grid_2d, y_grid_2d, c_final, levels=100,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_facecolor('black')
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y",rotation=0)
    ax.set_title("Final t = 1")
    ax.text(labelx, labely, "$(c)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/heat_forcing_plot_c.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    f, ax = plt.subplots(1,1,figsize=(4,4))
    im3 = ax.contourf(time_grid.T, theta_grid.T, f_grid.T, levels=100,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("$\\theta$",rotation=0)
    ax.set_title("Forcing Function")
    ax.text(labelx, labely, "$(d)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/heat_forcing_plot_d.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()

    f_cb, ax_cb = plt.subplots(figsize=(1.25, 4))
    cb = f_cb.colorbar(im3, cax=ax_cb)
    cb.set_label("Solution Value (c)")
    f_cb.tight_layout()
    f_cb.savefig(f"{plot_config['save_dir']}/heat_forcing_plot_d_colorbar.png",dpi=300)
    plt.close(f_cb)
    f_cb, ax_cb = plt.subplots(figsize=(1.25, 4))
    
    if config["solver_type"] == "FirstOrder":
        solver = f"{config['solver_type']}_{config['optimizer_name']}_{config['loss_strategy']}"
        steps = config["n_epochs"]
    else:
        solver = config["solver_type"]
        steps = config["max_steps"]
    plot_data = {
        "basis_Nx": config["basis_Nx"],
        "basis_Ny": config["basis_Ny"],
        "basis_Nt": config["basis_Nt"],
        "basis_bc": config["basis_bc"],
        "solver": solver,
        "steps": steps,
        "time_taken": float(config["time_taken"]),
        "n_pde": config["n_pde"],
        "n_bc": config["n_bc"],
        "n_target": config["n_target"],
        "n_t": config["n_t"],
    }
    with open("figure_info/heat_forcing_plot_info.yml", "w") as f:
        yaml.dump(plot_data, f)

def transport_targets(slice_name):
    image = plt.imread(f"data/fig_transport/{slice_name}")
    image = image[...,-1]
    image = ~image.astype(bool)
    image = jnp.flipud(image)
    # Apply Gaussian blur to smooth out gradients
    image = gaussian_filter(image.astype(float), sigma=15.0)
    x_grid, y_grid = jnp.meshgrid(jnp.linspace(-1,1,image.shape[1]), jnp.linspace(-1,1,image.shape[0]))
    r_grid = jnp.sqrt(x_grid**2 + y_grid**2)
    mask = r_grid < 1
    image = jnp.where(mask == 0, jnp.nan, image.astype(float))
    return x_grid, y_grid, image

def plot_transport_figure(plot_config):
    data_path = pathlib.Path("data/fig_transport")
    full_data = pr.load_dict_from_hdf5(data_path / "full_data.h5")
    config = full_data["config"]
    c_coeffs = full_data["fields"]["c"]
    u_coeffs = full_data["fields"]["u"]
    v_coeffs = full_data["fields"]["v"]
    target_list = ["H.png", "A.png", "R.png", "V.png", "A.png", "R.png", "D.png"]
    nx, ny = 200, 200
    slices = len(target_list)

    basis = pr.basis.BasisND([pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis, pr.basis.vectorized_chebyshev_basis], (config["basis_Nx"], config["basis_Ny"], config["basis_Nt"]))
    c_field = pr.BasisField(basis, pr.Coeffs(c_coeffs))
    u_field = pr.BasisField(basis, pr.Coeffs(u_coeffs))
    v_field = pr.BasisField(basis, pr.Coeffs(v_coeffs))
    t_sample = jnp.linspace(-1,1,slices)
    x_grid, y_grid = jnp.meshgrid(jnp.linspace(-1,1,nx), jnp.linspace(-1,1,ny))
    r_grid = jnp.sqrt(x_grid**2 + y_grid**2)
    mask = r_grid <= 1
    cmap = plot_config["cmap"]
    labelx, labely = 0.02, 0.1
    label_pad = 0.3
    letters = [(chr(ord('a')+i), chr(ord('a')+i+slices), chr(ord('a')+i+2*slices)) for i in range(slices)]

    c_eval = [c_field.evaluate(x_grid, y_grid, jnp.ones((ny,nx))*t_sample[i]).reshape(ny,nx) for i in range(slices)]
    u_eval = [u_field.evaluate(x_grid, y_grid, jnp.ones((ny,nx))*t_sample[i]).reshape(ny,nx) for i in range(slices)]
    v_eval = [v_field.evaluate(x_grid, y_grid, jnp.ones((ny,nx))*t_sample[i]).reshape(ny,nx) for i in range(slices)]
    c_eval = [jnp.where(mask == 0, jnp.nan, c_eval[i]) for i in range(slices)]
    u_eval = [jnp.where(mask == 0, jnp.nan, u_eval[i]) for i in range(slices)]
    v_eval = [jnp.where(mask == 0, jnp.nan, v_eval[i]) for i in range(slices)]
    c_vmin, c_vmax = 0, jnp.nanmax(jnp.array(c_eval))
    u_level = jnp.max(jnp.abs(jnp.array(u_eval)))
    v_level = jnp.max(jnp.abs(jnp.array(v_eval)))
    vel_mag = jnp.sqrt(jnp.array(u_eval)**2 + jnp.array(v_eval)**2)
    vel_mag_max = jnp.nanmax(vel_mag)

    size = (3,3)
    
    for i in range(slices):
        print(f"Processing slice {i}")
        f, ax = plt.subplots(1,1,figsize=size)
        im0 = ax.contourf(x_grid, y_grid, c_eval[i], levels=100,cmap=cmap,vmin=c_vmin,vmax=c_vmax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Time = {(t_sample[i]+1)/2:.2f}")
        ax.text(labelx, labely, f"$({letters[i][0]})$", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
        plt.tight_layout()
        plt.savefig(f"{plot_config['save_dir']}/transport_plot_{letters[i][0]}.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
        plt.close()

        f, ax = plt.subplots(1,1,figsize=size)
        vel_mag = jnp.sqrt(jnp.array(u_eval[i])**2 + jnp.array(v_eval[i])**2)
        # vel_mag = jnp.where(mask == 0, jnp.nan, vel_mag)
        im1 = ax.contourf(x_grid, y_grid, vel_mag, levels=100,cmap=cmap,vmin=0,vmax=vel_mag_max)
        im0 = ax.streamplot(np.asarray(x_grid), np.asarray(y_grid), np.asarray(u_eval[i]), np.asarray(v_eval[i]), color='black', density=1.5, linewidth=0.8,zorder=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Time = {(t_sample[i]):.2f}")
        ax.text(labelx, labely, f"$({letters[i][1]})$", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
        plt.tight_layout()
        plt.savefig(f"{plot_config['save_dir']}/transport_plot_{letters[i][1]}.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
        plt.close()
        
        target_x, target_y, target_data = transport_targets(target_list[i])
        f, ax = plt.subplots(1,1,figsize=size)
        im0 = ax.contourf(target_x, target_y, target_data, levels=100,cmap=cmap,vmin=c_vmin,vmax=c_vmax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Time = {(t_sample[i]+1)/2:.2f}")
        ax.text(labelx, labely, f"$({letters[i][2]})$", transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
        plt.tight_layout()
        plt.savefig(f"{plot_config['save_dir']}/transport_plot_{letters[i][2]}.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
        plt.close()

    standalone_colorbar(plot_config,(c_vmin, c_vmax),'Solution Value (c)','transport_plot_a')
    standalone_colorbar(plot_config,(0, vel_mag_max),'Velocity Magnitude','transport_plot_h')
    standalone_colorbar(plot_config,(c_vmin, c_vmax),'Target Value (c)','transport_plot_o')


    if config["solver_type"] == "FirstOrder":
        solver = f"{config['solver_type']}_{config['optimizer_name']}_{config['loss_strategy']}"
        steps = config["n_epochs"]
    else:
        solver = config["solver_type"]
        steps = config["max_steps"]
    plot_data = {
        "basis_Nx": config["basis_Nx"],
        "basis_Ny": config["basis_Ny"],
        "basis_Nt": config["basis_Nt"],
        "time_taken": float(config.get("time_taken",0.0)),
        "n_pde": config["n_pde"],
        "n_bc": config["n_bc"],
        "n_target": config["n_target"],
        "n_t": config["n_t"],
        "solver": solver,
        "steps": steps,
        "alpha": config["alpha"],
        "basis_fn": "Chebyshev",
    }
    with open("figure_info/transport_plot_info.yml", "w") as f:
        yaml.dump(plot_data, f)
        
def plot_amery(plot_config):
    data_path = pathlib.Path("data/fig_amery")

    full_data_25 = pr.load_dict_from_hdf5(data_path / "amery_25.h5")
    full_data_50 = pr.load_dict_from_hdf5(data_path / "amery_50.h5")
    config = full_data_25["config"]
    config_50 = full_data_50["config"]
    reference_data = pr.load_dict_from_hdf5(data_path / "Amery_data.h5")
    mu_coeffs_25 = full_data_25["fields"]["mu"]
    mu_coeffs_50 = full_data_50["fields"]["mu"]
    basis_25 = pr.basis.ChebyshevBasis2D((25, 25))
    basis_50 = pr.basis.ChebyshevBasis2D((50, 50))
    mu_field_25 = pr.LogBasisField(basis_25, pr.Coeffs(mu_coeffs_25))
    mu_field_50 = pr.LogBasisField(basis_50, pr.Coeffs(mu_coeffs_50))
    mask = reference_data["fields"]["u"]["mask"]
    ny, nx = reference_data["fields"]["u"]["x"].shape
    m0 = full_data_25["config"]["scaling_dict"]["mu"]
    x_vec, y_vec = jnp.linspace(-1,1,nx), jnp.linspace(-1,1,ny)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    mu_eval_25 = mu_field_25.evaluate(x_grid, y_grid)
    mu_eval_50 = mu_field_50.evaluate(x_grid, y_grid)
    mu_eval_25 = mu_eval_25.reshape(ny, nx)
    mu_eval_50 = mu_eval_50.reshape(ny, nx)
    mu_eval_25 = m0*mu_eval_25[::-1, :]
    mu_eval_50 = m0*mu_eval_50[::-1, :]
    mu_eval_25 = jnp.where(mask, mu_eval_25, jnp.nan)
    mu_eval_50 = jnp.where(mask, mu_eval_50, jnp.nan)
    cmap = plot_config["cmap"]
    labelx, labely = 0.02, 0.1
    label_pad = 0.3
    figsize = (4,2)
    f, ax = plt.subplots(1,1,figsize=figsize)
    im0 = ax.imshow(jnp.log10(mu_eval_25), cmap=cmap,vmin=13,vmax=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.text(labelx, labely, "$(a)$", transform=ax.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/amery_plot_a.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()
    f, ax = plt.subplots(1,1,figsize=figsize)
    im0 = ax.imshow(jnp.log10(mu_eval_50), cmap=cmap,vmin=13,vmax=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.text(labelx, labely, "$(b)$", transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{plot_config['save_dir']}/amery_plot_b.{plot_config["file_type"]}",dpi=300,format=plot_config["file_type"])
    plt.close()
    standalone_log_colorbar(plot_config,(1e13,1e16),"Viscosity (Pa $\\cdot$ s)","amery_plot_a")

    if config["solver_type"] == "FirstOrder":
        solver = f"{config['solver_type']}_{config['optimizer_name']}_{config['loss_strategy']}"
        steps = config["n_epochs"]
    else:
        solver = config["solver_type"]
        steps = config["max_steps"]

    if config_50["solver_type"] == "FirstOrder":
        solver_50 = f"{config_50['solver_type']}_{config_50['optimizer_name']}_{config_50['loss_strategy']}"
        steps_50 = config_50["n_epochs"]
    else:
        solver_50 = config_50["solver_type"]
        steps_50 = config_50["max_steps"]

    plot_data = {
        "basis_Nx": config["basis_Nx"],
        "basis_Ny": config["basis_Ny"],
        "basis_Nx_50": config_50["basis_Nx"],
        "basis_Ny_50": config_50["basis_Ny"],
        "basis_fn": "Chebyshev",
        "n_pde": config["n_pde"],
        "n_bc": config["n_bc"],
        "n_data": config["n_data"],
        "n_pde_50": config_50["n_pde"],
        "n_bc_50": config_50["n_bc"],
        "n_data_50": config_50["n_data"],
        "solver": solver,
        "steps": steps,
        "solver_50": solver_50,
        "steps_50": steps_50,
    }
    with open("figure_info/amery_plot_info.yml", "w") as f:
        yaml.dump(plot_data, f)

def coefficent_plot(coeff, f,axs, plot_config, cmap1, cmap2):
    _, S, _ = jnp.linalg.svd(coeff)
    S = S.reshape(coeff.shape[0])
    coeff_scale = jnp.max(jnp.abs(coeff))
    coeff_mask = jnp.where(coeff > 0, 1, -1)
    coeff_mask = jnp.where(coeff == 0, 0, coeff_mask)

    im0 = axs[0].imshow(coeff, cmap=cmap1, vmin=-coeff_scale, vmax=coeff_scale)
    im1 = axs[1].imshow(jnp.abs(coeff), cmap=cmap2, norm=mcolors.LogNorm())
    im2 = axs[2].imshow(coeff_mask, cmap=cmap1)
    im3 = axs[3].scatter(jnp.arange(coeff.shape[0]), S,color = plot_config["colors"][0])
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[2].invert_yaxis()
    images = [im0, im1, im2, im3]
    for i, (axis, img) in enumerate(zip(axs, images)):
        # Create a divider for the current axis
        divider = make_axes_locatable(axis)
        
        # Append a new axis to the right with a fixed size and padding
        # This new axis is where the colorbar will live.
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # --- Conditional Logic ---
        # Check if this is a plot that should have a colorbar
        if i in [0, 1]: # Indices of plots with colorbars
            # Create the colorbar in the new 'cax' axis
            f.colorbar(img, cax=cax)
        else:
            # For plots without a colorbar, just turn the new axis off
            cax.set_visible(False)
    axs[3].set_yscale("log")
    axs[0].set_box_aspect(1.0)
    axs[1].set_box_aspect(1.0)
    axs[2].set_box_aspect(1.0)
    axs[3].set_box_aspect(1.0)
    return im0, im1, im2, im3

def compare_coeff_lists(coeff_list, plot_config, name):
    coeffs_20, coeffs_40, coeffs_60 = coeff_list
    cmap1 = "bwr"
    cmap2 = "plasma"
    f, ax = plt.subplots(3,4,figsize=(13,8))
    ax = ax.flatten()
    im0, im1, im2, im3 = coefficent_plot(coeffs_20, f, ax[0:4], plot_config, cmap1, cmap2)
    im4, im5, im6, im7 = coefficent_plot(coeffs_40, f, ax[4:8], plot_config, cmap1, cmap2)
    im8, im9, im10, im11 = coefficent_plot(coeffs_60, f, ax[8:12], plot_config, cmap1, cmap2)

    ax[4].plot([-0.5, 20.5, 20.5, -0.5, -0.5], [-0.5, -0.5, 20.5, 20.5, -0.5], 'k-', linewidth=1)
    ax[5].plot([-0.5, 20.5, 20.5, -0.5, -0.5], [-0.5, -0.5, 20.5, 20.5, -0.5], 'w-', linewidth=1)
    ax[6].plot([-0.5, 20.5, 20.5, -0.5, -0.5], [-0.5, -0.5, 20.5, 20.5, -0.5], 'w-', linewidth=1)
    
    ax[8].plot([-0.5, 20.5, 20.5, -0.5, -0.5], [-0.5, -0.5, 20.5, 20.5, -0.5], 'k-', linewidth=1)
    ax[8].plot([-0.5, 40.5, 40.5, -0.5, -0.5], [-0.5, -0.5, 40.5, 40.5, -0.5], 'k-', linewidth=1)
    ax[9].plot([-0.5, 20.5, 20.5, -0.5, -0.5], [-0.5, -0.5, 20.5, 20.5, -0.5], 'w-', linewidth=1)
    ax[9].plot([-0.5, 40.5, 40.5, -0.5, -0.5], [-0.5, -0.5, 40.5, 40.5, -0.5], 'w-', linewidth=1)
    ax[10].plot([-0.5, 20.5, 20.5, -0.5, -0.5], [-0.5, -0.5, 20.5, 20.5, -0.5], 'w-', linewidth=1)
    ax[10].plot([-0.5, 40.5, 40.5, -0.5, -0.5], [-0.5, -0.5, 40.5, 40.5, -0.5], 'w-', linewidth=1)
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig(f"figure_primatives/{name}.png",dpi=300)
    plt.savefig(f"finished_figures/fig_{name}.pdf",format="pdf",dpi=300)
    plt.close()

def plot_laplace_coeffs(plot_config):
    data_path = pathlib.Path("data/fig_laplace_coeffs_error")
    full_data_60 = pr.load_dict_from_hdf5(data_path / "dogleg/60_laplace/full_data.h5")
    full_data_20 = pr.load_dict_from_hdf5(data_path / "dogleg/20_laplace/full_data.h5")
    full_data_40 = pr.load_dict_from_hdf5(data_path / "dogleg/40_laplace/full_data.h5")

    coeffs_20 = full_data_20["fields"]["u"]
    coeffs_40 = full_data_40["fields"]["u"]
    coeffs_60 = full_data_60["fields"]["u"]
   
    coeffs_20 = coeffs_20.reshape(21,21)
    coeffs_40 = coeffs_40.reshape(41,41)
    coeffs_60 = coeffs_60.reshape(61,61)

    compare_coeff_lists([coeffs_20, coeffs_40, coeffs_60], plot_config, "laplace_coeffs")

    full_data_20_adam = pr.load_dict_from_hdf5(data_path / "adam_50_40_01/20_laplace/full_data.h5")
    full_data_40_adam_2000 = pr.load_dict_from_hdf5(data_path / "adam_50_40_01/40_laplace/full_data.h5")
    full_data_60_adam_2000 = pr.load_dict_from_hdf5(data_path / "adam_50_40_01/60_laplace/full_data.h5")

    coeffs_20_adam = full_data_20_adam["fields"]["u"]
    coeffs_40_adam = full_data_40_adam_2000["fields"]["u"]
    coeffs_60_adam = full_data_60_adam_2000["fields"]["u"]

    coeffs_20_adam = coeffs_20_adam.reshape(21,21)
    coeffs_40_adam = coeffs_40_adam.reshape(41,41)
    coeffs_60_adam = coeffs_60_adam.reshape(61,61)

    compare_coeff_lists([coeffs_20_adam, coeffs_40_adam, coeffs_60_adam], plot_config, "laplace_coeffs_adam")

def plot_laplace_error_supplement(plot_config):
    error_data_Dogleg = jnp.load("data/fig_laplace_coeffs_error/dogleg/errors.npy")
    error_data_ADAM_01_20_10 = jnp.load("data/fig_laplace_coeffs_error/adam_01_20_10/errors.npy")
    error_data_ADAM_20_20_01 = jnp.load("data/fig_laplace_coeffs_error/adam_20_20_01/errors.npy")
    error_data_ADAM_50_20_01 = jnp.load("data/fig_laplace_coeffs_error/adam_50_20_01/errors.npy")
    error_data_ADAM_50_40_01 = jnp.load("data/fig_laplace_coeffs_error/adam_50_40_01/errors.npy")

    labeled_data = {
        "Dogleg": error_data_Dogleg,
        "ADAM 1": error_data_ADAM_01_20_10,
        "ADAM 2": error_data_ADAM_20_20_01,
        "ADAM 3": error_data_ADAM_50_20_01,
        "ADAM 4": error_data_ADAM_50_40_01,
    }
    labelx, labely = 0.02, 0.1
    label_pad = 0.3
    # First figure - L2 errors
    f1, ax1 = plt.subplots(1,1,figsize=plot_config["figsize"])
    colors = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]
    for i, (key, value) in enumerate(labeled_data.items()):
        basis_size = value[0,:]
        l2_errors = value[1,:]
        ax1.semilogy(basis_size, l2_errors,marker="o",label=key,color=colors[i])
    ax1.text(labelx, labely, "$(a)$", transform=ax1.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    ax1.set_xlabel("Basis Size")
    ax1.set_ylabel("$L_2$ Error")
    ax1.set_box_aspect(1.0)
    plt.tight_layout()
    plt.savefig("figure_primatives/laplace_basis_size_supp_a.png")
    plt.close()

    # Second figure - L∞ errors
    f2, ax2 = plt.subplots(1,1,figsize=plot_config["figsize"])
    for i, (key, value) in enumerate(labeled_data.items()):
        basis_size = value[0,:]
        linf_errors = value[2,:]
        ax2.semilogy(basis_size, linf_errors,marker="o",label=key,color=colors[i])
    ax2.text(labelx, labely, "$(b)$", transform=ax2.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    ax2.set_xlabel("Basis Size")
    ax2.set_ylabel("$L_\\infty$ Error")
    ax2.set_box_aspect(1.0)
    plt.tight_layout()
    plt.savefig("figure_primatives/laplace_basis_size_supp_b.png")
    plt.close()

def plot_coefficients(plot_config):
    data_path = pathlib.Path("data")
    amery_data_25 = pr.load_dict_from_hdf5(data_path / "fig_amery/amery_25.h5")
    amery_data_50 = pr.load_dict_from_hdf5(data_path / "fig_amery/amery_50.h5")
    coeffs_25 = amery_data_25["fields"]["u"]
    coeffs_50 = amery_data_50["fields"]["u"]
    coeffs_25 = coeffs_25.reshape(26,26)
    coeffs_50 = coeffs_50.reshape(51,51)
    cmap1 = "bwr"
    cmap2 = "plasma"
    f, ax = plt.subplots(2,4,figsize=(13,5.5))
    ax = ax.flatten()
    im0, im1, im2, im3 = coefficent_plot(coeffs_25, f, ax[0:4], plot_config, cmap1, cmap2)
    im4, im5, im6, im7 = coefficent_plot(coeffs_50, f, ax[4:8], plot_config, cmap1, cmap2)

    ax[4].plot([-0.5, 25.5, 25.5, -0.5, -0.5], [-0.5, -0.5, 25.5, 25.5, -0.5], 'k-', linewidth=1)
    ax[5].plot([-0.5, 25.5, 25.5, -0.5, -0.5], [-0.5, -0.5, 25.5, 25.5, -0.5], 'w-', linewidth=1)
    ax[6].plot([-0.5, 25.5, 25.5, -0.5, -0.5], [-0.5, -0.5, 25.5, 25.5, -0.5], 'w-', linewidth=1)
    
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/amery_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_amery_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

    taal_data = pr.load_dict_from_hdf5(data_path / "fig_laplace/laplace_taal_100.h5")
    coeffs_taal = taal_data["fields"]["c"]
    Nx_taal, Ny_taal = taal_data["config"]["basis_Nx"], taal_data["config"]["basis_Ny"]
    coeffs_taal = coeffs_taal.reshape(Nx_taal+1, Ny_taal+1)
    cmap1 = "bwr"
    cmap2 = "plasma"
    f, ax = plt.subplots(1,4,figsize=(13,3))
    ax = ax.flatten()
    im0, im1, im2, im3 = coefficent_plot(coeffs_taal, f, ax[0:4], plot_config, cmap1, cmap2)
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/taal_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_taal_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

    nls_data = pr.load_dict_from_hdf5(data_path / "fig_stiff/full_data_nls.h5")
    coeffs_nls_u = nls_data["fields"]["u"]
    coeffs_nls_v = nls_data["fields"]["v"]
    Nx_nls, Ny_nls = nls_data["config"]["basis_Nx"], nls_data["config"]["basis_Nt"]
    coeffs_nls_u = coeffs_nls_u.reshape(Nx_nls+1, Ny_nls+1)
    coeffs_nls_v = coeffs_nls_v.reshape(Nx_nls+1, Ny_nls+1)
    cmap1 = "bwr"
    cmap2 = "plasma"
    f, ax = plt.subplots(2,4,figsize=(13,5.5))
    ax = ax.flatten()
    im0, im1, im2, im3 = coefficent_plot(coeffs_nls_u, f, ax[0:4], plot_config, cmap1, cmap2)
    im4, im5, im6, im7 = coefficent_plot(coeffs_nls_v, f, ax[4:8], plot_config, cmap1, cmap2)
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/nls_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_nls_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

    ac_data = pr.load_dict_from_hdf5(data_path / "fig_stiff/full_data_ac.h5")   
    coeffs_ac = ac_data["fields"]["u"]
    Nx_ac, Ny_ac = ac_data["config"]["basis_Nx"], ac_data["config"]["basis_Nt"]
    coeffs_ac = coeffs_ac.reshape(Nx_ac+1, Ny_ac+1)
    coeffs_ac = coeffs_ac.T
    cmap1 = "bwr"
    cmap2 = "plasma"
    f, axs = plt.subplots(1,3,figsize=(12,3))
    axs = axs.flatten()
    coeff_scale = jnp.max(jnp.abs(coeffs_ac))
    coeff_mask = jnp.where(coeffs_ac > 0, 1, -1)
    coeff_mask = jnp.where(coeffs_ac == 0, 0, coeff_mask)

    im0 = axs[0].imshow(coeffs_ac, cmap=cmap1, vmin=-coeff_scale, vmax=coeff_scale,aspect="auto")
    im1 = axs[1].imshow(jnp.abs(coeffs_ac), cmap=cmap2, norm=mcolors.LogNorm(),aspect="auto")
    im2 = axs[2].imshow(coeff_mask, cmap=cmap1,aspect="auto")
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[2].invert_yaxis()
    images = [im0, im1, im2]
    for i, (axis, img) in enumerate(zip(axs, images)):
        # Create a divider for the current axis
        divider = make_axes_locatable(axis)
        
        # Append a new axis to the right with a fixed size and padding
        # This new axis is where the colorbar will live.
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # --- Conditional Logic ---
        # Check if this is a plot that should have a colorbar
        if i in [0, 1]: # Indices of plots with colorbars
            # Create the colorbar in the new 'cax' axis
            f.colorbar(img, cax=cax)
        else:
            # For plots without a colorbar, just turn the new axis off
            cax.set_visible(False)
    axs[0].set_box_aspect(1.0)
    axs[1].set_box_aspect(1.0)
    axs[2].set_box_aspect(1.0)
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax_val in enumerate(axs):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/ac_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_ac_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

    sphere_data = pr.load_dict_from_hdf5(data_path/"fig_sphere_diffusion/full_data.h5")
    sphere_coeffs = sphere_data["fields"]["c"]
    sphere_coeffs = sphere_coeffs.reshape(121,121)
    cmap1 = "bwr"
    cmap2 = "plasma"
    f, ax = plt.subplots(1,4,figsize=(13,3))
    ax = ax.flatten()
    im0, im1, im2, im3 = coefficent_plot(sphere_coeffs, f, ax[0:4], plot_config, cmap1, cmap2)
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/sphere_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_sphere_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

    wave_data = pr.load_dict_from_hdf5(data_path/"fig_wave/full_data.h5")
    wave_coeffs = wave_data["fields"]["u"]
    wave_coeffs = wave_coeffs.reshape(13,13,13)
    wave_max = jnp.max(jnp.abs(wave_coeffs))
    cmap1 = "bwr"

    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(3, 5, figure=fig)

    axes_plots = []
    ax_reference = None
    for i in range(13):
        row = i // 5
        col = i % 5
        if i == 0:
            ax = fig.add_subplot(gs[row, col])
            ax_reference = ax
        else:
            ax = fig.add_subplot(gs[row, col], sharex=ax_reference, sharey=ax_reference)
        axes_plots.append(ax)

    cax = fig.add_subplot(gs[2, 3])
    ax_to_hide = fig.add_subplot(gs[2, 4])
    ax_to_hide.set_visible(False)

    linthresh = 1e-3
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-wave_max, vmax=wave_max)
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax in enumerate(axes_plots):
        im = ax.imshow(wave_coeffs[:,:,i], cmap=cmap1, norm=norm)
        ax.set_box_aspect(1.0)
        ax.invert_yaxis()
        ax.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))

    cbar = fig.colorbar(im, cax=cax)
    cax.set_box_aspect(10.0)

    plt.tight_layout()
    plt.savefig("figure_primatives/wave_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_wave_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

    heat_forcing_data = pr.load_dict_from_hdf5(data_path/"fig_heat_forcing/full_data.h5")
    heat_forcing_coeffs = heat_forcing_data["fields"]["c"]
    heat_forcing_coeffs = heat_forcing_coeffs.reshape(11,11,11)
    forcing_fn_coeffs = heat_forcing_data["fields"]["f"]
    forcing_fn_coeffs = forcing_fn_coeffs.reshape(11,11)

    max_val = np.max(jnp.abs(heat_forcing_coeffs))
    max_forcing = np.max(jnp.abs(forcing_fn_coeffs))
    max_val = np.max([max_val, max_forcing])

    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(3, 5, figure=fig)

    axes_plots = []
    ax_reference = None
    for i in range(12):
        row = i // 5
        col = i % 5
        if i == 0:
            ax = fig.add_subplot(gs[row, col])
            ax_reference = ax
        else:
            ax = fig.add_subplot(gs[row, col], sharex=ax_reference, sharey=ax_reference)
        axes_plots.append(ax)

    cax = fig.add_subplot(gs[2, 3])
    ax_to_hide = fig.add_subplot(gs[2, 4])
    ax_to_hide.set_visible(False)

    linthresh = 1e-3
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-max_val, vmax=max_val)
    labelx, labely = -.4, 1.2
    label_pad = 0.0
    for i, ax in enumerate(axes_plots):
        if i <= 11:
            im = ax.imshow(heat_forcing_coeffs[:,:,i], cmap=cmap1, norm=norm)
        else:
            im = ax.imshow(forcing_fn_coeffs, cmap=cmap1, norm=norm)
        ax.set_box_aspect(1.0)
        ax.invert_yaxis()
        ax.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))

    cbar = fig.colorbar(im, cax=cax)
    cax.set_box_aspect(10.0)

    plt.tight_layout()
    plt.savefig("figure_primatives/heat_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_heat_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

    transport_data = pr.load_dict_from_hdf5(data_path/"fig_transport/full_data.h5")
    c_coeffs = transport_data["fields"]["c"].reshape(64,64)
    u_coeffs = transport_data["fields"]["u"].reshape(64,64)
    v_coeffs = transport_data["fields"]["v"].reshape(64,64)
    cmap1 = "bwr"
    cmap2 = "plasma"
    f, ax = plt.subplots(3,4,figsize=(12,7))
    ax = ax.flatten()
    im0, im1, im2, im3 = coefficent_plot(c_coeffs, f, ax[0:4], plot_config, cmap1, cmap2)
    im4, im5, im6, im7 = coefficent_plot(u_coeffs, f, ax[4:8], plot_config, cmap1, cmap2)
    im8, im9, im10, im11 = coefficent_plot(v_coeffs, f, ax[8:12], plot_config, cmap1, cmap2)
    labelx, labely = -.3, 1.1
    label_pad = 0.0
    for i, ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/transport_coeffs.png",dpi=300)
    plt.savefig("finished_figures/fig_transport_coeffs.pdf",format="pdf",dpi=300)
    plt.close()

def plot_synthetic_viscosity(plot_config):
    data_path = pathlib.Path("data/fig_viscosity_synthetic")
    reference_data = pr.load_dict_from_hdf5(data_path / "synthetic_data.h5")
    no_boundary_solve = pr.load_dict_from_hdf5(data_path / "full_data_no_boundary.h5")
    boundary_solve = pr.load_dict_from_hdf5(data_path / "full_data_boundary.h5")
    N = no_boundary_solve["config"]["basis_Nx"]
    basis = pr.ChebyshevBasis2D(deg=(N,N))
    ny, nx = reference_data["fields"]["u"]["x"].shape
    x_vec, y_vec = jnp.linspace(-1,1,nx), jnp.linspace(-1,1,ny)
    x_grid, y_grid = jnp.meshgrid(x_vec, y_vec)
    U0 = reference_data["scaling_dict"]["u"]
    H0 = reference_data["scaling_dict"]["h"]
    MU0 = reference_data["scaling_dict"]["mu"]
    vel_scaling_factor = 31536
    no_boundary_h_field = pr.BasisField(basis, pr.Coeffs(no_boundary_solve["fields"]["h"]))
    no_boundary_u_field = pr.BasisField(basis, pr.Coeffs(no_boundary_solve["fields"]["u"]))
    no_boundary_v_field = pr.BasisField(basis, pr.Coeffs(no_boundary_solve["fields"]["v"]))
    no_boundary_mu_field = pr.BasisField(basis, pr.Coeffs(no_boundary_solve["fields"]["mu"]))
    boundary_h_field = pr.BasisField(basis, pr.Coeffs(boundary_solve["fields"]["h"]))
    boundary_u_field = pr.BasisField(basis, pr.Coeffs(boundary_solve["fields"]["u"]))
    boundary_v_field = pr.BasisField(basis, pr.Coeffs(boundary_solve["fields"]["v"]))
    boundary_mu_field = pr.BasisField(basis, pr.Coeffs(boundary_solve["fields"]["mu"]))
    no_boundary_h_eval = H0*no_boundary_h_field.evaluate(x_grid, y_grid).reshape(ny, nx)/1000
    no_boundary_u_eval = U0*no_boundary_u_field.evaluate(x_grid, y_grid).reshape(ny, nx)*vel_scaling_factor
    no_boundary_v_eval = U0*no_boundary_v_field.evaluate(x_grid, y_grid).reshape(ny, nx)*vel_scaling_factor
    no_boundary_mu_eval = MU0*no_boundary_mu_field.evaluate(x_grid, y_grid).reshape(ny, nx)
    boundary_h_eval = H0*boundary_h_field.evaluate(x_grid, y_grid).reshape(ny, nx)/1000
    boundary_u_eval = U0*boundary_u_field.evaluate(x_grid, y_grid).reshape(ny, nx)*vel_scaling_factor
    boundary_v_eval = U0*boundary_v_field.evaluate(x_grid, y_grid).reshape(ny, nx)*vel_scaling_factor
    boundary_mu_eval = MU0*boundary_mu_field.evaluate(x_grid, y_grid).reshape(ny, nx)
    reference_u = reference_data["physical_data"]["u"]*vel_scaling_factor
    reference_v = reference_data["physical_data"]["v"]*vel_scaling_factor
    reference_h = reference_data["physical_data"]["h"]/1000
    reference_mu = reference_data["physical_data"]["mu"]

    x_ticks = [0, 20, 40, 60, 80]
    y_ticks = [0, 10, 20, 30, 40]
    
    mu_ticks = [0, 2e13, 4e13, 6e13, 8e13] 
    h_ticks = [0,0.13, 0.25, 0.38, 0.5]
    u_ticks = [0,0.5, 1.0, 1.5, 2.0]
    v_ticks = [-0.1, -0.05, 0.0, 0.05, 0.1]

    u_min, u_max = 0, 2.0
    v_min, v_max = np.min(reference_v), np.max(reference_v)
    h_min, h_max = 0, np.max(reference_h)
    mu_min, mu_max = 0, 8e13
    x_phys, y_phys = reference_data["physical_data"]["x"], reference_data["physical_data"]["y"]
    x_min, y_min = 0,0
    x_max, y_max = np.max(x_phys.reshape(-1))/1000, np.max(y_phys.reshape(-1))/1000
    cmap = "jet"
    f, ax = plt.subplots(4,3,figsize=(10,7),sharex=True,sharey=True)
    ax = ax.flatten()
    im0 = ax[0].imshow(reference_mu, cmap=cmap, vmin=mu_min, vmax=mu_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im1 = ax[1].imshow(no_boundary_mu_eval, cmap=cmap, vmin=mu_min, vmax=mu_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im2 = ax[2].imshow(boundary_mu_eval, cmap=cmap, vmin=mu_min, vmax=mu_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im3 = ax[3].imshow(reference_h, cmap=cmap, vmin=h_min, vmax=h_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im4 = ax[4].imshow(no_boundary_h_eval, cmap=cmap, vmin=h_min, vmax=h_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im5 = ax[5].imshow(boundary_h_eval, cmap=cmap, vmin=h_min, vmax=h_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im6 = ax[6].imshow(reference_u, cmap=cmap, vmin=u_min, vmax=u_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im7 = ax[7].imshow(no_boundary_u_eval, cmap=cmap, vmin=u_min, vmax=u_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im8 = ax[8].imshow(boundary_u_eval, cmap=cmap, vmin=u_min, vmax=u_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im9 = ax[9].imshow(reference_v, cmap=cmap, vmin=v_min, vmax=v_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im10 = ax[10].imshow(no_boundary_v_eval, cmap=cmap, vmin=v_min, vmax=v_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    im11 = ax[11].imshow(boundary_v_eval, cmap=cmap, vmin=v_min, vmax=v_max, extent=[x_min, x_max, y_min, y_max], origin="lower")
    cbar_mu = f.colorbar(im2, ax=ax[0:3])
    cbar_mu.set_ticks(mu_ticks)
    cbar_mu.set_label(r'$\mu$ (Pa s)') # <-- New line

    cbar_h = f.colorbar(im5, ax=ax[3:6])
    cbar_h.set_ticks(h_ticks)
    cbar_h.set_label(r'$h$ (km)') # <-- New line

    cbar_u = f.colorbar(im8, ax=ax[6:9])
    cbar_u.set_ticks(u_ticks)
    cbar_u.set_label(r'$u$ (km/yr)') # <-- New line

    cbar_v = f.colorbar(im11, ax=ax[9:12])
    cbar_v.set_ticks(v_ticks)
    cbar_v.set_label(r'$v$ (km/yr)') # <-- New line
    labelx, labely = -.18, 1.1
    label_pad = 0.0
    titles = ["Reference", "No Boundary", "Boundary"]
    for i,ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
        ax_val.set_xticks(x_ticks)
        ax_val.set_yticks(y_ticks)
        if i < 3:
            ax_val.set_title(titles[i])
        if i % 3 == 0:
            ax_val.set_ylabel("y (km)")
        if i >= 9:
            ax_val.set_xlabel("x (km)")
        ax_val.set_aspect("equal")
    plt.savefig("figure_primatives/synthetic_viscosity.png",dpi=300)
    plt.savefig("finished_figures/fig_synthetic_viscosity.pdf",format="pdf",dpi=300)
    plt.close()

    f,ax = plt.subplots(4,4,figsize=(12,9))
    ax = ax.flatten()
    cmap1 = "bwr"
    cmap2 = "plasma"
    im0, im1, im2, im3 = coefficent_plot(no_boundary_solve["fields"]["mu"].reshape(N+1,N+1), f, ax[:4], plot_config, cmap1, cmap2)
    im4, im5, im6, im7 = coefficent_plot(no_boundary_solve["fields"]["h"].reshape(N+1,N+1), f, ax[4:8], plot_config, cmap1, cmap2)
    im8, im9, im10, im11 = coefficent_plot(no_boundary_solve["fields"]["u"].reshape(N+1,N+1), f, ax[8:12], plot_config, cmap1, cmap2)
    im12, im13, im14, im15 = coefficent_plot(no_boundary_solve["fields"]["v"].reshape(N+1,N+1), f, ax[12:], plot_config, cmap1, cmap2)
    labelx, labely = -.4, 1.1
    label_pad = 0.0
    for i,ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/synthetic_viscosity_coefficients_no_boundary.png",dpi=300)
    plt.savefig("finished_figures/fig_synthetic_viscosity_coefficients_no_boundary.pdf",format="pdf",dpi=300)
    plt.close()

    f,ax = plt.subplots(4,4,figsize=(12,9))
    ax = ax.flatten()
    cmap1 = "bwr"
    cmap2 = "plasma"
    im0, im1, im2, im3 = coefficent_plot(boundary_solve["fields"]["mu"].reshape(N+1,N+1), f, ax[:4], plot_config, cmap1, cmap2)
    im4, im5, im6, im7 = coefficent_plot(boundary_solve["fields"]["h"].reshape(N+1,N+1), f, ax[4:8], plot_config, cmap1, cmap2)
    im8, im9, im10, im11 = coefficent_plot(boundary_solve["fields"]["u"].reshape(N+1,N+1), f, ax[8:12], plot_config, cmap1, cmap2)
    im12, im13, im14, im15 = coefficent_plot(boundary_solve["fields"]["v"].reshape(N+1,N+1), f, ax[12:], plot_config, cmap1, cmap2)
    labelx, labely = -.4, 1.1
    label_pad = 0.0
    for i,ax_val in enumerate(ax):
        ax_val.text(labelx, labely, f"$({chr(ord('a') + i)})$", transform=ax_val.transAxes, fontsize=16,fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', pad=label_pad))
    plt.tight_layout()
    plt.savefig("figure_primatives/synthetic_viscosity_coefficients_boundary.png",dpi=300)
    plt.savefig("finished_figures/fig_synthetic_viscosity_coefficients_boundary.pdf",format="pdf",dpi=300)
    plt.close()

    dx = x_grid[0,1] - x_grid[0,0]
    dy = y_grid[1,0] - y_grid[0,0]
    dh = dx*dy
    diff_mu_no_boundary = (no_boundary_mu_eval - reference_mu)/MU0
    diff_mu_boundary = (boundary_mu_eval - reference_mu)/MU0
    L2_error_no_boundary = jnp.sqrt(jnp.nansum(jnp.power(diff_mu_no_boundary, 2))*dh)
    L2_error_boundary = jnp.sqrt(jnp.nansum(jnp.power(diff_mu_boundary, 2))*dh)
    Linf_error_no_boundary = jnp.nanmax(jnp.abs(diff_mu_no_boundary))
    Linf_error_boundary = jnp.nanmax(jnp.abs(diff_mu_boundary))

    plot_data = {
        "L2_error_no_boundary": float(L2_error_no_boundary),
        "L2_error_boundary": float(L2_error_boundary),
        "Linf_error_no_boundary": float(Linf_error_no_boundary),
        "Linf_error_boundary": float(Linf_error_boundary),
        "basis_Nx": N,
        "solver": no_boundary_solve["config"]["solver_type"],
        "basis": "chebyshev",
        "boundary_n_pde": boundary_solve["config"]["n_pde"],
        "boundary_n_data": boundary_solve["config"]["n_data"],
        "boundary_n_bc": boundary_solve["config"]["n_bc"],
        "no_boundary_n_pde": no_boundary_solve["config"]["n_pde"],
        "no_boundary_n_data": no_boundary_solve["config"]["n_data"],
        "no_boundary_n_bc": no_boundary_solve["config"]["n_bc"]
    }

    with open("figure_info/synthetic_viscosity.yml", "w") as f:
        yaml.dump(plot_data, f)

def plot_svd_plots(plot_config):
    data_path = pathlib.Path("data")
    adam_data = pr.load_dict_from_hdf5(data_path / "fig_laplace_coeffs_error/adam_50_40_01/60_laplace/full_data.h5")
    dogleg_data = pr.load_dict_from_hdf5(data_path / "fig_laplace_coeffs_error/dogleg/60_laplace/full_data.h5")
    taal_data = pr.load_dict_from_hdf5(data_path / "fig_laplace/laplace_taal_100.h5")

    adam_coeffs = adam_data["fields"]["u"]
    dogleg_coeffs = dogleg_data["fields"]["u"]
    taal_coeffs = taal_data["fields"]["c"]

    adam_coeffs = adam_coeffs.reshape(61,61)
    dogleg_coeffs = dogleg_coeffs.reshape(61,61)
    taal_coeffs = taal_coeffs.reshape(101,101)

    _,adam_svd,_ = jnp.linalg.svd(adam_coeffs)
    _,dogleg_svd,_ = jnp.linalg.svd(dogleg_coeffs)
    _,taal_svd,_ = jnp.linalg.svd(taal_coeffs)
    adam_svd_norm = adam_svd
    dogleg_svd_norm = dogleg_svd
    taal_svd_norm = taal_svd
    f,ax = plt.subplots(figsize=(6,4))
    ax.semilogy(dogleg_svd_norm, color=plot_config["colors"][0], marker="o", label="Dogleg",linewidth=0,markersize=3,zorder=0)
    ax.semilogy(adam_svd_norm, color=plot_config["colors"][1], marker="o", label="Adam",linewidth=0,markersize=3,zorder=1)
    ax.semilogy(taal_svd_norm, color=plot_config["colors"][2],marker="o", label="Taal",linewidth=0,markersize=3,zorder=0)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(loc="upper right")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_box_aspect(1.0)
    plt.tight_layout()
    plt.savefig("figure_primatives/svd_plots.png",dpi=300)
    plt.savefig("finished_figures/fig_svd_plots.pdf",format="pdf",dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_config = {"cmap":"jet", "save_dir":"figure_primatives","colors":["#64CC33","#33BACC","#FF405E"],"cbar_size":(1.25, 4),"transparent":False,"figsize":(4,4),"file_type":"png"}
    # plot_wave_figure(plot_config)
    plot_laplace_figure(plot_config)
    # plot_schematic(plot_config)
    # plot_stiff_figure(plot_config)
    # plot_heat_forcing_figure(plot_config)
    # plot_transport_figure(plot_config)
    # plot_sphere_diffusion_figure(plot_config)
    # plot_amery(plot_config)
    # plot_laplace_coeffs(plot_config)
    # plot_coefficients(plot_config)
    # plot_laplace_error_supplement(plot_config)
    # plot_synthetic_viscosity(plot_config)
    # plot_svd_plots(plot_config)