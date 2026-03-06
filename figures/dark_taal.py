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

if __name__ == "__main__":
    plt.style.use('dark_background')
    cmap = "jet"
    data_path = pathlib.Path("data/fig_laplace")
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

    f,ax = plt.subplots(1,1,figsize=(4,4))
    ax.contourf(lake_x, lake_y, lake_eval, levels=100,cmap = cmap)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig("dark_taal.png",dpi=300)
    plt.close()