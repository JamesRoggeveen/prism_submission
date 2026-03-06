import jax
import jax.numpy as jnp
import numpy as np
import h5py

def sample_from_mask(key, x_grid, y_grid, ref, M=8000):
    mask = np.isfinite(ref)

    xi = x_grid[mask]
    yi = y_grid[mask]

    idx = jax.random.choice(key, len(xi), shape=(M,), replace=False)
    x_samples = jnp.array(xi[idx])
    y_samples = jnp.array(yi[idx])

    return x_samples, y_samples

def load_dict_from_hdf5(path):
    with h5py.File(path, "r") as f:
        data = _recursive_load_dict_from_hdf5(f)
    return data

def _recursive_load_dict_from_hdf5(h5_group):
    reconstructed_dict = {}
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):
            reconstructed_dict[key] = _recursive_load_dict_from_hdf5(item)
        elif isinstance(item, h5py.Dataset):
            if item.shape is None:
                value = None
            else:
                value = item[()]
                # h5py can load strings as bytes, so we decode them
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
            # Convert numpy types to standard Python or JAX types
            if isinstance(value, np.ndarray):
                # For arrays, convert to JAX arrays
                value = jnp.array(value)
            elif isinstance(value, (np.float64, np.float32)):
                # Handle all numpy float types
                value = float(value)
            elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                # Handle all numpy integer types
                value = int(value)
            # Add debugging to see what types might be causing issues
            # print(f"Key: {key}, Type: {type(value)}")
            reconstructed_dict[key] = value
    return reconstructed_dict



def save_dict_to_hdf5(path, data_dict):
    with h5py.File(path, "w") as f:
        _recursive_save_dict_to_hdf5(f, data_dict)

def _recursive_save_dict_to_hdf5(h5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, create a new group and recurse
            sub_group = h5_group.create_group(key)
            _recursive_save_dict_to_hdf5(sub_group, value)
        elif value is None:
            # HDF5 can't store None, so we use an empty dataset as a placeholder
            h5_group.create_dataset(key, h5py.Empty("f"))
        else:
            # Save other data types as datasets
            # Convert lists/tuples to numpy arrays for compatibility
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            h5_group.create_dataset(key, data=value)

