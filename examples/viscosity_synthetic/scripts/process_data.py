import numpy as np
from scipy.io import loadmat
import time
from prism import save_dict_to_hdf5

def strip_outer_nans(array, x, y):
    # Find rows that have any non-NaN values
    valid_rows = ~np.all(np.isnan(array), axis=1)
    # Find first and last valid row
    first_row = np.argmax(valid_rows)
    last_row = len(valid_rows) - np.argmax(valid_rows[::-1]) - 1
    
    # Find columns that have any non-NaN values
    valid_cols = ~np.all(np.isnan(array), axis=0)
    # Find first and last valid column
    first_col = np.argmax(valid_cols)
    last_col = len(valid_cols) - np.argmax(valid_cols[::-1]) - 1
    
    # Slice the arrays
    stripped_array = array[first_row:last_row+1, first_col:last_col+1]
    stripped_x = x[first_row:last_row+1, first_col:last_col+1]
    stripped_y = y[first_row:last_row+1, first_col:last_col+1]
    
    return stripped_array, stripped_x, stripped_y

if __name__ == "__main__":
    data = loadmat("data/original_synthetic.mat")
    h_data = {"data": data['hd'],"x":data['xd_h'],"y":data['yd_h']}
    u_data = {"data": data['ud'],"x":data['xd'],"y":data['yd']}
    v_data = {"data": data['vd'],"x":data['xd'],"y":data['yd']}
    bc_data = {"x":data['xct'],"y":data['yct'],"nx":data['nnct'][:,0],'ny':data['nnct'][:,1]}
    mu_data = {"data": data['mud'],"x":data['xd'],"y":data['yd']}

    for data_dict in [h_data, u_data, v_data]:
        data_dict['data'], data_dict['x'], data_dict['y'] = strip_outer_nans(data_dict['data'], data_dict['x'], data_dict['y'])
        data_dict["mask"] = ~np.isnan(data_dict['data'])

    h_phys, u_phys, v_phys, mu_phys = h_data["data"], u_data["data"], v_data["data"], mu_data["data"]
    X_phys_h, Y_phys_h = h_data["x"], h_data["y"]
    X_phys_u, Y_phys_u = u_data["x"], u_data["y"]
    X_phys_v, Y_phys_v = v_data["x"], v_data["y"]

    h_min, h_max = np.nanmin(h_phys), np.nanmax(h_phys)
    h_range = h_max if h_max > h_min else 1.0
    vel_mag = np.sqrt(u_phys**2 + v_phys**2)
    vel_max = np.nanmax(vel_mag)
    u_min, u_max = -vel_max, vel_max
    u_range = u_max - u_min if u_max > u_min else 1.0
    x_max = np.nanmax(np.concatenate([X_phys_h.reshape(-1), X_phys_u.reshape(-1), X_phys_v.reshape(-1)]))
    x_min = np.nanmin(np.concatenate([X_phys_h.reshape(-1), X_phys_u.reshape(-1), X_phys_v.reshape(-1)]))
    y_max = np.nanmax(np.concatenate([Y_phys_h.reshape(-1), Y_phys_u.reshape(-1), Y_phys_v.reshape(-1)]))
    y_min = np.nanmin(np.concatenate([Y_phys_h.reshape(-1), Y_phys_u.reshape(-1), Y_phys_v.reshape(-1)]))
    x_range = x_max - x_min
    y_range = y_max - y_min

    L0, H0 = x_range, h_range
    U0 = u_range / 2.0
    if U0 < 1e-9:
        U0 = 1.0
    rho_i, rho_w, g = 917.0, 1030.0, 9.8
    density_factor = rho_i * g * (1 - rho_i / rho_w)
    # Factor of 0.5 to account for the non-dimensional length scale has a range of 2
    MU0 = 0.5 * (density_factor * H0 * L0) / U0
    h_data["data"] = h_data["data"]/H0
    def map_data(array, min_val, range):
        return 2*(array - min_val)/range - 1
    u_data["data"] = map_data(u_data["data"], u_min, u_range)
    v_data["data"] = map_data(v_data["data"], u_min, u_range)
    mu_data["data"] = mu_data["data"]/MU0

    field_data = {"h":h_data,"u":u_data,"v":v_data,"mu":mu_data}

    for key, data_dict in field_data.items():
        data_dict["x"] = map_data(data_dict["x"], x_min, x_range)
        data_dict["y"] = map_data(data_dict["y"], y_min, y_range)
        field_data[key] = data_dict
    
    nx, ny = bc_data["nx"], bc_data["ny"]
    nx = nx/x_range
    ny = ny/y_range
    n_norm = np.sqrt(nx**2 + ny**2)
    bc_data["nx"] = nx/n_norm
    bc_data["ny"] = ny/n_norm
    bc_data["x"] = map_data(bc_data["x"], x_min, x_range)
    bc_data["y"] = map_data(bc_data["y"], y_min, y_range)
    physical_data = {"h":h_phys,"u":u_phys,"v":v_phys,"x":X_phys_h,"y":Y_phys_h,"mu":mu_phys}
    full_data_dict = {"bc": bc_data, "fields": field_data, "physical_data": physical_data}
    
    aspect_ratio = x_range/y_range

    scaling_dict = {
        "h": H0,
        "u": U0,
        "v": U0,
        "x": x_range,
        "y": y_range,
        "aspect_ratio": aspect_ratio,
        "mu": MU0,
        "rho_i": rho_i,
        "rho_w": rho_w,
        "g": g,
    }

    full_data_dict["scaling_dict"] = scaling_dict
    
    output_file = "data/synthetic_data.h5"

    time = time.strftime("%m%d%H%M")
    full_data_dict["metadata"] = {"time": time, "data_path": "data/original_synthetic.mat", "experiment_name": "visc_inv_synthetic", "data_source": "10.1126/science.adp3300"}

    save_dict_to_hdf5(output_file, full_data_dict)

    print(f"Data saved to {output_file}")

