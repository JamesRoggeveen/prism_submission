import matplotlib.pyplot as plt
import numpy as np
import pathlib

results_path = pathlib.Path("results")
errors_1 = np.loadtxt(results_path / "10182258_laplace_inverse/errors.csv", delimiter=',', skiprows=1)
n_bc_list_1 = errors_1[:,0]
l2_errors_1 = errors_1[:,1]
Linf_errors_1 = errors_1[:,2]
errors_2 = np.loadtxt(results_path / "10182145_laplace_inverse/errors.csv", delimiter=',', skiprows=1)
n_bc_list_2 = errors_2[:,0]
l2_errors_2 = errors_2[:,1]
Linf_errors_2 = errors_2[:,2]

plt.loglog(n_bc_list_1, l2_errors_1, label='L2 Error 1')
plt.loglog(n_bc_list_1, Linf_errors_1, label='Linf Error 1')
plt.loglog(n_bc_list_2, l2_errors_2, label='L2 Error 2')
plt.loglog(n_bc_list_2, Linf_errors_2, label='Linf Error 2')
plt.xlabel('Number of Boundary Conditions')
plt.ylabel('Error')
plt.title('Laplace Inverse Error')
plt.legend()
plt.savefig("figures/laplace_inverse_error.png")
plt.close()