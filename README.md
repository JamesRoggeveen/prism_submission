# PRISM: Meshless PDE Solutions on Irregular Geometries

**PRISM** is a Python library for solving partial differential equations (PDEs) and PDE inverse problems on arbitrary spatial domains—including irregular and time-varying geometries—without meshing. Solutions are represented using spectral bases on a hyperrectangle that contains the true domain, and the governing equations, boundary conditions, and optional data-assimilation or optimization targets are enforced via a single loss function, in the spirit of Physics-Informed Neural Networks (PINNs). Because the representation has native spectral (exponential) convergence, the optimization can achieve exponential convergence when combined with standard ML-style optimizers (e.g. Adam, Levenberg–Marquardt, Dogleg).

For full method details, convergence results, and applications, see the paper:

**[Meshless solutions of PDE inverse problems on irregular geometries](https://arxiv.org/abs/2510.25752)**  
James V. Roggeveen, Michael P. Brenner — *arXiv:2510.25752* [math.NA]

---

## Features

- **Meshless**: No mesh generation; the basis is defined on a bounding box and restricted to the domain via a mask or boundary residuals.
- **Spectral bases**: Chebyshev, Fourier, Legendre (and combinations) in 2D/3D for fast, smooth approximations.
- **Flexible residuals**: Combine equation residuals, boundary conditions, initial conditions, and observation/data terms in one loss.
- **JAX/Equinox**: Differentiable, JIT-compiled problem and solver code; compatible with Optax and Optimistix.
- **Inverse & optimization**: Naturally supports parameter inference, data assimilation, and optimization over PDE solutions.

---

## Installation

Requires Python ≥3.12. From the project root:

```bash
uv sync
# or: pip install -e .
```

Dependencies are listed in `pyproject.toml` (JAX, Equinox, Optax, Optimistix, etc.).

---

## Basic usage

1. **Define a basis** (e.g. Chebyshev or Fourier–Chebyshev) and **fields** (e.g. `BasisField`) whose coefficients you will optimize.
2. **Implement an `AbstractProblem`** that returns residual functions (equation, boundary, initial condition, observation, etc.) and a `ProblemConfig` with residual weights.
3. **Build `ProblemData`** with collocation points, boundary data, reference data, etc., and a `SystemConfig` (basis sizes, solver choice, max steps, etc.).
4. **Get a solver** (e.g. `get_solver("LevenbergMarquardt")` or Adam) and call `solver.solve(problem, problem_data, config, problem_config)`.

The problem’s fields (and thus the basis coefficients) are the optimization variables; the solver updates them to minimize the combined residual loss.

```python
import prism as pr

# Example: 2D Chebyshev basis and a single field
basis = pr.ChebyshevBasis2D(Nx=20, Ny=20)
coeffs = pr.Coeffs.make_zero((20, 20))
u_field = pr.BasisField(basis=basis, coeffs=coeffs)

# Define your problem (subclass AbstractProblem), build ProblemData and configs, then:
# solver = pr.get_solver("LevenbergMarquardt")
# solution, logs = solver.solve(problem, problem_data, config, problem_config)
```

---

## Examples

Each example under `examples/` contains scripts (often in `scripts/`) and configs (in `configs/`) that you can run and modify.

| Example | Description |
|--------|-------------|
| **`wave`** | Wave equation (2D space + time) with initial/boundary conditions; demonstrates time-dependent PDEs on a domain defined by a mask (e.g. from an image). |
| **`allencahn`** | Allen–Cahn equation (reaction–diffusion) in 2D; spectral basis in space and time with optional causal time-marching. |
| **`nls`** | Nonlinear Schrödinger equation (complex field, two real components); similar setup to Allen–Cahn. |
| **`circle`** | Heat equation on a disk; includes an analytic Fourier–Bessel reference solution for comparison. |
| **`peanut`** | Laplace and Helmholtz on a “peanut”-shaped 2D domain; includes forward solves, inverse problems (reconstruct boundary or coefficient from data), and visualization. |
| **`sphere_flow`** | Stokes flow (and variants) around obstacles (e.g. one or two spheres) in 2D/3D; velocity and pressure as basis fields with no-slip/inlet/outlet conditions. |
| **`viscosity_synthetic`** | Shallow shelf approximation (SSA) style equations with unknown viscosity field; **inverse problem**: infer viscosity from surface velocity data (synthetic). |
| **`viscosity_amery`** | Same SSA-style inverse problem using real Amery Ice Shelf data (with preprocessing scripts). |
| **`viscosity_ross`** | SSA-style viscosity inversion for Ross Ice Shelf. |

Running an example typically means:

1. `cd examples/<name>` (or run from repo root with paths adjusted).
2. Install deps (`uv sync` or `pip install -e .` from root).
3. Run the main script, e.g. `python scripts/solve_*.py` or `python scripts/<example>_experiment.py`, sometimes with a config path or `--N`-type arguments (see script docstrings or `argparse`).
4. Use the corresponding `scripts/visualize.py` (or similar) to plot results.

Config files are YAML (e.g. `configs/config.yml`) and set basis sizes, residual weights, solver type, step counts, and paths.

---

## Project layout

- **`src/prism/`** — Core library  
  - **`basis/`** — Spectral bases (1D/2D/3D: Chebyshev, Fourier, Legendre, mixed).  
  - **`fields/`** — `BasisField`, `LogBasisField`, coefficient types (`Coeffs`, `FactorizedCoeffs`, etc.).  
  - **`data.py`** — `ProblemData`, `SystemConfig`, `CollocationPoints`, `BoundaryData`, `ReferenceData`.  
  - **`_problem.py`** — `AbstractProblem`, `ProblemConfig`, residual combination and loss.  
  - **`_solver.py`** — `AbstractSolver`, Optimistix (Levenberg–Marquardt, Dogleg) and Optax (e.g. Adam) solvers, `get_solver`.  
  - **`solve_utils.py`** — Sampling and HDF5 I/O helpers.  
- **`examples/`** — Example problems and configs (wave, Allen–Cahn, NLS, circle, peanut, sphere flow, viscosity inversions).  
- **`test/`** — Tests for basis and fields.  
- **`figures/`** — Scripts and data used to generate figures for the paper.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{roggeveen2025meshless,
  title={Meshless solutions of PDE inverse problems on irregular geometries},
  author={Roggeveen, James V. and Brenner, Michael P.},
  journal={arXiv preprint arXiv:2510.25752},
  year={2025},
  url={https://arxiv.org/abs/2510.25752}
}
```

---

## License

See the repository for license information.
