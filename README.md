# Numerical Linear Algebra Scripts

This repository contains three standalone Python scripts implementing and comparing numerical methods for solving linear systems — particularly focused on tridiagonal matrices. These scripts were written for educational and experimental purposes, and are meant to provide insight into classic iterative and direct methods.

## Scripts Overview

### 1. `comparison-numerical-methods_tridiagonal_matplotlib.py`

This script compares four iterative solvers on tridiagonal systems of increasing size:

- **Jacobi method**
- **Gauss-Seidel method**
- **SOR (Successive Over-Relaxation)** — with a hardcoded relaxation factor `ω = 1.5`
- **Conjugate Gradient method**

It visualizes the number of iterations and approximation error (in ∞-norm) for each method across various system sizes using `matplotlib`. A reference solution is computed analytically via `np.linalg.solve()`.

### 2. `gauss-seidel-method.py`

This script implements the Gauss-Seidel method. It also includes functionality to approximate an optimal relaxation parameter (`ω`) for the **SOR method** via simple empirical testing.

The code is intentionally kept minimal and straightforward for educational clarity.

### 3. `LU-decomposition_tridiagonal.py`

This script implements LU decomposition specifically tailored for tridiagonal matrices. It demonstrates how such systems can be solved efficiently via forward and backward substitution once L and U are obtained.

## Notes

- Some values (e.g., relaxation parameters or matrix sizes) are **hardcoded**.
- Variable and comment names are partially in **German**.
- The code is not optimized for general-purpose use, but meant to **demonstrate key algorithmic ideas** clearly.

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for the comparison script)

You can install the requirements via:

```bash
pip install numpy matplotlib

