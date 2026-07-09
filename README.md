# Mathematical ML

Clean implementations and experiments in machine learning, numerical linear algebra, and computational mathematics.

This repository consolidates my computational mathematics work and my original MNIST neural-network project into one public package. The active code lives under `src/mathematical_ml/`; older coursework and historical scripts are preserved under `archive/`.

## Implementations

| Area | Implementation |
|---|---|
| Dimensionality reduction | PCA via eigendecomposition of centered data |
| Numerical linear algebra | Householder QR, modified Gram-Schmidt, block power iteration |
| Classification | Hard-margin linear SVM through the dual quadratic program |
| Spectral methods | kNN graph construction, graph Laplacian, spectral embedding utilities |
| Kernels | Linear, polynomial, and RBF kernels |
| Neural networks | Original NumPy MNIST feedforward / training / evaluation scripts |

## Repository structure

```text
src/mathematical_ml/
  decomposition.py      # PCA
  linear_algebra.py     # QR, Gram-Schmidt, block power iteration
  classification.py     # hard-margin SVM dual
  clustering.py         # graph Laplacian spectral clustering utilities
  kernels.py            # kernel functions
  mnist/
    feedforward.py      # original MNIST feedforward script
    train.py            # original MNIST training / weight script
    evaluate.py         # original MNIST evaluation utilities

tests/
  test_decomposition.py
  test_linear_algebra.py
  test_classification.py
  test_clustering.py
  test_mnist_original.py

archive/
  mnist-original/       # complete original standalone MNIST repo after merge
  coursework/           # historical computational math coursework archive
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Example: PCA smoke test

```python
import numpy as np
from mathematical_ml.decomposition import PCA

X = np.random.default_rng(0).normal(size=(30, 5))
Z = PCA(n_components=2).fit_transform(X)

print(Z.shape)  # (30, 2)
```

## Example: original MNIST feedforward script

```bash
printf "1 1\n1\n" > /tmp/weights.txt
python src/mathematical_ml/mnist/feedforward.py /tmp/weights.txt T1 1 2
```

Expected output:

```text
[3.0]
```

## Tests

Run the full test suite from the repository root:

```bash
pytest -q
```

The tests cover the active package surface, including PCA shape/reconstruction behavior, linear-algebra utilities, hard-margin SVM behavior on separable data, spectral-clustering utilities, and the preserved MNIST feedforward script.

## Design goals

- Keep the mathematical structure visible.
- Expose active implementations through a clean `src/` package.
- Preserve historical coursework and early projects without making them the public surface.
- Keep tests lightweight enough to run quickly in CI.

## Notes

Some utilities intentionally use standard scientific Python tools where appropriate. For example, the SVM implementation uses a quadratic-programming backend, and the spectral-clustering utilities use scikit-learn components for graph construction and clustering around the custom Laplacian workflow.

The goal of this repository is not to replace production ML libraries. It is to show the mathematical and implementation details behind core ML and numerical methods while keeping the public project organized, testable, and understandable.
