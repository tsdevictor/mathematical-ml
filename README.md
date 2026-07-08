cat > README.md <<'EOF'
# Mathematical ML from Scratch

Clean NumPy implementations of numerical linear algebra and mathematical machine learning algorithms, adapted from computational mathematics coursework and reorganized into a reusable Python package.

The goal of this repository is to make the underlying math visible: PCA, eigenvalue methods, QR decomposition, clustering, classification, kernels, and related experiments are implemented directly rather than hidden behind high-level library calls.

## Current clean implementations

| Area | Implemented in `src/` |
|---|---|
| Dimensionality reduction | PCA |
| Numerical linear algebra | QR decomposition, power iteration |
| Kernels | linear, polynomial, RBF |
| Clustering | k-means, spectral clustering |
| Classification | logistic regression, linear SVM |

## Archived coursework implementations

The original coursework files are preserved under `archive/coursework/`. These include earlier experiments with:

| Topic | Source |
|---|---|
| PCA / low-rank approximation | `hw2`, `hw3`, `hw4`, `hw14` |
| QR decomposition | `hw5` |
| Eigenvalue methods | `hw5`, `hw6` |
| Linear SVM | `hw12` |
| Spectral clustering / graph Laplacians | `hw13` |
| Logistic regression experiments | `hw13` |
| Autoencoders on MNIST | `hw14` |

## Repository structure

```text
src/mathematical_ml/
  decomposition.py      # PCA
  linear_algebra.py     # QR decomposition, power iteration
  kernels.py            # linear, polynomial, and RBF kernels
  clustering.py         # k-means and spectral clustering
  classification.py     # logistic regression and linear SVM

tests/                  # numerical correctness checks
examples/               # reproducible demos
figures/                # generated plots
archive/coursework/     # original coursework files
```

## Quickstart

```bash
python -m pip install -e ".[dev]"
pytest
python examples/run_digits_pca.py
```

## Example demo

The PCA demo projects handwritten digit images into two dimensions and saves the plot to:

```text
figures/digits_pca.png
```

## Design goals

- Implement core algorithms directly in NumPy
- Keep the mathematical structure readable
- Add small tests for numerical correctness
- Provide reproducible examples
- Preserve original coursework separately from the polished package interface

## Notes

These implementations are educational and intentionally minimal. They are meant to demonstrate mathematical understanding, not to replace optimized libraries such as NumPy, SciPy, or scikit-learn.
EOF