# Mathematical ML from Scratch

Computational mathematics and mathematical machine learning implementations adapted from coursework and reorganized into a reusable Python package.

The original coursework files are preserved in `archive/coursework/`; the cleaned implementations in `src/mathematical_ml/` expose the strongest algorithms through a smaller, more readable API.

## Clean implementations

| Area | Implementation |
|---|---|
| Dimensionality reduction | PCA via eigendecomposition of `X.T @ X` |
| Numerical linear algebra | Householder QR, Gram-Schmidt, block power iteration for `X X.T` |
| Classification | hard-margin linear SVM via the dual quadratic program |
| Spectral methods | kNN graph construction, graph Laplacian, spectral embedding, cluster homogeneity |
| Kernels | linear, polynomial, and RBF kernels |

## Archived coursework

The original coursework files are preserved under `archive/coursework/`.

| Topic | Source |
|---|---|
| PCA / low-rank approximation | `hw2`, `hw3`, `hw4`, `hw14` |
| Householder QR | `hw5` |
| Eigenvalue methods / block power iteration | `hw5`, `hw6` |
| Hard-margin SVM | `hw12` |
| Spectral clustering / graph Laplacians | `hw13` |
| Logistic regression experiments | `hw13` |
| Autoencoders on MNIST | `hw14` |

## Repository structure

```text
src/mathematical_ml/
  decomposition.py      # PCA
  linear_algebra.py     # Householder QR, Gram-Schmidt, block power iteration
  classification.py     # hard-margin SVM dual
  clustering.py         # graph Laplacian spectral clustering utilities
  kernels.py            # kernel functions

tests/                  # numerical correctness checks
examples/               # reproducible demos
figures/                # generated outputs
archive/coursework/     # original coursework files
```

## Quickstart

```bash
python -m pip install -e ".[dev]"
pytest
python examples/run_digits_pca.py
```

## Example output

The PCA demo projects handwritten digit data into two dimensions and saves the result to:

```text
figures/digits_pca.png
```

## Design goals

- Preserve the original mathematical ideas from coursework
- Refactor the strongest implementations into reusable modules
- Keep the linear algebra and optimization structure visible
- Add tests for numerical correctness
- Avoid presenting raw homework folders as the main public interface

## Notes

Some experiments use standard scientific Python tools such as SciPy, scikit-learn, and qpsolvers for numerical routines, graph construction, optimization, or comparison. The cleaned package focuses on making the mathematical pipeline explicit rather than replacing every optimized numerical backend.
