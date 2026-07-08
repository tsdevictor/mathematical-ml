# Mathematical ML from Scratch

Numerical methods and mathematical machine learning algorithms implemented from first principles in Python and NumPy.

This repository collects clean implementations of algorithms from computational mathematics, machine learning, and numerical linear algebra. The goal is not to replace optimized libraries, but to make the underlying mathematics explicit, readable, and reproducible.

## Implemented algorithms

| Area | Algorithms |
|---|---|
| Dimensionality reduction | PCA |
| Clustering | k-means, spectral clustering |
| Classification | logistic regression, linear SVM |
| Numerical linear algebra | QR decomposition, power iteration |
| Kernels | linear, polynomial, RBF |

## Repository structure

```text
src/mathematical_ml/
  decomposition.py      # PCA
  clustering.py         # k-means, spectral clustering
  classification.py     # logistic regression, linear SVM
  linear_algebra.py     # QR decomposition, power iteration
  kernels.py            # kernel functions
tests/                  # numerical correctness checks
examples/               # reproducible demos
figures/                # generated outputs
archive/coursework/     # original coursework files, preserved for reference