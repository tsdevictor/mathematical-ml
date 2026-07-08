# Mathematical ML

Machine learning and numerical linear algebra implementations from first principles.

This repository consolidates my computational mathematics work and my original MNIST neural-network project into one public package. The original standalone MNIST repository has been merged into this repo under `archive/mnist-original/`, and the original source scripts have been promoted into `src/mathematical_ml/mnist/`.

## Implementations

| Area | Implementation |
|---|---|
| Dimensionality reduction | PCA via eigendecomposition of centered data |
| Numerical linear algebra | Householder QR, modified Gram-Schmidt, block power iteration |
| Classification | Hard-margin linear SVM through the dual quadratic program |
| Spectral methods | kNN graph construction, graph Laplacian, spectral embedding |
| Kernels | Linear, polynomial, and RBF kernels |
| MNIST neural network | Original feedforward / training / evaluation scripts merged from the standalone MNIST repo |

## Repository Structure

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
      coursework/           # original computational math coursework archive, if retained

    docs/
      mnist-merge.md

## Quickstart

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    pytest

## Run the original MNIST feedforward script

Example smoke test using a tiny weight file:

    printf "1 1\n1\n" > /tmp/weights.txt
    python src/mathematical_ml/mnist/feedforward.py /tmp/weights.txt T1 1 2

Expected output:

    [3.0]

## Design Goals

- Keep the mathematical structure visible
- Preserve my original MNIST implementation rather than replacing it with a rewritten version
- Refactor the public surface enough that recruiters can understand the repo
- Keep raw historical files available under `archive/` while exposing the active code through `src/`

## Notes

The MNIST files under `src/mathematical_ml/mnist/` are intentionally kept close to the original standalone project. The goal of this merge is preservation and consolidation, not a rewrite.
