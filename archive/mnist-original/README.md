# MNIST Neural Network from Scratch

A from-scratch implementation of a feedforward neural network and backpropagation for MNIST handwritten digit classification.

The goal of this project was to understand the mechanics of neural network training without relying on high-level machine learning libraries. The implementation includes the forward pass, loss computation, backpropagation, weight updates, and evaluation loop.

## Result

- Task: MNIST handwritten digit classification
- Model: fully connected neural network
- Training: custom training loop
- Test accuracy: approximately 92%

## What this demonstrates

- Forward propagation through dense layers
- Manual backpropagation
- Gradient-based optimization
- Matrix and vector operations for neural networks
- End-to-end training and evaluation without high-level ML frameworks
- A lightweight implementation using only the Python standard library

## Repository structure

| Path | Purpose |
|---|---|
| `src/train.py` | Main training script |
| `src/feedforward.py` | Feedforward neural network logic |
| `src/evaluate.py` | Evaluation/testing script |
| `archive/` | Earlier experimental versions |
| `results/training_plot.png` | Training/result visualization |
| `results/weights.txt` | Saved weights from a trained run |

## How to run

Train the model:

    python3 src/train.py

Evaluate the model:

    python3 src/evaluate.py

## Notes

This project is intentionally small and educational. It is not intended to compete with optimized PyTorch or TensorFlow implementations. The purpose is to show the underlying mechanics of neural network training from first principles.

The earlier experimental files are preserved in `archive/` to show the development process, while the cleaned implementation lives in `src/`.
