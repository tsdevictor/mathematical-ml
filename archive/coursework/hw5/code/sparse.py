import scipy.io as sio
import numpy as np

# Read the MTX file
X = sio.mmread('X.mtx')

# To convert to a dense numpy array:
X_dense = X.toarray()

# Print basic information about the matrix
print(f"Matrix shape: {X.shape}")
print(f"Number of non-zero elements: {X.nnz}")

# Row sums of the sparse matrix (returns a dense array)
row_sums = X.sum(axis=1)
print(f"Row sums shape: {row_sums.shape}")

# Column sums
col_sums = X.sum(axis=0)
print(f"Column sums shape: {col_sums.shape}")

# Find maximum value in each row
row_max = X.max(axis=1).toarray()
print(f"Maximum value in first row: {row_max[0][0]}")

# Basic arithmetic operations are the same as numpy
Y = X * 2  # Multiply all elements by 2
Z = X + X  # Add the matrix to itself

# Matrix multiplication is done as in numpy
XtX = X.T @ X  # Transpose multiplication

# Get specific elements as in numpy
print(f"First row, first column element: {X[0, 0]}")

# Find non-zero elements in first row
first_row = X[0, :]
print(f"Non-zero elements in first row: {first_row.nnz}")

