import numpy as np

# Define the matrix with given vectors as rows
matrix_np = np.array([
    [1, 1, -2, 0, -1],
    [1, 2,  0, -4, 1],
    [0, 1,  3, -3, 2],
    [2, 3,  0, -2, 0]
])

# Perform row reduction using numpy's linear algebra functions to calculate the rank
rank = np.linalg.matrix_rank(matrix_np)
print(f'The rank of matrix_np is {rank}')

# Find the basis by reducing the matrix to row echelon form
# Using numpy.linalg.qr to assist in extracting basis
_, r_matrix = np.linalg.qr(matrix_np)  # QR decomposition for approximate reduction


# Extract the pivot rows corresponding to the basis
basis_indices = np.where(np.abs(r_matrix.diagonal()) > 1e-10)[0]

basis_vectors_np = matrix_np[basis_indices]
print(f'The basis of matrix_np  is')
print(basis_vectors_np)
