#!/usr/bin/python

import numpy as np


def power_iteration(A, num_simulations):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[0])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k

matrix = np.array([[0.5, 0.5], [0.2, 0.8]])

b_k = power_iteration(matrix, 10)
print("matrix",matrix)
print("egvector:",b_k)
print("egvalue:",b_k.dot(np.dot(matrix, b_k)) )

egvalue,egvector = np.linalg.eig(matrix)
print("egvalueBaseLinalg:",egvalue,egvector)

#http://mlwiki.org/index.php/Power_Iteration