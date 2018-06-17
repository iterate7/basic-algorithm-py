#!/usr/bin/python
"""
url: http://mlwiki.org/index.php/Power_Iteration

this class implements power iteration, which use for decomposition of matrix.

"""

import numpy as np


class PowIteration:

    def power_iteration(self, matrix, num_simulations):
        """
        algorithm:
        1. Ideally choose a random vector To decrease the chance that our vector
        2. get the vector of  vector dot matrix
        3. many iteration to decide the vector Is orthogonal to the eigenvector
        :param matrix:
        :param num_simulations:
        :return:
        """
        b_k = np.random.rand(matrix.shape[0])
        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = np.dot(matrix, b_k)

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm
        return b_k


def main():
    instance = PowIteration()
    matrix_input = np.array([[0.5, 0.5], [0.2, 0.8]])
    b_k = instance.power_iteration(matrix_input, 10)
    print("matrix", matrix_input)
    print("eigenvector:", b_k)
    print("eigenvalue:", b_k.dot(np.dot(matrix_input, b_k)))

    # use the linalg for validate
    eigen_value, eigen_vector = np.linalg.eig(matrix_input)
    print("egvalueBaseLinalg:", eigen_value, eigen_vector)


if __name__ == '__main__':
    main()

