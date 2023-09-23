"""
Example Linear Quadratic Resolver problem
=========================================

"""

import numpy as np
from control import lqr


if __name__ == '__main__':
    timestep = 1.0

    # Flying drone example
    # Want to keep the drone level at height zero.

    # Dynamics matrices
    # x_t+1 = A @ x_t + B @ u_t
    # In this case x is a 2D vector and u is a scalar
    A = np.array([[1.0, timestep], [0.0, 1.0]])
    B = np.array([[0.5 * timestep ** 2], [timestep]])

    # State cost matrix
    # x^T @ Q @ x
    Q = np.array([[1.0, 0.0], [0.0, 0.01]])

    # Action cost, increase to make moving the drone more costly.
    R = 1e3

    K, S, _  = lqr(A, B, Q, R)
    print('State feedback gains:', K)
    x = np.array([2.0, 3.0])
    print('Optimal action for state (2, 3):', -K @ x)
    x_prime = (A - B @ K) @ x
    print('Next state:', x_prime)
    print('Next optimal action:', -K @ x_prime)
