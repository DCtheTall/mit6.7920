"""
Example Linear Quadratic Resolver problem
=========================================

"""

import numpy as np
from control import lqr


if __name__ == '__main__':
    # Flying drone example
    # Want to keep the drone level at height zero.

    # Updates every 5 seconds
    timestep = 5.0

    # Dynamics matrices
    # x_t+1 = A @ x_t + B @ u_t
    # In this case x is a 2D vector and u is a scalar representing
    # vertical force on the drone.
    A = np.array([[1.0, timestep], [0.0, 1.0]])
    B = np.array([[0.5 * timestep ** 2], [timestep]])

    # State cost matrix
    # x^T @ Q @ x
    # Increase first element to make distance from desired height more
    # constly.
    # Increase the last to make high velocity more costly.
    Q = np.array([[1.0, 0.0], [0.0, 0.01]])

    # Action cost, increase to acceleration more costly.
    R = 1e6

    K, S, _  = lqr(A, B, Q, R)
    print('State feedback gains:', K)
    x = np.array([20, 0])
    print('Optimal action for state (20, 0):', -K @ x)
    x_prime = (A - B @ K) @ x
    print('Next state:', x_prime)
    print('Next optimal action:', -K @ x_prime)
