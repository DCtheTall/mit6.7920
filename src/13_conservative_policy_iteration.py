"""
Implementation of Conservative Policy Iteration (CPI)
=====================================================

"""

import numpy as np
from util.display import print_grid
from util.gridworld import GridWorld


def value_iteration(S, A, R, P, γ):
    """Value iteration algorithm implementation

    Complexity: O(S^2 * A)
    """
    V = {s: 0.0 for s in S}
    n_iter = 0
    while True:
        n_iter += 1
        V_prime = update_value_function(S, A, R, P, V, γ)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        else:
            V = V_prime
    return V, n_iter



def update_value_function(S, A, R, P, V, γ):
    """One round of the value function update step

    Complexity: O(S^2 * A)
    """
    return {
        s: max(bellman_operator(S, R, P, V, γ, s, a) for a in A)
        for s in S
    }


def bellman_operator(S, R, P, V, γ, s, a):
    """Bellman operator for value iteration

    Complexity: O(S)
    """
    return R[s] + γ * sum(
        P.get((s, a, s_prime), 0) * V[s_prime]
        for s_prime in S
    )


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75

    # Use value iteration to get value function
    V, n_iter = value_iteration(env.S, env.A, env.R, env.P, γ)
    print('Value iteration took', n_iter, 'iterations')
    print('Value function:')
    print_grid(V)
