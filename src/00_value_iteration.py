"""
Implementation of Value Iteration
=================================

"""

import numpy as np
from util.gridworld import GridWorld


def value_iteration(S, A, R, P, V, γ):
    """Value iteration algorithm implementation
    
    Complexity: O(S^2 * A)
    """
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


def optimal_bellman_operator_policy(S, A, R, P, V, γ):
    """Update the policy using the newly computed value function, V
    
    π[s] = argmax(bellman_operator(R, γ, V, s, a) for a in A)
    """
    π = {}
    for s in S:
        possible_actions = {
            a: bellman_operator(S, R, P, V, γ, s, a)
            for a in A
        }
        π[s] = max(possible_actions, key=possible_actions.get)
    return π


def print_grid(X):
    for y in range(3, -1, -1):
        print(*(str(X[(x, y)]) + '\t' for x in range(4)))


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Set of all states, 4x4 grid world
    S = env.S

    # Set of all actions
    A = env.A

    # Transition probabiltiy table
    P = env.P

    # Rewards
    R = env.R

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.75

    # Apply value iteration
    V_opt, n_iter = value_iteration(S, A, R, P, V, γ)

    # Derive optimal policy from value function using the optimal
    # Bellman operator.
    π_opt = optimal_bellman_operator_policy(S, A, R, P, V_opt, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal policy:')
    print_grid(π_opt)
    print(
        'Best first action:',
        max(A, key=lambda a: bellman_operator(S, R, P, V_opt, γ, (0, 0,), a))
    )
