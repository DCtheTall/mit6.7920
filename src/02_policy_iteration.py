"""
Implementation of Policy Iteration
==================================

"""

import numpy as np
from util.gridworld import Action, GridWorld


def policy_iteration(S, A, R, P, V, γ, π):
    """Value iteration algorithm implementation
    
    Complexity: O(S^2 * A)
    """
    n_iter = 0
    while True:
        n_iter += 1
        # Policy evaluation
        V_prime = evaluate_policy(S, R, P, V, γ, π)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        V = V_prime

        # Policy improvement
        π = optimal_bellman_operator_policy(S, A, R, P, V, γ)
    return V, π, n_iter


def evaluate_policy(S, R, P, V, γ, π):
    return {
        s: bellman_operator(S, R, P, V, γ, s, π[s])
        for s in S
    }


def bellman_operator(S, R, P, V, γ, s, a):
    """Bellman operator for value iteration
    
    Complexity: O(S)
    """
    return R.get(s, 0.0) + γ * sum(
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

    # Set of all states, 4x4 grid
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

    # Initialize stationary policy
    π = {s: Action.Right for s in S}

    # Apply policy iteration
    V_opt, π_opt, n_iter = policy_iteration(S, A, R, P, V, γ, π)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal policy:')
    print_grid(π_opt)
    print('Best first action:', π_opt[(0, 0)])

