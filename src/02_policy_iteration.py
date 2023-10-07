"""
Implementation of policy iteration algorithm
============================================

"""

import numpy as np


def get_reward(s):
    if s == (3, 3):
        return 1.0
    if s == (3, 2):
        return -1.0
    return 0.0


def policy_iteration(S, R, γ, V, π, A):
    """Value iteration algorithm implementation
    
    Complexity: O(S^2 * A)
    """
    n_iter = 0
    while True:
        n_iter += 1
        # Policy evaluation
        V_prime = evaluate_policy(S, R, γ, V, π)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        V = V_prime

        # Policy improvement
        π = update_policy(S, R, γ, V, A)
    return V, π, n_iter


def evaluate_policy(S, R, γ, V, π):
    return {
        s: bellman_operator(S, R, γ, V, s, π[s])
        for s in S
    }


def update_policy(S, R, γ, V, A):
    """Update the policy using the newly computed value function, V
    
    π[s] = argmax(bellman_operator(R, γ, V, s, a) for a in A)
    """
    π = {}
    for s in S:
        possible_actions = {
            a: bellman_operator(S, R, γ, V, s, a)
            for a in A
        }
        π[s] = max(possible_actions, key=possible_actions.get)
    return π


def bellman_operator(S, R, γ, V, s, a):
    """Bellman operator for value iteration
    
    Complexity: O(S)
    """
    return R[s] + γ * sum(
        transition_prob(s, a, s_prime) * V[s_prime]
        for s_prime in S
    )


def transition_prob(s, a, s_prime):
    """0% chance moving opposite direction, 33% change in any other.
    
    Can only move 1 cell at a time.
    """
    if s == {(3, 3), (3, 2)}:
        return 0.0
    dx, dy = s_prime[0] - s[0], s_prime[1] - s[1]
    if max(abs(dx), abs(dy), abs(dx) + abs(dy)) > 1:
        return 0.0
    if a == 'Up':
        return float(dy != -1) / 3.0
    if a == 'Down':
        return float(dy != 1) / 3.0
    if a == 'Left':
        return float(dx != 1) / 3.0
    assert a == 'Right'
    return float(dx != -1) / 3.0


if __name__ == '__main__':
    # Set of all states, 4x4 grid
    S = {
        (i // 4, i % 4)
        for i in range(16)
    }

    # Set of all actions
    A = {'Up', 'Down', 'Left', 'Right'}

    # Rewards
    R = {s: get_reward(s) for s in S}

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.75

    # Initialize stationary policy
    π = {s: 'Right' for s in S}

    # Apply policy iteration
    V_opt, π_opt, n_iter = policy_iteration(S, R, γ, V, π, A)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print(V_opt)
    print('Optimal policy:')
    print(π_opt)
    print('Best first action:', π_opt[(0, 0)])

