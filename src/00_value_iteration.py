"""
Implementation of value iteration algorithm
===========================================

"""

import numpy as np


def get_reward(s):
    if s == (3, 3):
        return 1.0
    if s == (3, 2):
        return -1.0
    return 0.0


def value_iteration(S, A, R, γ, V):
    """Value iteration algorithm implementation"""
    n_iter = 0
    while True:
        n_iter += 1
        V_prime = update_value_function(S, A, R, γ, V)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        else:
            V = V_prime
    return V, n_iter



def update_value_function(S, A, R, γ, V):
    """One round of the value function update step"""
    return {
        s: max(bellman_operator(S, R, γ, V, s, a) for a in A)
        for s in S
    }


def bellman_operator(S, R, γ, V, s, a):
    """Bellman operator for value iteration"""
    return R[s] + γ * sum(
        transition_prob(s, a, s_prime) * V[s_prime]
        for s_prime in S
    )


def transition_prob(s, a, s_prime, obstacles=[]):
    """0% chance moving opposite direction, 33% change in any other. Can only move 1 cell.
    
    Optional `obstacles` parameter allows you to consider the case when the agent
    cannot visit certain cells because they contain obstacles.

    With no obstacles, the optimal first action is Up. But, if you add a hole
    at (0, 1) the new first action is correctly Right.
    """
    if s_prime in obstacles:
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

    # Print final value function
    V_opt, n_iter = value_iteration(S, A, R, γ, V)
    print('Converged after', n_iter, 'iterations')
    print(V_opt)
    print(
        'Best first action:',
        'Up' if V_opt[(0, 1)] > V_opt[(1, 0)] else 'Right'
    )
