"""
Implementation of value iteration algorithm
===========================================

"""

import numpy as np


def build_transition_probs(S, A):
    """Build transition probability for 4x4 Frozen Lake MDP."""
    P = {}
    for s in S:
        for a in A:
            if s in {(3, 3), (3, 2)}:
                P[(s, a, s)] = 1.0
                continue
            possible_next_states = []
            for s_prime in S:
                dx, dy = s_prime[0] - s[0], s_prime[1] - s[1]
                if max(abs(dx), abs(dy), abs(dx) + abs(dy)) != 1:
                    continue
                if a == 'Left' and dx == 1:
                    continue
                if a == 'Right' and dx == -1:
                    continue
                if a == 'Up' and dy == -1:
                    continue
                if a == 'Down' and dy == 1:
                    continue
                possible_next_states.append(s_prime)
            for s_prime in possible_next_states:
                P[(s, a, s_prime)] = 1.0 / len(possible_next_states)
    return P


def get_reward(s):
    if s == (3, 3):
        return 1.0
    if s == (3, 2):
        return -1.0
    return 0.0


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


if __name__ == '__main__':
    # Set of all states, 4x4 grid
    S = {
        (i // 4, i % 4)
        for i in range(16)
    }

    # Set of all actions
    A = {'Up', 'Down', 'Left', 'Right'}

    # Transition probabiltiy table
    P = build_transition_probs(S, A)

    # Rewards
    R = {s: get_reward(s) for s in S}

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.75

    # Apply value iteration
    V_opt, n_iter = value_iteration(S, A, R, P, V, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print(V_opt)
    print(
        'Best first action:',
        max(A, key=lambda a: bellman_operator(S, R, P, V_opt, γ, (0, 0,), a))
    )
