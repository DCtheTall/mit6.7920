"""
Implementation of policy iteration algorithm
============================================

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
        π = update_policy(S, A, R, P, V, γ)
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


def update_policy(S, A, R, P, V, γ):
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
    R = {(3, 3): 1.0, (3, 2): -1.0}

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.75

    # Initialize stationary policy
    π = {s: 'Right' for s in S}

    # Apply policy iteration
    V_opt, π_opt, n_iter = policy_iteration(S, A, R, P, V, γ, π)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print(V_opt)
    print('Optimal policy:')
    print(π_opt)
    print('Best first action:', π_opt[(0, 0)])

