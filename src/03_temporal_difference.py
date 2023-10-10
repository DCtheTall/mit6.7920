"""
Implementation of Temporal Difference Learning
==============================================
This module implements the TD(0) temporal difference learning
algorithm.

"""

import numpy as np
import random


TERMINAL_NODES = {(3, 3), (3, 2)}


def td0_estimator(S, A, R, V, γ):
    N = {s: 0.0 for s in S}
    π = random_policy(S, A)
    n_iter = 0
    while True:
        n_iter += 1
        V_prime = update_value_function(S, R, V, N, π, γ)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        V = V_prime
    return V, n_iter


def random_policy(S, A):
    """Use random policy to show TD{0} still converges"""
    def π(s):
        assert s in S
        return random.sample(list(A), k=1)[0]
    return π


def update_value_function(S, R, V, N, π, γ, T=100):
    """One step of iterative temporal difference (TD) learning"""
    V = V.copy()
    s = (0, 0)
    for _ in range(T):
        # Update per-stat learning rate
        N[s] += 1.0
        η = learning_rate(N[s])

        # Take action
        a = π(s)
        s_prime = sample_next_state(S, s, a)

        # Temporal difference update step
        V[s] = V[s] + η * temporal_difference(V, R, γ, s, s_prime)

        # Stop if reached terminal node
        if s in TERMINAL_NODES:
            break
        s = s_prime
    return V


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


# Memoization table for function below
T = {}

def sample_next_state(S, s, a):
    """Sample next state from MDP
    
    TD(0) algorithm treats this as a black box.
    """
    if s in {(3, 3), (3, 2)}:
        return s
    if (s, a) in T:
        return random.sample(T[(s, a)], 1)[0]
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
    T[(s, a)] = possible_next_states
    return random.sample(possible_next_states, 1)[0]


def temporal_difference(V, R, γ, s, s_prime):
    """Compute temporal difference term in current step"""
    return R.get(s, 0.0) + γ * V[s_prime] - V[s]


if __name__ == '__main__':
    # Set of all states, 4x4 grid
    S = {
        (i // 4, i % 4)
        for i in range(16)
    }

    # Set of all actions
    A = {'Up', 'Down', 'Left', 'Right'}

    # Rewards
    R = {(3, 3): 1.0, (3, 2): -1.0}

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.75

    # Apply TD(0) iteration
    V_opt, n_iter = td0_estimator(S, A, R, V, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print(V_opt)
