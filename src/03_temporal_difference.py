"""
Implementation of Temporal Difference Learning
==============================================
This module implements the TD(0) temporal difference learning
algorithm.

"""

import random
import numpy as np


def td0_estimator(S, A, R, V, γ):
    t = 0
    while True:
        t += 1
        η = learning_rate(t)
        V_prime = temporal_difference_step(S, A, R, V, γ, η)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        V = V_prime
    return V, t


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def temporal_difference_step(S, A, R, V, γ, η):
    """One step of iterative temporal difference (TD)"""
    π = random_policy(S, A)
    V_prime = {}
    for s in S:
        a = π.get(s, None)
        if a is None:
            s_prime = s
        else:
            s_prime = sample_next_state(S, s, a)
        # Temporal difference update step
        V_prime[s] = V[s] + η * temporal_difference(V, R, γ, s, s_prime)
    return V_prime


def random_policy(S, A):
    """Use random policy to show TD{0} still converges"""
    return {s: random.sample(list(A), 1)[0] for s in S}

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
    # Upper right corner is the goal
    # Lower right corner is a failure
    R = {{3, 2}: -1.0, (3, 3): 1.0}

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.75

    # Apply value iteration
    V_opt, n_iter = td0_estimator(S, A, R, V, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print(V_opt)
