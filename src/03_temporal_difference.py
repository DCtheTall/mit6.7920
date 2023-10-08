"""
Implementation of Temporal Difference Learning
==============================================
This module implements the TD(0) temporal difference learning
algorithm.

"""

import random
import numpy as np


def get_reward(s):
    if s == (3, 3):
        return 1.0
    if s == (3, 2):
        return -1.0
    return 0.0


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
    π = greedy_policy(S, A, V)
    V_prime = {}
    for s in S:
        a = π.get(s, None)
        if a is None:
            s_prime = s
        else:
            s_prime = next_state_stochastic(S, s, a)
        # Temporal difference update step
        V_prime[s] = V[s] + η * temporal_difference(V, R, γ, s, s_prime)
    return V_prime


def greedy_policy(S, A, V):
    π = {}
    for s in S:
        possible_actions = {}
        for a in A:
            s_prime = next_state_deterministic(s, a)
            if s_prime != s:
                possible_actions[a] = V[s_prime]
        if len(possible_actions) > 0:
            π[s] = max(possible_actions, key=possible_actions.get)
    return π


def next_state_deterministic(s, a):
    if s in {(3, 3), (3, 2)}:
        return s
    if a == 'Up':
        return (s[0], min(s[1] + 1, 3))
    if a == 'Down':
        return (s[0], max(s[1] - 1, 0))
    if a == 'Left':
        return (max(s[0] - 1, 0), s[1])
    if a == 'Right':
        return (min(s[0] + 1, 3), s[1])
    raise ValueError(f'Unexpected action {a}')

# Memoization table for table below
T = {}

def next_state_stochastic(S, s, a):
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
    return R[s] + γ * V[s_prime] - V[s]


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

    # Apply value iteration
    V_opt, n_iter = td0_estimator(S, A, R, V, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print(V_opt)
    print(
        'Best first action:',
        'Up' if V_opt[(0, 1)] > V_opt[(1, 0)] else 'Right'
    )
