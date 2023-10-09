"""
Implementation of State-Action-Reward-State-Action (SARSA)
==========================================================

"""

import random
import numpy as np


def sarsa(S, A, R, Q, γ):
    t = 0
    while True:
        t += 1
        η = learning_rate(t)
        π = softmax_policy(S, A, Q, τ=1.0)
        Q_prime = update_q_function(S, R, Q, π, γ, η)
        if all(np.isclose(Q[(s, π[s])], Q_prime[(s, π[s])]) for s in S):
            break
        Q.update(Q_prime)
    return Q, t


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def update_q_function(S, R, Q, π, γ, η):
    """One step of iterative temporal difference (TD) learning"""
    Q_prime = {}
    for s in S:
        s_prime = sample_next_state(S, s, π[s])
        # Temporal difference update step
        Q_prime[(s, π[s])] = Q[(s, π[s])] + η * temporal_difference(Q, R, γ, π, s, s_prime)
    return Q_prime


def softmax_policy(S, A, Q, τ):
    """Softmax-based exploration policy"""
    π = {}
    for s in S:
        numerators = {}
        for a in A:
            numerators[a] = np.exp(Q[(s, a)] / τ)
        denom = sum(numerators.values())
        A_ordered = numerators.keys()
        p = [numerators[a] / denom for a in A_ordered]
        π[s] = np.random.choice(list(A_ordered), 1, p=p)[0]
    return π


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


def temporal_difference(Q, R, γ, π, s, s_prime):
    """Compute temporal difference term in current step"""
    return R.get(s, 0.0) + γ * Q[(s_prime, π[s_prime])] - Q[(s, π[s])]


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

    # Initialize Q function
    Q = {(s, a): 0.0 for s in S for a in A}

    # Discount factor
    γ = 0.75

    # Apply SARSA
    Q_opt, n_iter = sarsa(S, A, R, Q, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print(Q_opt)
    print('Best first action:', max(list(A), key=lambda a: Q[((0, 0), a)]))
