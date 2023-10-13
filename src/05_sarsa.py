"""
Implementation of State-Action-Reward-State-Action (SARSA)
==========================================================
SARSA is equivalent to the TD(0) algorithm in this repository
except it uses state-action (Q) values, instead of a value function
depending only on state.

SARSA is an on-policy learning algorithm, meaning that it uses the
same policy during exploration as it does for selecting the next
action for temporal distance learning.

"""

import random
import numpy as np


TERMINAL_NODES = {(3, 3), (3, 2)}


def sarsa(S, A, R, Q, γ):
    t = 0
    N = {s: 0.0 for s in S}
    while True:
        t += 1
        π = greedy_softmax_policy(S, A, Q, τ=1.0)
        Q_prime = update_q_function(S, R, Q, N, π, γ)
        if all(np.isclose(Q[(s, a)], Q_prime[(s, a)]) for s in S for a in A):
            break
        Q.update(Q_prime)
    return Q, t


def greedy_softmax_policy(S, A, Q, τ):
    """Softmax-based exploration policy"""
    p = {}
    A_ordered = list(A)
    for s in S:
        numerators = {}
        for a in A:
            numerators[a] = np.exp(Q[(s, a)] / τ)
        denom = sum(numerators.values())
        p[s] = [numerators[a] / denom for a in A_ordered]
    def π(s):
        assert s in S
        return np.random.choice(A_ordered, 1, p=p[s])[0]
    return π


def update_q_function(S, R, Q, N, π, γ, T=100):
    """One episode of iterative temporal difference (TD) learning"""
    Q = Q.copy()
    s = (0, 0)
    a_prime = π(s)
    for _ in range(T):
        # Update learning rate
        N[s] += 1.0
        η = learning_rate(N[s])

        # Take action
        a = a_prime
        s_prime = take_action(S, s, a)

        # Temporal difference update step
        a_prime = π(s_prime)
        Q[(s, a)] = Q[(s, a)] + η * temporal_difference(Q, R, γ, s, a, s_prime, a_prime)
        if s in TERMINAL_NODES:
            break
        s = s_prime
    return Q


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


# Memoization table for function below
T = {}

def take_action(S, s, a):
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


def temporal_difference(Q, R, γ, s, a, s_prime, a_prime):
    """Compute temporal difference term in current step"""
    return R.get(s, 0.0) + γ * Q[(s_prime, a_prime)] - Q[(s, a)]


def optimal_policy(S, A, Q):
    return {s: max(A, key=lambda a, s=s: Q[(s, a)]) for s in S}


def print_grid(X):
    for y in range(3, -1, -1):
        print(*(str(X[(x, y)]) + '\t' for x in range(4)))


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

    # Optimal policy
    π_opt = optimal_policy(S, A, Q_opt)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal policy:')
    print_grid(π_opt)
    print('Best first action:', π_opt[(0, 0)])
