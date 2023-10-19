"""
Implementation of Incremental Least Squares Temporal Difference (iLSTD) Learning
================================================================================

"""

import numpy as np
import random


D_MODEL = 8
TERMINAL_NODES = {(3, 3), (3, 2)}


def features(S):
    """Extract features for linear TD"""
    ϕ = {}
    for s in S:
        x, y = s
        l2_goal = ((x - 3) ** 2 + (y - 3) ** 2) ** 0.5
        l2_fail = ((x - 3) ** 2 + (y - 2) ** 2) ** 0.5
        ϕ[s] = np.array([
            float(x), float(y), # position
            float(s in TERMINAL_NODES), # if terminal
            l2_goal, # L2 distance from goal
            l2_fail, # L2 distance from failure
            0.0 if s == (3, 3) else np.arccos((y - 3) / l2_goal), # angle wrt goal
            0.0 if s == (3, 2) else np.arccos((y - 2) / l2_fail), # angle wrt failure
            (x ** 2.0 + y ** 2.0) ** 0.5, # L2 distance from origin
        ], dtype=np.float32)
    return ϕ


def initialize_parameters():
    return np.zeros((D_MODEL,))


def ilstd(S, A, R, V, γ, λ, θ, ϕ):
    N = {s: 0.0 for s in S}
    π = random_policy(S, A)
    n_iter = 0
    while True:
        n_iter += 1
        θ_prime = update_value_function(S, R, V, N, π, γ, λ, θ, ϕ)
        if all(np.isclose(a, b) for a, b in zip(θ, θ_prime)):
            break
        θ = θ_prime
    return θ, n_iter


def random_policy(S, A):
    """Use random policy for exploration"""
    def π(s):
        assert s in S
        return random.sample(list(A), k=1)[0]
    return π


def update_value_function(S, R, V, N, π, γ, λ, θ, ϕ, T=100):
    θ = θ.copy()
    s = (0, 0)
    z = {}
    for _ in range(T):
        # Update per-stat learning rate
        N[s] += 1.0

        # Take action
        a = π(s)
        s_prime = take_action(S, s, a)

        z[s] = z.get(s, 0.0) + 1.0

        # TD(λ) update step using LSTD update
        for sx in z.keys():
            N[sx] = N.get(sx, 0) + 1
            η = learning_rate(N[sx])
            θ += η * z[sx] * temporal_difference(V, R, γ, θ, s, s_prime) * ϕ[sx]
            z[sx] *= λ * γ

        # Stop if reached terminal node
        if s in TERMINAL_NODES:
            break
        s = s_prime
    return θ


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


def temporal_difference(V, R, γ, θ, s, s_prime):
    return R.get(s, 0.0) + γ * V(θ, s_prime) - V(θ, s)


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

    # Non-linear features
    ϕ = features(S)

    # Initialize parameters for linear TD
    θ = initialize_parameters()

    # Initialize parameterized value function
    def V(θ, s):
        return θ @ ϕ[s]

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Approximate value function with linear TD
    θ_opt, n_iter = ilstd(S, A, R, V, γ, λ, θ, ϕ)
    V_opt = {s: V(θ_opt, s) for s in S}

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal parameters:')
    print(θ_opt)
