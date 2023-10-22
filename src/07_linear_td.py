"""
Implementation of Linear Temporal Difference Learning
=====================================================

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld


N_FEATURES = 9


def features(env):
    """Extract features for linear TD"""
    ϕ = {}
    for s in env.S:
        x, y = s
        xg, yg = env.goal
        xf, yf = env.failure
        l2_goal = ((x - xg) ** 2 + (y - yg) ** 2) ** 0.5
        l2_fail = ((x - xf) ** 2 + (y - yf) ** 2) ** 0.5
        ϕ[s] = np.array([
            float(x), float(y), # position
            (x ** 2.0 + y ** 2.0) ** 0.5, # L2 distance from origin
            float(x + y), # L1 norm from origin
            l2_goal, # L2 distance from goal
            l2_fail, # L2 distance from failure
            0.0 if s == env.goal else np.arccos((y - yg) / l2_goal), # angle wrt goal
            0.0 if s == env.failure else np.arccos((y - yf) / l2_fail), # angle wrt failure
            float(env.is_terminal_state(s)),
        ], dtype=np.float32)
    return ϕ


def initialize_parameters():
    return np.zeros((N_FEATURES,))


def ilstd(env, V, γ, λ, θ, ϕ):
    N = {s: 0.0 for s in env.S}
    π = random_policy(env.S, env.A)
    n_iter = 0
    while True:
        n_iter += 1
        θ_prime = update_value_function(env, V, N, π, γ, λ, θ, ϕ)
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


def update_value_function(env, V, N, π, γ, λ, θ, ϕ, T=100):
    θ = θ.copy()
    s = (0, 0)
    z = {}
    for _ in range(T):
        # Update per-stat learning rate
        N[s] += 1.0

        # Take action
        a = π(s)
        s_prime = env.step(s, a)

        z[s] = z.get(s, 0.0) + 1.0

        # TD(λ) update step
        for sx in z.keys():
            N[sx] = N.get(sx, 0) + 1
            η = learning_rate(N[sx])
            dt = temporal_difference(V, env.R, γ, θ, s, s_prime)
            θ += η * z[sx] * dt * ϕ[sx]
            z[sx] *= λ * γ

        # Stop if reached terminal node
        if env.is_terminal_state(s):
            break
        s = s_prime
    return θ


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def temporal_difference(V, R, γ, θ, s, s_prime):
    return R[s] + γ * V(θ, s_prime) - V(θ, s)


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Non-linear features
    ϕ = features(env)

    # Initialize parameters for linear TD
    θ = initialize_parameters()

    # Initialize parameterized value function
    def V(θ, s):
        return θ @ ϕ[s]

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Approximate value function with linear TD
    θ_opt, n_iter = ilstd(env, V, γ, λ, θ, ϕ)
    V_opt = {s: V(θ_opt, s) for s in env.S}

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal parameters:')
    print(θ_opt)
