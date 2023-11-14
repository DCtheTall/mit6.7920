"""
Implementation of Linear Temporal Difference Learning
=====================================================

Result:
-------
Converged after 34237 iterations
Optimal value function:
0.6197100133372377	 0.5675408488428824	 0.47423490398675655	 1.663224340644876
0.020777519484171503	 -0.10161313610557865	 -0.3558205875475948	 -1.6019315639454315
-0.2759446825586251	 -0.3799280613453073	 -0.4932909168900485	 -0.5543027805244976
-0.5303302432843998	 -0.5598498893459154	 -0.6060757625938406	 -0.6502573562734688
Optimal parameters:
[ 0.31952767  0.70074957  0.51959239  0.51013862  0.25749992  1.00574523
 -2.8741527   1.16939335]

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld


N_FEATURES = 8


def initialize_parameters():
    return np.zeros((N_FEATURES,))


def ilstd(env, V, γ, λ, θ):
    N = {s: 0.0 for s in env.S}
    π = random_policy(env.S, env.A)
    n_iter = 0
    while True:
        n_iter += 1
        θ_prime = update_value_function(env, V, N, π, γ, λ, θ)
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


def update_value_function(env, V, N, π, γ, λ, θ, T=100):
    θ = θ.copy()
    s = env.start
    z = {}
    for _ in range(T):
        # Update per-stat learning rate
        N[s] += 1.0

        # Take action
        a = π(s)
        s_prime = env.step(s, a)

        z[s] = z.get(s, 0.0) + 1.0
        dt = temporal_difference(V, env.R, γ, θ, s, s_prime)

        # TD(λ) update step
        for sz in z.keys():
            N[sz] = N.get(sz, 0) + 1
            η = learning_rate(N[sz])
            θ += η * z[sz] * dt * env.ϕ[sz]
            z[sz] *= λ * γ

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

    # Initialize parameters for linear TD
    θ = initialize_parameters()

    # Initialize parameterized value function
    def V(θ, s):
        return θ @ env.ϕ[s]

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Approximate value function with linear TD
    θ_opt, n_iter = ilstd(env, V, γ, λ, θ)
    V_opt = {s: V(θ_opt, s) for s in env.S}

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal parameters:')
    print(θ_opt)
