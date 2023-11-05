"""
Implementation of Linear Temporal Difference Learning
=====================================================

Result:
-------
Converged after 23876 iterations
Optimal value function:
0.31046747178948664	 0.5071062847309171	 0.5198902963294577	 3.871262610983604
-0.3325531272310006	 -0.3689040577079704	 -0.8819454836249915	 -3.796414565562788
-0.19498098188606003	 -0.2727660311042817	 -0.5563914380429362	 -0.8165590136993452
0.06160465688481646	 -0.050828737704709685	 -0.22650545242896605	 -0.4194090843061706
Optimal parameters:
[ 0.58344505 -0.12527267 -0.13588442  0.4581726   0.38378023  0.32421862
 -2.38124734  0.8099467 ]

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
    s = (0, 0)
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
