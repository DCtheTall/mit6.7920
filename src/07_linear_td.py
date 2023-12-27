"""
Implementation of Linear Temporal Difference Learning
=====================================================

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 γ : Discount factor
 λ : TD(λ) parameter, see above.
 V : State value function
 π : Agent policy
 ϕ : Non-linear features from environment
 θ : Model parameters

Result:
-------
Converged after 34671 iterations
Optimal value function:
0.6177128013971709	 0.5611515595534329	 0.4659076250681236	 1.6434627167495766	
0.021653612098461394	 -0.10589137805841786	 -0.3637464187666175	 -1.5848980990755137	
-0.2680605762893823	 -0.3789686317673623	 -0.50112480145532	 -0.5766327068491073	
-0.5153089807239112	 -0.5532044507574105	 -0.6089858309794415	 -0.6643094647755023	
Optimal parameters:
[ 0.31508458  0.68942319  0.5137557   0.50225388  0.2729931   1.0245976
 -2.84141224  1.12146221]

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld

np.random.seed(42)
random.seed(42)


N_FEATURES = 8


def initialize_parameters():
    return np.zeros((N_FEATURES,))


def linear_td(env, V, γ, λ, θ):
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
    V = lambda θ, s: θ @ env.ϕ[s]

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Approximate value function with linear TD
    θ_opt, n_iter = linear_td(env, V, γ, λ, θ)
    V_opt = {s: V(θ_opt, s) for s in env.S}

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal parameters:')
    print(θ_opt)
