"""
Implementation of TD(λ) with Elgibility Traces
==============================================
Implments TD(λ) for 4x4 Frozen Lake MDP.
If you set λ=0, then this becomes TD(0).

Result:
-------
Converged after 16006 iterations
Optimal value function:
0.033723677017174	 0.13119574526971972	 0.6235409823432285	 3.598167796718273
0.0032197997971012207	 -0.01132062236114739	 -0.16839340098596253	 -3.680209919662842
-0.005925147282414969	 -0.029736214449579194	 -0.14119541625028553	 -0.7290011110784664
-0.00648537028429082	 -0.022989353956549052	 -0.08229763495664257	 -0.26665364084401494

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld


def td_lambda(env, V, γ, λ):
    π = random_policy(env.S, env.A)
    # State update counter, used for learning rate
    N = {}
    t = 0
    while True:
        t += 1
        V_prime = update_value_function(env, V, N, π, γ, λ)
        if all(np.isclose(V[s], V_prime[s]) for s in env.S):
            break
        V = V_prime
    return V, t


def random_policy(S, A):
    """Use random policy for exploration"""
    def π(s):
        assert s in S
        return random.sample(list(A), k=1)[0]
    return π


def learning_rate(t):
    """Decaying learning rate.

    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def update_value_function(env, V, N, π, γ, λ, T=100):
    """One episode of iterative temporal difference (TD) learning"""
    V = V.copy()
    s = env.start
    # Eligibility traces
    z = {}
    for _ in range(T):
        a = π(s)
        s_prime = env.step(s, a)
        z[s] = z.get(s, 0.0) + 1.0
        dt = temporal_difference(V, env.R, γ, s, s_prime)
        for sz in z.keys():
            # Temporal difference update step
            N[sz] = N.get(sz, 0) + 1
            η = learning_rate(N[sz])
            V[sz] += η * z[sz] * dt
            z[sz] *= λ * γ
        if env.is_terminal_state(s):
            break
        s = s_prime
    return V


def temporal_difference(V, R, γ, s, s_prime):
    """Compute temporal difference term in current step"""
    return R[s] + γ * V[s_prime] - V[s]


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Initialize value function
    V = {s: 0.0 for s in env.S}

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Apply TD(λ) iteration
    V_opt, n_iter = td_lambda(env, V, γ, λ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
