"""
Implementation of Temporal Difference Learning
==============================================
This module implements the TD(0) temporal difference learning
algorithm.

"""

import numpy as np
import random
from util.gridworld import GridWorld


def td0_estimator(env, V, γ):
    N = {s: 0.0 for s in env.S}
    π = random_policy(env.S, env.A)
    n_iter = 0
    while True:
        n_iter += 1
        V_prime = update_value_function(env, V, N, π, γ)
        if all(np.isclose(V[s], V_prime[s]) for s in env.S):
            break
        V = V_prime
    return V, n_iter


def random_policy(S, A):
    """Use random policy for exploration"""
    def π(s):
        assert s in S
        return random.sample(list(A), k=1)[0]
    return π


def update_value_function(env, V, N, π, γ, T=100):
    """One episode of iterative temporal difference (TD) learning"""
    V = V.copy()
    s = env.start
    for _ in range(T):
        # Update per-stat learning rate
        N[s] += 1.0
        η = learning_rate(N[s])

        # Take action
        a = π(s)
        s_prime = env.step(s, a)

        # Temporal difference update step
        V[s] = V[s] + η * temporal_difference(V, env.R, γ, s, s_prime)

        # Stop if reached terminal node
        if env.is_terminal_state(s):
            break
        s = s_prime
    return V


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def temporal_difference(V, R, γ, s, s_prime):
    """Compute temporal difference term in current step"""
    return R.get(s, 0.0) + γ * V[s_prime] - V[s]


def print_grid(X):
    for y in range(3, -1, -1):
        print(*(str(X[(x, y)]) + '\t' for x in range(4)))


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Initialize value function
    V = {s: 0.0 for s in env.S}

    # Discount factor
    γ = 0.75

    # Apply TD(0) iteration
    V_opt, n_iter = td0_estimator(env, V, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
