"""
Implementation of TD(λ) with Elgibility Traces
==============================================
Implments TD(λ) for 4x4 Frozen Lake MDP.

"""

import numpy as np
import random


TERMINAL_NODES = {(3, 3), (3, 2)}


def td_lambda(S, A, R, V, γ, λ):
    t = 0
    while True:
        t += 1
        η = learning_rate(t)
        V_prime = update_value_function(S, A, R, V, γ, λ, η)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        V = V_prime
    return V, t


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def update_value_function(S, A, R, V, γ, λ, η):
    """One step of iterative temporal difference (TD) learning"""
    π = random_policy(S, A)
    V_prime = {}
    # Eligibility traces
    z = {s: 0.0 for s in S}
    for s in S:
        a = π[s]
        s_prime = sample_next_state(S, s, a)
        # Temporal difference update step
        dt = episode_update(V, R, γ, λ, π, s, s_prime, z)
        V_prime[s] = V[s] + η * z[s] * dt
    return V_prime


def random_policy(S, A):
    """Use random policy to show TD{0} still converges"""
    return {s: random.sample(list(A), 1)[0] for s in S}


# Memoization table for function below
T = {}

def sample_next_state(S, s, a):
    """Sample next state from MDP
    
    TD(1) algorithm treats this as a black box.
    """
    if s in TERMINAL_NODES:
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


def episode_update(V, R, γ, λ, π, s, s_prime, z, T=100):
    discount = 1.0
    ret = 0.0
    for _ in range(T):
        ret += discount * temporal_difference(V, R, γ, s, s_prime)
        update_eligibility_trace(S, s, z, γ, λ)
        if s_prime in TERMINAL_NODES:
            break
        s = s_prime
        a = π[s]
        s_prime = sample_next_state(S, s, a)
        discount *= γ * λ
    return ret


def update_eligibility_trace(S, s, z, γ, λ):
    for s_z in S:
        if s == s_z:
            z[s] = 1.0 + γ * λ * z[s]
        else:
            z[s_z] = γ * λ * z[s_z]


def temporal_difference(V, R, γ, s, s_prime):
    """Compute temporal difference term in current step"""
    return R.get(s, 0.0) + γ * V[s_prime] - V[s]


if __name__ == '__main__':
    # Set of all states, 4x4 grid
    S = {
        (i // 4, i % 4)
        for i in range(16)
    }

    # Set of all actions
    A = {'Up', 'Down', 'Left', 'Right'}

    # Rewards
    # Upper right corner is the goal
    # One cell below right corner is a failure
    R = {(3, 3): 1.0, (3, 2): -1.0}

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Apply TD(λ) iteration
    V_opt, n_iter = td_lambda(S, A, R, V, γ, λ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print(V_opt)
