"""
Implementation of Conservative Policy Iteration (CPI)
=====================================================

"""

import numpy as np
from util.display import print_grid
from util.gridworld import GridWorld


def value_iteration_for_policy(S, R, P_π, γ):
    """Value iteration algorithm implementation

    Complexity: O(S^2 * A)
    """
    V_π = {s: 0.0 for s in S}
    while True:
        V_prime = {
            s: bellman_operator(S, R, P_π, V_π, γ, s)
            for s in S
        }
        if all(np.isclose(V_π[s], V_prime[s]) for s in S):
            break
        else:
            V_π = V_prime
    V_π = {s: (1 - γ) * V_π[s] for s in S}
    return V_π


def stochastic_policy(S, A, P):
    state_action_probs = {
        s: softmax([np.random.random() for _ in A])
        for s in S
    }
    P_π = {
        (s, s_prime): sum(
            p_a * P.get((s, a, s_prime), 0.0)
            for a, p_a in zip(A, state_action_probs[s])
        )
        for s in S
        for s_prime in S
    }
    def π(s):
        p_weights = state_action_probs[s]
        return np.random.choice(A, k=1, p=p_weights)[0]
    return π, P_π


def softmax(arr):
    logits = np.exp(arr)
    return logits / np.sum(logits)


def bellman_operator(S, R, P_π, V, γ, s):
    """Bellman operator for value iteration

    Complexity: O(S)
    """
    return  R[s] + γ * sum(
        P_π.get((s, s_prime), 0.0) * V[s_prime]
        for s_prime in S
    )


def q_function_for_policy(S, A, R, P, V_π):
    return {
        (s, a): (1 - γ) * R[s] + sum(
            P.get((s, a, s_prime), 0.0) * V_π[s_prime]
            for s_prime in S
        )
        for s in S
        for a in A
    }


def advantage_function(Q_π, V_π):
    return {
        (s, a): Q_π[(s, a)] - V_π[(s)]
        for s, a in Q_π.keys()
    }


def start_state_distribution(env):
    return {
        s: float(s == env.start)
        for s in env.S
    }


def discounted_future_state_distribution(S, P_π, µ, γ):
    d_π = {s: 1/len(S) for s in S}
    while True:
        d_π_prime = {
            s: µ[s] + sum(
                γ * P_π[(s, s_prime)] * d_π[s_prime]
                for s_prime in S
            )
            for s in S
        }
        if all(np.isclose(d_π[s], d_π_prime[s]) for s in S):
            break
        else:
            d_π = d_π_prime
    return {s: (1 - γ) * d_π[s] for s in S}


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75
    
    π, P_π = stochastic_policy(env.S, env.A, env.P)

    # Compute advantage function
    V_π = value_iteration_for_policy(env.S, env.R, P_π, γ)
    Q_π = q_function_for_policy(env.S, env.A, env.R, env.P, V_π)
    A_π = advantage_function(Q_π, V_π)
    µ = start_state_distribution(env)
    d_π = discounted_future_state_distribution(env.S, P_π, µ, γ)
    print(sum(d_π[s] for s in env.S))
