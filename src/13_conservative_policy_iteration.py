"""
Implementation of Conservative Policy Iteration (CPI)
=====================================================
CPI is an extension of policy iteration. This implementation is based
on the work in this paper:

https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf

This implementation is not model-free and requires the state-action-state
transition matrix for the MDP.

Result:
-------
Converged in 9429 iterations
Optimal policy:
Action.Up	 Action.Up	 Action.Up	 Action.Up	
Action.Up	 Action.Up	 Action.Left	 Action.Up	
Action.Up	 Action.Up	 Action.Left	 Action.Down	
Action.Up	 Action.Right	 Action.Left	 Action.Down	
Value function for optimal policy:
0.06824577808610179	 0.14071408908546346	 0.36629773073617006	 0.9999997610830491	
0.04229518366033263	 0.06363758921167877	 0.08855487329895684	 -0.9999997610830498	
0.022318357933392534	 0.027674662755099756	 0.0315890765007631	 -0.02064816966226994	
0.011723713051034035	 0.01355651691657542	 0.014660950631112097	 0.005846208903448203

"""

import numpy as np
from util.display import print_grid
from util.gridworld import GridWorld


def conservative_policy_iteration(env, γ, Ɛ):
    π, P_π = init_stochastic_policy(env.S, env.A, env.P)

    n_iter = 0
    while True:
        n_iter += 1
        # Compute advantage function
        V_π = value_iteration_for_policy(env.S, env.R, P_π, γ)
        Q_π = q_function_for_policy(env.S, env.A, env.R, env.P, V_π)
        A_π = advantage_function(Q_π, V_π)
        d_π = discounted_future_state_distribution(env.S, P_π, env.µ, γ)
        
        # Compute new policy and its advantage over current one
        π_prime, P_π_prime = epsilon_greedy_policy(env.S, env.A, env.P, A_π, Ɛ)
        A_π_prime = policy_advantage(env.S, env.A, d_π, A_π, π_prime)
        if A_π_prime < (2 * Ɛ / 3):
            break

        # Policy update
        π, P_π = policy_update(env.S, env.A, env.R, π, P_π, π_prime, P_π_prime, A_π_prime, γ, Ɛ)

    return optimal_policy(env.S, env.A, π), V_π, n_iter


def start_state_distribution(env):
    return {
        s: float(s == env.start)
        for s in env.S
    }


def init_stochastic_policy(S, A, P):
    state_action_probs = {
        s: softmax([np.random.random() for _ in A])
        for s in S
    }
    return policy_from_state_action_probs(S, A, P, state_action_probs)


def value_iteration_for_policy(S, R, P_π, γ):
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


def policy_from_state_action_probs(S, A, P, state_action_probs):
    P_π = {
        (s, s_prime): sum(
            p_a * P.get((s, a, s_prime), 0.0)
            for a, p_a in zip(A, state_action_probs[s])
        )
        for s in S
        for s_prime in S
    }
    π = {
        (s, a): state_action_probs[s][i]
        for s in S
        for i, a in enumerate(A)
    }
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


def discounted_future_state_distribution(S, P_π, µ, γ):
    d_π = {s: 0.0 for s in S}
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


def policy_advantage(S, A, d_π, A_π, π_prime):
    return sum(
        d_π[s] * sum(
            π_prime[(s, a)] * A_π[(s, a)]
            for a in A
        )
        for s in S
    )


def epsilon_greedy_policy(S, A, P, A_π, Ɛ):
    state_action_probs = {
        s: [Ɛ / len(A)] * len(A)
        for s in S
    }
    for s in S:
        i = np.argmax([A_π[(s, a)] for a in A])
        state_action_probs[s][i] += 1.0 - Ɛ
    return policy_from_state_action_probs(S, A, P, state_action_probs)


def policy_update(S, A, R, π, P_π, π_prime, P_π_prime, A_π_prime, γ, Ɛ):
    α = (1 - γ) * (A_π_prime - Ɛ / 3) / (4 * max(R.values()))
    for s in S:
        for a in A:
            π[(s, a)] *= 1 - α
            π[(s, a)] += α * π_prime[(s, a)]
        for s_prime in S:
            P_π[(s, s_prime)] *= 1 - α
            P_π[(s, s_prime)] += α * P_π_prime[(s, s_prime)]
    return π, P_π


def optimal_policy(S, A, π):
    return {
        s: A[np.argmax([π[(s, a)] for a in A])]
        for s in S
    }


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75

    # Epsilon for policy update
    Ɛ = 0.005

    π_opt, V_opt, n_iter = conservative_policy_iteration(env, γ, Ɛ)

    print('Converged in', n_iter, 'iterations')
    print('Optimal policy:')
    print_grid(π_opt)
    print('Value function for optimal policy:')
    print_grid(V_opt)
