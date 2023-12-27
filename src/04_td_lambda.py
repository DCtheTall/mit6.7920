"""
Implementation of TD(λ) with Elgibility Traces
==============================================
Implments TD(λ) for 4x4 Frozen Lake MDP.
If you set λ=0, then this becomes TD(0).
If you set λ=1, then this becomes Monte Carlo value iteration.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 γ : Discount factor
 λ : TD(λ) parameter, see above.
 V : State value function
 π : Agent policy

Result:
-------
Converged after 22029 iterations
Optimal value function:
-0.0016352802220612926	 0.023697866371326762	 0.29717431930371757	 3.6294914824661295	
-0.018186379847187788	 -0.08588570203442387	 -0.6776627981836699	 -3.7045548287999135	
-0.013380220025547724	 -0.045733345466611265	 -0.21421901527002882	 -0.6220527256181357	
-0.008149942776048804	 -0.022257505972251172	 -0.0784547448295287	 -0.16726770998643875

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld

random.seed(42)
np.random.seed(42)


def td_lambda(env, γ, λ):
    # Initialize value function
    V = {s: 0.0 for s in env.S}
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

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Apply TD(λ) iteration
    V_opt, n_iter = td_lambda(env, γ, λ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
