"""
Implementation of Temporal Difference Learning
==============================================
This module implements the TD(0) temporal difference learning
algorithm on 4x4 GridWorld.

This is the first example of a model-free algorithm since
it does not need to know the state-action-state transition
probabilities.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 γ : Discount factor
 V : State value function
 π : Agent policy

Result
------
Converged after 46881 iterations
Optimal value function:
0.021579184136287444	 0.12948176959259905	 0.7126360689559769	 3.693637870689115	
-0.030275064893229415	 -0.11845293519983191	 -0.6384552031075401	 -3.755284825452571	
-0.03240013047460124	 -0.09668898617234213	 -0.335087540331132	 -0.9802221769218665	
-0.023149037932043984	 -0.05671401659967302	 -0.1507542748053437	 -0.31537988832232666

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld

random.seed(42)
np.random.seed(42)


def td0_estimator(env, γ):
    # Initialize value function
    V = {s: 0.0 for s in env.S}
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
        dt = temporal_difference(V, env.R, γ, s, s_prime)
        V[s] += η * dt

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
    return R[s] + γ * V[s_prime] - V[s]


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75

    # Apply TD(0) iteration
    V_opt, n_iter = td0_estimator(env, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
