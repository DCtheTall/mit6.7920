"""
Implementation of Least Squares Temporal Difference (LSTD) Learning
===================================================================

TODO implement

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld


N_FEATURES = 9


def features(env):
    """Extract features for linear TD"""
    ϕ = {}
    for s in env.S:
        x, y = s
        xg, yg = env.goal
        xf, yf = env.failure
        l2_goal = ((x - xg) ** 2 + (y - yg) ** 2) ** 0.5
        l2_fail = ((x - xf) ** 2 + (y - yf) ** 2) ** 0.5
        ϕ[s] = np.array([
            float(x), float(y), # position
            (x ** 2.0 + y ** 2.0) ** 0.5, # L2 distance from origin
            float(x + y), # L1 norm from origin
            l2_goal, # L2 distance from goal
            l2_fail, # L2 distance from failure
            0.0 if s == env.goal else np.arccos((y - yg) / l2_goal), # angle wrt goal
            0.0 if s == env.failure else np.arccos((y - yf) / l2_fail), # angle wrt failure
            float(env.is_terminal_state(s)),
        ], dtype=np.float32)
    return ϕ


def initialize_parameters():
    return np.eye(N_FEATURES), np.zeros([N_FEATURES])


def lstd(env, γ, λ, A, b, ϕ):
    N = {s: 0.0 for s in env.S}
    π = random_policy(env.S, env.A)
    n_iter = 0
    while True:
        n_iter += 1
        A_prime, b_prime = update_parameters(env, N, π, γ, λ, A, b, ϕ)
        if (all(np.isclose(a, b)
                for a, b in zip(A.reshape((-1,)), A_prime.reshape((-1,))))
            and all(np.isclose(a, b) for a, b in zip(b, b_prime))):
            break
        A, b = A_prime, b_prime
    return A, b, n_iter


def random_policy(S, A):
    """Use random policy for exploration"""
    def π(s):
        assert s in S
        return random.sample(list(A), k=1)[0]
    return π


def update_parameters(env, N, π, γ, λ, A, b, ϕ, T=100):
    A, b = A.copy(), b.copy()
    s = env.start
    # Eligibility trace for TD(λ)
    z = np.zeros((N_FEATURES,))
    for _ in range(T):
        # Update per-stat learning rate
        N[s] += 1.0

        # Take action
        a = π(s)
        s_prime = env.step(s, a)

        z *= λ
        z += ϕ[s]

        A += np.outer(z, ϕ[s] - γ * ϕ[s_prime])
        b += env.R[s] * z

        # Stop if reached terminal node
        if env.is_terminal_state(s):
            break
        s = s_prime
    return A, b


def learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Non-linear features
    ϕ = features(env)

    # Initialize parameters for linear TD
    A, b = initialize_parameters()

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Approximate value function with linear TD
    A_opt, b_opt, n_iter = lstd(env, γ, λ, A, b, ϕ)

    # Initialize parameterized value function
    def V(A, b, s):
        return np.linalg.pinv(A) @ b @ ϕ[s]
    V_opt = {s: V(A_opt, b_opt, s) for s in env.S}

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal parameters:')
    print(A_opt)
    print(b_opt)
