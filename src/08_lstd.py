"""
Implementation of Least Squares Temporal Difference (LSTD) Learning
===================================================================

Result:
-------
Converged after 306727 iterations
Optimal value function:
0.08230684041545826	 0.4246897769685453	 0.6244879248650408	 5.140573387423212
-0.03692957233910154	 -0.01650878479563766	 -0.545922826045412	 -3.5920099949973356
-0.10129394641173659	 -0.1513275916490555	 -0.45920893934113327	 -0.858405607610629
-0.0014306927962319301	 -0.038276842252042434	 -0.18969110944334583	 -0.4033979521225268
Optimal parameters:
[[1.35277206e+07 5.49534273e+06 1.59361540e+07 1.90230623e+07
  2.11129482e+07 1.58364468e+07 1.76650853e+07 1.71904381e+07]
 [5.08572923e+06 1.67688402e+07 1.83060240e+07 2.18545684e+07
  2.28271906e+07 2.17777675e+07 1.44659459e+07 1.19686864e+07]
 [1.52559861e+07 1.82645784e+07 2.89093514e+07 3.35205645e+07
  3.88557128e+07 3.34457672e+07 2.74852315e+07 2.51884814e+07]
 [1.86134488e+07 2.22641819e+07 3.42421780e+07 4.08776317e+07
  4.39401388e+07 3.76142143e+07 3.21310312e+07 2.91591245e+07]
 [1.59378213e+07 1.84203826e+07 3.16296835e+07 3.43582039e+07
  1.29442738e+08 1.05402145e+08 6.37450501e+07 5.91834387e+07]
 [1.13047288e+07 1.82491016e+07 2.73604597e+07 2.95538304e+07
  1.04884589e+08 8.88349872e+07 5.08532246e+07 4.66310801e+07]
 [1.55470235e+07 1.24664288e+07 2.46794920e+07 2.80134523e+07
  6.75370159e+07 5.40842633e+07 3.84158361e+07 3.61029067e+07]
 [1.36722644e+07 9.19135443e+06 2.05877017e+07 2.28636188e+07
  6.00343883e+07 4.78302397e+07 3.37444017e+07 3.25245700e+07]]
[ -824324.74886158  -230743.99619352  -819255.69580472 -1055068.7450549
  -807606.96904475  -188961.30674559 -1245613.69877088  -568839.01705657]

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld


N_FEATURES = 8


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
            float(abs(x - xg) + abs(y - yg)), # L1 distance from goal
            float(abs(x - xf) + abs(y - yf)), # L1 distance from failure
            0.0 if s == env.goal else np.arccos((y - yg) / l2_goal), # angle wrt goal
            0.0 if s == env.failure else np.arccos((y - yf) / l2_fail), # angle wrt failure
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
