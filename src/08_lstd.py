"""
Implementation of Least Squares Temporal Difference (LSTD) Learning
===================================================================

Result:
-------
Converged after 303411 iterations
Optimal value function:
0.0815413549354722	 0.42268815258178094	 0.6212195714057054	 5.132912995583676
-0.03744659525134075	 -0.017887611347628507	 -0.5477244130831016	 -3.594339774829386
-0.101390363931934	 -0.15203792123759108	 -0.46006487600401025	 -0.8588160798373359
-0.001268747735318355	 -0.03851519721807484	 -0.19012318896042935	 -0.40382995907831454
Optimal parameters:
[[ 834468.53459948  340196.80894964  695545.4150686   587332.17177398
   651304.1631874   488628.20149203 1387639.16128565 1350051.00655491]
 [ 314458.885472   1038054.47909603  801179.9263062   676256.18228379
   706092.7494599   673978.92142002 1139255.94888316  942584.50046712]
 [ 665602.57140073  799569.19067048  893401.46718526  732585.88103699
   848652.11340374  730912.82007186 1528162.05573561 1400131.89841926]
 [ 574463.21003558  689125.14402281  748362.67068595  631795.17703091
   678698.45632395  581303.56145534 1263447.55508415 1146317.75350979]
 [ 491418.09669606  569768.52301063  690666.14501515  530593.30985382
  1996765.30352784 1626091.04817562 2504010.46622424 2324702.39116157]
 [ 348565.36221852  564744.30405312  597816.09025541  456654.83313492
  1618124.64264588 1370856.62078866 1997881.79978616 1831884.65697838]
 [1220990.79444697  982200.12996673 1372161.21072022 1101595.46220519
  2653213.56328547 2125029.34914398 3843014.85935162 3611210.33350487]
 [1073371.95853331  724125.26423676 1144238.98029861  898748.61138404
  2358174.82955419 1879026.0081168  3375046.00869925 3252787.39790395]]
[-203898.8827356   -57379.67114586 -143421.32441862 -130639.2769407
  -99865.67658899  -23404.1325249  -392205.90075214 -178990.74591267]

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld


N_FEATURES = 8


def initialize_parameters():
    return np.eye(N_FEATURES), np.zeros([N_FEATURES])


def lstd(env, γ, λ, A, b):
    N = {s: 0.0 for s in env.S}
    π = random_policy(env.S, env.A)
    n_iter = 0
    while True:
        n_iter += 1
        A_prime, b_prime = update_parameters(env, N, π, γ, λ, A, b)
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


def update_parameters(env, N, π, γ, λ, A, b, T=100):
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
        z += env.ϕ[s]

        A += np.outer(z, env.ϕ[s] - γ * env.ϕ[s_prime])
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

    # Initialize parameters for linear TD
    A, b = initialize_parameters()

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Approximate value function with linear TD
    A_opt, b_opt, n_iter = lstd(env, γ, λ, A, b)

    # Initialize parameterized value function
    def V(A, b, s):
        return np.linalg.pinv(A) @ b @ env.ϕ[s]
    V_opt = {s: V(A_opt, b_opt, s) for s in env.S}

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal parameters:')
    print(A_opt)
    print(b_opt)
