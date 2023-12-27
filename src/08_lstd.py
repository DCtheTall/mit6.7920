"""
Implementation of Least Squares Temporal Difference (LSTD) Learning
===================================================================

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 γ : Discount factor
 λ : TD(λ) parameter, see above.
 V : State value function
 π : Agent policy
 ϕ : Non-linear features from environment
 A, b : Model parameters

Result:
-------
Converged after 305451 iterations
Optimal value function:
0.08376946711617117	 0.4267784987540376	 0.62703915237774	 5.143498106637905	
-0.03717121696463854	 -0.0163258087758682	 -0.5457113640309985	 -3.5903547988245466	
-0.10125313442367889	 -0.1512067604565246	 -0.4596550981875105	 -0.8599743919194456	
-0.001066339979304498	 -0.03859784048930259	 -0.19080646427921755	 -0.4054361529697035	
Optimal parameters:
[[ 840312.62830712  342327.9583593   700317.99490535  591319.79333268
   656261.29312885  492349.10723241 1397700.59270088 1359929.07977956]
 [ 316694.4105095  1044355.51813852  806153.77655     680524.46432384
   710524.2995757   678002.61142659 1146825.96938918  948669.9539562 ]
 [ 670275.39247054  804346.96041274  899152.1137663   737311.17644302
   854526.8099551   735793.4360287  1538782.86214262 1409881.69894081]
 [ 578503.01940825  693341.2382488   753235.88572629  635923.12883046
   683392.79635268  585175.85932884 1272263.28104498 1154299.51686673]
 [ 495257.15348937  573356.8697504   695479.01771377  534307.01162048
  2011359.95113904 1637922.60927178 2522401.05532244 2341751.21552653]
 [ 351266.72180273  568151.38144869  601811.3344346   459709.05162489
  1629921.93216517 1380702.52896161 2012522.27199974 1845277.66694265]
 [1229930.61742117  988320.39570955 1381553.61921986 1109125.50656352
  2672589.88924198 2140464.25567129 3870899.35899019 3637491.59523518]
 [1081231.81754581  728600.88376117 1152109.62133594  904916.35065261
  2375490.99865032 1892786.46483583 3399601.77857456 3276564.2323718 ]]
[-205104.88076347  -57097.77264615 -143978.43789396 -131101.32670479
 -100453.16146694  -23385.25033196 -394676.01696024 -180324.99951568]

"""

import numpy as np
import random
from util.display import print_grid
from util.gridworld import GridWorld

np.random.seed(42)
random.seed(42)


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
    V = lambda A, b, s: np.linalg.pinv(A) @ b @ env.ϕ[s]
    V_opt = {s: V(A_opt, b_opt, s) for s in env.S}

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal parameters:')
    print(A_opt)
    print(b_opt)
