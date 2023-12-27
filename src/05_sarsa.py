"""
Implementation of State-Action-Reward-State-Action (SARSA)
==========================================================
SARSA is equivalent to the TD(0) algorithm in this repository
except it uses state-action (Q) values, instead of a value function
depending only on state.

SARSA is an on-policy learning algorithm, meaning that it uses the
same policy during exploration as it does for selecting the next
action for temporal difference learning.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 P : State-action-state transition probabilities
 γ : Discount factor
 Q : State-action value function
 π : Agent policy

Result:
-------
Converged after 129660 iterations
Optimal policy:
Action.Up	 Action.Up	 Action.Up	 Action.Left	
Action.Left	 Action.Left	 Action.Left	 Action.Up	
Action.Left	 Action.Left	 Action.Left	 Action.Down	
Action.Left	 Action.Left	 Action.Down	 Action.Down

"""

import numpy as np
from util.display import print_grid
from util.gridworld import GridWorld

np.random.seed(42)


def sarsa(env, γ):
    # Initialize Q function
    Q = {(s, a): 0.0 for s in env.S for a in env.A}
    t = 0
    N = {s: 0.0 for s in env.S}
    while True:
        t += 1
        π = greedy_softmax_policy(env.S, env.A, Q, τ=1.0)
        Q_prime = update_q_function(env, Q, N, π, γ)
        if all(np.isclose(Q[(s, a)], Q_prime[(s, a)])
               for s in env.S for a in env.A):
            break
        Q.update(Q_prime)
    return Q, t


def greedy_softmax_policy(S, A, Q, τ):
    """Softmax-based exploration policy"""
    p = {}
    for s in S:
        numerators = {}
        for a in A:
            numerators[a] = np.exp(Q[(s, a)] / τ)
        denom = sum(numerators.values())
        p[s] = [numerators[a] / denom for a in A]
    def π(s):
        assert s in S
        return np.random.choice(A, 1, p=p[s])[0]
    return π


def update_q_function(env, Q, N, π, γ, T=100):
    """One episode of iterative temporal difference (TD) learning"""
    Q = Q.copy()
    s = env.start
    a_prime = π(s)
    for _ in range(T):
        # Update learning rate
        N[s] += 1.0
        η = learning_rate(N[s])

        # Take action
        a = a_prime
        s_prime = env.step(s, a)

        # Temporal difference update step
        a_prime = π(s_prime)
        dt = temporal_difference(Q, env.R, γ, s, a, s_prime, a_prime)
        Q[(s, a)] += η * dt
        if env.is_terminal_state(s):
            break
        s = s_prime
    return Q


def learning_rate(t):
    """Decaying learning rate.

    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def temporal_difference(Q, R, γ, s, a, s_prime, a_prime):
    """Compute temporal difference term in current step"""
    return R[s] + γ * Q[(s_prime, a_prime)] - Q[(s, a)]


def optimal_policy(S, A, Q):
    return {s: max(A, key=lambda a, s=s: Q[(s, a)]) for s in S}


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75

    # Apply SARSA
    Q_opt, n_iter = sarsa(env, γ)

    # Optimal policy
    π_opt = optimal_policy(env.S, env.A, Q_opt)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal policy:')
    print_grid(π_opt)
    print('Best first action:', π_opt[(0, 0)])
