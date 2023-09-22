"""
Implementation of value iteration algorithm
===========================================

Example (from Bard):
Consider an agent that is trying to navigate a maze to reach a goal state.
The agent can take two actions: Up or Down.
The agent receives a reward of 1 for reaching the goal state,
and a reward of 0 for all other states.
The agent's transition probabilities are as follows:
State   | Action   | Next State | Probability
------- | -------- | ---------- | --------
Start   | Up       | Room A     | 0.5
Start   | Down     | Room B     | 0.5
Room A  | Up       | Goal       | 0.25
Room A  | Down     | Room B     | 0.75
Room B  | Up       | Room A     | 0.25
Room B  | Down     | Goal       | 0.75

"""

import numpy as np


def value_iteration(S, A, R, P, γ, V):
    """Value iteration algorithm implementation"""
    n_iter = 0
    while True:
        n_iter += 1
        V_prime = update_value_function(S, A, R, P, γ, V)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        else:
            V = V_prime
    return V, n_iter



def update_value_function(S, A, R, P, γ, V):
    """One round of the value function update step"""
    return {
        s: max(bellman_operator(S, R, P, γ, V, s, a) for a in A)
        for s in S
    }


def bellman_operator(S, R, P, γ, V, s, a):
    """Bellman operator for value iteration"""
    return R[s] + γ * sum(
        transition_prob(P, s, a, s_prime) * V[s_prime]
        for s_prime in S
    )


def transition_prob(P, s, a, s_prime):
    """Get transition probability if specified, 0 otherwise"""
    try:
        return P[(s, a, s_prime)]
    except KeyError:
        return 0.0


if __name__ == '__main__':
    # Set of all states
    S = {'Start', 'Room A', 'Room B', 'Goal'}

    # Set of all actions
    A = {'Up', 'Down'}

    # State transition probabilities
    # Keyed on: (old state, action, new state)
    P = {
        ('Start', 'Up', 'Room A'): 0.5,
        ('Start', 'Down', 'Room B'): 0.5,
        ('Room A', 'Up', 'Goal'): 0.25,
        ('Room A', 'Down', 'Room B'): 0.75,
        ('Room B', 'Up', 'Room A'): 0.25,
        ('Room B', 'Down', 'Goal'): 0.75,
    }

    # Rewards
    R = {s: 1.0 if s == 'Goal' else 0.0 for s in S}

    # Initialize value function
    V = {s: 0.0 for s in S}

    # Discount factor
    γ = 0.9

    # Print final value function
    V_opt, n_iter = value_iteration(S, A, R, P, γ, V)
    print('Converged after', n_iter, 'iterations')
    print(V_opt)
