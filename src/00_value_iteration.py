"""
Implementation of Value Iteration
=================================
Implementation of value iteration for 4x4 GridWorld.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 P : State-action-state transition probabilities
 γ : Discount factor

Result:
-------
Converged after 51 iterations
Optimal value function:
0.3140523824640529	 0.6281069041988825	 1.542700810265115	 3.999997734713374
0.2185574408028565	 0.34162206297324693	 0.5198260378022203	 -3.999997734713374
0.12825679779932486	 0.16621504491365266	 0.1949834098175992	 0.08485459602339823
0.07165242928353827	 0.086702543420937	 0.09389463568898715	 0.059582405402386074
Optimal policy:
Action.Up	 Action.Right	 Action.Up	 Action.Up
Action.Up	 Action.Up	 Action.Left	 Action.Up
Action.Up	 Action.Up	 Action.Left	 Action.Down
Action.Up	 Action.Right	 Action.Left	 Action.Up
Best first action: Action.Up

"""

import numpy as np
from util.display import print_grid
from util.gridworld import GridWorld


def value_iteration(S, A, R, P, γ):
    """Value iteration algorithm implementation

    Complexity: O(S^2 * A)
    """
    # Initialize value function
    V = {s: 0.0 for s in S}
    n_iter = 0
    while True:
        n_iter += 1
        V_prime = update_value_function(S, A, R, P, V, γ)
        if all(np.isclose(V[s], V_prime[s]) for s in S):
            break
        else:
            V = V_prime
    return V, n_iter



def update_value_function(S, A, R, P, V, γ):
    """One round of the value function update step

    Complexity: O(S^2 * A)
    """
    return {
        s: max(bellman_operator(S, R, P, V, γ, s, a) for a in A)
        for s in S
    }


def bellman_operator(S, R, P, V, γ, s, a):
    """Bellman operator for value iteration

    Complexity: O(S)
    """
    return R[s] + γ * sum(
        P.get((s, a, s_prime), 0) * V[s_prime]
        for s_prime in S
    )


def optimal_bellman_operator_policy(S, A, R, P, V, γ):
    """Update the policy using the newly computed value function, V

    π[s] = argmax(bellman_operator(R, γ, V, s, a) for a in A)
    """
    π = {}
    for s in S:
        possible_actions = {
            a: bellman_operator(S, R, P, V, γ, s, a)
            for a in A
        }
        π[s] = max(possible_actions, key=possible_actions.get)
    return π


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Set of all states, 4x4 grid world
    S = env.S

    # Set of all actions
    A = env.A

    # Transition probabiltiy table
    P = env.P

    # Rewards
    R = env.R

    # Discount factor
    γ = 0.75

    # Apply value iteration
    V_opt, n_iter = value_iteration(S, A, R, P, γ)

    # Derive optimal policy from value function using the optimal
    # Bellman operator.
    π_opt = optimal_bellman_operator_policy(S, A, R, P, V_opt, γ)

    # Display results
    print('Converged after', n_iter, 'iterations')
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal policy:')
    print_grid(π_opt)
    print(
        'Best first action:',
        max(A, key=lambda a: bellman_operator(S, R, P, V_opt, γ, (0, 0,), a))
    )
