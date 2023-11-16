"""
Implementation of Monte Carlo Search Trees
==========================================

Result
------
Optimal policy:
Action.Up	 Action.Right	 Action.Up	 Action.Up	
Action.Right	 Action.Left	 Action.Left	 None	
Action.Left	 Action.Up	 Action.Left	 Action.Down	
Action.Up	 Action.Up	 Action.Right	 Action.Up

"""

import numpy as np
from util.display import print_grid
from util.gridworld import GridWorld


N_TRAJECTORIES = 10
MAX_STEPS_PER_TRAJECTORY = 100
N_TRAJECTORIES_PER_ACTION = 250
N_ACTIONS = 4


def monte_carlo_search_tree_policy(env):
    π = {}
    s = env.start
    for _ in range(N_TRAJECTORIES):
        for _ in range(MAX_STEPS_PER_TRAJECTORY):
            if s not in π:
                expand_and_search_leaf(env, π, s)
            a = π[s]
            s_prime = env.step(s, a)
            if env.is_terminal_state(s):
                s = env.start
                break
            s = s_prime
    return π


def expand_and_search_leaf(env, π, s):
    best_r, best_a = -float('inf'), None
    for a in env.A:
        cur_r = 0.0
        for _ in range(N_TRAJECTORIES_PER_ACTION):
            s_start = env.step(s, a)
            cur_r += random_rollout(env, s_start)
        if cur_r > best_r:
            best_r = cur_r
            best_a = a
    π[s] = best_a


def random_rollout(env, s_start):
    s = s_start
    r = 0.0
    for _ in range(MAX_STEPS_PER_TRAJECTORY):
        a = env.A[np.random.randint(0, N_ACTIONS)]
        r += env.R[s]
        s_prime = env.step(s, a)
        if env.is_terminal_state(s):
            return r
        s = s_prime
    return 0.0


if __name__ == '__main__':
    env = GridWorld(size=4)
    π_opt = monte_carlo_search_tree_policy(env)
    print('Optimal policy:')
    print_grid(π_opt)
