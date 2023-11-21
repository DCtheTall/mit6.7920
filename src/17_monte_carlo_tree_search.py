"""
Implementation of Monte Carlo Tree Search (MCTS)
================================================
Implementation of MCTS on grid world for value estimation.
The backprop stage uses TD(0) updates.

Result:
-------
Optimal value function:
0.02245143036306701	 0.04637681806199216	 0.911367752810623	 1.53125
0.011865366305115796	 0.10652907384464531	 0.3479200081373322	 -1.9869791666666665
0.006133351768491495	 0.028175828895737835	 0.09987025008349251	 -0.4573523346851437
0.003483766632065051	 -0.02373441407737876	 -0.010394279081501747	 -0.3401467869707275
Optimal policy:
Action.Down	 Action.Left	 Action.Up	 Action.Up
Action.Left	 Action.Up	 Action.Left	 Action.Up
Action.Left	 Action.Up	 Action.Left	 Action.Left
Action.Left	 Action.Up	 Action.Left	 Action.Down

"""

from util.display import print_grid
from util.gridworld import GridWorld


N_TRAJECTORIES = 200
MAX_STEPS_PER_TRAJECTORY = 100
N_TRAJECTORIES_PER_ACTION = 100
N_ACTIONS = 4


def monte_carlo_search_tree_policy(env, γ):
    tree = SearchTree(env)
    for step in range(N_TRAJECTORIES):
        print('step', step)
        print('select')
        node, states, rewards = tree.selection_step(env)
        print('expand')
        new_states, new_rewards = tree.expand_and_simulate_step(node, env)
        print('backprop')
        tree.backprop_step(states + new_states, rewards + new_rewards, γ)
        print_grid(tree.value_function())
    return tree.value_function(), tree.policy(env.A)



class Node:
    def __init__(self, s, A):
        self.s = s
        self.v = 0.0
        self.parent = None
        self.children = {a: [] for a in A}

    def is_leaf(self):
        return all(len(self.children[a]) == 0 for a in self.children)

    def add_child(self, a, s_prime, A):
        child = Node(s_prime, A)
        self.children[a].append(child)
        return child


class SearchTree:
    def __init__(self, env):
        self.root = Node(env.start, env.A)
        self.nodes = {env.start: self.root}
        self.update_counts = {env.start: 0}

    def contains_leaf(self):
        return any(node.is_leaf() for node in self.nodes.values())

    def selection_step(self, env):
        cur = self.root
        states = []
        rewards = []
        while not cur.is_leaf():
            rewards.append(env.R[cur.s])
            if env.is_terminal_state(cur.s):
                break
            a = max(
                cur.children,
                key=lambda a: sum(node.v for node in cur.children[a]),
            )
            s_prime = env.step(cur.s, a)
            states.append([cur.s, s_prime])
            if s_prime not in cur.children[a]:
                if s_prime in self.nodes:
                    child = self.nodes[s_prime]
                    cur.children[a].append(child)
                else:
                    self.nodes[s_prime] = cur.add_child(a, s_prime, env.A)
                cur = self.nodes[s_prime]
            else:
                cur = self.nodes[s_prime]
        return cur, states, rewards

    def expand_and_simulate_step(self, node, env):
        if not node.is_leaf():
            return [], []
        for a in env.A:
            for _ in range(N_TRAJECTORIES_PER_ACTION):
                s = env.step(node.s, a)
                if s in self.nodes:
                    child = self.nodes[s]
                    if child not in node.children[a]:
                        node.children[a].append(child)
                else:
                    self.nodes[s] = node.add_child(a, s, env.A)
                states = []
                rewards = [env.R[s]]
                for _ in range(MAX_STEPS_PER_TRAJECTORY):
                    rewards.append(env.R[s])
                    s_prime = env.step(s, a)
                    states.append([s, s_prime])
                    if env.is_terminal_state(s):
                        break
                    s = s_prime
                    a = env.random_action()
        return states, rewards

    def backprop_step(self, states, rewards, γ):
        rewards = get_discounted_rewards(rewards, γ)
        for (s, s_prime), r in zip(states, rewards):
            if s in self.nodes:
                if s_prime in self.nodes:
                    dt = r - self.nodes[s].v + γ * self.nodes[s_prime].v
                else:
                    dt = r - self.nodes[s].v
                self.nodes[s].v += self._learning_rate(s) * dt

    def _learning_rate(self, s):
        t = self.update_counts.get(s, 0) + 1
        self.update_counts[s] = t
        return 1.0 / t

    def value_function(self):
        return {s: node.v for s, node in self.nodes.items()}

    def policy(self, A):
        return {
            s: max(A, key=lambda a: sum(child.v for child in node.children[a]))
            for s, node in self.nodes.items()
        }


def get_discounted_rewards(rewards, γ):
    result = [None] * len(rewards)
    r_sum = 0.0
    for i in range(len(rewards)-1, -1, -1):
        r_sum *= γ
        r_sum += rewards[i]
        result[i] = r_sum
    return result


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount
    γ = 0.75

    V_opt, π_opt = monte_carlo_search_tree_policy(env, γ)
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal policy:')
    print_grid(π_opt)