"""
Implementation of Monte Carlo Tree Search (MCTS)
================================================
Implementation of MCTS on grid world for value estimation.
The backprop stage uses TD(0) updates.

Result:
-------
Optimal value function:
0.2844944920415114	 0.6007109956912923	 1.2423077098355912	 1.75	
0.1539648532139492	 0.21868180064963455	 0.11386325660798145	 -1.9108428955078125	
0.038864991271832586	 0.05095245831363437	 -0.15978539749580978	 -0.585549731683253	
0.008259092778398579	 -0.01081149798324792	 -0.1331536793890871	 -0.22872147109426275

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
    return tree.value_function()



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
                    dt = r - self.nodes[s].v # + γ * r
                self.nodes[s].v += self._learning_rate(s) * dt

    def _learning_rate(self, s):
        t = self.update_counts.get(s, 0) + 1
        self.update_counts[s] = t
        return 1.0 / t

    def value_function(self):
        return {s: node.v for s, node in self.nodes.items()}


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

    V_opt = monte_carlo_search_tree_policy(env, γ)
    print('Optimal value function:')
    print_grid(V_opt)
