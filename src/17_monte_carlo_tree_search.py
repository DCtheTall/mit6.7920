"""
Implementation of Monte Carlo Tree Search (MCTS)
================================================
Implementation of MCTS on grid world for policy and value estimation.
The backprop stage uses TD(0) updates for Q-learning

While exporing the tree, the agent maximizes the Upper Confidence Bound (UCB)
instead of the Q-value.

The value function is computed using

V[s] = sum(π(a | s) * Q(s, a))

where π is a softmax policy using each state-action's Q-value.
The final policy displayed chooses the action which maximizes the Q-value.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 R : Reward function
 γ : Discount factor
 V : State value function
 Q : State-action value function
 π : Agent policy


Result:
-------
Optimal value function:
-0.25171246818913967	 -0.1009042245721357	 0.19878869372295796	 1.0
-0.31412406347088373	 -0.35864869920314807	 -0.40373236794589107	 -1.011183208370817
-0.3925497788050006	 -0.42624533606110376	 -0.5063655363045018	 -0.7083493531707351
-0.43107181034075226	 -0.46753220273203455	 -0.533413928971729	 -0.6295088675399246
Optimal policy:
Action.Up	 Action.Up	 Action.Up	 Action.Up
Action.Up	 Action.Up	 Action.Left	 Action.Left
Action.Right	 Action.Left	 Action.Left	 Action.Down
Action.Left	 Action.Left	 Action.Left	 Action.Down

"""

import numpy as np
from util.display import print_grid
from util.gridworld import GridWorld

np.random.seed(42)


N_STEPS = 30000
MAX_STEPS_PER_TRAJECTORY = 100
N_SIMULATIONS_PER_ACTION = 1
N_ACTIONS = 4


def monte_carlo_search_tree_policy(env, γ):
    tree = SearchTree(env)
    for step in range(N_STEPS):
        print('step', step)
        print('select')
        leaf_node = tree.selection_step(env)
        print('expand')
        tree.expand_and_simulate_step(env, leaf_node)
        print('backprop')
        tree.backprop_step(leaf_node, γ)
        print_grid(tree.softmax_policy_value_function(env))
    return tree.softmax_policy_value_function(env), tree.argmax_policy(env)



class Node:
    def __init__(self):
        self.children = {}
        self.parent = None

    def add_child(self, key, child):
        assert key not in self.children
        child.parent = self
        self.children[key] = child

    def is_leaf(self):
        return len(self.children) == 0


class StateNode(Node):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def upper_confidence_bound_policy_action(self, env):
        a = max(env.A, key=lambda a: self.children[a].upper_confidence_bound())
        return self.children[a]


class ActionNode(Node):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.q = 0.0
        self.update_count = 1

    def add_next_state(self, s):
        if s in self.children:
            return
        s_node = StateNode(s)
        self.add_child(s, s_node)

    def learning_rate(self):
        lr = 1.0 / self.update_count
        self.update_count += 1
        return lr

    def upper_confidence_bound(self):
        ret = self.q / self.update_count
        ret += np.sqrt(
            2.0 * np.log(self.parent_update_count()) / self.update_count)
        return ret

    def parent_update_count(self):
        node = self.parent.parent
        if node:
            return node.update_count
        return self.update_count


class SearchTree:
    def __init__(self, env):
        self.root = StateNode(env.start)

    def selection_step(self, env):
        cur = self.root
        while not cur.is_leaf():
            a_node = cur.upper_confidence_bound_policy_action(env)
            s_prime = env.step(cur.s, a_node.a)
            if s_prime in a_node.children:
                child = a_node.children[s_prime]
            else:
                child = StateNode(s_prime)
                a_node.add_child(s_prime, child)
            if env.is_terminal_state(cur.s):
                cur = child
                break
            cur = child
        return cur

    def expand_and_simulate_step(self, env, s_node):
        if not s_node.is_leaf():
            return
        for a in env.A:
            a_node = ActionNode(a)
            s_node.add_child(a, a_node)
            rewards = []
            for _ in range(N_SIMULATIONS_PER_ACTION):
                s = env.step(s_node.s, a)
                a_node.add_next_state(s)
                for _ in range(MAX_STEPS_PER_TRAJECTORY):
                    if env.is_terminal_state(s):
                        rewards.append(env.R[s])
                        break
                    s = env.step(s, a)
                    a = env.random_action()
                else:
                    rewards.append(0.0)
            a_node.q = np.mean(rewards)

    def backprop_step(self, node, γ):
        for a_prime_node in node.children.values():
            cur = node
            while cur.parent:
                a_node = cur.parent
                s_node = a_node.parent
                dt = env.R[s_node.s] - a_node.q + γ * a_prime_node.q
                a_node.q += a_node.learning_rate() * dt
                a_prime_node = a_node
                cur = s_node

    def q_function(self):
        Q = {}
        counts = {}
        frontier = [self.root]
        while frontier:  # BFS over tree
            new_frontier = []
            for node in frontier:
                s = node.s
                for child in node.children.values():
                    a = child.a
                    counts[(s, a)] = counts.get((s, a), 0) + 1
                    Q[(s, a)] = Q.get((s, a), 0.0) + child.q
                    new_frontier.extend(child.children.values())
            frontier = new_frontier
        return {sa: Q[sa] / counts[sa] for sa in Q.keys()}

    def softmax_policy_value_function(self, env):
        Q = self.q_function()
        V = {}
        for s in env.S:
            V[s] = 0.0
            for a in env.A:
                q = Q.get((s, a), 0.0)
                V[s] += np.exp(q) * q
            V[s] /= sum(np.exp(Q.get((s, a), 0.0)) for a in env.A)
        return V

    def argmax_policy(self, env):
        Q = self.q_function()
        π = {}
        for s in env.S:
            π[s] = max(env.A, key=lambda a: Q.get((s, a), -float('inf')))
        return π


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount
    γ = 0.75

    V_opt, π_opt = monte_carlo_search_tree_policy(env, γ)
    print('Optimal value function:')
    print_grid(V_opt)
    print('Optimal policy:')
    print_grid(π_opt)
