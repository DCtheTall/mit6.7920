"""
Implementation of MuZero
========================

"""

from util.gridworld import GridWorld


N_TRAIN_STEPS = 10
N_TREE_SIMULATIONS_PER_STEP = 10


def muzero(env, γ):
    for _ in range(N_TRAIN_STEPS):
        # step 1: train representation and dynamics network using MCTS
        # step 2: train value and policy networks
        pass


def monte_carlo_tree_search(env):
    tree = SearchTree(env)
    for _ in range(N_TREE_SIMULATIONS_PER_STEP):
        pass


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


class SearchTree:
    def __init__(self, env):
        self.root = StateNode(env.start)



if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount
    γ = 0.75
