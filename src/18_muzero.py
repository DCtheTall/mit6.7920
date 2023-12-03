"""
Implementation of MuZero
========================

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util.gridworld import GridWorld
from util.jax import MLP, Metrics, TrainState

jax.config.update('jax_enable_x64', True)


N_TRAIN_STEPS = 1
N_TRAJECTORIES_PER_STEP = 1
N_SIMULATIONS_PER_SEARCH = 1
N_UPDATES_PER_STEP = 1
BATCH_SIZE = 4
N_UNROLL_STEPS = 5
N_TD_STEPS = 10
N_FEATURES = 8
N_ACTIONS = 4
N_HIDDEN_LAYERS = 2
N_HIDDEN_FEATURES = 2 * N_FEATURES
N_REPRESENTATION_FEATURES = N_FEATURES // 2
LEARNING_RATE = 1e-3
ROOT_DIRICHLET_ALPHA = 3e-2
ROOT_EXPLORATION_FRACTION = 0.25
MAX_TRAJECTORIES_PER_STEP = 100
REPLAY_MEMORY_MAX_SIZE = 1000


def muzero(env, γ):
    net = Network()
    rng = jax.random.key(42)
    state = create_train_state(net, rng, features=N_FEATURES,
                               η=LEARNING_RATE, β=0.9)
    del rng
    memory = ReplayMemory(maxlen=REPLAY_MEMORY_MAX_SIZE,
                          batch_size=BATCH_SIZE)

    for _ in range(N_TRAIN_STEPS):
        # Step 1: Run MCTS simulation and collect history
        for _ in range(N_TRAJECTORIES_PER_STEP):
            history = run_self_play(env, state, γ)
            memory.save_game(history)

        # Step 2: Train the model
        for _ in range(N_UPDATES_PER_STEP):
            batch = memory.sample_batch(γ, unroll_steps=N_UNROLL_STEPS,
                                        td_steps=N_TD_STEPS)
            print(batch)


def run_self_play(env, state, γ):
    s = env.start
    history = GameHistory(env)
    for step in range(MAX_TRAJECTORIES_PER_STEP):
        tree = SearchTree(env, s)
        x = env.ϕ[s]
        policy_logits, q_values, hidden = state.apply_fn(
            {'params': state.params}, x, method=Network.initial_inference)
        tree.expand_node(tree.root, policy_logits, q_values, hidden)
        tree.add_exploration_noise()
        monte_carlo_tree_search(tree, state, γ)

        a = tree.softmax_action_with_temperature(int(step < 20))
        history.save_step(s, a, tree)
        s_prime = env.step(s, a)

        if env.is_terminal_state(s):
            break
        s = s_prime
    return history


def monte_carlo_tree_search(tree, state, γ):
    for _ in range(N_SIMULATIONS_PER_SEARCH):
        leaf_node = tree.selection_step()
        h = leaf_node.parent.parent.hidden
        a = leaf_node.parent.int_action()
        policy_logits, q_values, hidden = state.apply_fn(
            {'params': state.params}, h, np.array([a]),
            method=Network.recurrent_inference)
        tree.expand_node(leaf_node, policy_logits, q_values, hidden)
        tree.backprop_step(leaf_node, γ)


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
        self.hidden = None
    
    def upper_confidence_bound_policy_action(self, env):
        a = max(env.A, key=lambda a: self.children[a].upper_confidence_bound())
        return self.children[a]


class ActionNode(Node):
    def __init__(self, a, q, prior):
        super().__init__()
        self.a = a
        self.q = q
        self.prior = prior
        self.visit_count = 1
    
    def upper_confidence_bound(self):
        ret = self.q / self.visit_count
        ret += self.prior * np.sqrt(
            2.0 * np.log(self.parent_visit_count() / self.visit_count))
        return ret
    
    def parent_visit_count(self):
        node = self.parent.parent
        if node:
            return node.visit_count
        return self.visit_count
    
    def learning_rate(self):
        lr = 1.0 / self.visit_count
        self.visit_count += 1
        return lr
    
    def int_action(self):
        return self.a.value


class SearchTree:
    def __init__(self, env, start):
        self.env = env
        self.root = StateNode(start)

    def expand_node(self, node, policy_logits, q_values, hidden):
        if not node.is_leaf():
            return
        node.hidden = hidden
        for i, (a, q) in enumerate(zip(self.env.A, q_values)):
            a_node = ActionNode(a, q, policy_logits[i])
            node.add_child(a, a_node)

    def add_exploration_noise(self):
        noise = np.random.dirichlet([ROOT_DIRICHLET_ALPHA] * len(self.env.A))
        frac = ROOT_EXPLORATION_FRACTION
        for a, n in zip(self.env.A, noise):
            p = self.root.children[a].prior
            self.root.children[a].prior = (1 - frac) * p + frac * n

    def selection_step(self):
        cur = self.root
        while not cur.is_leaf():
            a_node = cur.upper_confidence_bound_policy_action(self.env)
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
    
    def softmax_action_with_temperature(self, temp):
        if temp == 0:
            return max(self.env.A, key=lambda a: self.root.children[a].q)
        logits = np.array(
            [np.exp(a_node.q) for a_node in self.root.children.values()])
        logits /= sum(logits)
        return np.random.choice(self.env.A, p=logits)


class GameHistory:
    def __init__(self, env):
        self.env = env
        self.states = []
        self.actions = []
        self.rewards = []
        self.root_q_values = []
        self.visit_counts = []

    def save_step(self, s, a, tree):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(self.env.R[s])
        q_values = np.array([child.q for child in tree.root.children.values()])
        self.root_q_values.append(q_values)
        visit_counts = np.array(
            [child.visit_count for child in tree.root.children.values()],
            dtype=np.float64)
        visit_counts /= sum(visit_counts)
        self.visit_counts.append(visit_counts)

    @property
    def length(self):
        return len(self.rewards)
    
    def make_targets(self, start_pos, unroll_steps, td_steps, γ):
        target_q = []
        target_p = []
        for cur_idx in range(start_pos, start_pos + unroll_steps):
            bootstrap_idx = cur_idx + td_steps
            if bootstrap_idx >= len(self.root_q_values):
                q_values = [0.0 for _ in self.env.A]
            else:
                q_values = self.root_q_values[bootstrap_idx] * (γ ** td_steps)
            a = self.actions[cur_idx].value
            for i, r in enumerate(self.rewards):
                q_values[a] += r * (γ ** i)
            target_q.append(q_values)
            target_p.append(self.visit_counts[cur_idx])
        return target_q, target_p


class ReplayMemory:
    def __init__(self, maxlen, batch_size):
        self.maxlen = maxlen
        self.buffer = [None] * maxlen
        self.index = 0
        self.length = 0
        self.batch_size = batch_size

    def save_game(self, game_history):
        self.buffer[self.index] = game_history
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample_batch(self, γ, unroll_steps, td_steps):
        games = [self.sample_game() for _ in range(self.batch_size)]
        positions = [np.random.randint(0, game.length - unroll_steps)
                     for game in games]
        targets = [game.make_targets(pos, unroll_steps, td_steps, γ)
                   for (pos, game) in zip(positions, games)]
        return [
            {
                'state': game.states[pos],
                'actions': game.actions[pos:pos+unroll_steps],
                'target_q_values': [t[0] for t in targets],
                'target_policy': [t[1] for t in targets],
            }
            for pos, game in zip(positions, games)
        ]

    def sample_game(self):
        i = np.random.randint(0, self.length)
        return self.buffer[i]


class Representation(nn.Module):
    hidden_dim: int
    n_layers: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(self.output_dim,
                     dtype=jnp.float64)(x)
        return x


class Dynamics(nn.Module):
    input_dim: int
    hidden_dim: int
    n_layers: int
    n_actions: int

    @nn.compact
    def __call__(self, h, a):
        x = jnp.concatenate(
            [h, nn.one_hot(a[...,0], self.n_actions, dtype=jnp.float64)],
            axis=-1)
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(self.input_dim,
                     dtype=jnp.float64)(x)
        return x


class QValue(nn.Module):
    hidden_dim: int
    n_layers: int
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x


class Policy(QValue):
    @nn.compact
    def __call__(self, x):
        x = super().__call__(x)
        x = nn.softmax(x)
        return x


class Network(nn.Module):
    def setup(self):
        self.h = Representation(hidden_dim=N_HIDDEN_FEATURES,
                                n_layers=N_HIDDEN_LAYERS,
                                output_dim=N_REPRESENTATION_FEATURES)
        self.g = Dynamics(input_dim=N_REPRESENTATION_FEATURES,
                          hidden_dim=N_HIDDEN_FEATURES,
                          n_layers=N_HIDDEN_LAYERS,
                          n_actions=N_ACTIONS)
        self.policy = Policy(hidden_dim=N_HIDDEN_FEATURES,
                             n_layers=N_HIDDEN_LAYERS,
                             n_actions=N_ACTIONS)
        self.qvalue = QValue(hidden_dim=N_HIDDEN_FEATURES,
                             n_layers=N_HIDDEN_LAYERS,
                             n_actions=N_ACTIONS)

    def initial_inference(self, x):
        h = self.h(x)
        return self.policy(h), self.qvalue(h), h

    def recurrent_inference(self, h, a):
        h_prime = self.g(h, a)
        return self.policy(h_prime), self.qvalue(h_prime), h_prime
    
    def __call__(self, x0, actions):
        p, q, h = self.initial_inference(x0)
        ps, qs = [p], [q]
        for i in range(actions.shape[-2]):
            a = actions[:,i,:]
            print(a.shape)
            p, q, h = self.recurrent_inference(h, a)
            ps.append(p)
            qs.append(q)
        return jnp.stack(ps), jnp.stack(qs)
        

def create_train_state(net, rng, η, features, β=None):
    params = net.init(rng,
                      jnp.ones([1, features], dtype=jnp.float64),
                      jnp.ones([1, 1, 1], dtype=jnp.int32))['params']
    tx = optax.sgd(η, β)
    return TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount
    γ = 0.75

    muzero(env, γ)
