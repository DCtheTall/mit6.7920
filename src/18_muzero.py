"""
Implementation of MuZero
========================

Result after 100 steps, LR=1e-3
Optimal policy:
Action.Up	 Action.Up	 Action.Up	 Action.Up
Action.Up	 Action.Up	 Action.Up	 Action.Up
Action.Right	 Action.Right	 Action.Up	 Action.Right
Action.Right	 Action.Right	 Action.Right	 Action.Right

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util.display import print_grid
from util.gridworld import GridWorld
from util.jax import MLP, Metrics, TrainState

jax.config.update('jax_enable_x64', True)


N_TRAIN_STEPS = 10
N_TRAJECTORIES_PER_STEP = 10
N_SIMULATIONS_PER_SEARCH = 10
N_UPDATES_PER_STEP = 64
BATCH_SIZE = 32
N_UNROLL_STEPS = 6
N_TD_STEPS = 0
N_FEATURES = 8
N_ACTIONS = 4
N_HIDDEN_LAYERS = 4
N_HIDDEN_FEATURES = 2 * N_FEATURES
N_REPRESENTATION_FEATURES = N_FEATURES
LEARNING_RATE = 1e-5
LR_DECAY_RATE = 0.1
LR_DECAY_STEPS = 1000
ROOT_DIRICHLET_ALPHA = 3e-2
ROOT_EXPLORATION_FRACTION = 0.25
MAX_STEPS_PER_TRAJECTORY = 100
REPLAY_MEMORY_MAX_SIZE = (N_TRAIN_STEPS * N_TRAJECTORIES_PER_STEP) // 2
UCB_COEFF = 2.0 ** 0.5


def muzero(env, γ):
    net = Network()
    rng = jax.random.key(42)
    state = create_train_state(net, rng, features=N_FEATURES,
                               η=LEARNING_RATE)
    del rng
    memory = ReplayMemory(maxlen=REPLAY_MEMORY_MAX_SIZE,
                          batch_size=BATCH_SIZE)
    for step in range(N_TRAIN_STEPS):
        print('Train step', step)
        # Step 1: Run MCTS simulation and collect history
        print('Running self play...')
        for _ in range(N_TRAJECTORIES_PER_STEP):
            history = run_self_play(env, state, γ)
            memory.save_game(history)

        # Step 2: Train the model
        print('Updating model...')
        for _ in range(N_UPDATES_PER_STEP):
            batch = memory.sample_batch(γ, unroll_steps=N_UNROLL_STEPS,
                                        td_steps=N_TD_STEPS)
            init_features, actions, target_q, target_p = preprocess_batch(
                env, batch)
            grads = train_step(state, init_features, actions,
                               target_q, target_p)
            state = state.apply_gradients(grads=grads)
    return state


def run_self_play(env, state, γ):
    s = env.start
    history = GameHistory(env)
    for step in range(MAX_STEPS_PER_TRAJECTORY):
        tree = SearchTree(env, s)
        x = env.ϕ[s]
        policy_logits, q_values, hidden = state.apply_fn(
            {'params': state.params}, x, method=Network.initial_inference)
        print('S', step, s)
        print('P', policy_logits)
        print('Q', q_values)
        print('H', hidden)
        tree.expand_node(tree.root, policy_logits, q_values, hidden)
        tree.add_exploration_noise()
        monte_carlo_tree_search(tree, state, γ)

        a = tree.softmax_action_with_temperature(1)
        history.save_step(s, a, tree)
        s_prime = env.step(s, a)
        print('A', a)
        print('S\'', s_prime)

        if env.is_terminal_state(s):
            break
        s = s_prime
    return history


def monte_carlo_tree_search(tree, state, γ):
    for _ in range(N_SIMULATIONS_PER_SEARCH):
        leaf_node = tree.selection_step()
        h = leaf_node.parent.parent.hidden
        a = leaf_node.parent.a.value
        policy_logits, q_values, hidden = state.apply_fn(
            {'params': state.params}, np.array([h]), np.array([a]),
            method=Network.recurrent_inference)
        tree.expand_node(leaf_node, policy_logits[0], q_values[0], hidden[0])
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
        ret += self.prior * UCB_COEFF * np.sqrt(
            np.log(self.parent_visit_count() / self.visit_count))
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


class SearchTree:
    def __init__(self, env, start):
        self.env = env
        self.root = StateNode(start)

    def expand_node(self, node, policy_logits, q_values, hidden):
        if not node.is_leaf():
            return
        node.hidden = hidden
        for a, q, p in zip(self.env.A, q_values, policy_logits):
            a_node = ActionNode(a, q, p)
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
            a = self.actions[cur_idx].value
            bootstrap_idx = cur_idx + td_steps
            if bootstrap_idx >= len(self.root_q_values):
                q = 0.0
            else:
                q = self.root_q_values[bootstrap_idx][a]
                q *= γ ** td_steps
            rs = self.rewards[cur_idx:]
            for i, r in enumerate(reversed(rs)):
                q += self.visit_counts[cur_idx][a] * r * γ ** (len(rs) - 1 - i)
            target_q.append([q])
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
        positions = [
            (np.random.randint(0, game.length - unroll_steps)
             if game.length > unroll_steps else 0)
            for game in games
        ]
        targets = [game.make_targets(pos, unroll_steps, td_steps, γ)
                   for (pos, game) in zip(positions, games)]
        return [
            {
                'state': game.states[pos],
                'actions': np.array([
                    a.value for a in game.actions[pos:pos+unroll_steps]
                ], dtype=np.int32),
                'target_q_values': np.array(targets[i][0]),
                'target_policy': np.array(targets[i][1]),
            }
            for i, (pos, game) in enumerate(zip(positions, games))
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
        a = nn.Embed(num_embeddings=self.n_actions,
                     features=self.input_dim,
                     dtype=jnp.float64)(a)
        x = jnp.concatenate([h, a], axis=-1)
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


# class Policy(QValue):
#     @nn.compact
#     def __call__(self, x):
#         x = super().__call__(x)
#         x = nn.softmax(x)
#         return x


class Network(nn.Module):
    def setup(self):
        self.h = Representation(hidden_dim=N_HIDDEN_FEATURES,
                                n_layers=N_HIDDEN_LAYERS,
                                output_dim=N_REPRESENTATION_FEATURES)
        self.g = Dynamics(input_dim=N_REPRESENTATION_FEATURES,
                          hidden_dim=N_HIDDEN_FEATURES,
                          n_layers=N_HIDDEN_LAYERS,
                          n_actions=N_ACTIONS)
        # self.policy = Policy(hidden_dim=N_HIDDEN_FEATURES,
        #                      n_layers=N_HIDDEN_LAYERS,
        #                      n_actions=N_ACTIONS)
        self.policy_ff = nn.Dense(N_ACTIONS)
        self.qvalue = QValue(hidden_dim=N_HIDDEN_FEATURES,
                             n_layers=N_HIDDEN_LAYERS,
                             n_actions=N_ACTIONS)

    def policy(self, h):
        x = self.qvalue(h)
        x = self.policy_ff(x)
        return nn.softmax(x)


    def initial_inference(self, x):
        h = self.h(x)
        return self.policy(h), self.qvalue(h), h

    def recurrent_inference(self, h, a):
        h_prime = self.g(h, a)
        return self.policy(h_prime), self.qvalue(h_prime), h_prime

    def __call__(self, x0, actions):
        _, _, h = self.initial_inference(x0)
        qs, ps = [], []
        for i in range(actions.shape[-1]):
            a = actions[:,i]
            p, q, h = self.recurrent_inference(h, a)
            qs.append(q)
            ps.append(p)
        return jnp.stack(qs, axis=1), jnp.stack(ps, axis=1)


def create_train_state(net, rng, η, features, β=0.9,
                       decay_rate=LR_DECAY_RATE,
                       decay_steps=LR_DECAY_STEPS):
    params = net.init(rng,
                      jnp.ones([1, features], dtype=jnp.float64),
                      jnp.ones([1, 1], dtype=jnp.int32))['params']
    lr_sched = optax.exponential_decay(
        init_value=η,
        decay_rate=decay_rate,
        transition_steps=decay_steps,
        end_value=0.0)
    tx = optax.sgd(lr_sched, β)
    return TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


def preprocess_batch(env, batch):
    init_features = []
    actions = []
    target_q = []
    target_p = []
    for example in batch:
        s = example['state']
        init_features.append(env.ϕ[s])
        actions.append(example['actions'])
        target_q.append(example['target_q_values'])
        target_p.append(example['target_policy'])
    return (np.array(init_features), np.array(actions),
            np.array(target_q), np.array(target_p))


@jax.jit
def train_step(state, init_features, actions, target_q, target_p):
    def loss_fn(params):
        pred_q, pred_p = state.apply_fn({'params': params}, init_features, actions)
        a_idx = jnp.expand_dims(actions, axis=-1)
        pred_q = jnp.take_along_axis(pred_q, a_idx, axis=-1)
        q_loss = jnp.mean((target_q - pred_q) ** 2.0) ** 0.5
        p_loss = -jnp.sum(target_p * jnp.log2(pred_p))
        return q_loss + p_loss
    return jax.grad(loss_fn)(state.params)


def optimal_policy(env, state):
    π = {}
    for s in env.S:
        x = env.ϕ[s]
        policy_logits, _, _ = state.apply_fn(
            {'params': state.params}, x, method=Network.initial_inference)
        π[s] = max(env.A, key=lambda a: policy_logits[a.value])
    return π


if __name__ == '__main__':
    env = GridWorld(size=4, transition_probs=(1.0, 0.0))

    # Discount
    γ = 0.75

    state = muzero(env, γ)
    π_opt = optimal_policy(env, state)

    print('Optimal policy:')
    print_grid(π_opt)
