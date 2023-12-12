"""
Implementation of MuZero
========================

Learns static GridWorld.

TODO build Stochastic MuZero for stochastic GridWorld.

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


N_TRAIN_STEPS = 200
N_TRAJECTORIES_PER_STEP = 10
N_SIMULATIONS_PER_SEARCH = 20
N_UPDATES_PER_STEP = 10
BATCH_SIZE = 64
N_UNROLL_STEPS = 8
N_TD_STEPS = 4
N_FEATURES = 8
N_ACTIONS = 4
N_HIDDEN_LAYERS = 2
N_HIDDEN_FEATURES = 4 * N_FEATURES
N_REPRESENTATION_FEATURES = N_FEATURES
LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.1
LR_DECAY_STEPS = 400
ROOT_DIRICHLET_ALPHA = 3e-2
ROOT_EXPLORATION_FRACTION = 0.25
MAX_STEPS_PER_TRAJECTORY = 100
REPLAY_MEMORY_MAX_SIZE = 200
UCB_COEFF = 2.0 ** 0.5


def muzero(env, γ):
    net = Network()
    rng = jax.random.key(13)
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
            batch = memory.sample_batch(N_UNROLL_STEPS, N_TD_STEPS, γ)
            init_features, actions, target_p, target_v, target_r = (
                preprocess_batch(env, batch))
            grads = train_step(state, init_features, actions, target_p,
                               target_v, target_r)
            state = state.apply_gradients(grads=grads)
        π, V = optimal_policy_and_value_function(env, state)
        print('Policy for step', step)
        print_grid(π)
        print('Value function for step', step)
        print_grid(V)
    return state


def run_self_play(env, state, γ):
    s = env.start
    history = GameHistory(env)
    for step in range(MAX_STEPS_PER_TRAJECTORY):
        tree = SearchTree(env, s)
        x = env.ϕ[s]
        policy_logits, value, hidden = state.apply_fn(
            {'params': state.params}, x, method=Network.initial_inference)
        print('S', step, s)
        print('P', policy_logits)
        print('V', value)
        print('H', hidden)
        tree.expand_node(tree.root, policy_logits, value, hidden)
        tree.add_exploration_noise()
        monte_carlo_tree_search(tree, state, γ)

        a = tree.softmax_action_with_temperature(int(step < 32))
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
        policy_logits, values, hidden, _ = state.apply_fn(
            {'params': state.params}, np.array([h]), np.array([a]),
            method=Network.recurrent_inference)
        tree.expand_node(leaf_node, policy_logits[0], values[0], hidden[0])
        tree.backprop_step(leaf_node, γ)


class Node:
    def __init__(self):
        self.parent = None
        self.visit_count = 0

    def add_child(self, child):
        child.parent = self


class StateNode(Node):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self.hidden = None
        self.value = 0.0
        self.children = {}

    def add_child(self, key, child):
        super().add_child(child)
        assert key not in self.children
        self.children[key] = child

    def learning_rate(self):
        return 1.0 / self.visit_count

    def upper_confidence_bound_policy_action(self, env):
        a = max(env.A, key=lambda a: self.children[a].upper_confidence_bound())
        return self.children[a]

    def is_leaf(self):
        return len(self.children) == 0


class ActionNode(Node):
    def __init__(self, a, prior):
        super().__init__()
        self.a = a
        self.prior = prior
        self.child = None

    def add_child(self, child):
        super().add_child(child)
        self.child = child

    @property
    def value(self):
        return self.child.value

    def upper_confidence_bound(self):
        if self.visit_count == 0:
            # Hack to execute each action at least once.
            # This works for GridWorld because the action space is very small.
            return float('inf')
        ret = self.value
        ret += self.prior
        ret += UCB_COEFF * np.sqrt(
            np.log(self.parent_visit_count) / self.visit_count)
        return ret

    @property
    def parent_visit_count(self):
        node = self.parent
        if node:
            return node.visit_count
        return self.visit_count

    def is_leaf(self):
        return self.child is None



class SearchTree:
    def __init__(self, env, start):
        self.env = env
        self.root = StateNode(start)

    def expand_node(self, node, policy_logits, value, hidden):
        if not node.is_leaf():
            return
        node.hidden = hidden
        node.value = value[0]
        for a, p in zip(self.env.A, policy_logits):
            a_node = ActionNode(a, p)
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
            if a_node.is_leaf():
                child = StateNode(s_prime)
                a_node.add_child(child)
            else:
                child = a_node.child
            if env.is_terminal_state(cur.s):
                cur = child
                break
            cur = child
        return cur

    def backprop_step(self, node, γ):
        v_prime = node.value
        cur = node
        while cur.parent:
            a_node = cur.parent
            s_node = a_node.parent
            s_node.visit_count += 1
            a_node.visit_count += 1
            dt = env.R[s_node.s] - s_node.value + γ * v_prime
            s_node.value += s_node.learning_rate() * dt
            cur = s_node
            v_prime = s_node.value


    def softmax_action_with_temperature(self, temp):
        if temp == 0:
            return max(self.env.A,
                       key=lambda a: self.root.children[a].visit_count)
        p = np.array([float(a_node.visit_count)
                      for a_node in self.root.children.values()])
        p /= sum(p)
        return np.random.choice(self.env.A, p=p)


class GameHistory:
    def __init__(self, env):
        self.env = env
        self.states = []
        self.actions = []
        self.rewards = []
        self.visit_counts = []
        self.root_values = []

    def save_step(self, s, a, tree):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(self.env.R[s])

        visit_counts = np.array([child.visit_count
                                 for child in tree.root.children.values()],
                                dtype=np.float64)
        visit_counts /= sum(visit_counts)
        print('Pt', s, visit_counts)
        self.visit_counts.append(visit_counts)
        self.root_values.append(tree.root.value)

    @property
    def length(self):
        return len(self.rewards)

    def make_targets(self, start_pos, unroll_steps, td_steps, γ):
        target_p = []
        target_v = []
        target_r = []
        v = 0.0
        discounted_rewards = self.discounted_rewards(γ)
        for cur_idx in range(start_pos, start_pos + unroll_steps):
            if cur_idx >= len(self.rewards):
                target_p.append([0.0] * len(self.visit_counts[0]))
                target_v.append([0.0])
                target_r.append([0.0])
                continue
            target_p.append(self.visit_counts[cur_idx])
            v = discounted_rewards[cur_idx]
            if cur_idx + td_steps < self.length:
                v += self.root_values[cur_idx] * (γ ** td_steps)
            target_v.append([v])
            target_r.append([self.rewards[cur_idx]])
        return target_p, target_v, target_r

    def discounted_rewards(self, γ):
        r = 0.0
        ret = []
        for cur in reversed(self.rewards):
            r *= γ
            r += cur
            ret.append(r)
        return list(reversed(ret))


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

    def sample_batch(self, unroll_steps, td_steps, γ):
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
                'actions': np.array(self.batch_actions(pos, game,
                                                       unroll_steps),
                                    dtype=np.int32),
                'target_p': np.array(targets[i][0]),
                'target_v': np.array(targets[i][1]),
                'target_r': np.array(targets[i][2]),
            }
            for i, (pos, game) in enumerate(zip(positions, games))
        ]

    def batch_actions(self, pos, game, unroll_steps):
        actions = [a.value for a in game.actions[pos:pos+unroll_steps]]
        if len(actions) < unroll_steps:
            actions += [-1] * (unroll_steps - len(actions))
        return actions


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
        return nn.sigmoid(x)


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
        x = nn.Dense(self.input_dim + 1,
                     dtype=jnp.float64)(x)
        x, r = jnp.split(x, [self.input_dim], axis=-1)
        return nn.sigmoid(x), r


class Policy(nn.Module):
    hidden_dim: int
    n_layers: int
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(self.n_actions)(x)
        return nn.softmax(x)


class Value(nn.Module):
    hidden_dim: int
    n_layers: int
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(1)(x)
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
        self.value = Value(hidden_dim=N_HIDDEN_FEATURES,
                           n_layers=N_HIDDEN_LAYERS,
                           n_actions=N_ACTIONS)


    def initial_inference(self, x):
        h = self.h(x)
        return self.policy(h), self.value(h), h

    def recurrent_inference(self, h, a):
        h_prime, r = self.g(h, a)
        return self.policy(h_prime), self.value(h_prime), h_prime, r

    def __call__(self, x0, actions):
        p, v, h = self.initial_inference(x0)
        ps, vs = [p], [v]
        rs = []
        for i in range(actions.shape[-1]):
            a = actions[:,i]
            p, v, h, r = self.recurrent_inference(h, a)
            ps.append(p)
            vs.append(v)
            rs.append(r)
        ps.pop()
        vs.pop()
        return (jnp.stack(ps, axis=1), jnp.stack(vs, axis=1),
                jnp.stack(rs, axis=1))


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
    target_p = []
    target_v = []
    target_r = []
    for example in batch:
        s = example['state']
        init_features.append(env.ϕ[s])
        actions.append(example['actions'])
        target_p.append(example['target_p'])
        target_v.append(example['target_v'])
        target_r.append(example['target_r'])
    return (np.array(init_features), np.array(actions), np.array(target_p),
            np.array(target_v), np.array(target_r))


@jax.jit
def train_step(state, init_features, actions, target_p, target_v, target_r,
               c=1e-2):
    def loss_fn(params):
        pred_p, v, r = state.apply_fn(
            {'params': params}, init_features, actions)
        mask = jnp.expand_dims(actions != -1, axis=-1).astype(jnp.int32)
        v_loss = jnp.mean(mask * (v - target_v) ** 2.0) ** 0.5
        p_loss = -jnp.sum(target_p * jnp.log2(pred_p))
        r_loss = jnp.mean(mask * (r - target_r) ** 2.0) ** 0.5
        l2_loss = 0.0
        for x in jax.tree_util.tree_leaves(params):
            l2_loss += jnp.sum(x ** 2.0)
        l2_loss *= c
        return v_loss + p_loss + r_loss + l2_loss
    return jax.grad(loss_fn)(state.params)


def optimal_policy_and_value_function(env, state):
    π = {}
    V = {}
    for s in env.S:
        x = env.ϕ[s]
        policy_logits, value, _ = state.apply_fn(
            {'params': state.params}, x, method=Network.initial_inference)
        π[s] = max(env.A, key=lambda a: policy_logits[a.value])
        V[s] = value[0]
    return π, V


if __name__ == '__main__':
    env = GridWorld(size=4, transition_probs=(1.0, 0.0))

    # Discount
    γ = 0.75

    state = muzero(env, γ)
    π_opt, V_opt = optimal_policy_and_value_function(env, state)

    print('Optimal policy:')
    print_grid(π_opt)
    print('Optimal value function:')
    print_grid(V_opt)
