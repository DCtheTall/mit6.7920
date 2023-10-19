"""
Implementation of Deep Q Network (DQN)
======================================
This file implements a simple DQN for GridWord using JAX.

"""

from clu import metrics
from flax import struct
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import random


D_MODEL = 8
TERMINAL_NODES = {(3, 3), (3, 2)}
TRAIN_STEPS = 10000
TRAIN_START = 200
MEMORY_SIZE = 2000
BATCH_SIZE = 32
EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
EPSILON_DECAY_STEPS = 5000
COPY_N_STEPS = 100
N_ACTIONS = 4


def features(S):
    """Extract features for linear TD"""
    ϕ = {}
    for s in S:
        x, y = s
        l2_goal = ((x - 3) ** 2 + (y - 3) ** 2) ** 0.5
        l2_fail = ((x - 3) ** 2 + (y - 2) ** 2) ** 0.5
        ϕ[s] = np.array([
            float(x), float(y), # position
            float(s in TERMINAL_NODES), # if terminal
            l2_goal, # L2 distance from goal
            l2_fail, # L2 distance from failure
            0.0 if s == (3, 3) else np.arccos((y - 3) / l2_goal), # angle wrt goal
            0.0 if s == (3, 2) else np.arccos((y - 2) / l2_fail), # angle wrt failure
            (x ** 2.0 + y ** 2.0) ** 0.5, # L2 distance from origin
        ], dtype=np.float32)
    return ϕ


def train_dqn(S, A, R, γ, ϕ):
    dqn = DQN(n_actions=len(A),
              hidden_dim=2*D_MODEL,
              n_layers=2)
    memory = ReplayMemory(maxlen=MEMORY_SIZE)

    rng = jax.random.key(42)
    state = create_train_state(dqn, rng)
    target_params = state.params.copy()
    del rng

    s = (0, 0)
    for step in range(TRAIN_STEPS):
        ε = compute_epsilon_for_step(step)
        π = epsilon_greedy_policy(state, A, ε, ϕ)
        a_idx = π(s)
        a = A[a_idx]
        r = R.get(s, 0.0)
        s_prime = take_action(S, s, a)
        memory.append(s, a_idx, r, s_prime)
        if s in TERMINAL_NODES:
            s = (0, 0)
        else:
            s = s_prime
        if step < TRAIN_START:
            continue
        X_batch, a_batch, r_batch, X_prime_batch = memory.sample(ϕ)
        q_target = state.apply_fn({'params': target_params}, X_prime_batch)
        q_target = np.max(q_target, axis=1, keepdims=False)
        q_target = r_batch + γ * q_target
        state = train_step(state, X_batch, a_batch, q_target)
        if step > TRAIN_START and step % COPY_N_STEPS == 0:
            target_params = state.params
    return state


class DQN(nn.Module):
    """A simple ANN model"""
    n_actions: int
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x


class ReplayMemory:
    """Replay memory for sampling training examples for the DQN"""
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = [None] * maxlen
        self.index = 0
        self.length = 0

    def append(self, s, a, r, s_prime):
        self.buffer[self.index] = (s, a, r, s_prime)
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, ϕ):
        X_batch = []
        a_batch = []
        r_batch = []
        X_prime_batch = []
        for i in np.random.randint(self.length, size=BATCH_SIZE):
            s, a, r, s_prime = self.buffer[i]
            X_batch.append(ϕ[s])
            a_batch.append(a)
            r_batch.append(r)
            X_prime_batch.append(ϕ[s_prime])
        return (np.array(X_batch), np.array(a_batch),
                np.array(r_batch), np.array(X_prime_batch))



@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(Q, rng, η=1e-3, β=0.95):
    params = Q.init(rng, jnp.ones([BATCH_SIZE, D_MODEL]))['params']
    tx = optax.sgd(η, β)
    return TrainState.create(
        apply_fn=Q.apply, params=params, tx=tx,
        metrics=Metrics.empty())


def compute_epsilon_for_step(step):
    a, b = EPSILON_MIN, EPSILON_MAX
    return max(a, b - ((b - a) * (step / EPSILON_DECAY_STEPS)))


def epsilon_greedy_policy(state, A, ε, ϕ):
    def π(s):
        if np.random.rand() < ε:
            return np.random.randint(len(A))
        x = ϕ[s]
        q_pred = state.apply_fn({'params': state.params}, np.array([x]))[0]
        return np.argmax(q_pred, keepdims=True)[0]
    return π


# Memoization table for function below
T = {}

def take_action(S, s, a):
    """Sample next state from MDP
    
    TD(0) algorithm treats this as a black box.
    """
    if s in {(3, 3), (3, 2)}:
        return s
    if (s, a) in T:
        return random.sample(T[(s, a)], 1)[0]
    possible_next_states = []
    for s_prime in S:
        dx, dy = s_prime[0] - s[0], s_prime[1] - s[1]
        if max(abs(dx), abs(dy), abs(dx) + abs(dy)) != 1:
            continue
        if a == 'Left' and dx == 1:
            continue
        if a == 'Right' and dx == -1:
            continue
        if a == 'Up' and dy == -1:
            continue
        if a == 'Down' and dy == 1:
            continue
        possible_next_states.append(s_prime)
    T[(s, a)] = possible_next_states
    return random.sample(possible_next_states, 1)[0]


@jax.jit
def train_step(state, X_batch, a_batch, q_target):
    def loss_fn(params):
        q_pred = state.apply_fn({'params': params}, X_batch)
        q_pred *= nn.one_hot(a_batch, N_ACTIONS)
        q_pred = jnp.sum(q_pred, axis=-1)
        return dqn_loss(q_pred, q_target)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def dqn_loss(q_pred, q_actual):
    error = jnp.abs(q_pred - q_actual)
    clipped_error = jnp.clip(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = jnp.mean(jnp.square(clipped_error) + linear_error)
    return loss


def optimal_policy(S, A, ϕ, state):
    π = {}
    for s in S:
        x = ϕ[s]
        q = state.apply_fn({'params': state.params}, np.array([x]))[0]
        π[s] = A[np.argmax(q)]
    return π


def print_grid(X):
    for y in range(3, -1, -1):
        print(*(str(X[(x, y)]) + '\t' for x in range(4)))


if __name__ == '__main__':
    # Set of all states, 4x4 grid
    S = {
        (i // 4, i % 4)
        for i in range(16)
    }

    # Set of all actions, list to allow indexing
    A = ['Up', 'Down', 'Left', 'Right']

    # Rewards
    R = {(3, 3): 1.0, (3, 2): -1.0}

    # Non-linear features
    ϕ = features(S)

    # Discount factor
    γ = 0.75

    opt_state = train_dqn(S, A, R, γ, ϕ)
    π_opt = optimal_policy(S, A, ϕ, opt_state)

    print('Optimal policy:')
    print_grid(π_opt)
