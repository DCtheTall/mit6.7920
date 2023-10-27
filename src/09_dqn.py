"""
Implementation of Deep Q Network (DQN)
======================================
This file implements a simple DQN for 4x4 GridWord using JAX.

Result:
-------
Optimal policy:
Action.Right	 Action.Right	 Action.Up	 Action.Left
Action.Right	 Action.Right	 Action.Left	 Action.Left
Action.Up	 Action.Right	 Action.Right	 Action.Down
Action.Right	 Action.Right	 Action.Right	 Action.Right
Loss:
0.5851697325706482
0.038698699325323105
0.025229347869753838
0.11659318953752518
0.026050325483083725
0.1154707670211792
0.07987924665212631
0.12255693972110748
0.9259278178215027
0.6909330487251282
0.6375937461853027
0.06970922648906708
0.07109434902667999
0.18300622701644897
0.2304687649011612
0.13453637063503265
0.1315186619758606
0.09592777490615845
0.08334122598171234
0.10568971186876297
0.14841288328170776
0.5190666317939758
0.5443905591964722
0.04820887744426727
Avg. Q Value:
0.2959030866622925
0.022004982456564903
0.033189885318279266
0.0678509920835495
0.034410785883665085
0.07644041627645493
0.10203485935926437
0.352797269821167
0.08633407950401306
0.3846694529056549
0.38886791467666626
0.17382757365703583
0.33108335733413696
0.13860435783863068
0.2881055176258087
0.3613390028476715
0.14946387708187103
0.3783504366874695
0.3182215094566345
0.129811093211174
0.2277454435825348
0.16887833178043365
0.4026474952697754
0.27707308530807495

"""

from clu import metrics
from flax import struct
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util.display import print_grid
from util.gridworld import GridWorld


N_FEATURES = 8
LEARNING_RATE = 1e-3
TRAIN_STEPS = 25000
TRAIN_START = 1000
MEMORY_SIZE = 5000
BATCH_SIZE = 16
EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
EPSILON_DECAY_STEPS = 15000
LOG_N_STEPS = 1000
COPY_N_STEPS = 250
N_ACTIONS = 4


def features(env):
    """Extract features for linear TD"""
    ϕ = {}
    for s in env.S:
        x, y = s
        xg, yg = env.goal
        xf, yf = env.failure
        l2_goal = ((x - xg) ** 2 + (y - yg) ** 2) ** 0.5
        l2_fail = ((x - xf) ** 2 + (y - yf) ** 2) ** 0.5
        ϕ[s] = np.array([
            float(x), float(y), # position
            (x ** 2.0 + y ** 2.0) ** 0.5, # L2 distance from origin
            float(x + y), # L1 norm from origin
            float(abs(x - xg) + abs(y - yg)), # L1 distance from goal
            float(abs(x - xf) + abs(y - yf)), # L1 distance from failure
            0.0 if s == env.goal else np.arccos((y - yg) / l2_goal), # angle wrt goal
            0.0 if s == env.failure else np.arccos((y - yf) / l2_fail), # angle wrt failure
        ], dtype=np.float32)
    return ϕ


def train_dqn(env, γ, ϕ, ddqn=False):
    dqn = DQN(hidden_dim=2*N_FEATURES,
              n_layers=2)
    memory = ReplayMemory(maxlen=MEMORY_SIZE)

    rng = jax.random.key(42)
    state = create_train_state(dqn, rng)
    target_params = state.params.copy()
    del rng

    metrics_history = {'loss': [], 'avg_q_value': []}

    s = env.start
    for step in range(TRAIN_STEPS):
        ε = compute_epsilon_for_step(step)
        π = epsilon_greedy_policy(state, ε, ϕ)
        a_idx = π(s)
        a = env.A[a_idx]
        r = env.R[s]
        s_prime = env.step(s, a)
        memory.append(s, a_idx, r, s_prime)
        if env.is_terminal_state(s):
            s = env.start
        else:
            s = s_prime
        if step < TRAIN_START:
            continue
        X_batch, a_batch, r_batch, X_prime_batch = memory.sample(ϕ)
        # For DDQN, we use the online network to select which action to use.
        # In vanilla DQN we use the max of the target network output.
        if ddqn:
            a_prime = state.apply_fn({'params': state.params}, X_prime_batch)
            a_prime = np.argmax(a_prime, axis=1, keepdims=False)
            q_target = state.apply_fn({'params': target_params}, X_prime_batch)
            q_target = np.take_along_axis(q_target,
                                          np.expand_dims(a_prime, -1),
                                          axis=1)
            q_target = np.squeeze(q_target, axis=-1)
        else:
            q_target = state.apply_fn({'params': target_params}, X_prime_batch)
            q_target = np.max(q_target, axis=1, keepdims=False)
        q_target = r_batch + γ * q_target
        state = train_step(state, X_batch, a_batch, q_target)
        if step % LOG_N_STEPS == 0:
            state = compute_metrics(state, X_batch, a_batch, q_target)
            for metric, value in state.metrics.compute().items():
                metrics_history[metric].append(float(value))
            state = state.replace(metrics=state.metrics.empty())
        if step % COPY_N_STEPS == 0:
            target_params = state.params
    return state, metrics_history


class DQN(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = nn.standardize(x)
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(features=N_ACTIONS)(x)
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
    avg_q_value: metrics.Average.from_output('avg_q_value')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(dqn, rng, η=LEARNING_RATE, β1=0.9, β2=0.99):
    params = dqn.init(rng, jnp.ones([BATCH_SIZE, N_FEATURES]))['params']
    tx = optax.adam(η, β1, β2)
    return TrainState.create(
        apply_fn=dqn.apply, params=params, tx=tx,
        metrics=Metrics.empty())


def compute_epsilon_for_step(step):
    a, b = EPSILON_MIN, EPSILON_MAX
    return max(a, b - ((b - a) * (step / EPSILON_DECAY_STEPS)))


def epsilon_greedy_policy(state, ε, ϕ):
    """Note that π returns the index in `A`, not the action"""
    def π(s):
        if np.random.rand() < ε:
            return np.random.randint(N_ACTIONS)
        x = ϕ[s]
        q_pred = state.apply_fn({'params': state.params}, np.array([x]))[0]
        return np.argmax(q_pred, keepdims=True)[0]
    return π


@jax.jit
def train_step(state, X_batch, a_batch, q_target):
    def loss_fn(params):
        q_pred = state.apply_fn({'params': params}, X_batch)
        q_pred *= nn.one_hot(a_batch, N_ACTIONS)
        q_pred = jnp.sum(q_pred, axis=-1)
        return rmse(q_pred, q_target)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(state, X_batch, a_batch, q_target):
    q_out = state.apply_fn({'params': state.params}, X_batch)
    q_pred = q_out * nn.one_hot(a_batch, N_ACTIONS)
    q_pred = jnp.sum(q_pred, axis=-1)
    loss = rmse(q_pred, q_target)
    metric_updates = state.metrics.single_from_model_output(
        loss=loss, avg_q_value=np.mean(q_out))
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


def rmse(q_pred, q_actual):
    """Root mean squared error (RMSE) loss"""
    return jnp.mean((q_pred - q_actual) ** 2.0) ** 0.5


def optimal_policy(state, S, A, ϕ):
    π = {}
    for s in S:
        x = ϕ[s]
        q = state.apply_fn({'params': state.params}, np.array([x]))[0]
        π[s] = A[np.argmax(q)]
    return π


def print_metric_history(history, metric):
    print('\n'.join([str(x) for x in history[metric]]))


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Non-linear features
    ϕ = features(env)

    # Discount factor
    γ = 0.75

    opt_state, metrics_history = train_dqn(env, γ, ϕ, ddqn=True)
    π_opt = optimal_policy(opt_state, env.S, env.A, ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
    print('Loss:')
    print_metric_history(metrics_history, 'loss')
    print('Avg. Q Value:')
    print_metric_history(metrics_history, 'avg_q_value')
