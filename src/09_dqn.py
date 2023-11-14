"""
Implementation of Deep Q Network (DQN)
======================================
This file implements a simple DQN for 4x4 GridWord using JAX.

Result:
-------
Optimal policy:
Action.Up	 Action.Up	 Action.Up	 Action.Up
Action.Up	 Action.Up	 Action.Left	 Action.Up
Action.Up	 Action.Up	 Action.Up	 Action.Down
Action.Up	 Action.Up	 Action.Up	 Action.Up
Loss:
0.6033509142502013
0.024827079457217634
0.12312317844911444
0.5869663645411252
0.6630905703503154
0.7734383597456392
0.7216847927385729
0.6343053951178764
0.02636962941571541
0.01629512498073732
1.0309935507979453
0.746993019839786
0.1594837666990678
0.18346446367961525
0.48158267895374096
0.7886761432468407
0.756206691240497
0.028250660716155864
0.09218173814059989
0.20873912809311904
0.15281865598540997
0.6453047087332597
0.5729455588776085
0.19968222636200397
Avg. Q Value:
0.1648217036218988
0.025058447516828314
0.18483667264152312
-4.470097605902977e-05
0.010941969107091349
0.013580140157612598
0.024024291421337315
-0.21922145261461146
-0.25803499034683286
-0.0037664343507158053
0.028254373644340917
0.036165262739584564
0.08429375666505479
0.09272074044304046
0.08870024687281747
0.15491103393540007
0.11192862573727966
0.2800735487471657
0.8471766196925983
0.3461497105193604
0.30376128489395426
0.21173183002682125
0.5385737263303318
0.05439975564499128

"""

from clu import metrics
import copy
from flax import struct
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util.display import print_grid
from util.jax import MLP, Metrics
from util.gridworld import GridWorld

jax.config.update('jax_enable_x64', True)


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
        ], dtype=np.float64)
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
            target_params = copy.deepcopy(state.params)
    return state, metrics_history


class DQN(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
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
