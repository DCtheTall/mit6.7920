"""
Implementation of Actor-Critic
==============================
This file implements vanilla Actor-Critic

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

jax.config.update('jax_enable_x64', True)


N_FEATURES = 8
N_STATES = 16
N_Y_COORDS = 4
N_ACTIONS = 4
N_HIDDEN_LAYERS = 2
N_HIDDEN_FEAFURES = 4 * N_FEATURES
ACTOR_LEARNING_RATE = 1e-5
CRITIC_LEARNING_RATE = 1e-4
N_EPISODES = 10


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
            l2_goal, # L2 distance from goal
            l2_fail, # L2 distance from failure
            0.0 if s == env.goal else np.arccos((y - yg) / l2_goal), # angle wrt goal
            0.0 if s == env.failure else np.arccos((y - yf) / l2_fail), # angle wrt failure
        ], dtype=np.float64)
    return ϕ


def actor_critic(env, γ, ϕ, T=100):
    # Initialize critic first
    Q_net = Critic(hidden_dim=N_HIDDEN_FEAFURES,
                   n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    Q_state = create_train_state(Q_net, rng, η=CRITIC_LEARNING_RATE)
    del rng

    # Initialize actor but its parameters will be copied
    # from the critic after the first step
    π_net = Actor(hidden_dim=N_HIDDEN_FEAFURES,
                  n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(0)
    π_state = create_train_state(π_net, rng, η=ACTOR_LEARNING_RATE)
    del rng

    for _ in range(N_EPISODES):
        s = env.start
        for _ in range(T):
            x = ϕ[s]
            a_logits = π_state.apply_fn({'params': π_state.params},
                                        np.array([x]))[0]
            a_idx = np.random.multinomial(1, pvals=a_logits)
            a_idx = np.argmax(a_idx)
            a = env.A[a_idx]
            r = env.R[s]
            s_prime = env.step(s, a)

            # Update critic
            grads = compute_critic_gradients(Q_state, r, γ, np.array([x]),
                                             np.array([ϕ[s_prime]]))
            Q_state = Q_state.apply_gradients(grads=grads)
            # Copy update to policy net
            π_state = copy_network_params(from_net=Q_state, to_net=π_state)

            # Update actor
            grads = compute_actor_gradients(π_state, np.array([x]),
                                            np.array([a_idx]))
            q_values = Q_state.apply_fn({'params': Q_state.params},
                                        np.array([x]))[0]
            q_value = q_values[a_idx]
            grads = policy_gradient(grads, q_value)
            π_state = π_state.apply_gradients(grads=grads)
            # Copy update to critic net
            Q_state = copy_network_params(from_net=π_state, to_net=Q_state)

            if env.is_terminal_state(s):
                break
            s = s_prime

    return π_state, Q_state


class Critic(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = nn.standardize(x)
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        q_values = nn.Dense(features=N_ACTIONS)(x)
        return q_values


class Actor(Critic):
    @nn.compact
    def __call__(self, x):
        q_values = super().__call__(x)
        # Use softmax so output is probability of each action
        logits = nn.softmax(q_values)
        return logits


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(π_net, rng, η, β1=0.9, β2=0.99):
    params = π_net.init(rng, jnp.ones([1, N_FEATURES]))['params']
    tx = optax.adam(η, β1, β2)
    return TrainState.create(
        apply_fn=π_net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


@jax.jit
def compute_critic_gradients(Q_state, r, γ, x, x_prime):
    def loss_fn(params):
        q = Q_state.apply_fn({'params': params}, x)[0]
        q_prime = Q_state.apply_fn({'params': params}, x_prime)[0]
        return r + jnp.sum(γ * q_prime - q)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(Q_state.params)
    return grads


@jax.jit
def compute_actor_gradients(state, x, a_idx):
    def loss_fn(params):
        a_logits = state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits, jnp.expand_dims(a_idx, axis=-1),
                                axis=1)
        return -jnp.sum(jnp.log(a))
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    return grads


def policy_gradient(grads, reward):
    for k1, k2 in iterate_over_gradients(grads):
        grads[k1][k2] *= reward
    return grads


def iterate_over_gradients(grads):
    return [(k1, k2) for k1 in grads.keys() for k2 in grads[k1].keys()]


def copy_network_params(from_net, to_net):
    params = jax.tree_map(lambda x: x, from_net.params)
    return to_net.replace(params=params)


def optimal_policy(π_state, S, A, ϕ):
    π = {}
    for s in S:
        x = ϕ[s]
        a_logits = π_state.apply_fn({'params': π_state.params}, np.array([x]))[0]
        π[s] = A[np.argmax(a_logits)]
    return π


def optimal_value_function(Q_state, S, ϕ):
    V = {}
    for s in S:
        x = ϕ[s]
        q_values = Q_state.apply_fn({'params': Q_state.params}, np.array([x]))[0]
        V[s] = np.max(q_values)
    return V


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Non-linear features
    ϕ = features(env)

    # Discount factor
    γ = 0.75

    π_state, Q_state = actor_critic(env, γ, ϕ)
    π_opt = optimal_policy(π_state, env.S, env.A, ϕ)
    V_opt = optimal_value_function(Q_state, env.S, ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
    print('Optimal value function:')
    print_grid(V_opt)
