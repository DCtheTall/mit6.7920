"""
Implementation of Actor-Critic
==============================
This file implements the Actor-Critic with Advantage Function (A2C)
variant of the Actor-Critic policy gradient method.

TODO share network between actor critic

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


N_FEATURES = 9
N_STATES = 16
N_Y_COORDS = 4
N_ACTIONS = 4
ACTOR_LEARNING_RATE = 1e-3
N_EPISODES = 1000


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
            float(env.is_terminal_state(s)),
        ], dtype=np.float64)
    return ϕ


def initialize_critic_parameters():
    return np.zeros((N_FEATURES,))


def actor_critic(env, V, γ, λ, ϕ, ω, T=100):
    π_net = Actor(hidden_dim=2 * N_FEATURES,
                  n_layers=2)
    rng = jax.random.key(42)
    state = create_train_state(π_net, rng)
    del rng

    # For critic learning rate
    N = {s: 0.0 for s in env.S}

    for _ in range(N_EPISODES):
        s = env.start
        # Eligibility traces
        z = {}
        for _ in range(T):
            x = ϕ[s]
            a_logits = state.apply_fn({'params': state.params},
                                        np.array([x]))[0]

            # For floating pt error
            a_logits = np.asarray(a_logits).astype(np.float64)
            a_logits /= a_logits.sum()

            a_idx = np.random.multinomial(1, pvals=a_logits)[0]

            a = env.A[a_idx]
            r = env.R[s]
            s_prime = env.step(s, a)

            # Update critic
            dt = temporal_difference(V, r, γ, ω, s, s_prime)
            z[s] = z.get(s, 0.0) + 1.0
            for sz in z.keys():
                N[sz] += 1.0
                η = critic_learning_rate(N[sz])
                ω += η * z[sz] * dt * ϕ[sz]
                z[sz] *= λ * γ

            # Update actor
            grads = compute_gradients(state, np.array([x]), np.array([a_idx]))
            grads = policy_gradient(grads, V(ω, s))
            state = state.apply_gradients(grads=grads)

            if env.is_terminal_state(s):
                break
            s = s_prime

    return state, ω


class Actor(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(features=N_ACTIONS)(x)
        # Use softmax so output is probability of each action
        logits = nn.softmax(x)
        return logits


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(π_net, rng, η=ACTOR_LEARNING_RATE, β1=0.9, β2=0.99):
    params = π_net.init(rng, jnp.ones([1, N_FEATURES]))['params']
    tx = optax.adam(η, β1, β2)
    return TrainState.create(
        apply_fn=π_net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


def critic_learning_rate(t):
    """Decaying learning rate.
    
    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


def temporal_difference(V, r, γ, ω, s, s_prime):
    return r + γ * V(ω, s_prime) - V(ω, s)


@jax.jit
def compute_gradients(state, x, a_idx):
    def loss_fn(params):
        a_logits = state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits,
                                jnp.expand_dims(a_idx, axis=-1),
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


def optimal_policy(state, S, A, ϕ):
    π = {}
    for s in S:
        x = ϕ[s]
        a_logits = state.apply_fn({'params': state.params}, np.array([x]))[0]
        π[s] = A[np.argmax(a_logits)]
    return π


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Non-linear features
    ϕ = features(env)

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Parameters for linear critic model
    ω = initialize_critic_parameters()

    # Initialize value function
    def V(ω, s):
        return ω @ ϕ[s]

    opt_state, ω_opt = actor_critic(env, V, γ, λ, ϕ, ω)
    π_opt = optimal_policy(opt_state, env.S, env.A, ϕ)
    V_opt = {s: V(ω_opt, s) for s in env.S}

    print('Optimal policy:')
    print_grid(π_opt)
    print('Optimal value function:')
    print_grid(V_opt)
