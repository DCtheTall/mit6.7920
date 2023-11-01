"""
Implementation of Natural Policy Gradients (NPG)
================================================

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util.gridworld import GridWorld
from util.jax import MLP, Metrics, TrainState

jax.config.update('jax_enable_x64', True)


N_FEATURES = 8
N_HIDDEN_FEAFURES = 4 * N_FEATURES
N_HIDDEN_LAYERS = 2
N_ACTIONS = 4
CRITIC_LEARNING_RATE = 1e-2
ACTOR_LEARNING_RATE = 1e-2
TRAINING_STEPS = 10
N_TRAJECTORIES_PER_STEP = 10
MAX_STEPS_PER_TRAJECTORY = 100


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


def natural_policy_gradient(env, γ, ϕ):
    # Initialize actor first
    π_net = Actor(hidden_dim=N_HIDDEN_FEAFURES,
                  n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    π_state = create_train_state(π_net, rng, η=ACTOR_LEARNING_RATE)
    del rng

    # Initialize critic which will copy the policy network
    # at the start of each step
    Q_net = Critic(hidden_dim=N_HIDDEN_FEAFURES,
                   n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(0)
    Q_state = create_train_state(Q_net, rng, η=CRITIC_LEARNING_RATE)
    del rng

    for _ in range(TRAINING_STEPS):
        Q_state = copy_network_params(from_net=π_state, to_net=Q_state)

        # Q-value estimation using current policy
        for _ in range(N_TRAJECTORIES_PER_STEP):
            s = env.start
            for _ in range(MAX_STEPS_PER_TRAJECTORY):
                x = ϕ[s]
                a_logits = π_state.apply_fn({'params': π_state.params},
                                            np.array([x]))[0]
                a_idx = np.random.multinomial(1, pvals=a_logits)
                a_idx = np.argmax(a_idx)
                a = env.A[a_idx]
                r = env.R[s]
                s_prime = env.step(s, a)
                x_prime = ϕ[s_prime]
                a_prime_logits = π_state.apply_fn({'params': π_state.params},
                                                np.array([x_prime]))[0]
                a_prime_idx = np.random.multinomial(1, pvals=a_prime_logits)
                a_prime_idx = np.argmax(a_prime_idx)

                # Compute current Q values
                q_values = Q_state.apply_fn({'params': Q_state.params},
                                            np.array([x, x_prime]))
                q = q_values[0][a_idx]
                q_prime = q_values[1][a_prime_idx]

                dt = temporal_difference(r, γ, q, q_prime)
                grads = compute_critic_gradients(Q_state, dt, np.array([x]),
                                                 np.array([a_idx]))
                Q_state = Q_state.apply_gradients(grads=grads)

                if env.is_terminal_state(s):
                    break
                s = s_prime
        
        # Policy improvement
        π_prime_state = copy_network_params(from_net=Q_state, to_net=π_state)
        # Compute Hessian of KL divergence with candidate policy
        # Apply natural gradient update
        pass


class Critic(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        q_values = nn.Dense(features=N_ACTIONS)(x)
        return q_values


class Actor(Critic):
    @nn.compact
    def __call__(self, x):
        q_values = super().__call__(x)
        # Use softmax so output is probability of each action
        logits = nn.softmax(q_values)
        return logits


def create_train_state(net, rng, η):
    params = net.init(rng, jnp.ones([1, N_FEATURES]))['params']
    tx = optax.sgd(η)
    return TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
        metrics=Metrics.empty()) 


def copy_network_params(from_net, to_net):
    params = jax.tree_map(lambda x: x, from_net.params)
    return to_net.replace(params=params)


def temporal_difference(r, γ, q, q_prime):
    return r - q + γ * q_prime


@jax.jit
def compute_critic_gradients(Q_state, dt, x, a_idx):
    def loss_fn(params):
        q = Q_state.apply_fn({'params': params}, x)
        q = jnp.take_along_axis(q, jnp.expand_dims(a_idx, axis=-1), axis=1)
        return -dt * jnp.sum(q)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(Q_state.params)
    return grads


if __name__ == '__main__':
    env = GridWorld(size=4)

     # Non-linear features
    ϕ = features(env)

    # Discount factor
    γ = 0.75

    natural_policy_gradient(env, γ, ϕ)
