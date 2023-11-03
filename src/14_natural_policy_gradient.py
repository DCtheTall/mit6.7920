"""
Implementation of Natural Policy Gradients (NPG)
================================================

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


N_FEATURES = 8
N_HIDDEN_FEAFURES = 4 * N_FEATURES
N_HIDDEN_LAYERS = 2
N_ACTIONS = 4
CRITIC_LEARNING_RATE = 1e-2
ACTOR_LEARNING_RATE = 1e-2
N_TRAJECTORIES = 3
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


def natural_policy_gradient(env, γ, λ, δ, ϕ):
    # Initialize actor
    π_net = Actor(hidden_dim=N_HIDDEN_FEAFURES,
                  n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    π_state = create_train_state(π_net, rng, η=ACTOR_LEARNING_RATE)
    del rng

    # Initialize critic
    V_net = Critic(hidden_dim=N_HIDDEN_FEAFURES,
                   n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(43)
    V_state = create_train_state(V_net, rng, η=CRITIC_LEARNING_RATE)
    del rng

    for _ in range(N_TRAJECTORIES):
        s = env.start
        # Eligibility traces
        z = {}
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

            v_out = V_state.apply_fn({'params': V_state.params},
                                    np.array([x, x_prime]))
            v = v_out[0]
            v_prime = v_out[1]

            dt = temporal_difference(r, γ, v, v_prime)

            # Update critic using GAE, which uses TD(λ) updates
            z[s] = z.get(s, 0.0) + 1.0
            for sz in z.keys():
                xz = ϕ[sz]
                grads = compute_critic_gradients(V_state, z[sz], dt,
                                                np.array([xz]))
                V_state = V_state.apply_gradients(grads=grads)
                z[sz] *= γ * λ

            # Compute policy gradient for this step
            # Ignore when the discounted reward is zero since
            # it will cancel out the gradient anyway
            grads, F = policy_gradient(π_state, dt, np.array([x]), np.array([a_idx]))
            grads = natural_gradients(grads, F, δ, dt)
            π_state = π_state.apply_gradients(grads=grads)

    return π_state, V_state


class NetworkBase(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(features=N_ACTIONS)(x)
        return x


class Actor(NetworkBase):
    @nn.compact
    def __call__(self, x):
        x = super().__call__(x)
        x = nn.softmax(x)
        return x


class Critic(NetworkBase):
    @nn.compact
    def __call__(self, x):
        x = super().__call__(x)
        return jnp.sum(x, axis=-1)


def create_train_state(net, rng, η):
    params = net.init(rng, jnp.ones([1, N_FEATURES]))['params']
    tx = optax.sgd(η)
    return TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


def temporal_difference(r, γ, v, v_prime):
    return r - v + γ * v_prime


@jax.jit
def compute_critic_gradients(V_state, z, dt, x):
    def loss_fn(params):
        v = V_state.apply_fn({'params': params}, x)
        return -dt * z * jnp.sum(v)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(V_state.params)
    return grads


@jax.jit
def policy_gradient(π_state, r, x, a_idx):
    def loss_fn(params):
        a_logits = π_state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits,
                                jnp.expand_dims(a_idx, axis=-1),
                                axis=1)
        return jax.lax.cond(
            r < 0.0,
            lambda: jnp.sum(jnp.log(1.0 - a)),
            lambda: jnp.sum(jnp.log(a)),
        )
    return jax.grad(loss_fn)(π_state.params), jax.hessian(loss_fn)(π_state.params)


def natural_gradients(grads, F, δ, r):
    g_flat, _ = jax.flatten_util.ravel_pytree(grads)
    F_flat, _ = jax.flatten_util.ravel_pytree(F)
    F_inv = np.linalg.pinv(F_flat.reshape(g_flat.shape[:1] * 2))

    # Scale summed gradient
    g_nat = F_inv @ g_flat

    # Multiply by step size
    # Negative sign bc NN opt does gradient descent
    scale = g_flat @ g_nat
    step_size = np.sqrt(2 * δ / (np.abs(scale) + 1e-12))
    g_nat *= step_size

    # Multiply my -|reward|, negative sign for NN optimizer
    g_nat *= -abs(r)

    # Re-ravel gradients into PyTree
    g_leaves, g_tree = jax.tree_util.tree_flatten(grads)
    i = 0
    unflatten_input = []
    for shape in [x.shape for x in g_leaves]:
        di = np.prod(shape)
        unflatten_input.append(g_nat[i:i+di].reshape(shape))
        i += di
    return jax.tree_util.tree_unflatten(g_tree, unflatten_input)


def optimal_policy(π_state, S, A, ϕ):
    π = {}
    for s in S:
        x = ϕ[s]
        a_logits = π_state.apply_fn({'params': π_state.params}, np.array([x]))[0]
        π[s] = A[np.argmax(a_logits)]
    return π


def optimal_value_function(V_state, S, ϕ):
    V = {}
    for s in S:
        x = ϕ[s]
        v = V_state.apply_fn({'params': V_state.params}, np.array([x]))[0]
        V[s] = v
    return V


if __name__ == '__main__':
    env = GridWorld(size=4)

     # Non-linear features
    ϕ = features(env)

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Step size scale
    δ = 1e-2

    π_state, V_state = natural_policy_gradient(env, γ, λ, δ, ϕ)
    π_opt = optimal_policy(π_state, env.S, env.A, ϕ)
    V_opt = optimal_value_function(V_state, env.S, ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
    print('Optimal value function:')
    print_grid(V_opt)
