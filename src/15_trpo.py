"""
Implementation of Trust Region Policy Optimization (TRPO)
=========================================================
Note: One difference TRPO paper authors mention is they use actual KL-divergence Hessian for NPG

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from util.gridworld import GridWorld
from util.display import print_grid
from util.jax import MLP, create_sgd_train_state

jax.config.update('jax_enable_x64', True)


N_FEATURES = 8
N_HIDDEN_FEATURES = 2 * N_FEATURES
N_HIDDEN_LAYERS = 2
N_ACTIONS = 4
LEARNING_RATE = 1e-2
TRAIN_STEPS = 1
N_TRAJECTORIES_PER_STEP = 10
N_VALUE_ESTIMATE_ITERATIONS = 10
MAX_STEPS_PER_TRAJECTORY = 100
CONJUGATE_GRADIENT_THRESHOLD = 1e-2
MAX_CONJUGATE_GRADIENT_STEPS = 10


def trpo(env, γ, λ, δ):
    # Initialize policy net
    π_net = PolicyNet(hidden_dim=N_HIDDEN_FEATURES,
                      n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    π_state = create_sgd_train_state(π_net, rng, η=LEARNING_RATE,
                                     features=N_FEATURES)
    del rng

    for _ in range(TRAIN_STEPS):
        V_π = td_lambda_value_estimate(env, π_state, γ, λ)
        xs = [[], []]
        a_idxs = [[], []]
        dts = [[], []]
        for _ in range(N_TRAJECTORIES_PER_STEP):
            s = env.start
            for _ in range(MAX_STEPS_PER_TRAJECTORY):
                x = env.ϕ[s]
                a_logits = π_state.apply_fn({'params': π_state.params},
                                            np.array(x))
                a_idx = np.random.multinomial(1, pvals=a_logits)
                a_idx = np.argmax(a_idx)
                a = env.A[a_idx]
                r = env.R[s]
                s_prime = env.step(s, a)
                dt = temporal_difference(r, γ, V_π[s], V_π[s_prime])

                i = 0 if dt < 0 else 1
                xs[i].append(x)
                a_idxs[i].append(a_idx)
                dts[i].append(dt)

                if env.is_terminal_state(s):
                    break
                s = s_prime
        for x, a_idx, dt in zip(xs, a_idxs, dts):
            if not x:
                continue
            grads = policy_gradient(π_state, np.array(dt), np.array(x),
                                    np.array(a_idx))
            π_prime_state = π_state.apply_gradients(
                # Negative sign because NN opt does gradient descent
                grads=jax.tree_util.tree_map(lambda x: -x, grads))
            q_logits = π_prime_state.apply_fn({'params': π_prime_state.params},
                                              np.array(x))
            F = fisher_information_matrix(π_state, q_logits, np.array(x))
            grads = natural_gradient(F, grads)
            # TODO use conjugate gradient method to approx natural grad
            # TODO natural gradient update
            # TODO check update is in trust region
            # grads = natural_gradients(grads, δ, np.array(dt))
            # π_state = π_state.apply_gradients(grads=grads)
    return π_state


class PolicyNet(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(features=N_ACTIONS)(x)
        x = nn.softmax(x)
        return x


def td_lambda_value_estimate(env, π_state, γ, λ):
    # Initialize value function
    V = {s: 0.0 for s in env.S}
    N = {}
    t = 0
    while True:
        t += 1
        V_prime = update_value_function(env, V, N, π_state, γ, λ)
        if (all(np.isclose(V[s], V_prime[s]) for s in env.S)
            or t == N_VALUE_ESTIMATE_ITERATIONS):
            break
        V = V_prime
    return V


def update_value_function(env, V, N, π_state, γ, λ):
    """One episode of iterative temporal difference (TD) learning"""
    V = V.copy()
    s = env.start
    # Eligibility traces
    z = {}
    for _ in range(MAX_STEPS_PER_TRAJECTORY):
        x = env.ϕ[s]
        a_logits = π_state.apply_fn({'params': π_state.params},
                                    np.array(x))
        a_idx = np.random.multinomial(1, pvals=a_logits)
        a_idx = np.argmax(a_idx)
        a = env.A[a_idx]
        s_prime = env.step(s, a)
        z[s] = z.get(s, 0.0) + 1.0
        dt = temporal_difference(env.R[s], γ, V[s], V[s_prime])
        for sz in z.keys():
            # Temporal difference update step
            N[sz] = N.get(sz, 0) + 1
            η = td_lambda_learning_rate(N[sz])
            V[sz] += η * z[sz] * dt
            z[sz] *= λ * γ
        if env.is_terminal_state(s):
            break
        s = s_prime
    return V


def temporal_difference(r, γ, v, v_prime):
    return r - v + γ * v_prime


def td_lambda_learning_rate(t):
    """Decaying learning rate.

    Using harmonic series since it meets Robbins-Monro conditions.
    """
    return 1.0 / t


@jax.jit
def policy_gradient(π_state, r, x, a_idx):
    def loss_fn(params):
        a_logits = π_state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits,
                                jnp.expand_dims(a_idx, axis=-1),
                                axis=0)
        return jnp.mean(
            jax.vmap(
                lambda r_i, a_i: jax.lax.cond(
                    r_i < 0.0,
                    lambda: -r_i * jnp.log(1.0 - a_i),
                    lambda: r_i * jnp.log(a_i),
                )
            )(r, a)
        )
    return jax.grad(loss_fn)(π_state.params)


@jax.jit
def fisher_information_matrix(π_state, q_logits, x):
    def kl_divergence(params):
        p_logits = π_state.apply_fn({'params': params}, x)
        return jnp.sum(p_logits * jnp.log(p_logits / q_logits))
    return jax.hessian(kl_divergence)(π_state.params)


def natural_gradient(F, grads):
    # Unravel gradient tree into a flat vector
    g_leaves, g_tree = jax.tree_util.tree_flatten(grads)
    g_flat = []
    for leaf in g_leaves:
        g_flat.extend(leaf.flatten())
    g_flat = np.array(g_flat, dtype=np.float64)
    # Unravel Hessian of KL-divergence into a matrix
    F_leaves, _ = jax.tree_util.tree_flatten(F)
    F_flat = []
    for i, g_leaf in enumerate(g_leaves):
        start = i * len(g_leaves)
        end = start + len(g_leaves)
        n_dims = len(g_leaf.shape)
        f_flat = [[] for _ in range(np.prod(g_leaf.shape))]
        for F_leaf in F_leaves[start:end]:
            F_leaf = F_leaf.reshape([-1] + [np.prod(F_leaf.shape[n_dims:])])
            for i, f_leaf in enumerate(F_leaf):
                f_flat[i].extend(f_leaf)
        F_flat.extend(f_flat)
    F_flat = np.array(F_flat, dtype=np.float64)
    g_nat = conjugate_gradient(F_flat, g_flat)
    # Re-ravel gradients into PyTree
    i = 0
    unflatten_input = []
    for shape in [x.shape for x in g_leaves]:
        di = np.prod(shape)
        unflatten_input.append(g_nat[i:i+di].reshape(shape))
        i += di
    return jax.tree_util.tree_unflatten(g_tree, unflatten_input)


def conjugate_gradient(F, g, Ɛ=CONJUGATE_GRADIENT_THRESHOLD):
    """
    Use conjugate gradients to solve:

    (F^T @ F) @ x = F^T @ b

    We multiply by F^T on the right since F is not guaranteed
    to be positive definite.

    """
    # Initial guess: natural gradient == normal gradient
    x = g
    A = F.transpose() @ F
    b = F.transpose() @ g - A @ x
    b_l1 = np.sum(np.abs(b))
    r = b  # No Ax term since x is zero vector
    p = r
    n_iter = 0
    for _ in range(MAX_CONJUGATE_GRADIENT_STEPS):
        n_iter += 1
        x, r_prime = conjugate_gradient_step(A, r, p, x)
        error = np.dot(r_prime, r_prime) ** 0.5
        if error < Ɛ * b_l1:
            break
        β = np.dot(r_prime, r_prime) / np.dot(r, r)
        p = r + β * p
        r = r_prime
    return x


@jax.jit
def conjugate_gradient_step(A, r, p, x):
    p_prime = jnp.matmul(A, p)
    α = jnp.dot(r, r) / jnp.dot(p, p_prime)
    x_prime = x + α * p
    r_prime = r - α * p_prime
    return x_prime, r_prime



if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Trust region size
    δ = 1e-2

    π_state = trpo(env, γ, λ, δ)
