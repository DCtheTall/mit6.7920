"""
Implementation of Trust Region Policy Optimization (TRPO)
=========================================================
TRPO improves on natural policy gradient (NPG) by using the
Hessian vector product to approximate the Fisher information
matrix (the latter being the negative of Hessian of the log-loss).

It also uses the conjugate gradients to solve the equation

F @ x = g

to approximate the natural gradient instead of expensively computing
the inverse of the Fisher matrix.

TRPO also only accepts updates that improve the expected return
of the policy and where the new policy's KL-divergence from the old
policy is within a certain threshold.

This algorithm trains for the same number of steps as NPG but
has a significantly faster wall clock time due to the speed
improvements.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 γ : Discount factor
 λ : TD(λ) parameter
 V : State value function
 π : Agent policy
 ϕ : Non-linear features from environment
 F : Fisher information matrix
 δ : Trust region size parameter

Result:
-------
Optimal policy:
Action.Right	 Action.Right	 Action.Right	 Action.Right
Action.Right	 Action.Right	 Action.Left	 Action.Left
Action.Right	 Action.Right	 Action.Left	 Action.Left
Action.Right	 Action.Right	 Action.Right	 Action.Left

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from util.gridworld import GridWorld
from util.display import print_grid
from util.jax import MLP, create_sgd_train_state

np.random.seed(42)
jax.config.update('jax_enable_x64', True)


N_FEATURES = 8
N_HIDDEN_FEATURES = 4 * N_FEATURES
N_HIDDEN_LAYERS = 2
N_ACTIONS = 4
TRAIN_STEPS = 200
N_TRAJECTORIES_PER_STEP = 10
N_VALUE_ESTIMATE_ITERATIONS = 100
MAX_STEPS_PER_TRAJECTORY = 100
MAX_CONJUGATE_GRADIENT_STEPS = 10
LINE_SEARCH_STEPS = 10


def trpo(env, γ, λ, δ):
    # Initialize policy net
    π_net = PolicyNet(hidden_dim=N_HIDDEN_FEATURES,
                      n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    # LR is 1 because TRPO uses line search for each step's LR
    π_state = create_sgd_train_state(π_net, rng, η=1.0, features=N_FEATURES)
    del rng

    # Initialize value function
    V_π = {s: 0.0 for s in env.S}

    for step in range(TRAIN_STEPS):
        V_π = td_lambda_value_estimate(env, π_state, V_π, γ, λ)
        print('Value estimate at step', step)
        print_grid(V_π)
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
            x, a_idx, dt = np.array(x), np.array(a_idx), np.array(dt)
            grads = policy_gradient(π_state, dt, x, a_idx)
            Fvp = fisher_vector_product(π_state, dt, x, a_idx)
            g_nat = conjugate_gradient(grads, Fvp)
            g_nat = scale_step_size(g_nat, Fvp)
            π_state = line_search_update(π_state, g_nat, x, a_idx, dt, δ)
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


def td_lambda_value_estimate(env, π_state, V, γ, λ):
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


def create_loss_fn(π_state, r, x, a_idx):
    def loss_fn(params):
        a_logits = π_state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits,
                                jnp.expand_dims(a_idx, axis=-1),
                                axis=1)
        return jnp.sum(
            jax.vmap(
                lambda r_i, a_i: jax.lax.cond(
                    r_i < 0.0,
                    lambda: -r_i * jnp.log(1.0 - a_i),
                    lambda: r_i * jnp.log(a_i),
                )
            )(r, a)
        )
    return loss_fn


@jax.jit
def policy_gradient(π_state, r, x, a_idx):
    loss_fn = create_loss_fn(π_state, r, x, a_idx)
    return jax.grad(loss_fn)(π_state.params)


def fisher_vector_product(π_state, r, x, a_idx):
    @jax.jit
    def fvp(v):
        loss_fn = create_loss_fn(π_state, r, x, a_idx)
        # Compute Hessian
        H = jax.jvp(
            jax.grad(loss_fn),
            [tree_to_float64(π_state.params)],
            [tree_to_float64(v)],
        )[1]
        # Fisher matrix is negative Hessian
        return jax.tree_map(lambda x: -x, H)
    return fvp


def tree_to_float64(pytree):
    return jax.tree_map(lambda x: x.astype(np.float64), pytree)


def conjugate_gradient(grads, Fvp, Ɛ=1e-2):
    """
    Use conjugate gradients to solve for x in:

    F @ x = g

    For an initial guess, set x=g and compute
    r = g - F @ g

    """
    g_leaves, g_tree, g = unravel(grads)
    x = g
    Fg = unravel(Fvp(grads))[2]
    r = g - Fg
    p = r
    n_iter = 0
    r_sq = np.dot(r, r)
    for _ in range(MAX_CONJUGATE_GRADIENT_STEPS):
        error = r_sq ** 0.5
        if error < Ɛ:
            break
        n_iter += 1
        p_raveled = ravel(g_leaves, g_tree, p)
        p_prime = unravel(Fvp(p_raveled))[2]
        α = r_sq / np.dot(p, p_prime)
        x += α * p
        r -= α * p_prime
        r_sq_prime = np.dot(r, r)
        β = r_sq_prime / r_sq
        p = r + β * p
        r_sq = r_sq_prime
    return ravel(g_leaves, g_tree, x)


def unravel(grads):
    g_leaves, g_tree = jax.tree_util.tree_flatten(grads)
    g_flat = []
    for leaf in g_leaves:
        g_flat.extend(leaf.flatten())
    g_flat = np.array(g_flat, dtype=np.float64)
    return g_leaves, g_tree, g_flat


def ravel(g_leaves, g_tree, g_flat):
    i = 0
    unflatten_input = []
    for shape in [x.shape for x in g_leaves]:
        di = np.prod(shape)
        unflatten_input.append(
            g_flat[i:i+di].reshape(shape))
        i += di
    return jax.tree_util.tree_unflatten(g_tree, unflatten_input)


def scale_step_size(g_nat, Fvp):
    _, _, x = unravel(g_nat)
    _, _, Hx = unravel(Fvp(g_nat))
    scale = np.sqrt(2.0 / (np.abs(np.dot(x, Hx)) + 1e-12))
    return jax.tree_map(lambda x: x * scale, g_nat)


def line_search_update(π_state, g_nat, x, a_idx, dt, δ):
    for j in range(LINE_SEARCH_STEPS):
        α = 10.0 ** -j
        # Negative sign for NN opt which does gradient descent
        g_nat_step = jax.tree_map(lambda x: -α * x, g_nat)
        π_prime_state = π_state.replace().apply_gradients(grads=g_nat_step)
        L = importance_weighted_policy_advantage(π_state, π_prime_state, x,
                                                 a_idx, dt)
        if L < 0:
            continue
        d_kl = kl_divergence(π_state, π_prime_state, np.array(x))
        if d_kl < δ:
            print('accepted update', j)
            return π_prime_state
    print('no update accepted')
    return π_state


@jax.jit
def importance_weighted_policy_advantage(π_state, π_prime_state, x, a_idx, dt):
    π_logits = π_state.apply_fn({'params': π_state.params}, x)
    π_prime_logits = π_prime_state.apply_fn({'params': π_prime_state.params}, x)
    π_a = jnp.take_along_axis(π_logits,
                              jnp.expand_dims(a_idx, axis=-1),
                              axis=1)
    π_prime_a = jnp.take_along_axis(π_prime_logits,
                                    jnp.expand_dims(a_idx, axis=-1),
                                    axis=1)
    return jnp.sum((π_prime_a / π_a - 1.0) * dt)


@jax.jit
def kl_divergence(π_state, π_prime_state, x):
    p_logits = π_state.apply_fn({'params': π_state.params}, x)
    q_logits = π_prime_state.apply_fn({'params': π_prime_state.params}, x)
    return jnp.sum(p_logits * jnp.log2(p_logits / q_logits))


def optimal_policy(π_state, S, A, ϕ):
    π = {}
    for s in S:
        x = ϕ[s]
        a_logits = π_state.apply_fn({'params': π_state.params}, np.array([x]))[0]
        π[s] = A[np.argmax(a_logits)]
    return π


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75
    λ = 0.6

    # Trust region size
    δ = 5.0

    π_state = trpo(env, γ, λ, δ)
    π_opt = optimal_policy(π_state, env.S, env.A, env.ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
