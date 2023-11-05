"""
Implementation of Natural Policy Gradients (NPG)
================================================
NPG improves on policy gradient methods by taking into account the geometry
of the parameter space you are training a policy with.

It does so by multiplying the gradients by the inverse Fisher information
matrix, the Hessian of the KL-divergence between the current and candidate
policy. It can be shown that the Fisher matrix can be approximated using

E(∇log(π(a|s)) x ∇log(π(a|s)))

i.e. the expectation of the outer product of the policy gradient.

This algorithm is able to converge to a decent policy after just 1,000
trajectories, less than 20% of what REINFORCE required.

Result:
-------
Optimal policy:
Action.Up	 Action.Up	 Action.Right	 Action.Right
Action.Left	 Action.Left	 Action.Left	 Action.Right
Action.Left	 Action.Left	 Action.Left	 Action.Down
Action.Left	 Action.Left	 Action.Left	 Action.Down

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
N_HIDDEN_FEATURES = 4 * N_FEATURES
N_HIDDEN_LAYERS = 2
N_ACTIONS = 4
LEARNING_RATE = 1e-3
TRAIN_STEPS = 100
N_TRAJECTORIES_PER_STEP = 10
MAX_STEPS_PER_TRAJECTORY = 100
N_VALUE_ESTIMATE_ITERATIONS = 100


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
    π_net = PolicyNet(hidden_dim=N_HIDDEN_FEATURES,
                      n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    π_state = create_train_state(π_net, rng, η=LEARNING_RATE)
    del rng

    for _ in range(TRAIN_STEPS):
        V_π = td_lambda_value_estimate(env, π_state, γ, λ, ϕ)
        xs = [[], []]
        a_idxs = [[], []]
        dts = [[], []]
        for _ in range(N_TRAJECTORIES_PER_STEP):
            s = env.start
            for _ in range(MAX_STEPS_PER_TRAJECTORY):
                x = ϕ[s]
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
            grads = natural_gradients(grads, δ, np.array(dt))
            π_state = π_state.apply_gradients(grads=grads)

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


def create_train_state(net, rng, η):
    params = net.init(rng, jnp.ones([1, N_FEATURES]))['params']
    tx = optax.sgd(η)
    return TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


def td_lambda_value_estimate(env, π_state, γ, λ, ϕ):
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
        x = ϕ[s]
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
def policy_gradient(π_state, dts, xs, a_idxs):
    def loss_fn(params, r, x, a_idx):
        a_logits = π_state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits,
                                jnp.expand_dims(a_idx, axis=-1),
                                axis=0)
        return jax.lax.cond(
                r < 0.0,
                lambda: -r * jnp.sum(jnp.log(1.0 - a)),
                lambda: r * jnp.sum(jnp.log(a)),
            )
    return jax.vmap(
        lambda dt, x, a_idx: jax.grad(loss_fn)(π_state.params, dt, x, a_idx)
    )(dts, xs, a_idxs)


def natural_gradients(grads, δ, dt):
    g_leaves, g_tree = jax.tree_util.tree_flatten(grads)
    n = g_leaves[0].shape[0]
    g_flat = [[] for _ in range(n)]
    for leaf in g_leaves:
        for i, grad in enumerate(leaf):
            g_flat[i].extend(grad.flatten())
    g_flat = np.array(g_flat)
    g_nat = compute_natural_gradient(g_flat, δ, dt)

    # Re-ravel gradients into PyTree
    i = 0
    unflatten_input = []
    for shape in [x.shape for x in g_leaves]:
        shape = shape[1:]
        di = np.prod(shape)
        unflatten_input.append(g_nat[i:i+di].reshape(shape))
        i += di
    return jax.tree_util.tree_unflatten(g_tree, unflatten_input)


@jax.jit
def compute_natural_gradient(g_flat, δ, dt):
    F = jnp.mean(jnp.array([jnp.outer(g, g) for g in g_flat]), axis=0)
    F_inv = jnp.linalg.pinv(F)

    # Scale summed gradient
    g_nat = jnp.einsum('ab,cb->ca', F_inv, g_flat)

    # Scale step size
    scale = jnp.mean(jnp.einsum('ab,ab->a', g_nat, g_flat))
    step_size = jnp.sqrt(2 * δ / (jnp.abs(scale) + 1e-12))
    g_nat *= step_size

    # Negative sign for NN optimizer
    return -jnp.sum(g_nat, axis=0)


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

    π_state = natural_policy_gradient(env, γ, λ, δ, ϕ)
    π_opt = optimal_policy(π_state, env.S, env.A, ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
