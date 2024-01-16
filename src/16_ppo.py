"""
Implementation of Proximal Policy Optimization (PPO)
====================================================
This implementation uses PPO clip with adaptive KL divergence penalty.
It uses TD(λ) for value estimation and each action's advantage is
estimated using the temporal difference between the new and old states.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 γ : Discount factor
 λ : TD(λ) parameter
 V : State value function
 π : Agent policy
 ϕ : Non-linear features from environment
 δ : Trust region size parameter
 β : Adaptive KL-divergence penalty scale

Result:
-------
Optimal policy:
Action.Right	 Action.Right	 Action.Right	 Action.Right
Action.Left	 Action.Left	 Action.Left	 Action.Right
Action.Left	 Action.Left	 Action.Left	 Action.Left
Action.Left	 Action.Left	 Action.Left	 Action.Left

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util.display import print_grid
from util.gridworld import GridWorld
from util.jax import MLP, Metrics, TrainState

np.random.seed(42)
jax.config.update('jax_enable_x64', True)


N_FEATURES = 8
N_HIDDEN_FEATURES = 4 * N_FEATURES
N_HIDDEN_LAYERS = 2
N_ACTIONS = 4
LEARNING_RATE = 1e-3
TRAIN_STEPS = 200
N_TRAJECTORIES_PER_STEP = 10
N_VALUE_ESTIMATE_ITERATIONS = 100
MAX_STEPS_PER_TRAJECTORY = 100
N_UPDATES_PER_STEP = 10


def ppo(env, γ, λ, δ, β0):
    # Initialize policy net
    π_net = PolicyNet(hidden_dim=N_HIDDEN_FEATURES,
                      n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    π_state = create_train_state(π_net, rng)
    del rng

    # Initialize value function
    V_π = {s: 0.0 for s in env.S}

    # Adaptive KL-divergence penalty scale
    β = β0

    for step in range(TRAIN_STEPS):
        V_π = td_lambda_value_estimate(env, π_state, V_π, γ, λ)
        print('Value estimate at step', step)
        print_grid(V_π)
        step_x = []
        step_a_idx = []
        step_a_logits = []
        step_dt = []
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

                step_x.append(x)
                step_a_idx.append(a_idx)
                step_a_logits.append(a_logits)
                step_dt.append(dt)
        x = np.array(step_x, dtype=np.float64)
        a_idx = np.array(step_a_idx, dtype=np.int32)
        a_logits = np.array(step_a_logits, dtype=np.float64)
        dt = np.array(step_dt, dtype=np.float64)

        # Update model
        for _ in range(N_UPDATES_PER_STEP):
          ppo_grads = ppo_clip_gradients(π_state, x, a_idx, a_logits, dt)
          kl_grads = kl_divergence_gradients(π_state, x, a_logits)
          grads = jax.tree_map(lambda p, k: -p + β * k, ppo_grads, kl_grads)
          π_state = π_state.apply_gradients(grads=grads)

        # Update KL penalty
        d_kl = kl_divergence(π_state, π_state.params, x, a_logits)
        print('KL divergence:', d_kl)
        if d_kl >= 1.5 * δ:
            β *= 2.0
            print('increasing β to', β)
        elif d_kl <= δ / 1.5:
            β /= 2.0
            print('decreasing β to', β)
        else:
            print('no change to β')
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


def create_train_state(π_net, rng, η=LEARNING_RATE, β1=0.9, β2=0.99):
    params = π_net.init(rng, jnp.ones([1, N_FEATURES]))['params']
    tx = optax.adam(η, β1, β2)
    return TrainState.create(
        apply_fn=π_net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


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


@jax.jit
def ppo_clip_gradients(π_state, x, a_idx, a_logits, dt, Ɛ=1e-1):
    def loss_fn(params):
        logits = π_state.apply_fn({'params': params}, x)
        a_prob = jnp.take_along_axis(a_logits,
                                     jnp.expand_dims(a_idx, axis=-1),
                                     axis=1)
        a_prime_prob = jnp.take_along_axis(logits,
                                           jnp.expand_dims(a_idx, axis=-1),
                                           axis=1)
        r = a_prime_prob / a_prob
        return jnp.sum(jnp.minimum(r * dt, jnp.clip(r, 1.0 - Ɛ, 1.0 + Ɛ) * dt))
    return jax.grad(loss_fn)(π_state.params)


@jax.jit
def kl_divergence_gradients(π_state, x, q_logits):
    def loss_fn(params):
        return kl_divergence(π_state, params, x, q_logits)
    return jax.grad(loss_fn)(π_state.params)


@jax.jit
def kl_divergence(π_state, params, x, q_logits):
    p_logits = π_state.apply_fn({'params': params}, x)
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

    # Adaptive KL divergence trust-region threshold
    δ = 0.1
    # Initial value for KL divergence penalty scale
    β0 = 1.0

    π_state = ppo(env, γ, λ, δ, β0)
    π_opt = optimal_policy(π_state, env.S, env.A, env.ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
