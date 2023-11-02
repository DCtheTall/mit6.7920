"""
Implementation of Natural Policy Gradients (NPG)
================================================

"""

import copy
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
TRAINING_STEPS = 1
N_TRAJECTORIES_PER_STEP = 1
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


def natural_policy_gradient(env, γ, δ, ϕ):
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

        # Estimate of state transition probabilities
        # given current policy
        P_π = {}

        # Q-value estimation using current policy
        all_rewards = []
        all_grad_inputs = []
        for _ in range(N_TRAJECTORIES_PER_STEP):
            cur_rewards = []
            cur_grad_inputs = []
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
                P_π[(s, s_prime)] = P_π.get((s, s_prime), 0.0) + 1.0
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

                cur_rewards.append(q)
                cur_grad_inputs.append((x, a_idx))

                if env.is_terminal_state(s):
                    break
                s = s_prime
            all_rewards.append(cur_rewards)
            all_grad_inputs.append(cur_grad_inputs)

        # Estimate 
        P_π = normalize_state_transition_probs(P_π)
        d_π = discounted_future_state_distribution(env.S, P_π, env.µ, γ)
        
        # Policy improvement
        π_prime_state = copy_network_params(from_net=Q_state, to_net=π_state)

        # Compute Fisher information matrix
        x = np.array([
            np.concatenate([ϕ[s], np.array([d_π[s]])])
            for s in env.S
        ])
        F = fisher_information_matrix(π_state, π_prime_state,
                                      x[:,:N_FEATURES],
                                      x[:,N_FEATURES:])
        F_inv = jax.tree_map(lambda f: np.linalg.pinv(f), F)
        

        # Apply natural policy gradient update
        all_rewards = discount_all_rewards(all_rewards, γ)
        grads = mean_policy_gradient(π_state, all_rewards, all_grad_inputs)
        grads = natural_gradients(F_inv, grads, δ)
        π_state = π_state.apply_gradients(grads=grads)

    return π_state


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
    return jax.grad(loss_fn)(Q_state.params)


def normalize_state_transition_probs(P_π):
    total = sum(P_π.values())
    return {k: v / total for k, v in P_π.items()}


def discounted_future_state_distribution(S, P_π, µ, γ):
    d_π = {s: 0.0 for s in S}
    while True:
        d_π_prime = {
            s: µ[s] + sum(
                γ * P_π.get((s, s_prime), 0.0) * d_π[s_prime]
                for s_prime in S
            )
            for s in S
        }
        if all(np.isclose(d_π[s], d_π_prime[s]) for s in S):
            break
        else:
            d_π = d_π_prime
    return {s: (1 - γ) * d_π[s] for s in S}


@jax.jit
def fisher_information_matrix(π_state, π_prime_state, x, p_x):
    def kl_divergence(params):
        p_logits = π_state.apply_fn({'params': params}, x)
        q_logits = π_prime_state.apply_fn({'params': π_prime_state.params}, x)
        return jnp.sum(p_x * p_logits * jnp.log(p_logits / q_logits))
    return jax.hessian(kl_divergence)(π_state.params)


def discount_all_rewards(all_rewards, γ):
    all_discounted = [discount_rewards(r, γ) for r in all_rewards]
    return all_discounted


def discount_rewards(rewards, γ):
    result = [None] * len(rewards)
    r_sum = 0.0
    for i in range(len(rewards)-1, -1, -1):
        r_sum *= γ
        r_sum += rewards[i]
        result[i] = r_sum
    return result


def mean_policy_gradient(π_state, all_rewards, all_grad_inputs):
    grads = None
    m = len(all_rewards)
    for cur_rewards, cur_grad_inputs in zip(all_rewards,
                                            all_grad_inputs):
        n = m * len(cur_rewards)
        for r, (x, a_idx) in zip(cur_rewards, cur_grad_inputs):
            cur_grads = policy_gradient(π_state, r, np.array([x]), np.array([a_idx]))
            if grads is None:
                grads = copy.deepcopy(cur_grads)
                grads = jax.tree_map(lambda x: x / n, grads)
            else:
                grads = jax.tree_map(lambda x, y: x + y / n, grads, cur_grads)
    return grads


@jax.jit
def policy_gradient(π_state, r, x, a_idx):
    def loss_fn(params):
        a_logits = π_state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits,
                                jnp.expand_dims(a_idx, axis=-1),
                                axis=1)
        return r * jax.lax.cond(
            r < 0.0,
            lambda: jnp.sum(jnp.log(1.0 - a)),
            lambda: -jnp.sum(jnp.log(a)),
        )
    return jax.grad(loss_fn)(π_state.params)


def natural_gradients(F_inv, grads, δ):
    F_inv_flat, _ = jax.flatten_util.ravel_pytree(F_inv)
    g_flat, _ = jax.flatten_util.ravel_pytree(grads)

    # Multiply gradient with inverse Fisher information matrix
    F_inv_flat = F_inv_flat.reshape([len(g_flat)] * 2)
    g_nat = F_inv_flat @ g_flat
    # Multiply by step size
    g_nat *= np.sqrt(2 * δ / np.abs(g_flat.transpose() @ g_nat + 1e-12))

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


if __name__ == '__main__':
    env = GridWorld(size=4)

     # Non-linear features
    ϕ = features(env)

    # Discount factor
    γ = 0.75

    # Step size scale
    δ = 1.0

    π_state = natural_policy_gradient(env, γ, δ, ϕ)
    π_opt = optimal_policy(π_state, env.S, env.A, ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
