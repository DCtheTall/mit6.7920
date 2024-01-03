"""
Implementation of Actor-Critic
==============================
This file implements vanilla Actor-Critic using shared neural network parameters.

Terms:
 S : Set of all states in the MDP
 A : Set of all actions
 γ : Discount factor
 Q : State-action value function
 π : Agent policy
 ϕ : Non-linear features from environment

Result
------
Optimal policy:
Action.Up	 Action.Up	 Action.Up	 Action.Right
Action.Up	 Action.Left	 Action.Left	 Action.Up
Action.Left	 Action.Left	 Action.Left	 Action.Down
Action.Up	 Action.Left	 Action.Down	 Action.Down
Optimal value function:
0.05547957750818869	 0.09679753546955988	 0.349915101235855	 2.7741836203542753
-0.09539952916895389	 -0.2228283860463518	 -0.7845030576699474	 -3.692328350054204
-0.0774243971347982	 -0.19235196919886166	 -0.483701472415494	 -0.9912019012305577
-0.033861328623924575	 -0.11558709175320808	 -0.27065158535773737	 -0.4871237806125439

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from util.display import print_grid
from util.jax import MLP, create_sgd_train_state
from util.gridworld import GridWorld

np.random.seed(42)
jax.config.update('jax_enable_x64', True)


N_FEATURES = 8
N_ACTIONS = 4
N_HIDDEN_LAYERS = 2
N_HIDDEN_FEATURES = 4 * N_FEATURES
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-2
N_TRAJECTORIES = 1000


def actor_critic(env, γ, T=100):
    # Initialize critic first
    Q_net = Critic(hidden_dim=N_HIDDEN_FEATURES,
                   n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(42)
    Q_state = create_sgd_train_state(Q_net, rng, η=CRITIC_LEARNING_RATE,
                                     features=N_FEATURES)
    del rng

    # Initialize actor but its parameters will be copied
    # from the critic after the first step
    π_net = Actor(hidden_dim=N_HIDDEN_FEATURES,
                  n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(0)
    π_state = create_sgd_train_state(π_net, rng, η=ACTOR_LEARNING_RATE,
                                     features=N_FEATURES)
    del rng

    for _ in range(N_TRAJECTORIES):
        s = env.start
        for _ in range(T):
            x = env.ϕ[s]
            a_logits = π_state.apply_fn({'params': π_state.params},
                                        np.array([x]))[0]
            a_idx = np.random.multinomial(1, pvals=a_logits)
            a_idx = np.argmax(a_idx)
            a = env.A[a_idx]
            r = env.R[s]
            s_prime = env.step(s, a)
            x_prime = env.ϕ[s_prime]
            a_prime_logits = π_state.apply_fn({'params': π_state.params},
                                              np.array([x_prime]))[0]
            a_prime_idx = np.random.multinomial(1, pvals=a_prime_logits)
            a_prime_idx = np.argmax(a_prime_idx)

            # Compute current Q values
            q_values = Q_state.apply_fn({'params': Q_state.params},
                                        np.array([x, x_prime]))
            q = q_values[0][a_idx]
            q_prime = q_values[1][a_prime_idx]

            # Update critic
            dt = temporal_difference(r, γ, q, q_prime)
            grads = compute_critic_gradients(Q_state, dt, np.array([x]),
                                             np.array([a_idx]))
            Q_state = Q_state.apply_gradients(grads=grads)
            # Copy update to policy net
            π_state = copy_network_params(from_net=Q_state, to_net=π_state)

            # Update actor
            grads = compute_actor_gradients(π_state, q, np.array([x]),
                                            np.array([a_idx]))
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


@jax.jit
def compute_actor_gradients(state, q, x, a_idx):
    def loss_fn(params):
        a_logits = state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits, jnp.expand_dims(a_idx, axis=-1),
                                axis=1)
        return q * jax.lax.cond(
            q < 0.0,
            lambda: jnp.sum(jnp.log(1.0 - a)),
            lambda: -jnp.sum(jnp.log(a)),
        )
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    return grads


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

    # Discount factor
    γ = 0.75

    π_state, Q_state = actor_critic(env, γ)
    π_opt = optimal_policy(π_state, env.S, env.A, env.ϕ)
    V_opt = optimal_value_function(Q_state, env.S, env.ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
    print('Optimal value function:')
    print_grid(V_opt)
