"""
Implementation of Advantage Actor-Critic (A2C)
==============================================
This implementation uses Jax to implement A2C with
Generalized Advantage Estimation (GAE).

A2C was invented after Asynchronous Advantage Actor-Critic
(aka A3C), and is found to be just as good for learning
and more data efficient than A3C.

Result:
-------
Optimal policy:
Action.Up	 Action.Up	 Action.Down	 Action.Down
Action.Left	 Action.Left	 Action.Left	 Action.Down
Action.Left	 Action.Left	 Action.Left	 Action.Down
Action.Left	 Action.Left	 Action.Left	 Action.Left
Optimal value function:
-0.24725349974234986	 -0.08145740584122518	 0.42139891590555445	 2.554816192694035
-0.3817363674729299	 -0.40616133199433485	 -0.7857642737751733	 -4.174560193698902
-0.38620141797720287	 -0.40257019105638125	 -0.5752793719865422	 -1.1547905627224417
-0.3651403501054477	 -0.40294576413652916	 -0.5105077772981813	 -0.8728789968697053

"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util.display import print_grid
from util.jax import MLP, Metrics, TrainState
from util.gridworld import GridWorld

jax.config.update('jax_enable_x64', True)


N_FEATURES = 8
N_ACTIONS = 4
N_HIDDEN_LAYERS = 2
N_HIDDEN_FEATURES = 4 * N_FEATURES
ACTOR_LEARNING_RATE = 1e-2
CRITIC_LEARNING_RATE = 1e-2
N_TRAJECTORIES = 750


def a2c(env, γ, λ, T=100):
    # Initialize critic first
    V_net = Critic(hidden_dim=N_HIDDEN_FEATURES,
                   n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(13)
    V_state = create_train_state(V_net, rng, η=CRITIC_LEARNING_RATE)
    del rng

    # Initialize actor but its parameters will be copied
    # from the critic after the first step
    π_net = Actor(hidden_dim=N_HIDDEN_FEATURES,
                  n_layers=N_HIDDEN_LAYERS)
    rng = jax.random.key(0)
    π_state = create_train_state(π_net, rng, η=ACTOR_LEARNING_RATE)
    del rng

    for _ in range(N_TRAJECTORIES):
        s = env.start
        # Eligibility traces
        z = {}
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

            v_out = V_state.apply_fn({'params': V_state.params},
                                     np.array([x, x_prime]))
            v = v_out[0]
            v_prime = v_out[1]

            dt = temporal_difference(r, γ, v, v_prime)

            # Update critic using GAE, which uses TD(λ) updates
            z[s] = z.get(s, 0.0) + 1.0
            for sz in z.keys():
                xz = env.ϕ[sz]
                grads = compute_critic_gradients(V_state, z[sz], dt,
                                                 np.array([xz]))
                V_state = V_state.apply_gradients(grads=grads)
                z[sz] *= γ * λ
            # Copy update to policy net
            π_state = copy_network_params(from_net=V_state, to_net=π_state)

            # Update actor
            grads = compute_actor_gradients(π_state, dt, np.array([x]),
                                            np.array([a_idx]))
            π_state = π_state.apply_gradients(grads=grads)
            # Copy update to critic net
            V_state = copy_network_params(from_net=π_state, to_net=V_state)

            if env.is_terminal_state(s):
                break
            s = s_prime
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
def compute_actor_gradients(π_state, dt, x, a_idx):
    def loss_fn(params):
        a_logits = π_state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits, jnp.expand_dims(a_idx, axis=-1),
                                axis=1)
        return dt * jax.lax.cond(
            dt < 0.0,
            lambda: jnp.sum(jnp.log(1.0 - a)),
            lambda: -jnp.sum(jnp.log(a)),
        )
    return jax.grad(loss_fn)(π_state.params)


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


def optimal_value_function(V_state, S, ϕ):
    V = {}
    for s in S:
        x = ϕ[s]
        v = V_state.apply_fn({'params': V_state.params}, np.array([x]))[0]
        V[s] = v
    return V


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75
    λ = 0.6

    π_state, V_state = a2c(env, γ, λ)

    π_opt = optimal_policy(π_state, env.S, env.A, env.ϕ)
    V_opt = optimal_value_function(V_state, env.S, env.ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
    print('Optimal value function:')
    print_grid(V_opt)
