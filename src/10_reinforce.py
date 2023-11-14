"""
Implementation of REINFORCE Policy Gradient Algorithm
=====================================================
Implementation of REINFORCE policy gradient learning algorithm
for GridWorld 4x4.

Result:
-------
Optimal policy:
Action.Up	 Action.Up	 Action.Up	 Action.Up
Action.Up	 Action.Left	 Action.Left	 Action.Up
Action.Left	 Action.Left	 Action.Left	 Action.Down
Action.Left	 Action.Left	 Action.Left	 Action.Down

"""

import copy
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
LEARNING_RATE = 1e-2
N_TRAJECTORIES_PER_UPDATE = 100
TRAIN_STEPS = 75


def reinforce(env, γ, T=100):
    π_net = PolicyNet(hidden_dim=2*N_FEATURES,
                      n_layers=4)
    rng = jax.random.key(42)
    state = create_train_state(π_net, rng)
    del rng

    for _ in range(TRAIN_STEPS):
        all_rewards = []
        all_grad_inputs = []
        for _ in range(N_TRAJECTORIES_PER_UPDATE):
            cur_rewards = []
            cur_grad_inputs = []
            s = env.start
            for _ in range(T):
                x = env.ϕ[s]
                a_logits = state.apply_fn({'params': state.params},
                                          np.array([x]))[0]
                a_idx = np.random.multinomial(1, pvals=a_logits)
                a_idx = np.argmax(a_idx)
                cur_grad_inputs.append((x, a_idx))

                a = env.A[a_idx]
                s_prime = env.step(s, a)

                r = env.R[s_prime]
                cur_rewards.append(r)

                if env.is_terminal_state(s_prime):
                    break
                s = s_prime
            all_grad_inputs.append(cur_grad_inputs)
            all_rewards.append(cur_rewards)

        all_rewards = discount_all_rewards(all_rewards, γ)
        grads = mean_policy_gradient(state, all_rewards, all_grad_inputs)
        state = state.apply_gradients(grads=grads)
    return state


class PolicyNet(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        x = MLP(features=self.hidden_dim,
                n_layers=self.n_layers)(x)
        x = nn.Dense(features=N_ACTIONS,
                     dtype=jnp.float64)(x)
        # Use softmax so output is probability of each action
        logits = nn.softmax(x)
        return logits


def create_train_state(π_net, rng, η=LEARNING_RATE, β1=0.9, β2=0.99):
    params = π_net.init(rng, jnp.ones([1, N_FEATURES]))['params']
    tx = optax.adam(η, β1, β2)
    return TrainState.create(
        apply_fn=π_net.apply, params=params, tx=tx,
        metrics=Metrics.empty())


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


def mean_policy_gradient(state, all_rewards, all_grad_inputs):
    grads = None
    m = len(all_rewards)
    for cur_rewards, cur_grad_inputs in zip(all_rewards,
                                            all_grad_inputs):
        n = m * len(cur_rewards)
        for r, (x, a_idx) in zip(cur_rewards, cur_grad_inputs):
            cur_grads = policy_gradient(state, r, np.array([x]), np.array([a_idx]))
            if grads is None:
                grads = copy.deepcopy(cur_grads)
                grads = jax.tree_map(lambda x: x / n, grads)
            else:
                grads = jax.tree_map(lambda x, y: x + y / n, grads, cur_grads)
    return grads



@jax.jit
def policy_gradient(state, r, x, a_idx):
    def loss_fn(params):
        a_logits = state.apply_fn({'params': params}, x)
        a = jnp.take_along_axis(a_logits,
                                jnp.expand_dims(a_idx, axis=-1),
                                axis=1)
        return r * jax.lax.cond(
            r < 0.0,
            lambda: jnp.sum(jnp.log(1.0 - a)),
            lambda: -jnp.sum(jnp.log(a)),
        )
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    return grads


def optimal_policy(state, S, A, ϕ):
    π = {}
    for s in S:
        x = ϕ[s]
        a_logits = state.apply_fn({'params': state.params}, np.array([x]))[0]
        π[s] = A[np.argmax(a_logits)]
    return π


if __name__ == '__main__':
    env = GridWorld(size=4)

    # Discount factor
    γ = 0.75

    opt_state = reinforce(env, γ)
    π_opt = optimal_policy(opt_state, env.S, env.A, env.ϕ)

    print('Optimal policy:')
    print_grid(π_opt)
