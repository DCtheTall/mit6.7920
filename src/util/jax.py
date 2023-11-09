"""
Utility functions for using Jax for function approximation

"""

from clu import metrics
from flax import struct
import flax.linen as nn
from flax.training import train_state
import jax.numpy as jnp
import optax
from typing import Callable


Array = jnp.ndarray


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) utility class"""
    features: int
    n_layers: int
    use_bias: bool = True
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for _ in range(self.n_layers):
            x = nn.Dense(self.features, self.use_bias,
                         dtype=jnp.float64)(x)
            x = self.activation(x)
        return x


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_sgd_train_state(net, rng, η, features):
    params = net.init(rng, jnp.ones([1, features]))['params']
    tx = optax.sgd(η)
    return TrainState.create(
        apply_fn=net.apply, params=params, tx=tx,
        metrics=Metrics.empty())
