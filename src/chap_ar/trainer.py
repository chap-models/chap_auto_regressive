from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from more_itertools import peekable

from .data_loader import DataLoader


class TrainState(train_state.TrainState):
    key: jax.Array


def l2_regularization(params, scale=1.0):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim == 2) * scale


class Trainer:
    def __init__(self, model, n_iter=3000, learning_rate=1e-5, validation_loader: Optional[DataLoader] = None):
        self.model = model
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._validation_loader = validation_loader

    def train(self, data_loader: DataLoader, loss_fn):
        ix, iar_y, iy = peekable(iter(data_loader)).peek()
        params = self.model.init(jax.random.PRNGKey(0), ix, iar_y, training=False)
        dropout_key = jax.random.PRNGKey(40)

        training_state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optax.adam(self.learning_rate), key=dropout_key
        )

        @jax.jit
        def train_step(state: TrainState, dropout_key, x, ar_y, y) -> Tuple[TrainState, jnp.ndarray]:
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_func(params):
                eta = state.apply_fn(params, x, ar_y, training=True, rngs={"dropout": dropout_train_key})
                return loss_fn(eta, y) + l2_regularization(params, 0.001)

            grad_func = jax.value_and_grad(loss_func)
            loss, grad = grad_func(state.params)
            state = state.apply_gradients(grads=grad)
            return state, loss

        @jax.jit
        def get_validation_loss(state: TrainState, x, ar_y, y):
            return loss_fn(state.apply_fn(state.params, x, ar_y, training=False), y)

        for i in range(self.n_iter):
            total_loss = 0
            for x, ar_y, y in iter(data_loader):
                training_state, cur_loss = train_step(training_state, dropout_key, x, ar_y, y)
                total_loss += cur_loss
            if i % 10 == 0:
                validation_loss = 0
                if self._validation_loader is not None:
                    v_loss = 0
                    for v_x, v_ar, v_y in iter(self._validation_loader):
                        v_loss += get_validation_loss(training_state, v_x, v_ar, v_y)
                    validation_loss = v_loss
                print(f"Loss: {cur_loss}, Validation Loss: {validation_loss}")

        return training_state
