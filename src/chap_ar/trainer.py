"""The training loop.

[`Trainer`][chap_ar.trainer.Trainer] fits a flax network by minimizing a
caller-supplied loss with the Adam optimizer. The per-step update is JIT-compiled
with ``jax.jit`` and uses ``jax.value_and_grad`` for the gradient; a small L2
penalty on the weight matrices is added for regularization.
"""

import logging
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from more_itertools import peekable

from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    """Flax training state extended with a PRNG key for dropout.

    Attributes:
        key: The PRNG key folded per step to drive dropout.
    """

    key: jax.Array


def l2_regularization(params: Any, scale: float = 1.0) -> Any:
    """Sum of squared weight-matrix entries, used as an L2 penalty.

    Only rank-2 parameters (weight matrices) are penalized; biases and other
    parameters are left out.

    Args:
        params: A pytree of model parameters.
        scale: Multiplier applied to the summed squared weights.

    Returns:
        The scaled L2 penalty as a scalar.
    """
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim == 2) * scale


class Trainer:
    """Fits a flax model to windowed data with Adam.

    The trainer initializes the model from the first window, then for ``n_iter``
    epochs runs a JIT-compiled gradient step over every training window. When a
    validation loader is supplied, the validation loss is reported every ten
    epochs.
    """

    def __init__(
        self,
        model: Any,
        n_iter: int = 3000,
        learning_rate: float = 1e-5,
        validation_loader: Optional[DataLoader] = None,
    ):
        """Configure the trainer.

        Args:
            model: The flax module to fit.
            n_iter: Number of epochs (full passes over the windows).
            learning_rate: Adam learning rate.
            validation_loader: Optional loader providing a held-out window for
                periodic validation-loss reporting.
        """
        self.model = model
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._validation_loader = validation_loader

    def train(self, data_loader: DataLoader, loss_fn: Callable) -> "TrainState":
        """Train the model and return the final state.

        Initializes parameters from the first window, then repeatedly applies a
        JIT-compiled Adam step over all windows. Each step adds an L2 penalty and
        folds a fresh dropout key from the step counter for stochastic
        regularization.

        Args:
            data_loader: Yields ``(x, ar_y, y)`` training windows.
            loss_fn: Maps ``(eta, y)`` to a scalar loss (the negative
                log-likelihood under the model's output distribution).

        Returns:
            The final [`TrainState`][chap_ar.trainer.TrainState] holding the
            trained parameters.
        """
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
                logger.info("epoch %d: loss=%s validation_loss=%s", i, cur_loss, validation_loss)

        return training_state
