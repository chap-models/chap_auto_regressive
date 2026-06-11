"""The recurrent network architecture.

The model that the auto-regressive estimator trains is
[`ARModel2`][chap_auto_regressive.rnn_model.ARModel2], assembled by the
[`model_makers`][chap_auto_regressive.rnn_model.model_makers] factories. It processes a window
in four steps: a per-location [`Preprocess`][chap_auto_regressive.rnn_model.Preprocess] stage,
an auto-regressive join ([`ARAdder`][chap_auto_regressive.rnn_model.ARAdder]), and two stacked
``SimpleCell`` RNNs that encode the context and decode the forecast horizon.

Array convention: ``batch x location x time x feature``.
"""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import SimpleCell

# Dimensions: batch_dim x location_dim x time_dim x feature_dim

#: Recurrent cell types selectable by name. ``"gru"`` (gated) handles the
#: multi-step forecast horizon markedly better than the plain ``"simple"`` RNN.
CELL_TYPES = {"simple": SimpleCell, "gru": nn.GRUCell}


class Preprocess(nn.Module):
    """Per-location feature embedding and projection.

    Each location is given a learned embedding vector (so the shared network can
    still distinguish regions), which is concatenated onto the per-period features
    and passed through a small dense stack with dropout. The result is a
    low-dimensional per-period representation the RNNs consume.

    Attributes:
        n_hidden: Width of the hidden dense layer.
        n_locations: Number of locations (unused at call time; the count is taken
            from the input shape).
        embedding_dim: Size of the per-location embedding.
        output_dim: Size of the produced per-period representation.
        dropout_rate: Dropout probability applied during training.
    """

    n_hidden: int = 4
    n_locations: int = 1
    embedding_dim: int = 4
    output_dim: int = 1
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x: Any, training: bool = False) -> Any:
        """Embed locations, concatenate, and project the features.

        Args:
            x: Features, shape ``(..., locations, time, features)``.
            training: Whether dropout is active.

        Returns:
            The projected features, shape ``(..., locations, time, output_dim)``.
        """
        n_locations = x.shape[-3]
        loc = nn.Embed(num_embeddings=n_locations, features=self.embedding_dim)(jnp.arange(n_locations))
        axis = -2
        loc = jnp.repeat(loc[..., None, :], x.shape[axis], axis=axis)
        if x.ndim == 4:
            loc = jnp.repeat(loc[None, ...], x.shape[0], axis=0)
        x = jnp.concatenate([x, loc], axis=-1)  # batch x embedding_dim
        layers = [self.n_hidden]
        for i in range(len(layers)):
            x = nn.Dense(features=layers[i])(x)
            x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.output_dim)(x)
        return nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)


class ARAdder(nn.Module):
    """The auto-regressive join: append past observations to the features.

    This is the step that makes the model auto-regressive. It concatenates the
    observed target ``y`` onto the processed features so the recurrent encoder
    sees the series' own recent values alongside the covariates.
    """

    @nn.compact
    def __call__(self, x: Any, y: Any) -> Any:
        """Concatenate past targets onto the features.

        Args:
            x: Processed features, shape ``(..., time, features)``.
            y: Observed targets over the context, shape ``(..., n_context)``.

        Returns:
            Features with the lagged target appended on the feature axis.
        """
        n_y = y.shape[-1]
        # log1p the auto-regressive input so it sits on a comparable scale to the
        # z-scored covariate features instead of swamping them with raw counts.
        return jnp.concatenate([jnp.log1p(y)[..., None], x[..., 1 : n_y + 1, :]], axis=-1)


class MultiValueARAdder(nn.Module):
    """Auto-regressive join that also mixes targets across locations.

    A variant of [`ARAdder`][chap_auto_regressive.rnn_model.ARAdder] that, in addition to each
    series' own past, passes the targets through a dense layer across locations so
    a region can borrow signal from the others.
    """

    @nn.compact
    def __call__(self, x: Any, y: Any) -> Any:
        """Concatenate the lagged target and a cross-location mix onto the features.

        Args:
            x: Processed features, shape ``(..., time, features)``.
            y: Observed targets over the context, shape ``(locations, n_context)``.

        Returns:
            Features with both the per-location lag and a learned cross-location
            target mix appended on the feature axis.
        """
        n_y = y.shape[-1]
        collected_y = jnp.moveaxis(nn.Dense(features=y.shape[0])(jnp.moveaxis(y, 0, -1)), -1, 0)
        collected_y = nn.relu(collected_y)
        return jnp.concatenate([collected_y[..., None], y[..., None], x[..., 1 : n_y + 1, :]], axis=-1)


class ARModel2(nn.Module):
    """The full encoder/decoder auto-regressive network.

    The forward pass preprocesses the features, joins in the past targets, encodes
    the context window with one ``SimpleCell`` RNN, then continues from that RNN's
    final state across the forecast horizon with a second ``SimpleCell`` RNN
    (where observed targets are no longer available). Dense layers map the
    combined hidden states to the two output channels (``eta``) consumed by the
    negative-binomial head.

    Attributes:
        preprocess: The per-location feature stage.
        cell_pre: The recurrent cell used to encode the context.
        cell_post: The recurrent cell used to decode the forecast horizon.
        ar_adder: The auto-regressive join module.
        output_dim: Number of output channels per period (2 for the NB head).
        head_features: Width of the hidden dense layer mapping the combined
            encoder/decoder states to ``eta``.
    """

    preprocess: nn.Module
    cell_pre: nn.RNNCellBase
    cell_post: nn.RNNCellBase
    ar_adder: ARAdder = ARAdder()
    output_dim: int = 2
    head_features: int = 6

    @nn.compact
    def __call__(self, x: Any, y: Any, training: bool = False) -> Any:
        """Run the encoder/decoder forward pass over a window.

        Args:
            x: Features over the full window, shape ``(..., time, features)``.
            y: Observed targets over the context, shape ``(..., n_context)``.
            training: Whether dropout in the preprocess stage is active.

        Returns:
            The per-period output ``eta``, shape ``(..., time, output_dim)``.
        """
        x = self.preprocess(x, training=training)
        n_y = y.shape[-1]
        prev_x = self.ar_adder(x, y)
        states = nn.RNN(self.cell_pre)(prev_x)
        new_states = nn.RNN(self.cell_post)(x[..., n_y + 1 :, :], initial_carry=states[..., -1, :])
        x = jnp.concatenate([states, new_states], axis=-2)
        x = nn.Dense(features=self.head_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


def build_network(
    n_locations: int,
    *,
    cell: str = "gru",
    rnn_features: int = 16,
    preprocess_hidden: int = 16,
    preprocess_output: int = 8,
    embedding_dim: int = 8,
    head_features: int = 24,
    dropout_rate: float = 0.2,
) -> ARModel2:
    """Build an [`ARModel2`][chap_auto_regressive.rnn_model.ARModel2] from explicit hyperparameters.

    Args:
        n_locations: Number of locations (passed to the preprocess stage).
        cell: Recurrent cell type, a key of
            [`CELL_TYPES`][chap_auto_regressive.rnn_model.CELL_TYPES] (``"gru"`` or ``"simple"``).
        rnn_features: Hidden width of the encoder and decoder cells.
        preprocess_hidden: Hidden width of the per-location preprocess stack.
        preprocess_output: Output width of the preprocess stage.
        embedding_dim: Size of the per-location embedding.
        head_features: Hidden width of the output head.
        dropout_rate: Dropout probability in the preprocess stage.

    Returns:
        The configured network.
    """
    cell_cls = CELL_TYPES[cell]
    return ARModel2(
        Preprocess(
            n_locations=n_locations,
            n_hidden=preprocess_hidden,
            embedding_dim=embedding_dim,
            output_dim=preprocess_output,
            dropout_rate=dropout_rate,
        ),
        cell_cls(features=rnn_features),
        cell_cls(features=rnn_features),
        head_features=head_features,
    )


# Back-compat: the original named factories. ``"base"`` builds the default network
# (now GRU-based); ``"multi_value"`` swaps in the cross-location auto-regressive join.
model_makers = {
    "base": lambda n_locations: build_network(n_locations),
    "multi_value": lambda n_locations: ARModel2(
        Preprocess(n_locations=n_locations, output_dim=2, dropout_rate=0.2),
        nn.SimpleCell(features=4),
        nn.SimpleCell(features=4),
        ar_adder=MultiValueARAdder(),
    ),
}
