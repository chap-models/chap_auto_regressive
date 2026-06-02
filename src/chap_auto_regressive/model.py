"""The public model and its trained predictor.

[`AutoRegressiveModel`][chap_auto_regressive.model.AutoRegressiveModel] is the estimator CHAP
trains; its ``train`` method returns a
[`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor] that produces probabilistic
forecasts. Both share the same output head — the network emits two channels per
period (``eta``) which [`nb_head`][chap_auto_regressive.distributions.nb_head] turns into a
negative-binomial distribution that is then sampled.

The public API speaks tidy :class:`pandas.DataFrame` objects (one row per location
and time period); the model itself has no dependency on chap-core.
"""

import pickle
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .data_loader import DataSet as DLDataSet
from .data_loader import SimpleDataLoader
from .distributions import nb_head
from .rnn_model import model_makers
from .trainer import Trainer
from .transforms import ZScaler, get_series, location_groups


def _forecast_frame(future: pd.DataFrame, samples: Any) -> pd.DataFrame:
    """Assemble the prediction output frame from per-location samples.

    Args:
        future: The future-covariate frame the forecast was made for.
        samples: Sampled counts with a leading location axis, each entry shaped
            ``(periods, n_samples)`` (aligned with
            [`location_groups`][chap_auto_regressive.transforms.location_groups]).

    Returns:
        A long frame with columns ``time_period``, ``location`` and one
        ``sample_i`` column per draw.
    """
    groups = list(location_groups(future))
    time_period = groups[0][1]["time_period"].to_numpy()
    frames = []
    for (location, _sub), location_samples in zip(groups, samples):
        location_samples = np.asarray(location_samples)
        frame = pd.DataFrame({f"sample_{i}": location_samples[:, i] for i in range(location_samples.shape[-1])})
        frame.insert(0, "location", location)
        frame.insert(0, "time_period", time_period)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class FlaxPredictor:
    """A trained model that turns history and future covariates into samples.

    Produced by [`AutoRegressiveModel.train`][chap_auto_regressive.model.AutoRegressiveModel.train].
    It holds the fitted network parameters and the feature scaler, and can be
    pickled to disk and reloaded.

    Attributes:
        distribution_head: Maps the network's two-channel output to a sampling
            distribution (the negative-binomial head).
    """

    distribution_head = staticmethod(nb_head)

    def __init__(self, params: Any, transform: Callable, model: Any, prediction_length: int, context_length: int):
        self.model = model
        self._params = params
        self._transform = transform
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.rng_key = jax.random.PRNGKey(1234)

    def get_samples(self, eta: Any, n_samples: int) -> Any:
        """Draw samples from the output distribution for each period.

        Args:
            eta: The network's per-period output, shape ``(..., periods, 2)``.
            n_samples: Number of samples to draw per period.

        Returns:
            Sampled counts, with a leading sample axis of length ``n_samples``.
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))

    def predict(self, historic: pd.DataFrame, future: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
        """Forecast the future periods as samples per location.

        The last ``context_length`` periods of history (with observed cases) are
        concatenated with the future covariates, run through the network, and the
        forecast-horizon outputs are sampled.

        Args:
            historic: Recent history including observed ``disease_cases``.
            future: The periods to forecast, with covariates but no cases.
            num_samples: Number of samples to draw per location and period.

        Returns:
            A frame with columns ``time_period``, ``location`` and one ``sample_i``
            column per draw.
        """
        assert historic["location"].unique().tolist() == future["location"].unique().tolist()
        x, _ = get_series(future)
        prev_values, prev_y = get_series(historic)
        prev_values = prev_values[:, -self.context_length :]
        prev_y = prev_y[:, -self.context_length :]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length)
        dataset.set_transform(self._transform)
        x, y = dataset.prediction_instance()
        eta = self.model.apply(self._params, x, y)
        n_prev = prev_values.shape[1]
        samples = self.get_samples(eta[:, n_prev - 1 :], num_samples)
        return _forecast_frame(future, samples)

    def save(self, path: str) -> None:
        """Pickle the trained parameters and feature scaler to ``path``.

        Args:
            path: Destination file path.
        """
        with open(path, "wb") as f:
            pickle.dump((self._params, self._transform), f)

    @classmethod
    def load(cls, path: str, *args: Any, **kwargs: Any) -> "FlaxPredictor":
        """Reload a predictor previously written by ``save``.

        Args:
            path: Path to the pickled parameters and scaler.
            *args: Forwarded to the constructor (``model``, ``prediction_length``,
                ``context_length``).
            **kwargs: Forwarded to the constructor.

        Returns:
            The reconstructed ``FlaxPredictor``.
        """
        with open(path, "rb") as f:
            params, transform = pickle.load(f)
        return cls(params, transform, *args, **kwargs)


class AutoRegressiveModel:
    """Deep auto-regressive forecaster for disease case counts.

    The estimator wraps the [`ARModel2`][chap_auto_regressive.rnn_model.ARModel2] network: it
    extracts and scales features, slices each location's series into
    ``context_length + prediction_length`` windows, and fits the network by
    maximizing the negative-binomial likelihood of the observed cases. Calling
    ``train`` returns a [`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor].

    All input/output is via tidy :class:`pandas.DataFrame` objects with the columns
    ``location``, ``time_period``, ``rainfall``, ``mean_temperature``,
    ``population`` and (for training) ``disease_cases``.

    Attributes:
        rnn_model_name: Which [`model_makers`][chap_auto_regressive.rnn_model.model_makers]
            architecture to build (``"base"`` or ``"multi_value"``).
        prediction_length: Number of periods to forecast ahead.
        n_iter: Number of training epochs.
        context_length: Number of past periods read as context.
        learning_rate: Adam learning rate.
        distribution_head: Maps the network output to the sampling distribution.
    """

    rnn_model_name = "base"
    prediction_length = 3
    n_iter: int = 1000
    context_length = 24
    learning_rate = 1e-4
    distribution_head = staticmethod(nb_head)

    def __init__(self, rng_key=jax.random.PRNGKey(100)):
        self.rng_key = rng_key
        self._model = None
        self._n_locations = None
        self._params = None
        self._validation_loader = None

    def set_model(self, model: Any) -> None:
        """Override the network instance instead of building one lazily.

        Args:
            model: A flax module to use in place of the default architecture.
        """
        self._model = model

    @property
    def model(self) -> Any:
        """The flax network, built lazily from ``rnn_model_name`` on first access."""
        if self._model is None:
            self._model = model_makers[self.rnn_model_name](self._n_locations)
        return self._model

    def _get_dataset(self, data: pd.DataFrame) -> DLDataSet:
        """Build a windowed dataset from the input frame."""
        x, y = get_series(data)
        return DLDataSet(x, y, forecast_length=self.prediction_length, context_length=self.context_length)

    def set_validation_data(self, historic: pd.DataFrame, future: pd.DataFrame) -> None:
        """Provide a held-out window for validation-loss reporting during training.

        The supplied history and future are concatenated into a single window and
        wrapped in a loader the trainer evaluates periodically.

        Args:
            historic: History including observed cases.
            future: The matching future periods with their covariates.
        """
        x, y = get_series(historic)
        x = x[:, -self.context_length :]
        y = y[:, -self.context_length :]
        fx, fy = get_series(future)
        full_x = np.concatenate([x, fx], axis=1)
        full_y = np.concatenate([y, fy], axis=1)
        self._validation_loader = SimpleDataLoader(
            DLDataSet(full_x, full_y, forecast_length=self.prediction_length, context_length=self.context_length)
        )

    def train(self, data: pd.DataFrame) -> FlaxPredictor:
        """Fit the model and return a predictor.

        Extracts features, fits the feature scaler, builds the windowed loader,
        and runs the [`Trainer`][chap_auto_regressive.trainer.Trainer] to minimize the negative
        log-likelihood.

        Args:
            data: Training frame, one row per location and period, with observed
                ``disease_cases``.

        Returns:
            A [`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor] holding the trained
            parameters and the fitted scaler.
        """
        data_set = self._get_dataset(data)
        self._transform = ZScaler.from_data(data_set)
        data_set.set_transform(self._transform)
        self._n_locations = data["location"].nunique()
        data_loader = SimpleDataLoader(data_set)
        trainer = Trainer(
            self.model, self.n_iter, learning_rate=self.learning_rate, validation_loader=self._validation_loader
        )
        state = trainer.train(data_loader, self._loss)
        self._params = state.params
        return FlaxPredictor(self._params, self._transform, self.model, self.prediction_length, self.context_length)

    def load_predictor(self, path: str) -> FlaxPredictor:
        """Load a saved predictor, attaching this model's architecture and lengths.

        Args:
            path: Path to a file written by
                [`FlaxPredictor.save`][chap_auto_regressive.model.FlaxPredictor.save].

        Returns:
            The reconstructed [`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor].
        """
        return FlaxPredictor.load(path, self.model, self.prediction_length, self.context_length)

    def predict(self, historic: pd.DataFrame, future: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
        """Forecast the future periods directly from this (trained) model.

        Equivalent to the predictor's ``predict``; useful when forecasting from a
        model trained in the same process.

        Args:
            historic: Recent history including observed ``disease_cases``.
            future: The periods to forecast, with covariates but no cases.
            num_samples: Number of samples per location and period.

        Returns:
            A frame with columns ``time_period``, ``location`` and one ``sample_i``
            column per draw.
        """
        assert historic["location"].unique().tolist() == future["location"].unique().tolist()
        x, _ = get_series(future)
        prev_values, prev_y = get_series(historic)
        prev_values = prev_values[:, -self.context_length :]
        prev_y = prev_y[:, -self.context_length :]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length)
        dataset.set_transform(self._transform)
        x, y = dataset.prediction_instance()
        eta = self.model.apply(self._params, x, y)
        n_prev = prev_values.shape[1]
        samples = self.get_samples(eta[:, n_prev - 1 :], num_samples)
        return _forecast_frame(future, samples)

    def loss_func(self, eta_pred: Any, y_true: Any) -> jnp.ndarray:
        """Per-period negative log-likelihood of the observed cases.

        Args:
            eta_pred: The network output, shape ``(..., periods, 2)``.
            y_true: The window's target; ``y_true[..., 0]`` is the lagged value, so
                the likelihood is evaluated against ``y_true[..., 1:]``.

        Returns:
            The element-wise negative log-likelihood under the negative binomial.
        """
        return -self.distribution_head(eta_pred).log_prob(y_true[..., 1:])

    def _loss(self, y_pred: Any, y_true: Any) -> Any:
        """Aggregate the per-period loss, emphasizing the forecast horizon."""
        L = self.loss_func(y_pred, y_true)
        return jnp.mean(L[:, -self.prediction_length :]) / self.context_length + jnp.mean(
            L[:, -self.prediction_length :]
        )

    def get_samples(self, eta: Any, n_samples: int) -> Any:
        """Draw samples from the output distribution for each period.

        Args:
            eta: The network's per-period output, shape ``(..., periods, 2)``.
            n_samples: Number of samples to draw per period.

        Returns:
            Sampled counts, with a leading sample axis of length ``n_samples``.
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))
