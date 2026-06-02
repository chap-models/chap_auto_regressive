import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import optax
import scipy
from chap_core.datatypes import ClimateHealthTimeSeries, FullData, Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from flax import linen as nn
from flax.training import train_state
from matplotlib import pyplot as plt

from .distributions import NegativeBinomial3, Normal, Poisson, skip_nan_distribution
from .rnn_model import RNNModel
from .transforms import year_position_from_datetime

PoissonSkipNaN = skip_nan_distribution(Poisson)


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y):
    y = y.copy()
    for row in y:
        nans, x = nan_helper(row)
        if np.all(nans):
            row[:] = 0
        else:
            row[nans] = np.interp(x(nans), x(~nans), row[~nans])
    return y


def l2_regularization(params, scale=1.0):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim == 2) * scale


class TrainState(train_state.TrainState):
    key: jax.Array


class FlaxModel:
    model: nn.Module  # = RNNModel()
    n_iter: int = 3000

    def __init__(self, rng_key: jax.random.PRNGKey = jax.random.PRNGKey(100), n_iter: int = None):
        self.rng_key = rng_key
        self._losses = []
        self._params = None
        self._saved_x = None
        self._validation_x = None
        self._validation_y = None
        self._model = None
        self._n_locations = None
        if n_iter is not None:
            self.n_iter = n_iter

    @property
    def model(self):
        if self._model is None:
            self._model = RNNModel(n_locations=self._saved_x.shape[0])
        return self._model

    def set_validation_data(self, data: DataSet[FullData]):
        x, y = self._get_series(data)
        self._validation_x = x
        self._validation_y = y

    def _get_series(self, data: DataSet[FullData]):
        x = []
        y = []
        for series in data.values():
            year_position = [year_position_from_datetime(period.start_timestamp.date) for period in series.time_period]
            x.append(np.array((series.rainfall, series.mean_temperature, series.population, year_position)).T)  # type: ignore
            if hasattr(series, "disease_cases"):
                y.append(series.disease_cases)
        assert not np.any(np.isnan(x))
        return np.array(x), np.array(y)

    def _loss(self, y_pred, y_true):
        return jnp.mean(self.loss_func(y_pred, y_true))

    def loss_func(self, y_pred, y_true):
        return -PoissonSkipNaN(jnp.exp(y_pred.ravel())).log_prob(y_true.ravel())

    def get_validation_y(self, params):
        x = np.concatenate([self._saved_x, (self._validation_x - self._mu) / self._std], axis=1)
        y_pred = self.model.apply(params, x)
        return y_pred[:, self._saved_x.shape[1] :]

    def _state_apply(self, state, params, x, y, dropout_train_key=None, training=True):
        if training:
            return state.apply_fn(params, x, training=training, rngs={"dropout": dropout_train_key})
        else:
            return state.apply_fn(params, x, training=training)

    def init_params(self, x, y):
        params = self.model.init(self.rng_key, x, training=False)
        y_pred = self.model.apply(params, x)
        assert np.all(np.isfinite(y_pred))
        assert np.all(~np.isnan(y_pred)), y_pred
        return params

    def train(self, data: DataSet[ClimateHealthTimeSeries]):

        x, y = self._get_series(data)
        self._mu = np.mean(x, axis=(0, 1))
        self._std = np.std(x, axis=(0, 1))
        x = (x - self._mu) / self._std
        self._saved_x = x
        params = self.init_params(x, y)

        dropout_key = jax.random.PRNGKey(40)

        state = TrainState.create(apply_fn=self.model.apply, params=params, tx=optax.adam(1e-2), key=dropout_key)

        @jax.jit
        def train_step(state: TrainState, dropout_key) -> TrainState:
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_func(params):
                eta = self._state_apply(state, params, x, y, dropout_train_key, training=True)
                return self._loss(eta, y) + l2_regularization(params, 0.001)

            grad_func = jax.value_and_grad(loss_func)
            loss, grad = grad_func(state.params)
            state = state.apply_gradients(grads=grad)
            return state

        for i in range(self.n_iter):
            state = train_step(state, dropout_key)
            if i % 1000 == 0:
                eta = self._state_apply(state, state.params, x, y, training=False)
                loss = self._loss(eta, y)
                print(f"Loss: {loss}")
                if self._validation_x is not None:
                    validation_y = self.get_validation_y(state.params)
                    val_loss = self._loss(validation_y, self._validation_y)
                    print(f"Validation Loss: {val_loss}")
                    eta = self._state_apply(state, state.params, x, y, training=False)
                    mean = self._get_mean(eta)
                    j = 0
                    for series, true in zip(mean, y):
                        plt.plot(series)
                        plt.plot(true)
                        plt.show()
                        j = j + 1
                        if j > 10:
                            break
                    print(f"Loss: {self._loss(eta, y)}")

        self._params = state.params
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet, num_samples: int = 100):
        assert list(historic_data.keys()) == list(future_data.keys())
        x, y = self._get_series(future_data)
        x = (x - self._mu) / self._std
        prev_values = self._get_series(historic_data)[0]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        eta = self.model.apply(self._params, full_x)
        n_prev = prev_values.shape[1]
        samples = self.get_samples(eta[:, n_prev:], num_samples)
        time_period = next(iter(future_data.values())).time_period
        return DataSet({key: Samples(time_period, s) for key, s in zip(future_data.keys(), samples)})

    def _get_q(self, eta, q=0.95):
        mu = self._get_mean(eta)
        q95 = scipy.stats.poisson.ppf(q, mu)
        print(eta.shape, mu.shape, q95.shape)
        return q95

    def _get_mean(self, eta):
        return np.exp(eta)

    def diagnose(self):
        import matplotlib.pyplot as plt

        plt.plot(self._losses)
        plt.show()


NormalSkipNaN = skip_nan_distribution(Normal)
NBSkipNaN = skip_nan_distribution(NegativeBinomial3)


class ProbabilisticFlaxModel(FlaxModel):
    n_iter: int = 32000

    @property
    def model(self):
        if self._model is None:
            self._model = RNNModel(n_locations=self._saved_x.shape[0], output_dim=2, n_hidden=4, embedding_dim=4)
        return self._model

    def _get_dist(self, eta):
        return NBSkipNaN(jax.nn.softplus(eta[..., 0]), eta[..., 1])

    def get_samples(self, eta, n_samples):
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self._get_dist(eta).sample(eta, (n_samples,))

    def _get_mean(self, eta):
        return self._get_dist(eta).mean

    def _get_q(self, eta, q=0.95):
        return self._get_dist(eta).icdf(q)

    def loss_func(self, eta_pred, y_true):
        return -self._get_dist(eta_pred).log_prob(y_true).ravel()
