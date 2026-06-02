import jax
import jax.numpy as jnp
import numpy as np

from chap_ar.data_loader import DataSet, SimpleDataLoader
from chap_ar.rnn_model import model_makers
from chap_ar.trainer import Trainer


def _toy_loader(n_loc=2, n_periods=12, n_feat=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n_loc, n_periods, n_feat)).astype("float32")
    y = rng.poisson(3, (n_loc, n_periods)).astype("float32")
    return SimpleDataLoader(DataSet(X, y, forecast_length=2, context_length=4))


def test_network_forward_produces_two_finite_channels():
    loader = _toy_loader()
    x, ar_y, _ = next(iter(loader))
    model = model_makers["base"](x.shape[0])
    params = model.init(jax.random.PRNGKey(0), x, ar_y, training=False)
    eta = np.asarray(model.apply(params, x, ar_y))
    assert eta.shape[-1] == 2
    assert np.isfinite(eta).all()


def test_trainer_runs_a_couple_of_steps():
    loader = _toy_loader()
    x, ar_y, _ = next(iter(loader))
    model = model_makers["base"](x.shape[0])
    state = Trainer(model, n_iter=2, learning_rate=1e-3).train(loader, lambda eta, y: jnp.mean(eta**2))
    assert state.params is not None
