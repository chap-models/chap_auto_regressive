import numpy as np

from chap_auto_regressive.transforms import ZScaler, period_start_date, year_position_from_period


def test_zscaler_standardizes_first_element():
    mu = np.array([1.0, 2.0])
    std = np.array([2.0, 4.0])
    feats = np.array([[3.0, 6.0]])
    out = ZScaler(mu, std)((feats,))
    assert np.allclose(out[0], (feats - mu) / std)


def test_year_position_bounds():
    assert year_position_from_period("2024-01") < 0.02
    assert year_position_from_period("2024-12") > 0.9


def test_period_start_date_monthly_and_weekly():
    assert period_start_date("2010-03").month == 3
    weekly = period_start_date("2003-12-29/2004-01-04")
    assert (weekly.year, weekly.month, weekly.day) == (2003, 12, 29)


class _Predictors:
    """Minimal stand-in exposing ``predictors(0)`` for ``ZScaler.from_data``."""

    def __init__(self, features):
        self._features = features

    def predictors(self, i):
        return self._features


def test_zscaler_handles_zero_variance_features():
    # A constant feature (single location or static covariate) has std 0; the
    # scaler must not divide by zero and produce inf/NaN inputs.
    features = np.array([[[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]])  # second feature constant
    scaler = ZScaler.from_data(_Predictors(features))
    assert scaler.std[1] == 1.0  # zero variance floored to 1
    scaled = scaler((features,))[0]
    assert np.isfinite(scaled).all()
    assert np.allclose(scaled[..., 1], 0.0)  # constant feature standardizes to 0
