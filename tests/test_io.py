import numpy as np
import pandas as pd
import pytest

from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.transforms import get_series


def _small_trained_model():
    train_df = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 13)])
    model = AutoRegressiveModel()
    model.context_length = 4
    model.prediction_length = 2
    model.n_iter = 2
    return model, model.train(train_df), train_df


def _frame(locations, periods, with_target=True, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for index, location in enumerate(locations):
        for period in periods:
            row = {
                "location": location,
                "time_period": period,
                "rainfall": float(rng.rand() * 100),
                "mean_temperature": float(20 + rng.rand() * 10),
                "population": 1000.0 * (index + 1),  # vary by location so the scaler's std > 0
            }
            if with_target:
                row["disease_cases"] = float(rng.poisson(5))
            rows.append(row)
    return pd.DataFrame(rows)


def test_get_series_shapes_and_target():
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    x, y = get_series(_frame(["A", "B"], months))
    assert x.shape == (2, 6, 4)  # 2 locations, 6 periods, 4 features
    assert y.shape == (2, 6)


def test_get_series_future_has_no_target():
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    x, y = get_series(_frame(["A", "B"], months, with_target=False))
    assert x.shape == (2, 6, 4)
    assert y.size == 0


def test_train_predict_roundtrip_returns_sample_frame():
    train_months = [f"2020-{m:02d}" for m in range(1, 13)]
    future_months = ["2021-01", "2021-02"]
    train_df = _frame(["A", "B"], train_months)
    future_df = _frame(["A", "B"], future_months, with_target=False, seed=1)

    model = AutoRegressiveModel()
    model.context_length = 4
    model.prediction_length = 2
    model.n_iter = 2
    predictor = model.train(train_df)

    out = predictor.predict(train_df, future_df, num_samples=10)

    sample_cols = [c for c in out.columns if c.startswith("sample_")]
    assert {"time_period", "location"}.issubset(out.columns)
    assert len(sample_cols) == 10
    assert len(out) == len(["A", "B"]) * len(future_months)
    assert set(out["location"]) == {"A", "B"}
    assert np.isfinite(out[sample_cols].to_numpy()).all()


def test_predict_is_insensitive_to_future_location_row_order():
    # Locations are grouped canonically, so a future frame whose locations appear
    # in a different row order than the history must still be accepted and labeled
    # correctly (the validation used to be order-sensitive and would reject this).
    model, predictor, train_df = _small_trained_model()
    future_months = ["2021-01", "2021-02"]
    future_reversed = _frame(["B", "A"], future_months, with_target=False, seed=1)

    out = predictor.predict(train_df, future_reversed, num_samples=8)

    assert set(out["location"]) == {"A", "B"}
    assert len(out) == 2 * len(future_months)
    assert np.isfinite(out[[c for c in out.columns if c.startswith("sample_")]].to_numpy()).all()


def test_predict_rejects_wrong_future_length():
    # The model forecasts a fixed horizon (prediction_length); a future frame with
    # a different number of periods must fail loudly rather than silently misalign.
    model, predictor, train_df = _small_trained_model()
    too_many = _frame(["A", "B"], ["2021-01", "2021-02", "2021-03"], with_target=False, seed=1)

    with pytest.raises(ValueError, match="prediction_length"):
        predictor.predict(train_df, too_many, num_samples=8)
