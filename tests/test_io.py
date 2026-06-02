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


def test_predict_rejects_future_longer_than_horizon():
    # The model can forecast at most prediction_length periods (its trained
    # horizon); asking for more must fail loudly rather than extrapolate silently.
    model, predictor, train_df = _small_trained_model()  # prediction_length = 2
    too_many = _frame(["A", "B"], ["2021-01", "2021-02", "2021-03"], with_target=False, seed=1)

    with pytest.raises(ValueError, match="prediction_length"):
        predictor.predict(train_df, too_many, num_samples=8)


def test_predict_accepts_future_shorter_than_horizon():
    # A chap eval backtest forecasts fewer periods than prediction_length; a short
    # future must be accepted and produce one row per location and period.
    model, predictor, train_df = _small_trained_model()  # prediction_length = 2
    short_future = _frame(["A", "B"], ["2021-01"], with_target=False, seed=1)

    out = predictor.predict(train_df, short_future, num_samples=8)

    assert len(out) == 2  # 2 locations x 1 period
    assert sorted(out["time_period"].unique()) == ["2021-01"]
    assert np.isfinite(out[[c for c in out.columns if c.startswith("sample_")]].to_numpy()).all()


def test_predict_labels_each_location_with_its_own_periods():
    # Each location must be labeled with its own forecast periods, not the first
    # location's (the output used to copy group-0's time_period onto every group).
    model, predictor, train_df = _small_trained_model()
    future = pd.concat(
        [
            _frame(["A"], ["2021-01", "2021-02"], with_target=False, seed=1),
            _frame(["B"], ["2021-03", "2021-04"], with_target=False, seed=2),
        ],
        ignore_index=True,
    )

    out = predictor.predict(train_df, future, num_samples=6)

    assert sorted(out[out["location"] == "A"]["time_period"]) == ["2021-01", "2021-02"]
    assert sorted(out[out["location"] == "B"]["time_period"]) == ["2021-03", "2021-04"]


def test_set_validation_data_requires_observed_cases():
    # Validation loss is computed against observed cases, so a future without a
    # disease_cases column must fail with a clear message (not a cryptic concat).
    model = AutoRegressiveModel()
    model.context_length, model.prediction_length = 4, 2
    historic = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 7)])
    future_no_target = _frame(["A", "B"], ["2020-07", "2020-08"], with_target=False, seed=3)

    with pytest.raises(ValueError, match="disease_cases"):
        model.set_validation_data(historic, future_no_target)


def test_predict_rejects_unseen_locations():
    # The network learns per-location embeddings by sorted index; predicting on a
    # location set the model was not trained on would silently reuse another
    # location's embedding, so it must be rejected.
    model, predictor, train_df = _small_trained_model()  # trained on A, B
    unseen_hist = _frame(["C", "D"], [f"2020-{m:02d}" for m in range(1, 13)], seed=9)
    unseen_future = _frame(["C", "D"], ["2021-01", "2021-02"], with_target=False, seed=10)

    with pytest.raises(ValueError, match="training locations"):
        predictor.predict(unseen_hist, unseen_future, num_samples=8)


def test_saved_predictor_preserves_and_enforces_locations(tmp_path):
    # The training locations must survive a save/load round-trip so a reloaded
    # predictor still rejects unseen locations.
    model, predictor, train_df = _small_trained_model()  # trained on A, B
    path = str(tmp_path / "model.pkl")
    predictor.save(path)
    loaded = model.load_predictor(path)

    assert loaded.locations == ["A", "B"]
    out = loaded.predict(train_df, _frame(["A", "B"], ["2021-01", "2021-02"], with_target=False, seed=1), num_samples=4)
    assert set(out["location"]) == {"A", "B"}
    with pytest.raises(ValueError, match="training locations"):
        loaded.predict(
            _frame(["C", "D"], [f"2020-{m:02d}" for m in range(1, 13)], seed=9),
            _frame(["C", "D"], ["2021-01", "2021-02"], with_target=False, seed=10),
            num_samples=4,
        )


def test_validation_features_are_scaled_after_train():
    # The fitted scaler must be applied to the validation window too, otherwise
    # the reported validation loss is computed on raw features and is not
    # comparable to the training loss.
    model = AutoRegressiveModel()
    model.context_length, model.prediction_length, model.n_iter = 4, 2, 2
    historic = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 7)])
    future = _frame(["A", "B"], ["2020-07", "2020-08"], with_target=True, seed=4)
    model.set_validation_data(historic, future)

    before = model._validation_loader.dataset[0][0]
    model.train(_frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 13)]))
    after = model._validation_loader.dataset[0][0]

    assert not np.allclose(before, after)  # raw features were standardized in place
