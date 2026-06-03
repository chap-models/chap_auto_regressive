"""Build a deterministic test fixture for the Rust inference port.

Trains a tiny ``base`` model, saves the predictor pickle, exports the portable
weights, writes ``historic.csv``/``future.csv``, and dumps the exact
``(scaled_x, ar_y, eta)`` the Python forward pass produces so the Rust port can be
checked to ~1e-5. The eta dump is the deterministic part of prediction (everything
before negative-binomial sampling).

Usage:
    uv run python scripts/make_fixture.py out_dir
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import save_file

from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.data_loader import DataSet as DLDataSet
from chap_auto_regressive.transforms import get_series

from export_weights import export

CONTEXT_LENGTH = 4
PREDICTION_LENGTH = 2
LOCATIONS = ["A", "B"]
TRAIN_PERIODS = [f"2020-{m:02d}" for m in range(1, 13)]
FUTURE_PERIODS = ["2021-01", "2021-02"]


def _frame(locations: list, periods: list, with_target: bool = True, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic tidy frame (mirrors the test helper)."""
    rng = np.random.RandomState(seed)
    rows = []
    for index, location in enumerate(locations):
        for period in periods:
            row = {
                "location": location,
                "time_period": period,
                "rainfall": float(rng.rand() * 100),
                "mean_temperature": float(20 + rng.rand() * 10),
                "population": 1000.0 * (index + 1),
            }
            if with_target:
                row["disease_cases"] = float(rng.poisson(5))
            rows.append(row)
    return pd.DataFrame(rows)


def _forward_dump(predictor, historic: pd.DataFrame, future: pd.DataFrame) -> dict:
    """Replicate the deterministic predict path and capture its arrays."""
    x, _ = get_series(future)
    prev_values, prev_y = get_series(historic)
    prev_values = prev_values[:, -predictor.context_length :]
    prev_y = prev_y[:, -predictor.context_length :]
    full_x = np.concatenate([prev_values, x], axis=1)
    dataset = DLDataSet(
        full_x, prev_y, forecast_length=predictor.prediction_length, context_length=predictor.context_length
    )
    dataset.set_transform(predictor._transform)
    scaled_x, ar_y = dataset.prediction_instance()
    eta = predictor.model.apply(predictor._params, scaled_x, ar_y)
    return {
        "scaled_x": np.asarray(scaled_x, dtype=np.float32),
        "ar_y": np.asarray(ar_y, dtype=np.float32),
        "eta": np.asarray(eta, dtype=np.float32),
    }


def main() -> None:
    """Generate the fixture into ``out_dir``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    out = Path(parser.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = _frame(LOCATIONS, TRAIN_PERIODS)
    model = AutoRegressiveModel()
    model.context_length = CONTEXT_LENGTH
    model.prediction_length = PREDICTION_LENGTH
    model.n_iter = 5
    predictor = model.train(train_df)

    pickle_path = out / "model.pkl"
    predictor.save(str(pickle_path))
    export(str(pickle_path), str(out), CONTEXT_LENGTH, PREDICTION_LENGTH, "base")

    # The CSVs the Rust binary will read. historic carries the observed cases; the
    # future frame carries covariates only.
    historic = train_df
    future = _frame(LOCATIONS, FUTURE_PERIODS, with_target=False, seed=1)
    historic.to_csv(out / "historic.csv", index=False)
    future.to_csv(out / "future.csv", index=False)

    dump = _forward_dump(predictor, historic, future)
    save_file(dump, str(out / "parity.safetensors"))
    print(f"fixture written to {out}  eta shape {dump['eta'].shape}")


if __name__ == "__main__":
    main()
