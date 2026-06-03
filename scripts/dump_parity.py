"""Build an external parity fixture from an already-trained model.

Given a predictor pickle and a historic/future CSV pair, this writes a directory
the Rust parity test can consume (point `CHAP_FIXTURE_DIR` at it):

- ``weights.safetensors`` + ``meta.json`` (via ``export_weights.export``),
- ``historic.csv`` / ``future.csv`` (copied verbatim),
- ``parity.safetensors`` holding the exact ``(scaled_x, ar_y, eta)`` the Python
  forward pass produces for that input.

Used to validate the Rust port against the real monthly/weekly CHAP models.

Usage:
    PYTHONPATH=scripts uv run python scripts/dump_parity.py \
        model.bin historic.csv future.csv out_dir \
        --context-length 12 --prediction-length 3
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import save_file

from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.data_loader import DataSet as DLDataSet
from chap_auto_regressive.transforms import get_series

from export_weights import export


def forward_dump(predictor, historic: pd.DataFrame, future: pd.DataFrame) -> dict:
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
    """Write the parity fixture into ``out_dir``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("historic")
    parser.add_argument("future")
    parser.add_argument("out_dir")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--prediction-length", type=int, required=True)
    parser.add_argument("--rnn-model-name", default="base")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export(args.model, str(out), args.context_length, args.prediction_length, args.rnn_model_name)

    model = AutoRegressiveModel()
    model.context_length = args.context_length
    model.prediction_length = args.prediction_length
    model.rnn_model_name = args.rnn_model_name
    predictor = model.load_predictor(args.model)

    historic = pd.read_csv(args.historic)
    future = pd.read_csv(args.future)
    shutil.copy(args.historic, out / "historic.csv")
    shutil.copy(args.future, out / "future.csv")

    dump = forward_dump(predictor, historic, future)
    save_file(dump, str(out / "parity.safetensors"))
    print(f"parity fixture written to {out}  eta shape {dump['eta'].shape}")


if __name__ == "__main__":
    main()
