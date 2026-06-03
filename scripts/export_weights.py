"""Export a trained predictor to a portable format for the Rust inference port.

The pickle written by :meth:`FlaxPredictor.save` holds only ``(params, scaler,
locations)`` -- the context/prediction lengths and the architecture name are model
configuration supplied at load time, so they are passed in here as arguments.

Outputs (into ``out_dir``):

- ``weights.safetensors`` -- every network parameter as a named float32 tensor,
  keyed by its flattened Flax pytree path (e.g. ``params/preprocess/Dense_0/kernel``).
- ``meta.json`` -- lengths, architecture name, embedding/hidden dimensions, the
  feature order, the fitted scaler ``mu``/``std`` and the sorted training locations.

Usage:
    uv run python scripts/export_weights.py model.pkl out_dir \
        --context-length 24 --prediction-length 3 --rnn-model-name base
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file

# The fixed per-period feature order produced by transforms.get_series.
FEATURE_ORDER = ["rainfall", "mean_temperature", "population", "year_position"]


def _load_pickle(path: str) -> tuple:
    """Load the ``(params, scaler, locations)`` tuple written by ``save``."""
    if not Path(path).is_file():
        raise SystemExit(
            f"no such predictor pickle: {path!r}\n"
            "Pass the path to a file written by FlaxPredictor.save(...). To create one:\n"
            "    from chap_auto_regressive import AutoRegressiveModel\n"
            "    AutoRegressiveModel().train(df).save('model.pkl')\n"
            "or reuse the bundled example at "
            "../chap_auto_regressive_rs/tests/fixtures/model.pkl "
            "(trained with --context-length 4 --prediction-length 2)."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if len(payload) == 2:  # legacy pickle without tracked locations
        params, scaler = payload
        locations = None
    else:
        params, scaler, locations = payload
    return params, scaler, locations


def _flat_f32(params: dict) -> dict:
    """Flatten the Flax param pytree to ``{slash/path: float32 ndarray}``."""
    flat = flatten_dict(params, sep="/")
    return {key: np.asarray(value, dtype=np.float32) for key, value in flat.items()}


def export(
    pickle_path: str,
    out_dir: str,
    context_length: int,
    prediction_length: int,
    rnn_model_name: str,
) -> None:
    """Write ``weights.safetensors`` and ``meta.json`` from a saved predictor."""
    params, scaler, locations = _load_pickle(pickle_path)
    tensors = _flat_f32(params)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "weights.safetensors"))

    embedding = tensors["params/preprocess/Embed_0/embedding"]
    meta = {
        "rnn_model_name": rnn_model_name,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "n_locations": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]),
        "feature_order": FEATURE_ORDER,
        "scaler": {
            "mu": [float(v) for v in np.asarray(scaler.mu).ravel()],
            "std": [float(v) for v in np.asarray(scaler.std).ravel()],
        },
        "locations": list(locations) if locations is not None else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out/'weights.safetensors'} ({len(tensors)} tensors) and {out/'meta.json'}")


def main() -> None:
    """Parse arguments and run the export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_path", help="Path to a predictor pickle written by FlaxPredictor.save")
    parser.add_argument("out_dir", help="Directory to write weights.safetensors and meta.json into")
    parser.add_argument("--context-length", type=int, default=24)
    parser.add_argument("--prediction-length", type=int, default=3)
    parser.add_argument("--rnn-model-name", default="base", choices=["base", "multi_value"])
    args = parser.parse_args()
    export(
        args.pickle_path,
        args.out_dir,
        args.context_length,
        args.prediction_length,
        args.rnn_model_name,
    )


if __name__ == "__main__":
    main()
