# chap_auto_regressive

The deep auto-regressive (RNN) flax model (`AutoRegressiveModel`) used by the
CHAP `auto_regressive_monthly` and `auto_regressive_weekly` models.

```python
from chap_auto_regressive import AutoRegressiveModel
```

It is a minimal, modernized fork of
[`knutdrand/ch_modelling`](https://github.com/knutdrand/ch_modelling), reduced to
just the flax auto-regressive model path. The upstream package carried a large
amount of experimental code (PyMC models, JAX HMC/Bayesian samplers, SSM
forecasters, multi-country variants) that this model never used but which broke
on modern `jax`/`chap-core`. This fork keeps only what the model needs and runs
on the current stack with no compatibility shims.

## Layout

```
src/chap_auto_regressive/
  __init__.py        # exposes AutoRegressiveModel
  model.py           # AutoRegressiveModel + FlaxPredictor
  rnn_model.py       # the RNN architectures (model_makers)
  trainer.py         # the training loop
  data_loader.py     # windowed dataset / loaders
  transforms.py      # feature scaling and series extraction
  distributions.py   # Normal / NegativeBinomial / Poisson + the nb_head adapter
```

`distributions.py` holds the distribution primitives (`Normal`, `Poisson`,
`NegativeBinomial2/3`, `skip_nan_distribution`) plus `nb_head`, which maps the
network's two-channel output to a NaN-tolerant negative binomial. These were
lifted out of the upstream `jax_models.model_spec` so the HMC/Bayesian machinery
(and its dependency on the removed `chap_core.training_control`) could be dropped.

## Fixed issues (inherited from upstream)

This fork has been through iterative review. The bugs below all existed in the
original `ch_modelling` code and have been **fixed** here; they are recorded for
provenance and as regression-test anchors (each has a test under `tests/`). None
were introduced by the modernization, and the feature extraction remains
numerically identical to the legacy model (verified to `0.0` difference).

| Area | Symptom | Cause | Resolution |
| --- | --- | --- | --- |
| `transforms.ZScaler.from_data` | `inf`/`NaN` inputs, training then fails | A zero-variance feature (single location, or a static covariate such as constant population) divided by zero | Zero standard deviations are treated as `1`, so a constant feature standardizes to `0` |
| `data_loader.interpolate_nans` | `ValueError: array of sample points is empty` | A location whose target is entirely `NaN` had nothing to interpolate from | All-`NaN` rows are filled with `0`; the raw target keeps its `NaN`s so the likelihood still skips those periods |
| `distributions.NegativeBinomial3.sample` | Forecasts not reproducible from the model RNG | The JAX key was accepted but ignored; SciPy used the global NumPy RNG | The key now seeds SciPy's generator via `random_state` |
| `model.predict` (location check) | A valid call rejected when locations appear in a different row order | Validation compared **ordered** lists of locations | Location **sets** are compared instead |
| `model.predict` (horizon) | Cryptic shape errors / wrong period alignment | The fixed forecast horizon (`prediction_length`) was never validated against the future frame | `predict` now requires exactly `prediction_length` periods per location, with a clear error |
| `model._forecast_frame` | Every location's output labeled with the **first** location's periods | The first group's `time_period` was reused for all groups | Each location is labeled with its own sorted forecast periods |
| `model.set_validation_data` | Cryptic NumPy concat dimension error | A validation `future` without `disease_cases` was accepted, then failed downstream | Raises a clear `ValueError` — validation needs observed cases as labels |
| `model.train` (validation scaling) | Validation loss not comparable to training loss | The fitted scaler was never applied to the validation window, so it ran on raw features | `train` attaches the fitted scaler to the validation loader |
| `transforms.location_groups` | Location→embedding index depended on input row order | Grouping used first-seen (CSV) order rather than a canonical order | Locations are grouped in sorted order, matching the legacy `DataSet` and keeping `train`/`predict` aligned |
| `model.predict` (location identity) | Valid-looking forecasts for the **wrong** locations | A predictor did not remember its training locations, so an unseen set (e.g. C/D) reused the embeddings learned for A/B by position | `train` records the canonical training locations (persisted on save); prediction/validation frames must cover exactly that set |
| `transforms.get_series` | `ValueError: setting an array element with a sequence` | Locations with differing period counts (ragged input) broke the dense array construction | A targeted `ValueError` names the per-location period counts before the array is built |

## Environment

- Python 3.13, managed with [uv](https://docs.astral.sh/uv/), `uv_build` backend
- Runs on the current stack: `flax 0.12`, `jax 0.10` (pure pandas/numpy I/O, no
  `chap-core` dependency)

```bash
make install   # uv sync
make check     # ruff (format + lint) + mypy + pyright, no changes
make lint      # ruff format + autofix, then type-check
make docs      # serve the documentation locally
```

Full documentation — usage, data format, concepts, a glossary, libraries, and the
API reference — lives in `docs/` and is built with mkdocs.
