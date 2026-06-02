# Usage

## Install

`chap_ar` is managed with [uv](https://docs.astral.sh/uv/) and targets Python 3.13.

```bash
# as a git dependency
uv add "chap_ar @ git+https://github.com/mortenoh/chap_ar"
```

Or, working on the library itself:

```bash
git clone https://github.com/mortenoh/chap_ar
cd chap_ar
make install   # uv sync
```

## The model API

The public surface is a single class, `AutoRegressiveModel`, plus the
`FlaxPredictor` it produces.

```python
from chap_ar import AutoRegressiveModel

model = AutoRegressiveModel()
model.context_length = 12      # past periods read as context
model.prediction_length = 3    # periods to forecast ahead
model.learning_rate = 1e-5
model.n_iter = 1000            # training iterations
```

### Configuration

The model is configured by setting attributes after construction. All have
defaults, so you only set the ones you need:

| attribute | default | what it does |
| --- | --- | --- |
| `n_iter` | `1000` | Number of training epochs (full passes over the training windows). More iterations train longer and can fit better at the cost of time; very high values risk overfitting. |
| `context_length` | `24` | How many past periods the model reads as history before forecasting — roughly a year is typical (12 months or 52 weeks). |
| `prediction_length` | `3` | How many periods ahead to forecast (the horizon). |
| `learning_rate` | `1e-4` | The Adam step size. Smaller is more stable but converges more slowly. |
| `rnn_model_name` | `"base"` | Which architecture to build — `"base"`, or `"multi_value"` which also mixes the target across locations. |

In the CHAP models these are set in `main.py`. There, `n_iter` is read from the
`AR_N_ITER` environment variable (defaulting to `1000`) so the test suite can run
a fast pass — see each model's README.

### Train

`train` takes a `chap_core` `DataSet` of `FullData` (per-location time series with
`rainfall`, `mean_temperature`, `population`, and `disease_cases`) and returns a
`FlaxPredictor`:

```python
predictor = model.train(training_data)
predictor.save("model.bin")     # pickles the trained parameters + scaler
```

### Predict

`predict` takes the history (with observed cases) and the future covariates
(climate for the periods to forecast) and returns a `DataSet` of `Samples` — by
default 100 sampled trajectories per location:

```python
predictor = model.load_predictor("model.bin")
forecasts = predictor.predict(historic_data, future_data, num_samples=100)
```

The output is **probabilistic**: each location/horizon gets a distribution of
sampled case counts, which is what CHAP uses to build prediction intervals.

## Inside a CHAP model

In a CHAP model repository this class is wrapped with `chap_core`'s CLI adaptor,
which turns it into the `train` / `predict` commands CHAP calls. A complete
`main.py` is just:

```python
import os
from chap_core.adaptors.command_line_interface import generate_app
from chap_ar import AutoRegressiveModel

model = AutoRegressiveModel()
model.n_iter = int(os.environ.get("AR_N_ITER", "1000"))
model.context_length = 12
model.prediction_length = 3
model.learning_rate = 1e-5

app = generate_app(model)
app()
```

The model repo's `MLproject` points CHAP at that `main.py` via the uv runner. See
`auto_regressive_monthly` and `auto_regressive_weekly` for the full setup, which
differ only in the period type and the context/horizon lengths:

| Model | period | context_length | prediction_length |
| --- | --- | --- | --- |
| `auto_regressive_monthly` | month | 12 | 3 |
| `auto_regressive_weekly` | week | 52 | 12 |

## Evaluating through CHAP

With a chap-core checkout you can backtest a model directly from its GitHub URL:

```bash
uv run chap eval \
    --model-name https://github.com/mortenoh/auto_regressive_monthly \
    --dataset-csv example_data/laos_subset.csv \
    --output-file /tmp/ar_eval.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 1
```
