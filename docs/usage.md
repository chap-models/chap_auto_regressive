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

All input and output is plain [pandas](https://pandas.pydata.org/) — `chap_ar`
has no dependency on chap-core.

### Train

`train` takes a tidy `DataFrame` with one row per location and time period and the
columns `location`, `time_period`, `rainfall`, `mean_temperature`, `population`,
and `disease_cases`. It returns a `FlaxPredictor`:

```python
import pandas as pd

predictor = model.train(pd.read_csv("training_data.csv"))
predictor.save("model.bin")     # pickles the trained parameters + scaler
```

### Predict

`predict` takes the history (with observed cases) and the future covariates
(climate for the periods to forecast, no `disease_cases`) and returns a
`DataFrame` with columns `time_period`, `location`, and one `sample_i` column per
draw (100 by default):

```python
predictor = model.load_predictor("model.bin")
forecasts = predictor.predict(historic_df, future_df, num_samples=100)
# columns: time_period, location, sample_0 … sample_99
```

The output is **probabilistic**: each location/period gets a distribution of
sampled case counts, which is what CHAP uses to build prediction intervals. See
[Data format](data.md) for the full CSV contract.

## Inside a CHAP model

A CHAP model repository wraps this class in two tiny scripts — `train.py` and
`predict.py` — that do the CSV I/O with pandas and call `chap_ar`. No chap-core
import is needed; CHAP runs the scripts via the `MLproject` uv runner. A shared
`model.py` builds the configured estimator:

```python
# model.py
import os
from chap_ar import AutoRegressiveModel

def build_model() -> AutoRegressiveModel:
    model = AutoRegressiveModel()
    model.n_iter = int(os.environ.get("AR_N_ITER", "1000"))
    model.context_length = 12
    model.prediction_length = 3
    model.learning_rate = 1e-5
    return model
```

```python
# train.py — invoked as: python train.py {train_data} {model}
import sys, pandas as pd
from model import build_model

build_model().train(pd.read_csv(sys.argv[1])).save(sys.argv[2])
```

```python
# predict.py — invoked as: python predict.py {model} {historic} {future} {out}
import sys, pandas as pd
from model import build_model

predictor = build_model().load_predictor(sys.argv[1])
predictor.predict(pd.read_csv(sys.argv[2]), pd.read_csv(sys.argv[3])).to_csv(sys.argv[4], index=False)
```

See `auto_regressive_monthly` and `auto_regressive_weekly` for the full setup,
which differ only in the period type and the context/horizon lengths:

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
