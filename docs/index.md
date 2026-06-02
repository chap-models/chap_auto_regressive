# chap_auto_regressive

`chap_auto_regressive` is the deep **auto-regressive** (RNN) model used by the CHAP
disease-forecasting models `auto_regressive_monthly` and `auto_regressive_weekly`.
It forecasts disease case counts from a region's own recent history together with
climate covariates, and returns a full probabilistic forecast (samples), not just
a point estimate.

```python
from chap_auto_regressive import AutoRegressiveModel

model = AutoRegressiveModel()
model.context_length = 12      # how many past periods the model reads
model.prediction_length = 3    # how many periods ahead to forecast
model.n_iter = 1000            # training iterations

predictor = model.train(training_data)
forecasts = predictor.predict(historic_data, future_data)
```

## At a glance

| Property | Value |
| --- | --- |
| Task | Forecast `disease_cases` per location and period |
| Inputs | `rainfall`, `mean_temperature`, `population`, day-of-year |
| History used | `context_length` past periods |
| Horizon | `prediction_length` future periods |
| Architecture | Location embedding → MLP → two RNN (SimpleCell) stages |
| Output | Negative-binomial distribution per period (probabilistic) |
| Stack | Python 3.13, `jax` + `flax` + `optax`, `pandas` I/O (no chap-core) |

## Where to go next

- [Usage](usage.md) — install it, train and predict, and run it through CHAP.
- [Data format](data.md) — the input and output CSVs, with samples and an
  input-vs-output breakdown.
- [Concepts](concepts.md) — what regression and auto-regression are, and exactly
  how this model handles them.
- [Libraries used](libraries.md) — what each dependency does here.
- [API Reference](api-reference.md) — the generated reference for the public API.

## Origin

`chap_auto_regressive` is a minimal, modernized fork of
[`knutdrand/ch_modelling`](https://github.com/knutdrand/ch_modelling), reduced to
just the flax auto-regressive model and updated to run on the current
`jax` / `flax` stack with a plain pandas interface and no chap-core dependency.
