# Auto-regression explained

This page explains what regression and auto-regression are, why they fit disease
forecasting, and exactly how `chap_ar` implements them.

## Regression

**Regression** predicts a numeric outcome from a set of input features. A model
learns a function

```
y ≈ f(x₁, x₂, …, xₙ)
```

from examples, then applies it to new inputs. In disease forecasting the outcome
`y` is the number of cases, and the features might be rainfall, temperature, and
population. Plain regression treats each period independently — it has no notion
that this week follows last week.

## Auto-regression

**Auto-regression** adds the missing ingredient: the series' own past. An
auto-regressive model predicts the next value from earlier values of the *same*
series:

```
yₜ ≈ f(yₜ₋₁, yₜ₋₂, …, yₜ₋k,  external features)
```

The "auto" means *self* — the target is regressed on its own history (the last
`k` periods, the **context**). This matters for disease because case counts are
strongly serially correlated: an outbreak this month makes more cases next month
likely, regardless of climate. A model that ignores recent cases throws that
signal away.

`chap_ar` is auto-regressive in exactly this sense: when forecasting, it feeds the
recently observed case counts back into the network alongside the climate
covariates.

## Why a *deep* auto-regressive model

A classical auto-regressive model (e.g. ARIMA) assumes a fixed, linear
relationship to a few lagged values. `chap_ar` instead uses a small **recurrent
neural network (RNN)**, which:

- learns a non-linear function of the history rather than fixed lag coefficients;
- maintains a hidden **state** that summarizes everything seen so far, so it isn't
  limited to a handful of explicit lags;
- shares one set of weights across all locations, while still letting each
  location differ through a learned **embedding**.

## How chap_ar handles it, step by step

### 1. Inputs and target

For every location and period the model builds a feature vector from:

- `rainfall`
- `mean_temperature`
- `population`
- **day-of-year position** (`day / 365`) — a simple seasonal signal

The target is `disease_cases`. Features are extracted in `transforms.get_series`
and z-scored (mean 0, unit variance) by `transforms.ZScaler` so that no single
feature dominates training.

### 2. Context and horizon

Two settings define the auto-regressive window:

- **`context_length`** — how many past periods the model reads before forecasting
  (12 months for the monthly model, 52 weeks for the weekly one — i.e. about a
  year of history in both).
- **`prediction_length`** — how many periods ahead to forecast (3 and 12
  respectively).

The data loader (`data_loader.py`) slices each location's series into
`(context + horizon)` windows for training.

### 3. The network

The architecture (`rnn_model.ARModel2`) processes a window in stages:

1. **Preprocess** — each location gets a learned embedding (so the model can tell
   regions apart), which is concatenated with the features and passed through a
   small dense layer with dropout.
2. **Auto-regressive join** — the recently observed cases (`y`) are concatenated
   onto the processed features. *This is the auto-regressive step*: the network
   literally sees the series' own past values as input.
3. **Recurrent encoder** — a `SimpleCell` RNN runs over the context window,
   rolling the history up into a hidden state.
4. **Recurrent decoder** — a second `SimpleCell` RNN continues from that state
   across the forecast horizon, where observed cases are no longer available.
5. **Output head** — dense layers emit two numbers per period (`eta`).

### 4. From network output to a distribution

Counts are non-negative integers and are typically **overdispersed** (variance
larger than the mean), so a Poisson is too rigid. `chap_ar` uses a **negative
binomial** instead. The two network outputs are mapped to its parameters by
`distributions.nb_head`:

- channel 0, passed through `softplus`, becomes the count parameter;
- channel 1 becomes the logits.

The wrapper `skip_nan_distribution` makes the likelihood **NaN-tolerant**: missing
observations simply contribute nothing to the loss instead of breaking it.

### 5. Training

Training (`trainer.py`) minimizes the negative log-likelihood of the observed
cases under the predicted negative binomial, using the `optax` Adam optimizer for
`n_iter` steps. Because the whole pipeline is written in `jax`, the loss and its
gradients are JIT-compiled.

### 6. Forecasting

At prediction time the model:

1. reads the last `context_length` periods of real history (with observed cases);
2. runs the encoder/decoder forward across the future periods using the supplied
   future climate covariates;
3. draws `num_samples` (default 100) samples from the negative binomial at each
   future period.

The result is a set of sampled trajectories per location — a probabilistic
forecast that CHAP turns into medians and prediction intervals.

## In one sentence

`chap_ar` rolls each region's recent case history and climate into an RNN state,
continues that state across the forecast horizon, and reads out a negative
binomial distribution of future cases at each step — auto-regression, done with a
small neural network and a count-appropriate likelihood.
