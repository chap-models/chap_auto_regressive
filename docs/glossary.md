# Glossary

Plain-language definitions of the terms used throughout these docs.

## Modelling

**Regression**
: Predicting a numeric outcome from input features — learning a function
`y ≈ f(features)` from examples.

**Auto-regression**
: Regression where the inputs include the series' *own past values*. The model
predicts the next value partly from earlier values of the same series. See
[Concepts](concepts.md).

**Covariate**
: An input feature used to help predict the target. Here: `rainfall`,
`mean_temperature`, and `population`.

**Target**
: The quantity being predicted. Here: `disease_cases`.

**Context length**
: How many past periods the model reads before forecasting (`context_length`).

**Prediction length / horizon**
: How many periods ahead the model forecasts (`prediction_length`).

**Window**
: One training instance — a contiguous slice of `context_length + prediction_length`
periods cut from a location's series.

## Network

**RNN (recurrent neural network)**
: A network that processes a sequence step by step, carrying a **hidden state**
that summarizes everything seen so far. It can use the whole history rather than
a fixed number of lags.

**Hidden state**
: The vector an RNN carries from one time step to the next; the model's running
"memory" of the sequence.

**Encoder / decoder**
: The encoder RNN rolls the context window up into a hidden state; the decoder RNN
continues from that state across the forecast horizon.

**Embedding**
: A small learned vector assigned to each location, letting one shared network
still distinguish regions.

**Dropout**
: A regularization trick that randomly zeroes part of the network during training
so it doesn't rely too heavily on any one path.

**`eta`**
: The network's raw two-number-per-period output, which the head turns into a
distribution's parameters.

## Probability

**Distribution**
: A description of which outcomes are possible and how likely each is. The model
predicts one per period rather than a single point value.

**Negative binomial**
: A probability distribution for count data that allows the variance to exceed
the mean (see *overdispersion*) — a better fit for case counts than a Poisson.

**Overdispersion**
: When real counts vary more than a simple Poisson would predict. The negative
binomial accounts for it, giving appropriately wider intervals.

**Likelihood**
: How probable the observed data is under a predicted distribution. Higher is
better.

**Negative log-likelihood (NLL)**
: The negative logarithm of the likelihood. Minimizing it is equivalent to
maximizing the likelihood, and it is the model's training loss.

**Sample / probabilistic forecast**
: A draw from the predicted distribution. The model outputs many samples per
period (default 100) instead of a single number, so you can compute a median and
prediction intervals.

**Prediction interval**
: A range that the outcome is expected to fall within with some probability (e.g.
an 80% interval from the 10th to the 90th percentile of the samples).

## Training

**Loss function**
: A single number measuring how wrong the model's predictions are. Training
searches for the weights that make it as small as possible. Here the loss is the
negative log-likelihood.

**Gradient**
: The direction and rate at which the loss changes as each weight changes — it
tells the optimizer which way to step.

**Optimizer / Adam**
: The algorithm that updates the weights from the gradients. `chap_ar` uses Adam,
a widely used adaptive optimizer.

**Epoch**
: One full pass over all the training windows. Training runs `n_iter` of them.

**L2 regularization**
: A penalty on large weights added to the loss, discouraging overfitting.

**Z-scaling (standardization)**
: Rescaling each feature to zero mean and unit variance so no single covariate
dominates training.

## Stack

**JAX**
: The array/autodiff library that computes the loss, its gradients, and compiles
the training step.

**Flax**
: The neural-network library (on top of JAX) used to define the architecture.

**Optax**
: The optimizer library that provides Adam.

**chap-core**
: The CHAP platform that runs the model through the MLproject CSV contract.
`chap_ar` does not depend on it — they only meet at the CSV files.

**pandas**
: The library for the tidy DataFrames `chap_ar` takes and returns.
