# Libraries used

`chap_auto_regressive` is deliberately small and leans on a few well-established libraries.

## Modelling stack

### JAX

[JAX](https://jax.readthedocs.io) is the numerical engine. It provides
NumPy-style array operations with three features the model relies on:

- **autodiff** (`jax.grad` / `jax.value_and_grad`) to get loss gradients,
- **JIT compilation** (`jax.jit`) to fuse the training step into fast code,
- **explicit PRNG keys** (`jax.random`) for reproducible sampling.

### Flax

[Flax](https://flax.readthedocs.io) is the neural-network library on top of JAX.
`chap_auto_regressive` uses Flax's `linen` API to define the network in `rnn_model.py`
(`nn.Embed`, `nn.Dense`, `nn.Dropout`, and the `SimpleCell` RNN), and
`flax.training.train_state` to hold parameters during training.

### Optax

[Optax](https://optax.readthedocs.io) supplies the optimizer. The trainer uses
`optax.adam` to update the network weights from the gradients JAX computes.

### pandas, SciPy and NumPy

[pandas](https://pandas.pydata.org/) is the public I/O type — `train`/`predict`
take and return DataFrames. [NumPy](https://numpy.org) handles feature assembly
and scaling (`transforms.py`), and [SciPy](https://scipy.org) provides the
negative-binomial sampling/PMF used by the distributions in `distributions.py`.

## Integration with CHAP

`chap_auto_regressive` itself has **no chap-core dependency** — it speaks plain pandas. The CHAP
platform ([chap-core](https://github.com/dhis2-chap/chap-core)) runs a model purely
through the `MLproject` CSV contract: it writes the train/historic/future CSVs,
invokes the model's `train.py` / `predict.py`, and reads the `sample_*` output CSV.
What happens inside the model is invisible to it, which is what lets the model drop
the chap-core dependency entirely.

## Tooling

- [uv](https://docs.astral.sh/uv/) — environment and dependency management, plus
  the `uv_build` backend for packaging.
- [ruff](https://docs.astral.sh/ruff/), [mypy](https://mypy.readthedocs.io), and
  [pyright](https://microsoft.github.io/pyright/) — linting and type checking,
  configured from the same baseline as `chapkit`.
- [mkdocs](https://www.mkdocs.org) with Material and mkdocstrings — these docs.
