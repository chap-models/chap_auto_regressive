# chap_ar

The deep auto-regressive (RNN) flax model (`AutoRegressiveModel`) used by the
CHAP `auto_regressive_monthly` and `auto_regressive_weekly` models.

```python
from chap_ar import AutoRegressiveModel
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
src/chap_ar/
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

## Environment

- Python 3.13, managed with [uv](https://docs.astral.sh/uv/), `uv_build` backend
- Runs on the current stack: `flax 0.12`, `jax 0.10`, `chap-core` (git master)

```bash
make install   # uv sync
make check     # ruff (format + lint) + mypy + pyright, no changes
make lint      # ruff format + autofix, then type-check
make docs      # serve the documentation locally
```

Full documentation — usage, data format, concepts, a glossary, libraries, and the
API reference — lives in `docs/` and is built with mkdocs.
