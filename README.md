# chap_ar

The deep auto-regressive (RNN) flax model (`ARModelTV1`) used by the CHAP
`auto_regressive_monthly` and `auto_regressive_weekly` models.

It is a minimal, modernized fork of
[`knutdrand/ch_modelling`](https://github.com/knutdrand/ch_modelling), reduced to
just the flax auto-regressive model path. The upstream package carried a large
amount of experimental code (PyMC models, JAX HMC/Bayesian samplers, SSM
forecasters, multi-country variants) that this model never used but which broke
on modern `jax`/`chap-core`. This fork keeps only what `ARModelTV1` needs and
runs on the current stack with no compatibility shims.

## Layout

```
src/chap_ar/
  __init__.py             # exposes ARModelTV1
  data_loader.py
  distribution_head.py
  distributions.py        # extracted from the old jax_models.model_spec
  flax_model.py
  flax_model_v1.py        # ARModelTV1
  rnn_model.py
  trainer.py
  transforms.py
```

`distributions.py` holds the few distribution primitives (`Normal`, `Poisson`,
`NegativeBinomial2/3`, `skip_nan_distribution`) that `flax_model` needs, lifted
out of the upstream `jax_models.model_spec` so the HMC/Bayesian machinery (and
its dependency on the removed `chap_core.training_control`) could be dropped.

## Environment

- Python 3.13, managed with [uv](https://docs.astral.sh/uv/)
- Runs on the current stack: `flax 0.12`, `jax 0.10`, `chap-core 1.4+`

```bash
make install   # uv sync
make check     # ruff (format + lint) + mypy + pyright, no changes
make lint      # ruff format + autofix, then type-check
```

## Linting

Config is derived from `chapkit` (ruff / mypy / pyright). The seven vendored
legacy numerical files are excluded from strict typing and from docstring rules
(see the per-file-ignores / overrides in `pyproject.toml`); the extracted
`distributions.py` is checked normally.
