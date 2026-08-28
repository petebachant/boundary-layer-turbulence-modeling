# Boundary layer turbulence modeling

Developing and validating RANS turbulence closures against boundary-layer DNS,
and a benchmarking harness — the "RANS gym" — for evaluating new closure ideas
against a common suite of flows.

The DNS reference data comes from the Johns Hopkins Turbulence Database (JHTDB)
transitional boundary layer and the Sillero/Jiménez zero-pressure-gradient
turbulent boundary layer.

## Reproducing

```sh
calkit run
```

## Benchmarking a closure

Write a closure class, register it, and it is solved and scored against every
case in the suite, with in-sample and out-of-sample results separated by
construction:

```sh
calkit run run-benchmark        # or: uv run --project envs/compute \
                                #       python scripts/run-benchmark.py
```

See [docs/rans-gym.md](docs/rans-gym.md) for how to add a closure or a case.

## Layout

| path | what it is |
|---|---|
| `pypkg/` | solvers, closures, and the benchmark registry |
| `pypkg/cases/` | benchmark cases, one module each |
| `scripts/` | pipeline stages |
| `scripts/standalone/` | **not** pipeline stages; token-gated or heavy deps |
| `sim/` | OpenFOAM cases and custom turbulence models |
| `envs/` | uv environments: `compute`, `viz`, `notebook` |
| `docs/` | design notes, ideas log, literature review, roadmap |
| `paper/` | the manuscript |

## Environments

Python environments are uv projects on Python 3.14, split so that adding a
plotting package does not invalidate expensive numerical results:

- `envs/compute` — numerics; every fit, search and benchmark depends on it
- `envs/viz` — the same numerics plus plotting
- `envs/notebook` — Jupyter, for `notebooks/main.ipynb`

`pypkg` is installed editable into each, so stages depend on the specific
modules they use rather than on the whole package.

OpenFOAM runs in the `blsim` Docker environment, built from `sim/Dockerfile`.
