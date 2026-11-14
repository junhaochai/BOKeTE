# BOKeTE

[![PyPI Version](https://img.shields.io/pypi/v/BOKeTE.svg)](https://pypi.org/project/BOKeTE/)
[![Licence: All rights reserved](https://img.shields.io/badge/Licence-All_rights_reserved-blue.svg)](LICENSE)

Personal PyTorch training and experimentation helpers, extracted from my
2024/2025 computer vision final year project at the University of Nottingham.
Named after Bad Bunny's *BoKeTe*.

<p align="center">
  <img src="https://raw.githubusercontent.com/junhaochai/BoKeTE/master/assets/BOKeTE.png" alt="BOKeTE music video homage" width="600" />
</p>

- **Training:** mixed precision, gradient clipping, early stopping and full-state checkpoint resume.
- **Experiments:** YAML configuration, repeated trials, grid sweeps and interruption handling.
- **Reporting:** metric tracking, loss curves and Markdown reports for trials and sweeps.

## Installation

Requires Python 3.10+ and PyTorch 2.0+.

```bash
pip install bokete
```

For local development, run `uv pip install -e .` from the repository root.
Editable installation uses your working copy directly.

## Quick start

The [starter template](src/bokete/templates/starter/main.py) trains an MLP on
MNIST and generates experiment reports. Generate a working project from the
repository root:

```bash
uv run bokete init my-experiment
cd my-experiment
uv run --with-editable .. python main.py --config example
```

`--with-editable ..` uses BOKeTE from this checkout. With a published BOKeTE
installation, use `bokete init my-experiment` and omit `--with-editable ..`
when running the generated project. `--config example` loads its
`configs/example.yaml`.

Keep the packaged template unchanged; run experiments in the generated project.

The supplied configuration runs a parameter sweep with three trials per
combination. Remove `param_grid` to run one configuration; set `trials: 1`
for a single trial.

## Documentation

See the [documentation](docs/index.md) for getting started, usage guides and
the generated API reference. Preview the documentation website locally:

```powershell
uv run --group docs mkdocs serve
```
