# Repository Guidelines

## Project Structure & Module Organization
This repository collects several encrypted-traffic classification implementations by model. `YaTC/refactor/` is the most modernized package and contains `config.py`, `models.py`, `data.py`, `engine.py`, `train.py`, and `tests.py`; `YaTC/github/` keeps the upstream variant. `FS-Net/`, `AppScanner/`, and `DeepFingerprinting/` are independent model directories with their own scripts, plus local `data/`, `output/`, and `checkpoints/` folders. Keep generated artifacts inside those module folders and avoid committing large files from `output/`, `checkpoints/`, `temp/`, or dataset directories.

## Build, Test, and Development Commands
There is no single root build step; install dependencies per module.

`pip install -r FS-Net/requirements.txt` installs FS-Net dependencies.

`pip install -r AppScanner/requirements.txt` installs AppScanner dependencies.

`cd YaTC/refactor && pip install -e .[dev]` installs YaTC plus `pytest`, `black`, `flake8`, and `mypy`.

Run training from the target module directory, for example:

`python train.py pretrain --data_path ./data` in `YaTC/refactor`

`python run_train.py` in `FS-Net`

`python train.py --mode train --data_dir ./data/apps` in `AppScanner`

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for modules and functions, and `PascalCase` for classes. Keep configuration in `config.py`, model definitions in `models.py`, and training entry points in `train.py` or `run_train.py` to match the existing layout. Preserve type hints and docstring style where they already exist. In `YaTC/refactor`, follow the configured `black`, `flake8`, and `mypy` rules, including the 100-character line limit.

## Testing Guidelines
Prefer targeted verification before broader runs. Existing unit tests live in `FS-Net/tests.py` and `YaTC/refactor/tests.py`; run `python -m pytest tests.py -v` from the relevant directory. Add regression coverage next to the code you change, using `tests.py` or `test_*.py` naming that matches `YaTC/refactor/pyproject.toml`. For data-heavy changes, record a minimal reproducible command when full training is too expensive.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, often written in Chinese. Keep messages specific, avoid empty summaries, and separate unrelated changes into different commits. Pull requests should identify the affected model, note dataset assumptions, list the commands you ran, and summarize any metric, dependency, or output-path changes. Update `README.md` or the relevant module README when usage or layout changes, and keep `CODEX_TASK_QUEUE.md` current while work is in progress.
