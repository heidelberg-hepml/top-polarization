# Top polarization

The codebase is forked from the [LloCa repository](https://github.com/heidelberg-hepml/lloca-experiments.git). See their README for more info about the codebase.

## Repository Layout
- `experiments/`: experiment base classes, data handling, embedding,
  - `experiments/tagging`: tagging experiment files
- `config/`: Hydra configuration trees for experiments and models
- `runs/`: default output location for trained checkpoints, configs, plots,
  and optional MLflow metadata.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
### Pre-commit hooks
We use [pre-commit](https://pre-commit.com/) to run hooks before each commit. To install the hooks, run:
```
pre-commit install
pre-commit run --all-files # Optional, to test that it works
```
The only hook that is currently active is [ruff](https://github.com/astral-sh/ruff), which performs linting and formatting. The hook is configured in `.pre-commit-config.yaml` and ruff is configured in `pyproject.toml`.
When committing, ruff will automatically fix any issues it can and report the rest. You then have to fix the issues and commit again.

## Running Experiments

Runs parameters are set via Hydra configs. There are different configuration files for each experiment type.
```bash
python run.py --config-name tagging
```

### Individual runs
```bash
python run.py -cn tagging \
    exp_name=test_tagging \
    run_name=lgatr_200k \
    data.dataset=mini \
    training.iterations=200000 \
    model=tag_lgatr \
    training=tag_lgatr
```
The model config is loaded from `config/model/<model_name>.yaml`. It has to correspond to the chosen experiment. The training config is loaded from `config/training/<training_name>.yaml` and should correspond to the chosen experiment and model.

Outputs are stored in `runs/<exp_name>/<run_name>/`, including:
- `config.yaml`, `config_<run_idx>.yaml`: frozen configs.
- `models/model_run*.pt`: checkpoints (model, optimizer, EMA, scheduler).
- `out_<run_idx>.log`: aggregated log.
- `plots/`: PDFs summarising losses and chosen observables.
- `source.zip` (optional): zipped source code at the beginning of the run.

## Test dataset

There is an additional test dataset and test experiment that can be started with the test config, for lgatr debugging. It does not work yet for gatr but works with a transformer.
