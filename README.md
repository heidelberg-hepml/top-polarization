Some elements from the old README as temporary explanations


# Top polarization

The codebase is forked from the [high-dim-unfolding repository](https://github.com/heidelberg-hepml/high-dim-unfolding.git) and uses the [`lgatr` library](https://github.com/heidelberg-hepml/lgatr.git) for L-GATr blocks.

## Highlights
- Use of `pytorch-geometric` graph objects for variable-length events with jets, constituents, scalar information at detector-level and particle-level.
- Lorentz-equivariant models via the `lgatr` library.

## Repository Layout
- `experiments/`: experiment base classes, data handling, embedding,
  conditional flow-matching models, utilities, and baseline networks.
  - `experiments/baselines`: baseline models, as opposed to the models imported from the `lgatr` package
  - `experiments/multiplicity`: multiplicity experiment files
- `config/`: Hydra configuration trees for experiments and models
- `runs/`: default output location for trained checkpoints, configs, plots,
  and optional MLflow metadata.

## Setup
1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Ensure the PyTorch build matches your CUDA toolkit.  Installing `xformers`
   may require wheels specific to your platform.

2. **Datasets**
   Place the datasets under `data/`. Existing configs
   expect the EnergyFlow `zplusjet` dataset or our generated top dataset, available upon request.
   See `experiments/dataset.py` to add new datasets.

## Running Experiments

Runs parameters are set via Hydra configs. There are different configuration files for each experiment type.
```bash
python run.py --config-name multiplicity
```

### Individual runs
```bash
python run.py -cn multiplicity \
    exp_name=test_mult \
    run_name=lgatr_200k \
    data.dataset=zplusjet \
    training.iterations=200000 \
    model=lgatr_multiplicity
```
The model config is loaded from `config/model/<model_name>.yaml`. It has to correspond to the chosen experiment.

Outputs are stored in `runs/<exp_name>/<run_name>/`, including:
- `config.yaml`, `config_<run_idx>.yaml`: frozen configs.
- `models/model_run*.pt`: checkpoints (model, optimizer, EMA, scheduler).
- `out_<run_idx>.log`: aggregated log.
- `plots/`: PDFs summarising losses and chosen observables.
- `source.zip` (optional): zipped source code at the beginning of the run.