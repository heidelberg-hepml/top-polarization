# Reproducing the paper runs

Commands to train and evaluate the networks behind the paper figures: the down tagger (tt_pol),
the VBF toy (`vbf_pseudoscalar`, `vbf_cp`) and tt_smeft (`lbl_0=sm`, `lbl_0=neg`), evaluated with
the true down quark (emulated dileptonic) and with the down-tagger probability (semileptonic).

All networks are L-GATr-slim. Steps 1-5 run once per seed, the paper uses seeds 0-4. tt_smeft takes
the down tag as one scalar on the non-b subjets and their constituents, the probability that the
subjet is the down quark: `data.down_tag=truth` (default) is the true one, `mix` weights the outputs
of the two assignments with the probability of a down tagger, read from `data.downtag_labels`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python data/data_scripts/fetch_data.py --process tt_pol
python data/data_scripts/fetch_data.py --process tt_smeft
python data/data_scripts/fetch_data.py --process vbf
```

Every setting not given on the command line comes from `config/`.
The frozen config of each run is written to `runs/<exp_name>/<run_name>/config.yaml`.

Set the seed at the start of your script, and change it for each run:

```bash
SEED=0
```

## 1. Down tagger

```bash
python run.py -cp config -cn downtagging \
    seed=${SEED} run_name=down_slim_s${SEED}
```

Evaluate on polarized data:

```bash
DOWNTAGGING_RUN_DIR=runs/downtagging/down_slim_s${SEED}
```

```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=polarized
```

## 2. VBF toy

Neg. vs Pos.
```bash
python run.py -cp config -cn vbf data.exp_type=pseudoscalar \
    seed=${SEED} run_name=pseudoscalar_slim_s${SEED}
```

SM vs Pos.
```bash
python run.py -cp config -cn vbf data.exp_type=cp \
    seed=${SEED} run_name=cp_slim_s${SEED}
```

## 3. tt_smeft

Neg. vs Pos.
```bash
python run.py -cp config -cn tt_smeft data.lbl_0=neg \
    seed=${SEED} run_name=neg_slim_s${SEED}
```

SM vs Pos.
```bash
python run.py -cp config -cn tt_smeft data.lbl_0=sm \
    seed=${SEED} run_name=sm_slim_s${SEED}
```

## 4. tt_smeft evaluated with the predicted down tag

This needs the slim down tagger from step 1 and the tt_smeft runs from step 3. The semileptonic
output is O = p O(d = q_hard) + (1 - p) O(d = q_soft) for the scalar and pseudoscalar logits,
with p the down-tagger probability of the harder subjet. `data.down_tag=mix` evaluates the
network with the wrong down tag and mixes it with the outputs of the true one, read from the
truth evaluation of the same events (`preds_0/` of the training run).

```bash
TTSMEFT_NEG_RUN_DIR=runs/tt_smeft/neg_slim_s${SEED}
TTSMEFT_SM_RUN_DIR=runs/tt_smeft/sm_slim_s${SEED}
```

The down tagger labels the tt_smeft test events:

```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=neg

python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=sm
```

Neg. vs Pos.
```bash
python run.py -cp ${TTSMEFT_NEG_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.down_tag=mix \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_neg/labels.txt
```

SM vs Pos.
```bash
python run.py -cp ${TTSMEFT_SM_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.down_tag=mix \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_sm/labels.txt
```

Results go to `{preds,plots}_0_mixdown/` in each tt_smeft run directory.

## 5. tt_smeft Neg. vs Pos. at Im(c_tG) = 0.1

The Neg. vs Pos. network trained at Im(c_tG) = 2.6 (step 3), evaluated on all Im(c_tG) = +-0.1
events (none of them is trained on) with the truth and the mixed down tag. The down tagger
first labels all events (`data.ttsmeft.split=all`).

```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=neg data.ttsmeft.c_value=0.1 \
    data.ttsmeft.split=all

python run.py -cp ${TTSMEFT_NEG_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] data.c_value=0.1 \
    data.eval_all=true

python run.py -cp ${TTSMEFT_NEG_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] data.c_value=0.1 \
    data.eval_all=true data.down_tag=mix \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_neg_c0p1_all/labels.txt
```

Results go to `{preds,plots}_0_c0p1_all/` and `{preds,plots}_0_mixdown_c0p1_all/` in the tt_smeft
run directory.

## 6. O1 asymmetry

O1 on all events of a tt_smeft sample, with the true down quark and with the probability of the
down tagger with the smallest validation loss. All down taggers first label all events of each
sample (step 5 covers `neg_c0p1_all`):

```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
        train=false model_load_idx=0 evaluation.eval_set=[test] \
        data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=neg data.ttsmeft.split=all
```

Once all seeds are done, this prints the asymmetries and draws Fig. 5:

```bash
python -m paper_plots.tt_smeft_o1 --lbl_0 neg --c_value 2.6 \
    --labels runs/downtagging/down_slim_s*/preds_0_ttsmeft_neg_all/labels.txt
python -m paper_plots.tt_smeft_o1 --lbl_0 neg --c_value 0.1 \
    --labels runs/downtagging/down_slim_s*/preds_0_ttsmeft_neg_c0p1_all/labels.txt
python -m paper_plots.tt_smeft_o1 --plot
```

## 7. Paper figures

From the runs of all seeds, written to `paper_plots/<task>/`. The scripts also print the AUCs,
`downtagging` the spin analyzing powers and `tt_smeft_posneg` the pooled asymmetries of the
pseudoscalar output. The training curves come from the run with the smallest validation loss, so
they change with the seeds.

```bash
python -m paper_plots.downtagging runs/downtagging/down_slim_s*
python -m paper_plots.vbf_pseudoscalar runs/vbf/pseudoscalar_slim_s*
python -m paper_plots.vbf_cp runs/vbf/cp_slim_s*
python -m paper_plots.tt_smeft_sm runs/tt_smeft/sm_slim_s*
python -m paper_plots.tt_smeft_posneg runs/tt_smeft/neg_slim_s*
```

| Figure | Files |
| --- | --- |
| 1 | `downtagging/roc.pdf`, `downtagging/calibration.pdf` |
| 2 | `downtagging/polarization.pdf`, `downtagging/polarization_efficiency.pdf` |
| 3 | `vbf_cp/output_scalar.pdf`, `vbf_cp/output_pseudo.pdf` |
| 4 | `vbf_pseudoscalar/output_scalar.pdf`, `vbf_pseudoscalar/output_pseudo.pdf` |
| 5 | `tt_smeft/tt_smeft_2p6_O1_delphes.pdf`, `tt_smeft/tt_smeft_2p6_O1_downtag_delphes.pdf` |
| 6 | `tt_smeft/output_scalar.pdf`, `tt_smeft/output_pseudo.pdf` |
| 7 | `tt_smeft_posneg/output_scalar.pdf`, `tt_smeft_posneg/output_pseudo.pdf` |
| 8 | `tt_smeft_posneg/score.pdf`, `tt_smeft_posneg/score_downtag.pdf` |

The GNN curve of `polarization_efficiency.pdf` reads `data/beta_opt_GNN.csv` with the columns
`efficiency`, `beta_opt_tL` and `beta_opt_tR`, the result of arXiv:2407.01663. The file is not
distributed, and the curve is skipped without it.
