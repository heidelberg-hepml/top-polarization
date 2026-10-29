# Reproducing the paper runs

Commands to train and evaluate the networks behind the paper figures: the down tagger (tt_pol),
the VBF toy (`vbf_pseudoscalar`, `vbf_cp`) and tt_smeft (`lbl_0=sm`, `lbl_0=neg`), evaluated on
both the truth-labelled test split and the test split relabelled by the down tagger.

L-GATr-slim runs use seeds 0-4, L-GATr runs seed 0. tt_smeft takes the down tag as one scalar on the non-b subjets and their constituents, the
probability that the subjet is the down quark: `data.down_tag=truth` (default) is the true one,
`hard` / `soft` the rounded / raw probability of a down tagger, read from `data.downtag_labels`.

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

For reproducibility, set the seed at the start of your script (and change it for each run) with:

```bash
SEED=0
```

Every command in steps 1-3 can be done with `L-GATr` instead of `L-GATr-slim` by adding `model=tag_lgatr training=tag_lgatr`.

## 1. Down tagger

L-GATr-slim:
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

This needs the slim down tagger from step 1 and both tt_smeft runs from step 3.

```bash
TTSMEFT_NEG_RUN_DIR=runs/tt_smeft/neg_slim_s${SEED}
TTSMEFT_SM_RUN_DIR=runs/tt_smeft/sm_slim_s${SEED}
```

**Step 1: saving the predicted labels.**

Neg. vs Pos.
```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=neg
```

SM vs Pos.
```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=sm
```

`${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_<TASK>/labels.txt` holds the predicted down tags.

**Step 2: tt_smeft on the relabelled test split.**

Neg. vs Pos.
```bash
python run.py -cp ${TTSMEFT_NEG_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.down_tag=hard \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_neg/labels.txt
```

SM vs Pos.
```bash
python run.py -cp ${TTSMEFT_SM_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.down_tag=hard \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_sm/labels.txt
```

Results go to `{preds,plots}_0_downtag/` in each tt_smeft run directory.

## 5. tt_smeft Neg. vs Pos. at Im(d_tG) = 0.1

The Neg. vs Pos. network trained at Im(d_tG) = 2.6 (step 3), evaluated on all Im(d_tG) = +-0.1
events (none of them is trained on) with the truth and the predicted down tag. The down tagger
first labels all events (`data.ttsmeft.split=all`).

```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=neg data.ttsmeft.c_value=0.1 \
    data.ttsmeft.split=all

LABELS=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_neg_c0p1_all/labels.txt
python run.py -cp ${TTSMEFT_NEG_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] data.c_value=0.1 \
    data.eval_all=true

python run.py -cp ${TTSMEFT_NEG_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] data.c_value=0.1 \
    data.eval_all=true data.down_tag=hard data.downtag_labels=${LABELS}
```

Results go to `{preds,plots}_0_c0p1_all/` and `{preds,plots}_0_downtag_c0p1_all/` in the tt_smeft
run directory. The asymmetries (`paper_plots/tt_smeft_posneg.py`) pool the +c and -c samples:
N+ counts O > 0 at +c and O < 0 at -c.

## 6. tt_smeft Neg. vs Pos. trained on the soft down tag

The down-tagger probability replaces the true down tag in training and evaluation, so the
down tagger first labels all events.

```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=neg data.ttsmeft.split=all

python run.py -cp config -cn tt_smeft data.lbl_0=neg \
    seed=${SEED} run_name=neg_slim_softdown_s${SEED} data.down_tag=soft \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_neg_all/labels.txt
```

## 7. O1 asymmetry

O1 on all events of a tt_smeft sample, with the true down quark and with the down-tagger
probability of every L-GATr-slim down tagger. The down taggers label all events of each sample
first (steps 5 and 6 cover `neg_c0p1_all` and `neg_all`).

```bash
python run.py -cp ${DOWNTAGGING_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=ttsmeft data.ttsmeft.lbl_0=sm data.ttsmeft.split=all
```

```bash
python -m paper_plots.tt_smeft_o1 --lbl_0 neg --c_value 2.6 \
    --labels runs/downtagging/down_slim_s*/preds_0_ttsmeft_neg_all/labels.txt
python -m paper_plots.tt_smeft_o1 --lbl_0 sm --c_value 2.6 \
    --labels runs/downtagging/down_slim_s*/preds_0_ttsmeft_sm_all/labels.txt
python -m paper_plots.tt_smeft_o1 --lbl_0 neg --c_value 0.1 \
    --labels runs/downtagging/down_slim_s*/preds_0_ttsmeft_neg_c0p1_all/labels.txt
python -m paper_plots.tt_smeft_o1 --plot
```
