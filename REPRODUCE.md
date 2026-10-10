# Reproducing the paper runs

Commands to train and evaluate the networks behind the paper figures: the down tagger (tt_pol),
the VBF toy (`vbf_pseudoscalar`, `vbf_cp`) and tt_smeft (`lbl_0=sm`, `lbl_0=neg`), evaluated on
both the truth-labelled test split and the test split relabelled by the down tagger.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python data/data_scripts/fetch_data.py --process tt_pol
python data/data_scripts/fetch_data.py --process tt_smeft
python data/data_scripts/fetch_data.py --process vbf
```

Every setting not given on the command line comes from `config/` (e.g. `training=tag_slim`:
Lion, lr 3e-5, weight decay 2, batch size 128, cosine schedule, 200k iterations; tt_smeft:
`bsm=im`, `c_value=2.6`, `tag_down=true`). The frozen config of each run is written to
`runs/<exp_name>/<run_name>/config.yaml`.

## 1. Down tagger

L-GATr-slim:
```bash
python run.py -cp config -cn downtagging run_name=slim
```

L-GATr (same training settings as L-GATr-slim):
```bash
python run.py -cp config -cn downtagging run_name=lgatr \
    model=tag_lgatr training=tag_slim
```

Every command in steps 1-3 runs with L-GATr by adding `model=tag_lgatr training=tag_slim`. The paper
uses seeds 0-4 (`seed=<k>`) for L-GATr-slim and seed 0 for L-GATr.

## 2. VBF toy

Neg. vs Pos.
```bash
python run.py -cp config -cn vbf run_name=pseudoscalar data.exp_type=pseudoscalar
```

SM vs Pos.
```bash
python run.py -cp config -cn vbf run_name=cp data.exp_type=cp
```

## 3. tt_smeft

Neg. vs Pos.
```bash
python run.py -cp config -cn tt_smeft run_name=neg data.lbl_0=neg
```

SM vs Pos.
```bash
python run.py -cp config -cn tt_smeft run_name=sm data.lbl_0=sm
```

## 4. tt_smeft evaluated with the predicted down tag

This needs the slim down tagger from step 1 and both tt_smeft runs from step 3.

```bash
DOWNTAGGING_RUN_DIR=runs/downtagging/slim # change to the path to the run
TTSMEFT_NEG_RUN_DIR=runs/tt_smeft/neg
TTSMEFT_SM_RUN_DIR=runs/tt_smeft/sm
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
    data.dataset_eval=downtag \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_neg/labels.txt
```

SM vs Pos.
```bash
python run.py -cp ${TTSMEFT_SM_RUN_DIR} -cn config \
    train=false model_load_idx=0 evaluation.eval_set=[test] \
    data.dataset_eval=downtag \
    data.downtag_labels=${DOWNTAGGING_RUN_DIR}/preds_0_ttsmeft_sm/labels.txt
```

Results go to `{preds,plots}_0_downtag/` in each tt_smeft run directory.
